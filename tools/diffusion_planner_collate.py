import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class DiffusionPlannerCollate:
    """DiffusionPlannerData 샘플들을 배치 텐서로 묶는 collate_fn.

    각 샘플은 agent / lane / route / static 개수에 상한이 걸려 있고,
    이 collate_fn 은
      1) (선택) 중심 기준 거리 크로핑
      2) 배치 내 최대 길이 계산
      3) 그 길이에 맞춰 0 패딩
    순서로 고정 shape 배치를 만든다.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """collate 설정을 초기화한다.

        Args:
            args (argparse.Namespace):
                - caching_max_agent_num (int)
                - caching_max_lane_num (int)
                - caching_max_static_num (int)
        """
        self.args = args


    _FIXED_STACK_EGO_KEYS: Tuple[str, ...] = (
        "ego_agent_past", # (time_len, 11)
        "ego_future_gt_3_dim", # (future_len, 3)
        "planner_future_11_dim", # (future_len, 11)
        "ego_agent_past_is_valid", # (time_len,)
        "ego_future_gt_is_valid", # (future_len,)
    )

    def _is_string_container_value(self, value: Any) -> bool:
        """값이 '문자열(또는 문자열 묶음)'인지 판별합니다.

        목적
        ----
        배치를 만들 때 숫자/배열 데이터는 torch.Tensor로 묶어야 하지만,
        문자열(str)은 torch.Tensor로 바꿀 수 없어 에러가 납니다.
        그래서 문자열은 "그대로 리스트로 모아서" 반환하기 위해 미리 구분합니다.

        문자열로 취급하는 케이스
        ----------------------
        1) value가 str(또는 numpy의 문자열 타입)인 경우
           - 예: "scenario_000123"  # shape: ()
        2) value가 list/tuple이고, 안의 원소가 전부 str인 경우
           - 예: ["a", "b"]  # 원소 개수는 샘플마다 다를 수 있음
        3) value가 numpy 배열이고 dtype이 문자열인 경우
           - 예: np.array(["a","b"])  # shape: (N,)
           - 예: np.array("abc")      # shape: ()

        Args:
            value: 샘플 dict 안의 어떤 값.

        Returns:
            bool:
                - True: 문자열(또는 문자열 묶음)
                - False: 숫자/배열 등(텐서로 묶을 수 있는 쪽)
        """
        if value is None:
            return False

        # 1) 단일 문자열
        if isinstance(value, (str, np.str_)):
            return True

        # 2) list/tuple 안이 전부 문자열인 경우
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return False
            return all(isinstance(x, (str, np.str_)) for x in value)

        # 3) numpy 배열이 문자열 dtype인 경우
        if isinstance(value, np.ndarray):
            # dtype.kind:
            #   'U' unicode string, 'S' byte string, 'O' object
            if value.dtype.kind in ("U", "S"):
                return True
            if value.dtype.kind == "O" and value.size > 0:
                # object 배열이라도 실제가 문자열이면 문자열로 취급
                first = value.flat[0]
                return isinstance(first, (str, np.str_))

        return False

    def _should_return_string_list_for_key(self, key: str,
                                           values: List[Any]) -> bool:
        """특정 key가 '문자열 배치(List[str])'로 반환되어야 하는지 결정합니다.

        규칙
        ----
        - values(길이 B) 중 None이 아닌 값들이 전부 문자열 계열이면:
            -> 이 key는 텐서로 묶지 않고 List[str]로 반환합니다.
        - 문자열 값과 숫자/배열 값이 섞여 있으면:
            -> 데이터 자체가 섞인 상태라 이후 로직이 예측 불가능해지므로
               명확히 에러를 냅니다.

        Args:
            key (str): 배치 dict의 key 이름. shape: ()
            values (List[Any]): 길이 B의 샘플 값 목록.
                - 문자열이면 보통: str (shape: ())
                - 또는 문자열 묶음(list[str], np.ndarray[str])도 가능

        Returns:
            bool:
                - True: 이 key는 List[str]로 반환
                - False: 이 key는 기존대로 torch.Tensor/None으로 반환

        Raises:
            ValueError: 문자열 값과 숫자/배열 값이 같은 key에서 섞여 있을 때
        """
        non_none = [v for v in values if v is not None]
        if len(non_none) == 0:
            return False

        is_str_flags = [self._is_string_container_value(v) for v in non_none]
        if all(is_str_flags):
            return True

        if any(is_str_flags) and (not all(is_str_flags)):
            raise ValueError(
                f"[Collate] key='{key}' 에 문자열 값과 숫자/배열 값이 섞여 있습니다. "
                f"한 key는 한 종류로만 들어오게 정리해야 합니다.")

        return False

    def _collate_string_values_to_list(self, values: List[Any]) -> List[str]:
        """문자열(또는 문자열 묶음)을 배치 단위 List[str]로 모아 반환합니다.

        반환 형태
        --------
        - 항상 길이 B의 리스트를 반환합니다.
        - 각 원소는 str 입니다.
        - None은 빈 문자열 "" 로 바꿉니다.

        예시
        ----
        values = ["a", None, "c"]  -> ["a", "", "c"]

        Args:
            values (List[Any]):
                길이 B 리스트.
                각 원소는 보통 str(shape: ()) 또는 None 입니다.

        Returns:
            List[str]:
                길이 B의 문자열 리스트.
                - length: B
        """
        out: List[str] = []
        for v in values:
            if v is None:
                out.append("")
            elif isinstance(v, (str, np.str_)):
                out.append(str(v))
            else:
                # list[str], np.ndarray[str] 같은 경우도 "문자열 1개"로 만들어 담습니다.
                # (요구사항이 List[str] 이므로, 샘플 1개당 문자열 1개로 표현)
                out.append(str(v))
        return out

    def _infer_max_sizes_for_agent_route_lane_order(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """agent_route_lane_order를 패딩하기 위한 (최대 agent 수, 최대 lane 수)를 계산합니다.

        왜 이 함수가 필요한가
        --------------------
        agent_route_lane_order는 (행=agent, 열=lane) 관계를 담고 있습니다.
        현재 로직에서는 agent_route_lane_order의 "행"이
        neighbor 전체가 아니라 near agent 기준으로 잘려서 들어올 수 있습니다.

        그래서 이 key를 배치로 패딩할 때,
        agent 축 최대값(max_agent_num)은 neighbor_agents_past 같은 다른 텐서가 아니라
        **agent_route_lane_order 자체의 행 개수**만 보고 정하는 것이 안전합니다.
        (그래야 agent_route_lane_order_is_valid 같은 agent 축 마스크와도 길이가 맞습니다.)

        lane 축(max_lane_num)은 lanes.shape[0]와 agent_route_lane_order.shape[1] 중
        더 큰 값을 사용해, lane 텐서 패딩과도 잘 맞도록 합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            Tuple[int, int]:
                - max_agent_num: shape (), int
                    agent_route_lane_order의 행 수(=agent 수) 최대값.
                - max_lane_num: shape (), int
                    lanes의 lane 개수 또는 agent_route_lane_order의 열 수 최대값.
        """
        max_agent_num: int = 0
        max_lane_num: int = 0

        for sample in batch:
            aro = sample.get("agent_route_lane_order", None)
            if aro is not None:
                aro_arr = np.asarray(aro)  # (agent_num, lane_num)
                # aro_arr: (A_i, L_i)
                if aro_arr.ndim == 2:
                    max_agent_num = max(max_agent_num, int(aro_arr.shape[0]))
                    max_lane_num = max(max_lane_num, int(aro_arr.shape[1]))

            lanes = sample.get("lanes", None)
            if lanes is not None:
                lanes_arr = np.asarray(lanes)  #
                # lanes_arr: (L_i, lane_len, feat) 또는 최소 (L_i, ...)
                if lanes_arr.ndim >= 1:
                    max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))

        return int(max_agent_num), int(max_lane_num)

    def _get_special_padding_category_spec(
        self,
        key: str,
    ) -> Optional[Tuple[int, int]]:
        """특정 key에 대해, '패딩을 어떤 카테고리로 채울지' 규칙을 돌려줍니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / left_line_type / right_line_type 같은 값은
        보통 (카테고리 개수)만큼의 길이를 가진 벡터(대부분 one-hot)로 들어옵니다.

        그런데 패딩을 전부 0으로 채우면,
        나중에 argmax 같은 방식으로 카테고리를 뽑을 때
        "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.

        그래서 아래 key들만은 패딩을 0이 아니라
        '모름/미정'에 해당하는 카테고리로 채우도록 규칙을 따로 둡니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()

        Returns:
            Optional[Tuple[int, int]]:
                - (unknown_category_index, num_categories)
                - lane_type: (3, 4)  -> (L, 4)에서 마지막(3번)을 1로 채움
                - left/right_line_type: (11, 13) -> (L, 13)에서 11번을 1로 채움
                - 그 외 key는 None
        """
        key_str = str(key)
        if key_str == "lane_type":
            return 3, 4
        if key_str in ("left_line_type", "right_line_type"):
            return 11, 13
        return None

    def _infer_max_lane_num_from_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> int:
        """배치에서 'lane 개수의 최대값'을 안전하게 구합니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / line_type 같은 값이 어떤 샘플에서는 None일 수 있습니다.
        그런데 그 샘플의 lanes 개수는 클 수 있어서,
        단순히 non-None 값들만 보고 max 길이를 정하면
        lanes 텐서와 lane_type 텐서의 lane 차원이 달라질 수 있습니다.

        그래서 lane 관련 key들은 lane 차원을 항상 lanes 기준으로 맞추기 위해
        batch 전체의 lanes.shape[0] 최대값을 구합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            int:
                - max_lane_num: 배치 내 lanes의 첫 번째 축 최대값. shape: ()
        """
        max_lane_num: int = 0
        for sample in batch:
            lanes_value = sample.get("lanes", None)
            if lanes_value is None:
                continue
            lanes_arr = np.asarray(lanes_value)
            if lanes_arr.ndim >= 1:
                max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))
        return int(max_lane_num)

    def _build_default_padded_tensor_for_key(
        self,
        key: str,
        out_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """key 성격에 맞는 '기본 패딩 텐서'를 만듭니다.

        기본 규칙
        --------
        - 대부분의 key: 0으로 채운 텐서를 만듭니다.
        - lane_type: (…, 4) 형태라면 마지막 축에서 3번 인덱스만 1로 채웁니다.
            즉, [0,0,0,1] 형태의 "미정" one-hot 패딩입니다.
        - left_line_type / right_line_type: (…, 13) 형태라면 마지막 축에서 11번만 1로 채웁니다.
            즉, "선은 있으나 타입 모름" one-hot 패딩입니다.

        주의
        ----
        - 만약 어떤 이유로 one-hot 형태가 아니라 (…, ) 라벨 형태로 들어온다면,
          그 경우는 텐서 전체를 unknown 카테고리 숫자(예: 3, 11)로 채웁니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()
            out_shape (Tuple[int, ...]):
                출력 텐서 shape.
                예:
                  - lane_type: (B, L_max, 4)
                  - left_line_type: (B, L_max, 13)
            dtype (torch.dtype):
                출력 텐서 dtype.

        Returns:
            torch.Tensor:
                out: out_shape 그대로의 텐서.
        """
        spec = self._get_special_padding_category_spec(key)
        if spec is None:
            return torch.zeros(out_shape, dtype=dtype)

        unknown_index, num_categories = int(spec[0]), int(spec[1])

        # out_shape가 충분히 길고, 마지막 축이 카테고리 개수와 맞으면 one-hot 패딩
        if len(out_shape) >= 2 and int(out_shape[-1]) == int(num_categories):
            out = torch.zeros(out_shape, dtype=dtype)
            if 0 <= unknown_index < num_categories:
                out[..., unknown_index] = torch.as_tensor(1, dtype=dtype)
            return out

        # one-hot이 아니라 라벨 형태(예: (B, L_max))로 들어오는 경우를 대비한 fallback
        if len(out_shape) >= 2:
            return torch.full(out_shape, fill_value=unknown_index, dtype=dtype)

        # 기대 형태가 너무 이상하면 안전하게 0으로
        return torch.zeros(out_shape, dtype=dtype)

    def _stack_fixed(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """모든 샘플에서 shape가 같은 키를 (B, ...) 텐서로 쌓는다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.
            key (str): 예: "ego_agent_past".
            dtype (torch.dtype): 출력 dtype.

        Returns:
            torch.Tensor: shape (B, *sample_shape)
        """
        return torch.stack(
            [torch.as_tensor(sample[key], dtype=dtype) for sample in batch],
            dim=0,
        )

    # ------------------------------------------------------------------
    # 2) (N_i, ...) → (B, target_len, ...) 패딩
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 3) (N_i, M_i) → (B, target0, target1) 패딩
    # ------------------------------------------------------------------



    def _is_collatable_value(self, value: Any) -> bool:
        """이 값이 '텐서로 묶을 수 있는 값'인지 판별합니다.

        변경 포인트
        ----------
        - 문자열(str) 또는 문자열 배열/리스트는 텐서로 묶을 수 없으므로 False
          (대신 별도 경로로 List[str]로 모아서 반환합니다)

        Args:
            value: 샘플 dict 안의 어떤 값.

        Returns:
            bool:
                - True: torch.Tensor로 묶을 수 있는 값
                - False: 문자열 계열(또는 텐서화 불가 값)
        """
        if value is None:
            return True

        # ✅ 문자열 계열은 텐서로 묶지 않는다.
        if self._is_string_container_value(value):
            return False

        if isinstance(value, torch.Tensor):
            return True

        if isinstance(value, np.ndarray):
            # numpy 배열도 문자열 dtype이면 제외
            if value.dtype.kind in ("U", "S"):
                return False
            if value.dtype.kind == "O" and value.size > 0 and isinstance(
                    value.flat[0], (str, np.str_)):
                return False
            return True

        if isinstance(value, (bool, int, float, np.number)):
            return True

        if isinstance(value, (list, tuple)):
            # list/tuple이라도 전부 문자열이면 제외
            if len(value) > 0 and all(
                    isinstance(x, (str, np.str_)) for x in value):
                return False
            return True

        return False

    def _get_value_shape(self, value: Any) -> Tuple[int, ...]:
        """입력 값의 shape를 튜플로 얻습니다.

        Args:
            value:
                - torch.Tensor 또는 numpy.ndarray 또는 숫자 리스트/튜플/스칼라.

        Returns:
            Tuple[int, ...]:
                - 예: (A, T, 11), (L, P, 2), (A, L), (T,), ()(스칼라)
        """
        if isinstance(value, torch.Tensor):
            return tuple(int(x) for x in value.shape)
        if isinstance(value, np.ndarray):
            return tuple(int(x) for x in value.shape)

        # 리스트/튜플/스칼라 등: numpy로 shape만 확인
        arr = np.asarray(value)
        return tuple(int(x) for x in arr.shape)

    def _choose_output_dtype(self, value: Any) -> torch.dtype:
        """배치 텐서를 만들 때 사용할 dtype을 결정합니다.

        규칙(속도/안정성 목적)
        -------------------
        - 입력이 float 계열이면: torch.float32 로 통일 (float64 방지)
        - bool은: torch.bool 유지
        - 정수는: torch dtype 그대로 유지(보통 int64)

        Args:
            value:
                - None이 아닌 샘플 값(배치 내 최소 1개는 실제 값이 있어야 dtype을 정할 수 있음)

        Returns:
            torch.dtype:
                - 출력 텐서의 dtype
        """
        t = torch.as_tensor(value)
        if t.is_floating_point():
            return torch.float32
        return t.dtype

    def _collect_batch_keys(self, batch: List[Dict[str, Any]]) -> List[str]:
        """ 배치 안에 등장한 key를 모아서 “collate 대상 key 목록”을 만든다. (중복 제거)

#### 내부에서 호출되는 함수들(핵심)

* `_is_collatable_value(v)`

  * **텐서로 묶을 수 있는 값인지** 판단
  * 숫자/배열/텐서/None은 True, **문자열은 False**
* `_is_string_container_value(v)`

  * **문자열(또는 문자열 묶음)** 인지 판단
  * 문자열이면 텐서로 바꾸지 않고 따로 처리하기 위해 사용

#### 결과

* `keys: List[str]` 생성
* 그리고 `_FIXED_STACK_KEYS`에 있는 5개 key가 있으면 **keys의 맨 앞으로** 배치합니다.
        """
        seen = set()
        keys: List[str] = []

        for sample in batch:
            for k, v in sample.items():
                if k in seen:
                    continue

                # ✅ 텐서로 묶을 수 있거나, 문자열 계열이면 key를 포함
                if self._is_collatable_value(
                        v) or self._is_string_container_value(v):
                    seen.add(k)
                    keys.append(k)

        fixed_front: List[str] = []
        for k in self._FIXED_STACK_EGO_KEYS:
            if k in seen and k in keys:
                fixed_front.append(k)

        if fixed_front:
            rest = [k for k in keys if k not in fixed_front]
            return fixed_front + rest

        return keys

    def _stack_fixed_key_for_named_key(
        self,
        key: str,
        values: List[Any],
    ) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다. validity면 bool로 강제합니다.

        동작
        ----
        - ref_shape: 첫 번째 non-None 샘플 shape를 기준으로 (B, *ref_shape) 텐서를 만듭니다.
        - None인 샘플은 0/False로 남습니다.
        - key가 "*_is_valid"면 dtype을 torch.bool로 강제합니다.
          (입력이 int(0/1)이어도 자동으로 False/True로 변환됩니다)

        Args:
            key (str): key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)

        # 고정 key이므로 shape가 다르면 에러
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"key={key}, ref_shape={ref_shape}, got={self._get_value_shape(v)}"
                )

        # out: shape (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _stack_fixed_key(self, values: List[Any]) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다.

        이 함수가 담당하는 경우
        ----------------------
        - ego_agent_past
        - ego_future_gt_3_dim
        - planner_future_11_dim
        - ego_agent_past_is_valid
        - ego_future_gt_is_valid

        위 5개는 "모든 샘플에서 길이가 항상 동일"하다는 전제이므로
        max length를 찾거나 0 padding으로 늘릴 필요가 없습니다.

        단, 값이 None일 수도 있으므로:
        - 배치 안에 실제 값이 하나라도 있으면 그 shape를 기준으로
          None인 샘플은 0으로 채운 텐서를 만들어 stack합니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S) 텐서
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)  # 예: (T_past, 11)

        # non-None 값들의 shape가 동일한지 검사 (고정 key이므로 같아야 정상)
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"ref_shape={ref_shape}, got={self._get_value_shape(v)}")

        # (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        # 값이 있는 샘플만 복사 (None은 0 그대로)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _pad_and_stack_agent_route_lane_order(
        self,
        batch: List[Dict[str, Any]],
        max_agent_num: int,
        max_lane_num: int,
    ) -> torch.Tensor:
        """agent_route_lane_order를 (B, A_max, L_max) 텐서로 만들고 -1로 패딩합니다.

        핵심 규칙(중요)
        -------------
        - A_max(최대 agent 수)는 **agent_route_lane_order의 행 개수만** 보고 결정합니다.

        - L_max(최대 lane 수)는
          lanes.shape[0] 와 agent_route_lane_order.shape[1] 중 더 큰 값을 사용합니다.
          (lane 텐서 패딩과 같이 쓰기 쉬운 형태를 유지)

        패딩 값
        -------
        - 없는 값은 전부 -1로 채웁니다. (연결/매칭 없음 의미)

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            torch.Tensor:
                - out: shape (B, A_max, L_max)
                - dtype: torch.int64
                - padding: -1
        """
        batch_size: int = int(len(batch))

        # max_agent_num, max_lane_num = self._infer_max_sizes_for_agent_route_lane_order(
        #     batch)
        out = torch.full(
            (batch_size, max_agent_num, max_lane_num),
            fill_value=-1,
            dtype=torch.int64,
        )

        for b_idx, sample in enumerate(batch):
            aro = sample.get("agent_route_lane_order", None)
            if aro is None:
                continue

            aro_arr = np.asarray(aro)
            # aro_arr: (A_i, L_i)
            if aro_arr.ndim != 2:
                continue

            a_i: int = min(int(aro_arr.shape[0]), max_agent_num)
            l_i: int = min(int(aro_arr.shape[1]), max_lane_num)

            if a_i <= 0 or l_i <= 0:
                continue

            out[b_idx, :a_i, :l_i] = torch.as_tensor(
                aro_arr[:a_i, :l_i],
                dtype=torch.int64,
            )

        return out

    def _pad_and_stack_variable_key_for_named_key(
        self,
        key: str,
        values: List[Any],
        batch: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape 기준으로 padding 후 쌓습니다. validity면 bool로 강제합니다.

        추가로 반영된 규칙
        ----------------
        1) lane_type (shape: (lane_num, 4))
           - 패딩을 0으로 하면 "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 (0,0,0,1)로 둡니다. (index 3이 1인 one-hot)

        2) left_line_type / right_line_type (shape: (lane_num, 13))
           - 패딩을 0으로 하면 특정 선 타입(0번)으로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 index 11이 1인 one-hot로 둡니다.
             (선은 있으나 타입을 모름)

        3) lane 관련 key는 lanes의 최대 lane_num에 맞춰 첫 번째 축 길이를 보정합니다.
           - 어떤 샘플에서 lane_type이 None이라도,
             lanes 개수는 클 수 있어서 lane 차원이 어긋나지 않게 보강합니다.

        Args:
            key (str):
                key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None
            batch (List[Dict[str, Any]]):
                - 원본 샘플 dict 리스트(길이 B).
                - lanes 개수 등을 참고하기 위해 사용합니다.

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)
        ndim: int = int(len(ref_shape))

        max_shape = list(ref_shape)
        all_same_shape: bool = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)

            if len(shape_i) != ndim:
                raise ValueError(
                    f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. key={key}, "
                    f"ref_ndim={ndim}, got_ndim={len(shape_i)}")

            if shape_i != ref_shape:
                all_same_shape = False

            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # ✅ lane_type / line_type 계열은 lanes 최대 lane_num에 맞춰 첫 축을 보정
        special_spec = self._get_special_padding_category_spec(key)
        if special_spec is not None and ndim >= 1:
            max_lane_num_from_lanes: int = self._infer_max_lane_num_from_batch(
                batch)
            if max_lane_num_from_lanes > int(max_shape[0]):
                max_shape[0] = int(max_lane_num_from_lanes)

        # 빠른 경로: 패딩이 전혀 필요 없으면 stack
        # (special_spec가 있어도 max_shape가 ref_shape와 같으면 패딩 영역이 없으므로 stack 가능)
        if all_same_shape and (len(non_none_idx)
                               == batch_size) and (tuple(max_shape)
                                                   == tuple(ref_shape)):
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values],
                dim=0,
            )

        # out: shape (B, *max_shape)
        out_shape: Tuple[int,
                         ...] = (batch_size, *tuple(int(x) for x in max_shape))
        out: torch.Tensor = self._build_default_padded_tensor_for_key(
            key=key,
            out_shape=out_shape,
            dtype=out_dtype,
        )

        # 값 복사 (있는 샘플만 앞쪽부터 채움)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                out[b_idx] = t
                continue

            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *slices)] = t[slices]

        return out

    def _pad_and_stack_variable_key(
            self, values: List[Any]) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape를 기준으로 0 padding 배치 텐서를 만듭니다.

        처리 방식(핵심)
        -------------
        - key 이름을 보고 분기하지 않습니다.
        - 값이 ndarray/tensor 라면 그 shape를 보고,
          배치 내에서 각 차원별 최대값(max)을 구해 큰 텐서를 한 번만 만들고,
          샘플 값은 "앞쪽부터" 그대로 복사합니다.
        - padding 영역은 전부 0 입니다.
          (bool이면 False, float이면 0.0, int면 0)

        이 로직이 자연스럽게 커버하는 예시 shape
        ----------------------------------------
        - neighbor_agents_past:        (A_i, T_past, 11)  -> (B, A_max, T_past, 11)
        - neighbor_future_gt_3_dim:        (A_i, T_fut, 3)    -> (B, A_max, T_fut, 3)
        - lanes:                       (L_i, P_lane, 12) -> (B, L_max, P_max, 12)
        - lanes_len_is_valid:          (L_i, P_lane)      -> (B, L_max, P_max)
        - agent_route_lane_order:      (A_i, L_i)         -> (B, A_max, L_max)
        - road_edge:                   (E_i, P_edge, 2)   -> (B, E_max, P_max, 2)
        - stop_sign_points:            (S_i, P, 2)        -> (B, S_max, P_max, 2)
        - 그리고 그 외 "첫 번째 축이 개수"인 모든 값들

        None 처리
        --------
        - 어떤 샘플에서 None이면 그 샘플은 전부 0으로 남습니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)
        ndim = int(len(ref_shape))

        # max_shape 계산: 배치 내 각 차원별 최대 크기
        max_shape = list(ref_shape)
        all_same_shape = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)
            if len(shape_i) != ndim:
                raise ValueError(f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. "
                                 f"ref_ndim={ndim}, got_ndim={len(shape_i)}")
            if shape_i != tuple(ref_shape):
                all_same_shape = False
            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # (빠른 경로) None도 없고, 모든 shape가 동일하면 stack이 제일 빠름
        if all_same_shape and (len(non_none_idx) == batch_size):
            # shape: (B, *ref_shape)
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values], dim=0)

        # 일반 경로: 0 padding 텐서 만들고 값 복사
        # out shape: (B, *max_shape)
        out = torch.zeros((batch_size, *max_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                # 스칼라: out[b_idx]에 바로 대입
                out[b_idx] = t
                continue

            # 각 차원별로 실제 길이만큼만 복사
            # 예: t.shape == (A_i, T, 11) 이면 slice(0,A_i), slice(0,T), slice(0,11)
            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *([slice(None)] * 0))]  # (형태 힌트용; 실제론 아래 라인만으로 충분)

            out[(b_idx, *slices)] = t[slices]

        return out

    def _build_collated_batch_tensors(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """배치(dict 리스트)를 최종 배치 텐서(dict)로 변환합니다.

        변경 포인트
        ----------
        - 문자열(str) 계열 key는 텐서로 만들지 않고 List[str]로 반환합니다.
          (길이 B, None은 ""로 치환)
        - 나머지 숫자/배열 key는 기존대로 torch.Tensor 또는 None을 반환합니다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 dict 리스트

        Returns:
            Dict[str, Any]:
                - 텐서로 만들 수 있으면 torch.Tensor
                - 배치 전체가 None이면 None
                - 문자열 계열이면 List[str] (length=B)
        """
        batch_size: int = int(len(batch))
        if batch_size <= 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        keys: List[str] = self._collect_batch_keys(batch)
        batch_out: Dict[str, Any] = {}

        # 1) 고정 5개 key (항상 텐서로)
        for k in self._FIXED_STACK_EGO_KEYS:
            if k not in keys:
                continue
            values = [sample.get(k, None) for sample in batch]
            t = self._stack_fixed_key_for_named_key(k, values)
            batch_out[k] = t  # torch.Tensor 또는 None

        # 2) 나머지 key
        fixed_set = set(self._FIXED_STACK_EGO_KEYS)
        for k in keys:
            if k in fixed_set:
                continue

            values = [sample.get(k, None) for sample in batch]

            # 전부 None이면 None 유지
            if all(v is None for v in values):
                batch_out[k] = None
                continue

            # ✅ 문자열 계열이면 List[str]로 반환 # 어떤 str은 None일 수도 있으므로 ""로 치환
            if self._should_return_string_list_for_key(k, values):
                batch_out[k] = self._collate_string_values_to_list(values)
                continue

            # agent_route_lane_order는 마지막에 전용 처리
            if k == "agent_route_lane_order":
                continue

            # 그 외는 기존 padding+stack 경로
            t = self._pad_and_stack_variable_key_for_named_key(k, values, batch)
            batch_out[k] = t  # torch.Tensor 또는 None

        # 3) agent_route_lane_order는 -1 padding 전용 처리
        neighbor_agents_past = batch_out.get("neighbor_agents_past", None)
        lanes = batch_out.get("lanes", None)

        if not isinstance(neighbor_agents_past, torch.Tensor):
            raise ValueError(
                "[Collate] neighbor_agents_past가 torch.Tensor가 아닙니다. "
                "문자열 key 처리와 무관하게, 입력 데이터가 깨졌을 가능성이 큽니다.")
        if not isinstance(lanes, torch.Tensor):
            raise ValueError("[Collate] lanes가 torch.Tensor가 아닙니다. "
                             "문자열 key 처리와 무관하게, 입력 데이터가 깨졌을 가능성이 큽니다.")

        max_agent_num = int(neighbor_agents_past.shape[1])  # shape: ()
        max_lane_num = int(lanes.shape[1])  # shape: ()

        batch_out[
            "agent_route_lane_order"] = self._pad_and_stack_agent_route_lane_order(
                batch=batch,
                max_agent_num=max_agent_num,
                max_lane_num=max_lane_num,
            )

        # 간단 정합 체크
        agent_route_lane_order = batch_out.get("agent_route_lane_order", None)
        if isinstance(agent_route_lane_order, torch.Tensor):
            assert int(agent_route_lane_order.shape[1]) == int(
                neighbor_agents_past.shape[1]
            ), ("agent_route_lane_order의 agent 수와 neighbor_agents_past의 agent 수가 "
                "일치하지 않습니다.")

        return batch_out

    def _is_validity_key_name(self, key: str) -> bool:
        """key 이름이 validity 마스크인지 빠르게 판별합니다.

        규칙
        ----
        - key가 "_is_valid" 로 끝나면 validity 마스크로 봅니다.
          예) ego_agent_past_is_valid, lanes_len_is_valid, lanes_is_valid, ...

        Args:
            key (str): sample dict의 key 이름. shape: ()

        Returns:
            bool:
                - True: validity 마스크 key
                - False: 그 외 key
        """
        return str(key).endswith("_is_valid")

    def _choose_output_dtype_for_key(self, key: str, value: Any) -> torch.dtype:
        """특정 key에 대해, collate 출력 dtype을 결정합니다.

        핵심 규칙
        --------
        - "*_is_valid" 류 key는 입력이 int(0/1) 이 섞여 있어도 **항상 torch.bool**로 강제합니다.
        - 그 외 key는 기존 규칙을 그대로 사용합니다.
          (float -> float32, 나머지 -> 입력 dtype 유지)

        Args:
            key (str): sample dict의 key 이름. shape: ()
            value (Any): None이 아닌 샘플 값(배치에서 dtype 기준으로 삼을 값)

        Returns:
            torch.dtype: 출력 텐서 dtype
        """
        if self._is_validity_key_name(key):
            return torch.bool
        return self._choose_output_dtype(value)


    # ------------------------------------------------------------------
    # 4) collate 본체
    # ------------------------------------------------------------------
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """단일 샘플 dict 리스트를 (B, ·) 배치 dict로 변환한다.

        Returns:
            Dict[str, Any]:
                - 숫자/배열 값: torch.Tensor 또는 None
                - 문자열 값: List[str] (length=B)
        """
        if len(batch) == 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        return self._build_collated_batch_tensors(batch)
