import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from typing import Optional, Tuple


class minADE(Metric):
    def __init__(self, only_eval_targets_to_predict: bool = True) -> None:
        super(minADE, self).__init__()

        # True이면 "targets_to_predict(ego 포함)"만 평가에 사용합니다.
        # False이면 기존처럼 입력으로 들어온 전체 agent를 평가합니다.
        self.only_eval_targets_to_predict = bool(only_eval_targets_to_predict)

        # (3) 기존 custom: agent마다 rollout 중 가장 좋은 1개로 ADE
        self.add_state("sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

        # (1) WOSAC average_displacement_error 방식(시나리오 단위 평균 후 전체 평균)
        self.add_state(
            "wosac_like_avg_sum", default=tensor(0.0), dist_reduce_fx="sum"
        )

        # (2) WOSAC min_average_displacement_error 방식(시나리오 단위 min 후 전체 평균)
        self.add_state(
            "wosac_like_min_sum", default=tensor(0.0), dist_reduce_fx="sum"
        )

        # (1)(2)에서 쓰는 “유효 시나리오 개수”
        self.add_state(
            "wosac_like_scenario_count", default=tensor(0.0), dist_reduce_fx="sum"
        )

    @staticmethod
    def _compute_per_agent_per_rollout_ade(
        pred: Tensor,
        target: Tensor,
        target_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """agent별/rollout별 ADE(시간 평균 오차)를 계산합니다.

        Args:
            pred (Tensor):
                예측 (x, y).
                shape: (N, R, T, 2)
                - N: agent 수
                - R: rollout 개수
                - T: 미래 길이
            target (Tensor):
                정답 (x, y).
                shape: (N, T, 2)
            target_valid (Tensor):
                정답이 “있는지/없는지” 표시.
                shape: (N, T)
                - True인 시간만 오차 계산에 포함

        Returns:
            Tuple[Tensor, Tensor]:
                - per_agent_per_rollout_ade:
                    shape: (N, R)
                    각 agent가 각 rollout에서 얼마나 틀렸는지(시간 평균)
                - valid_agent_mask:
                    shape: (N,)
                    해당 agent가 유효한 시간(True)이 1개라도 있으면 True
        """
        # dist: (N, R, T)
        dist = torch.norm(pred - target.unsqueeze(1), p=2, dim=-1)

        # valid_f: (N, 1, T)
        valid_f = target_valid.to(dtype=dist.dtype).unsqueeze(1)

        # dist_sum: (N, R)  (유효 시간만 합)
        dist_sum = (dist * valid_f).sum(-1)

        # valid_steps: (N,)
        valid_steps = target_valid.to(dtype=dist.dtype).sum(-1)

        # per_agent_per_rollout_ade: (N, R)
        per_agent_per_rollout_ade = dist_sum / (valid_steps.unsqueeze(1) + 1e-6)

        # valid_agent_mask: (N,)
        valid_agent_mask = target_valid.any(-1)

        return per_agent_per_rollout_ade, valid_agent_mask

    @staticmethod
    def _accumulate_wosac_like_metrics(
        per_agent_per_rollout_ade: Tensor,
        valid_agent_mask: Tensor,
        agent_batch: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """WOSAC 방식 (1)(2)를 “시나리오 단위”로 계산해 합/개수를 반환합니다.

        (1) wosac_like_avg:
            - 한 시나리오 안에서 (agent, rollout) 전체 평균

        (2) wosac_like_min:
            - 한 시나리오 안에서 rollout마다 agent 평균을 구한 뒤,
              rollout 중 최소 1개 선택

        Args:
            per_agent_per_rollout_ade (Tensor):
                shape: (N, R)
            valid_agent_mask (Tensor):
                shape: (N,)
            agent_batch (Optional[Tensor]):
                각 agent가 속한 시나리오 인덱스.
                shape: (N,)
                - None이면 “전부 하나의 시나리오”로 취급

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                (sum_avg, sum_min, scenario_count)
                - sum_avg: shape ()
                - sum_min: shape ()
                - scenario_count: shape ()  (float)
        """
        device = per_agent_per_rollout_ade.device
        dtype = per_agent_per_rollout_ade.dtype

        if agent_batch is None:
            agent_batch = torch.zeros(
                (int(per_agent_per_rollout_ade.shape[0]),),
                device=device,
                dtype=torch.long,
            )
        else:
            agent_batch = agent_batch.to(device=device, dtype=torch.long)

        unique_scenarios = torch.unique(agent_batch)

        sum_avg = torch.zeros((), device=device, dtype=dtype)
        sum_min = torch.zeros((), device=device, dtype=dtype)
        scenario_count = torch.zeros((), device=device, dtype=dtype)

        for sid in unique_scenarios:
            mask = (agent_batch == sid) & valid_agent_mask  # (N,)
            if not torch.any(mask):
                continue

            # ade_sid: (Ns, R)
            ade_sid = per_agent_per_rollout_ade[mask]

            # (1) 시나리오 안에서 rollout×agent 전체 평균: scalar
            scenario_avg = ade_sid.mean()

            # (2) rollout마다 agent 평균 -> rollout 중 최소: scalar
            rollout_means = ade_sid.mean(dim=0)  # (R,)
            scenario_min = rollout_means.min()   # ()

            sum_avg = sum_avg + scenario_avg
            sum_min = sum_min + scenario_min
            scenario_count = scenario_count + torch.ones((), device=device, dtype=dtype)

        return sum_avg, sum_min, scenario_count

    @staticmethod
    def _validate_eval_target_mask(
        eval_target_mask: Tensor,
        n_agent: int,
        device: torch.device,
    ) -> Tensor:
        """평가 대상(True) / 제외(False) 표시 텐서를 검증하고 정리합니다.

        Args:
            eval_target_mask (Tensor):
                평가에 포함할 agent는 True, 제외할 agent는 False인 표시 텐서.
                shape: (N,)
                - N: agent 수
            n_agent (int):
                agent 수 N. shape: ()
            device (torch.device):
                반환 텐서를 올려둘 device.

        Returns:
            Tensor:
                dtype: torch.bool
                shape: (N,)
        """
        if eval_target_mask.dim() != 1 or int(eval_target_mask.shape[0]) != int(n_agent):
            raise ValueError(
                "eval_target_mask는 (N,) 이어야 합니다. "
                f"expected N={int(n_agent)}, got shape={tuple(eval_target_mask.shape)}"
            )
        return eval_target_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _build_eval_target_mask_from_eval_object_ids(
        agent_id: Tensor,
        agent_batch: Optional[Tensor],
        eval_object_ids: Tensor,
    ) -> Tensor:
        """agent_id가 "평가 대상 id 목록"에 포함되는지 확인해 True/False 마스크를 만듭니다.

        이 함수는 배치 안에 여러 시나리오가 섞여 있어도 동작합니다.
        agent_batch로 agent가 어떤 시나리오에 속하는지 구분한 뒤,
        해당 시나리오의 eval_object_ids와 비교합니다.

        Args:
            agent_id (Tensor):
                agent의 object id.
                shape: (N,)
            agent_batch (Optional[Tensor]):
                각 agent가 속한 시나리오 인덱스.
                shape: (N,)
                - None이면 "전부 같은 시나리오(0번)"로 취급합니다.
            eval_object_ids (Tensor):
                시나리오별 평가 대상 object id 목록.
                shape:
                  - (K,)          : 시나리오가 1개일 때
                  - (B, K)        : 배치에 시나리오가 B개 있을 때
                - 0 이하 값(0, -1 등)은 "패딩"으로 보고 비교에서 자동으로 제외합니다.

        Returns:
            Tensor:
                eval_target_mask.
                True인 agent만 평가에 포함됩니다.
                shape: (N,)
                dtype: torch.bool
        """
        if agent_id.dim() != 1:
            raise ValueError(f"agent_id는 (N,) 이어야 합니다. shape={tuple(agent_id.shape)}")

        n_agent = int(agent_id.shape[0])
        device = agent_id.device

        agent_id_long = agent_id.to(device=device, dtype=torch.long)

        if agent_batch is None:
            agent_batch_long = torch.zeros((n_agent,), device=device, dtype=torch.long)
        else:
            if agent_batch.dim() != 1 or int(agent_batch.shape[0]) != n_agent:
                raise ValueError(
                    "agent_batch는 (N,) 이어야 합니다. "
                    f"expected N={n_agent}, got shape={tuple(agent_batch.shape)}"
                )
            agent_batch_long = agent_batch.to(device=device, dtype=torch.long)

        eval_ids = eval_object_ids.to(device=device, dtype=torch.long)
        if eval_ids.dim() == 1:
            eval_ids = eval_ids.unsqueeze(0)  # (1, K)
        elif eval_ids.dim() != 2:
            raise ValueError(
                "eval_object_ids는 (K,) 또는 (B, K) 이어야 합니다. "
                f"현재 shape={tuple(eval_object_ids.shape)}"
            )

        max_sid = int(agent_batch_long.max().item()) if n_agent > 0 else 0
        if int(eval_ids.shape[0]) <= max_sid:
            raise ValueError(
                "eval_object_ids의 첫 번째 차원(B)이 agent_batch의 시나리오 인덱스를 커버하지 못합니다. "
                f"B={int(eval_ids.shape[0])}, max_sid={max_sid}"
            )

        eval_target_mask = torch.zeros((n_agent,), device=device, dtype=torch.bool)
        unique_scenarios = torch.unique(agent_batch_long)

        for sid in unique_scenarios:
            sid_int = int(sid.item())
            sid_mask = (agent_batch_long == sid)

            ids_this = eval_ids[sid_int]  # (K,)
            ids_this = ids_this[ids_this > 0]  # padding(<=0) 제거
            if ids_this.numel() == 0:
                continue

            eval_target_mask[sid_mask] = torch.isin(agent_id_long[sid_mask], ids_this)

        return eval_target_mask

    def update(
        self,
        pred: Tensor,         # (N, R, T, 2)
        target: Tensor,       # (N, T, 2)
        target_valid: Tensor, # (N, T)
        agent_batch: Optional[Tensor] = None,  # (N,)
        *,
        agent_id: Optional[Tensor] = None,          # (N,)
        eval_object_ids: Optional[Tensor] = None,   # (K,) or (B, K)
        eval_target_mask: Optional[Tensor] = None,  # (N,)
    ) -> None:
        # (N, R), (N,)
        per_agent_per_rollout_ade, valid_agent_mask = self._compute_per_agent_per_rollout_ade(
            pred=pred,
            target=target,
            target_valid=target_valid,
        )

        # ✅ 옵션: targets_to_predict(ego 포함)만 평가하도록 필터링
        if self.only_eval_targets_to_predict:
            n_agent = int(per_agent_per_rollout_ade.shape[0])
            device = per_agent_per_rollout_ade.device

            if eval_target_mask is not None:
                eval_mask = self._validate_eval_target_mask(
                    eval_target_mask=eval_target_mask,
                    n_agent=n_agent,
                    device=device,
                )
            else:
                if agent_id is None or eval_object_ids is None:
                    raise ValueError(
                        "only_eval_targets_to_predict=True 인 경우, "
                        "eval_target_mask 또는 (agent_id + eval_object_ids)를 반드시 제공해야 합니다."
                    )
                eval_mask = self._build_eval_target_mask_from_eval_object_ids(
                    agent_id=agent_id,
                    agent_batch=agent_batch,
                    eval_object_ids=eval_object_ids,
                )

            valid_agent_mask = valid_agent_mask & eval_mask

        # (3) 기존 custom: agent마다 rollout 중 최소 1개
        # per_agent_min: (N,)
        per_agent_min = per_agent_per_rollout_ade.min(dim=1).values
        self.sum += per_agent_min[valid_agent_mask].sum()
        self.count += valid_agent_mask.sum()

        # (1)(2) WOSAC 방식(시나리오 단위)
        sum_avg, sum_min, scenario_count = self._accumulate_wosac_like_metrics(
            per_agent_per_rollout_ade=per_agent_per_rollout_ade,
            valid_agent_mask=valid_agent_mask,
            agent_batch=agent_batch,
        )
        self.wosac_like_avg_sum += sum_avg
        self.wosac_like_min_sum += sum_min
        self.wosac_like_scenario_count += scenario_count

    def compute(self) -> torch.Tensor:
        # (3) custom 결과(기존과 동일)
        return self.sum / self.count

    def compute_wosac_like_average_displacement_error(self) -> torch.Tensor:
        """(1) WOSAC average_displacement_error 방식(시나리오 평균)의 최종 값을 반환합니다."""
        if float(self.wosac_like_scenario_count.item()) <= 0.0:
            return tensor(0.0, device=self.wosac_like_avg_sum.device)
        return self.wosac_like_avg_sum / self.wosac_like_scenario_count

    def compute_wosac_like_min_average_displacement_error(self) -> torch.Tensor:
        """(2) WOSAC min_average_displacement_error 방식(시나리오 평균)의 최종 값을 반환합니다."""
        if float(self.wosac_like_scenario_count.item()) <= 0.0:
            return tensor(0.0, device=self.wosac_like_min_sum.device)
        return self.wosac_like_min_sum / self.wosac_like_scenario_count
