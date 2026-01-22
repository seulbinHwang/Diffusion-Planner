from timm.models.layers import Mlp
from timm.layers import DropPath
import torch.nn.functional as F
import math
from diffusion_planner.model.module.mixer import MixerBlock
from flash_attn.bert_padding import unpad_input, pad_input

# ==== (encoder.py 상단 import 근처에 추가) ====
from typing import Tuple  # 이미 있으면 중복 추가 불필요

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def profile_block(name: str,
                  enabled: bool = True,
                  device_type: str = "cuda") -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력하는 간단한 프로파일러.

    Args:
        name: 출력에 사용할 블록 이름.
        enabled: False 이면 아무 것도 하지 않고 그냥 통과.
        device_type: "cuda" 인 경우, GPU 연산 정합을 위해 앞/뒤에 synchronize 호출.
    """
    if not enabled:
        # 아무 것도 하지 않고 블록만 실행
        yield
        return

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time: float = time.perf_counter()

    yield  # 실제 코드 실행

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms: float = (time.perf_counter() - start_time) * 1000.0
    print(f"[PROFILE] {name}: {elapsed_ms:.3f} ms")


# ===== FlashAttention-2 varlen import (2.x 표준 경로 + 백업 경로) =====
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

    try:
        # 일부 버전(옛 코드) 표기
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_q_kvpacked_func as flash_attn_varlen_cross_func)
    except ImportError:
        # flash-attn 2.8.x의 정식 이름
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_kvpacked_func as flash_attn_varlen_cross_func)
    _FA2_AVAILABLE = True
    _FA2_IMPORT_ERR = None
except Exception as _e1:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func

        try:
            from flash_attn import (flash_attn_varlen_q_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        except ImportError:
            from flash_attn import (flash_attn_varlen_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        _FA2_AVAILABLE = True
        _FA2_IMPORT_ERR = None
    except Exception as _e2:
        _FA2_AVAILABLE = False
        _FA2_IMPORT_ERR = Exception(
            f"interface import err: {_e1}; top-level err: {_e2}")
        flash_attn_varlen_cross_func = None

from typing import Tuple, Dict, Optional, List, Iterable
import torch
import torch.nn as nn


def encode_time_with_fourier_features(
    t_sec: torch.Tensor,
    time_min: float,
    time_max: float,
    num_fourier_frequencies: int,
) -> torch.Tensor:
    """시간 값을 Fourier(2K) + 중심화 시간 스칼라(1)로 인코딩합니다(AMP 안전).

    Args:
        t_sec (torch.Tensor): (...,) 형태의 초 단위 시간 텐서.
        time_min (float): 시간 최소값.
        time_max (float): 시간 최대값.
        num_fourier_frequencies (int): K, 사용할 주파수 개수.

    Returns:
        torch.Tensor: (..., 2K + 1) 형태의 인코딩 텐서. 채널 구성은
            [cos(·), sin(·), t_hat]이며, 앞의 2K채널이 Fourier 성분,
            마지막 1채널이 중심화된 시간 스칼라(t_hat)입니다.

    Note:
        - 삼각함수 계산은 fp32에서 수행 후, 최종적으로 입력 텐서 dtype으로 캐스팅합니다.
        - 0으로 나누기 방지를 위해 작은 epsilon을 분모에 더합니다.
    """
    eps: float = 1e-6

    # 정규화된 시간: (...,)
    t_norm: torch.Tensor = (t_sec - time_min) / (time_max - time_min + eps
                                                )  # (...,)
    t_norm: torch.Tensor = t_norm.clamp(0.0, 1.0).to(torch.float32)  # (...,)

    # 주파수 인덱스: (K,)
    k_indices: torch.Tensor = torch.arange(1,
                                           num_fourier_frequencies + 1,
                                           device=t_sec.device,
                                           dtype=torch.float32)  # (K,)

    # 각도: (..., K)
    angles: torch.Tensor = t_norm.unsqueeze(
        -1) * k_indices * torch.pi  # (..., K)

    # Fourier 성분: (..., 2K)
    fourier: torch.Tensor = torch.cat(
        [torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., 2K)

    # 중심화 시간 스칼라: (..., 1),  t_hat = 2 * t_norm - 1
    t_scalar: torch.Tensor = (2.0 * t_norm - 1.0).unsqueeze(-1)  # (..., 1)

    # 최종 인코딩: (..., 2K + 1)  [cos | sin | t_hat]
    out: torch.Tensor = torch.cat([fourier, t_scalar],
                                  dim=-1).to(t_sec.dtype)  # (..., 2K + 1)
    return out


def timegrid_past_3d(dt: float,
                     total_steps: int,
                     B: int,
                     agents_num: int,
                     *,
                     device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    반환 모양: (B, agents_num, total_steps)
    마지막 축(시간축): [-dt*(N-1), ..., -dt*1, 0.0]
    항상 독립 메모리 텐서(복사본)를 반환합니다.
    """
    # (total_steps,) = [total_steps-1, ..., 1, 0]
    idx = torch.arange(total_steps - 1, -1, -1, device=device, dtype=dtype)
    base = -float(dt) * idx
    base[-1] = torch.tensor(0.0, dtype=dtype, device=device)  # -0.0 → 0.0

    # (1,1,N) → (B,A,N) 확장 후 clone()으로 복사본 보장
    return base.reshape(1, 1, total_steps).expand(B, agents_num,
                                                  total_steps).clone()


def timegrid_future_2d(dt: float,
                       total_steps: int,
                       B: int,
                       *,
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    반환 모양: (B, total_steps)
    마지막 축(시간축): [dt, 2*dt, ..., N*dt]
    항상 독립 메모리 텐서(복사본)를 반환합니다.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1")
    if B <= 0:
        raise ValueError("B must be >= 1")

    # (N,) = [1, 2, ..., N]
    idx = torch.arange(1, total_steps + 1, device=device, dtype=dtype)
    base = float(dt) * idx  # [dt, 2dt, ..., N*dt]

    # (1,N) → (B, N) 확장 후 clone()으로 복사본 보장
    return base.reshape(1, total_steps).expand(B, total_steps).clone()


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.route_order_drop_prob: float = getattr(config,
                                                    "route_order_drop_prob",
                                                    0.5)

        self.chunk_length = 20
        self.num_fourier_frequencies = 4
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.time_gap = 0.1
        self.time_min = -(config.time_len - 1) * self.time_gap
        self.time_max = config.future_len * self.time_gap
        self.config = config
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        self.road_safety_encoder = RoadSafetyFusionEncoder(
            hidden_dim=config.hidden_dim,
            num_seeds=getattr(config, "road_safety_num_seeds", 8),
            attn_drop_p=0.0,
            out_drop_p=0.0,
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth + 1,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        lane_summary_num: int = int(getattr(config, "lane_summary_num", 0))
        if lane_summary_num > 0:
            self.lane_summary_pooler = LaneSummaryTokenPooler(
                hidden_dim=config.hidden_dim,
                num_seeds=lane_summary_num,
                attn_drop_p=0.0,
                out_drop_p=0.0,
            )
        else:
            self.lane_summary_pooler = None

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device)

        # position embedding encode
        # [x, y, cos, sin] + type_onehot(5) = 9
        # type_onehot: (ego, neighbor, static, lane, road_safety)
        pos_emb_mlp_ratio: float = float(
            getattr(config, "pos_emb_mlp_ratio", 1.0))
        pos_emb_drop_p: float = float(getattr(config, "pos_emb_drop_p", 0.0))

        self.pos_emb = self._build_pos_embedding_module(
            in_dim=9,
            out_dim=config.hidden_dim,
            hidden_ratio=pos_emb_mlp_ratio,
            drop_p=pos_emb_drop_p,
        )
        # -----------------------------
        # 로컬 인코더 출력 흔들림(무작위 꺼짐 동작)을 막기 위한 설정
        # -----------------------------
        # True면: 로컬 인코더 파라미터가 "전부" 고정된 경우, 로컬 인코더 서브모듈을 eval로 내려서
        # 학습 모드에서만 발생하는 무작위 꺼짐 동작을 막습니다.
        self._eval_frozen_encoder_local: bool = True

        # None이면 자동 모드(위 규칙 사용).
        # True면 무조건 로컬 인코더를 eval로 강제(디버깅/검증용).
        # False면 로컬 인코더는 부모 모드(train/eval)를 그대로 따름.
        self._force_encoder_local_eval: Optional[bool] = None

    def _build_pos_embedding_module(
        self,
        in_dim: int,
        out_dim: int,
        hidden_ratio: float,
        drop_p: float,
    ) -> nn.Module:
        """위치/방향/종류 정보를 토큰 임베딩 길이로 바꾸는 모듈을 만듭니다.

        이 인코더는 각 토큰마다 아래 9개 값을 가지고 있습니다.
            - 위치/방향: [x, y, cos, sin]  -> 4개
            - 종류 표시: 0/1로 된 5개 값  -> 5개
            - 합계: 9개

        이 9개 값을 토큰 임베딩과 같은 길이(out_dim, 보통 hidden_dim)의 벡터로 바꿔서,
        토큰 임베딩에 더할 수 있게 합니다.

        입력/출력 모양(Shape)
            - 입력 텐서:  (N, in_dim)   보통 (N, 9)
            - 출력 텐서:  (N, out_dim)  보통 (N, hidden_dim)

        동작
            - hidden_ratio <= 0 이면:
                기존처럼 "한 번 변환"만 사용합니다. (in_dim -> out_dim)
            - hidden_ratio > 0 이면:
                "두 번 변환"을 사용합니다. (in_dim -> mid -> out_dim)
                mid 길이는 out_dim * hidden_ratio 로 정합니다.
                이렇게 하면 학습할 수 있는 값(파라미터) 수가 늘어,
                위치 정보를 더 다양한 형태로 바꿀 수 있습니다.

        Args:
            in_dim (int): 입력 길이. 기본 9.
            out_dim (int): 출력 길이. 보통 hidden_dim.
            hidden_ratio (float): 중간 길이를 out_dim 대비 얼마나 넓힐지 비율.
            drop_p (float): 학습 중에만 중간 결과의 일부를 0으로 만드는 비율(0이면 사용하지 않음).

        Returns:
            nn.Module: (N, in_dim) -> (N, out_dim) 변환 모듈.
        """
        if hidden_ratio <= 0.0:
            layer = nn.Linear(in_dim, out_dim, bias=True)
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
            return layer

        mid_dim: int = max(16, int(out_dim * float(hidden_ratio)))

        layer = Mlp(
            in_features=in_dim,
            hidden_features=mid_dim,
            out_features=out_dim,
            act_layer=nn.GELU,
            drop=float(drop_p),
        )

        # 초기 출력 크기가 과도하게 커지지 않도록, 기존 코드와 같은 std=0.02 초기화 사용
        nn.init.normal_(layer.fc1.weight, std=0.02)
        nn.init.zeros_(layer.fc1.bias)
        nn.init.normal_(layer.fc2.weight, std=0.02)
        nn.init.zeros_(layer.fc2.bias)
        return layer

    def iter_encoder_local_parameters(self) -> Iterator[nn.Parameter]:
        """로컬 인코더(Group A)에 속한 파라미터들을 순서대로 돌려줍니다.

        로컬 인코더는 개별 차량, 정적 물체, 차선, 도로 안전 요소, 위치 임베딩 등을 읽는 부분입니다.
        """
        local_modules = [
            self.static_encoder,
            self.lane_encoder,
            self.road_safety_encoder,
        ]
        for module in local_modules:
            for param in module.parameters():
                yield param

        for param in self.pos_emb.parameters():
            # param: (hidden_dim, 9) 또는 (hidden_dim,)
            yield param

        # pos_scale: shape () 스칼라 파라미터
        yield self.pos_scale

    def iter_encoder_global_parameters(self) -> Iterator[nn.Parameter]:
        """글로벌 인코더(Group B)에 속한 파라미터들을 순서대로 돌려줍니다.

        글로벌 인코더는 여러 객체를 합쳐 장면 전체를 요약하는 부분입니다.
        """
        global_modules = [
            self.fusion,
            self.lane_summary_pooler,
        ]
        for module in global_modules:
            if module is None:
                continue
            for param in module.parameters():
                yield param

    def set_eval_frozen_encoder_local_enabled(self, enabled: bool) -> None:
        """로컬 인코더가 고정된 경우(eval로 내릴지) 정책을 켜거나 끕니다.

        이 함수는 "로컬 인코더 파라미터를 전부 고정한 상태"에서,
        학습 모드일 때도 로컬 인코더 결과가 매번 달라지는 문제를 줄이기 위한 스위치입니다.

        동작 방식(쉽게 설명):
            - enabled=True:
                로컬 인코더 파라미터가 전부 고정(requires_grad=False)이라면,
                전체 모델이 학습 모드여도 로컬 인코더 서브모듈만 평가 모드로 내려서
                무작위로 값이 꺼지는 동작이 로컬 인코더에서 나오지 않게 합니다.
            - enabled=False:
                로컬 인코더 서브모듈도 부모 모드(train/eval)를 그대로 따릅니다.

        Args:
            enabled (bool): 위 정책을 사용할지 여부.
        """
        self._eval_frozen_encoder_local = bool(enabled)
        self._sync_encoder_local_train_eval_mode()

    def set_force_encoder_local_eval(self, force_eval: Optional[bool]) -> None:
        """로컬 인코더를 항상 eval로 둘지(또는 항상 부모 모드를 따를지) 강제 설정합니다.

        이 함수는 "파라미터를 얼렸다"를 lr=0 같은 방식으로 구현해서
        requires_grad가 True로 남아 있는 경우에도,
        로컬 인코더의 출력 흔들림(무작위 꺼짐 동작)을 확실히 막고 싶을 때 씁니다.

        동작:
            - force_eval=None:
                자동 모드(로컬 파라미터가 전부 고정된 경우에만 eval로 내림)
            - force_eval=True:
                무조건 로컬 인코더 서브모듈을 eval로 둠
            - force_eval=False:
                무조건 로컬 인코더 서브모듈이 부모 모드(train/eval)를 그대로 따름

        Args:
            force_eval (Optional[bool]): 위 설명의 강제 값.
        """
        self._force_encoder_local_eval = force_eval
        self._sync_encoder_local_train_eval_mode()

    def are_encoder_local_parameters_frozen(self) -> bool:
        """로컬 인코더 파라미터가 전부 '고정' 상태인지 확인합니다.

        여기서 "고정"은 PyTorch의 파라미터 옵션인 `requires_grad=False`를 의미합니다.
        (즉, 학습 중에 그 파라미터 값을 바꾸지 않는 상태)

        Returns:
            bool:
                - 로컬 인코더에 속한 파라미터가 하나라도 있고,
                  그 파라미터들이 전부 requires_grad=False 이면 True
                - 그 외에는 False
        """
        has_any_param: bool = False
        for p in self.iter_encoder_local_parameters():
            # p: (out_dim, in_dim) 또는 (dim,) 또는 () 등 여러 형태 가능
            has_any_param = True
            if p.requires_grad:
                return False
        return has_any_param

    def _encoder_local_modules(self) -> Tuple[nn.Module, ...]:
        """로컬 인코더로 취급할 서브모듈 목록을 반환합니다.

        Returns:
            Tuple[nn.Module, ...]: 로컬 인코더에 해당하는 모듈들.
        """
        return (
            self.static_encoder,
            self.lane_encoder,
            self.road_safety_encoder,
            self.pos_emb,  # 선형층이라 모드 영향은 거의 없지만, 일관성 위해 포함
        )

    @staticmethod
    def _set_modules_train_mode(modules: Iterable[nn.Module],
                                mode: bool) -> None:
        """주어진 모듈들을 한꺼번에 train/eval 모드로 바꿉니다.

        Args:
            modules (Iterable[nn.Module]): 모드 변경 대상 모듈들.
            mode (bool): True면 학습 모드, False면 평가 모드.
        """
        for m in modules:
            m.train(mode)

    def _sync_encoder_local_train_eval_mode(self) -> None:
        """현재 설정과 파라미터 고정 상태에 따라, 로컬 인코더 서브모듈의 모드를 맞춥니다.

        목표(쉽게 설명):
            - 전체 모델이 학습 모드(train)여도,
              로컬 인코더가 "고정" 상태라면 로컬 인코더 내부에서
              무작위로 일부 값이 꺼지는 동작이 나오지 않도록(=eval) 만든다.
            - 반대로 로컬 인코더가 학습 대상이면(=하나라도 requires_grad=True),
              로컬 인코더도 학습 모드를 유지한다.

        Note:
            - 이 함수는 값 자체를 바꾸지 않고 "모드(train/eval)만" 바꿉니다.
            - 모드 변경은 Dropout/DropPath 같은 "학습 모드에서만 흔들리는 동작"을 막는 데 목적이 있습니다.
        """
        local_modules = self._encoder_local_modules()

        # 1) 강제 설정이 있으면 그게 최우선
        if self._force_encoder_local_eval is True:
            self._set_modules_train_mode(local_modules, mode=False)
            return
        if self._force_encoder_local_eval is False:
            self._set_modules_train_mode(local_modules,
                                         mode=bool(self.training))
            return

        # 2) 자동 모드: "로컬 파라미터가 전부 고정"일 때만 eval로 내림
        if not self._eval_frozen_encoder_local:
            self._set_modules_train_mode(local_modules,
                                         mode=bool(self.training))
            return

        local_is_frozen: bool = self.are_encoder_local_parameters_frozen()
        if bool(self.training) and local_is_frozen:
            # 전체는 train이어도, 로컬은 eval로 내려서 흔들림 방지
            self._set_modules_train_mode(local_modules, mode=False)
        else:
            # 그 외는 부모 모드를 따름
            self._set_modules_train_mode(local_modules,
                                         mode=bool(self.training))

    def train(self, mode: bool = True) -> "Encoder":
        """PyTorch train()/eval() 호출 시, 로컬 인코더 모드까지 함께 정리합니다.

        Args:
            mode (bool): True면 학습 모드, False면 평가 모드.

        Returns:
            Encoder: 자기 자신(self).
        """
        super().train(mode)
        self._sync_encoder_local_train_eval_mode()
        return self

    def _zero_with_touch(self, ref: torch.Tensor,
                         params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        """ref와 같은 shape의 0 텐서를 반환하되, 주어진 파라미터들을 0계수로 터치해
        autograd 그래프를 연결(grad는 0)."""
        # 스칼라 누산(기기/dtype 맞춤)
        touch = ref.new_zeros(())
        for p in params:
            # 첫 원소만 살짝 참조 → 비용 최소화
            touch = touch + p.view(-1)[:1].sum()
        return ref.new_zeros(ref.shape) + touch * 0.0  # broadcast OK

    @staticmethod
    def _road_safety_inputs_are_all_none(
        stop_sign_points: Optional[torch.Tensor],
        stop_sign_is_valid: Optional[torch.Tensor],
        crosswalk_points: Optional[torch.Tensor],
        crosswalk_is_valid: Optional[torch.Tensor],
        speed_bump_points: Optional[torch.Tensor],
        speed_bump_is_valid: Optional[torch.Tensor],
        driveway_points: Optional[torch.Tensor],
        driveway_is_valid: Optional[torch.Tensor],
        road_edge: Optional[torch.Tensor],
        road_edge_is_valid: Optional[torch.Tensor],
        road_edge_type: Optional[torch.Tensor],
    ) -> bool:
        """road-safety 입력이 전부 None인지 확인합니다.

        이 함수의 목적은 아주 단순합니다.
        road-safety 관련 입력이 “하나도 안 들어온 경우”를 빠르게 감지해서,
        아래의 빈 토큰 생성 로직으로 안전하게 넘어가기 위함입니다.

        Args:
            stop_sign_points: (B, Ns, S, 2) 또는 None
            stop_sign_is_valid: (B, Ns) 또는 None
            crosswalk_points: (B, Nc, S, 2) 또는 None
            crosswalk_is_valid: (B, Nc) 또는 None
            speed_bump_points: (B, Nb, S, 2) 또는 None
            speed_bump_is_valid: (B, Nb) 또는 None
            driveway_points: (B, Nd, S, 2) 또는 None
            driveway_is_valid: (B, Nd) 또는 None
            road_edge: (B, Ne, S, 2) 또는 None
            road_edge_is_valid: (B, Ne) 또는 None
            road_edge_type: (B, Ne, 3) 또는 None

        Returns:
            bool:
                - 전부 None이면 True
                - 하나라도 텐서가 있으면 False
        """
        tensors = [
            stop_sign_points,
            stop_sign_is_valid,
            crosswalk_points,
            crosswalk_is_valid,
            speed_bump_points,
            speed_bump_is_valid,
            driveway_points,
            driveway_is_valid,
            road_edge,
            road_edge_is_valid,
            road_edge_type,
        ]
        return all(t is None for t in tensors)

    def _build_empty_road_safety_tokens(
        self,
        batch_size: int,
        ref_encoding: torch.Tensor,
        ref_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """road-safety 입력이 아예 없을 때 사용할 '빈 토큰'을 만듭니다.

        해야 하는 일은 2가지입니다.

        1) road-safety 토큰을 “있다고 가정”하되,
           실제 값은 전부 0이고, 마스크는 전부 True(=패딩, 무효)로 만들어
           이후 과정(Fusion 등)이 끊기지 않고 그대로 진행되게 합니다.

        2) 여러 GPU로 학습할 때(여러 장치로 나눠 학습하는 경우),
           road_safety_encoder 쪽 파라미터가 “이번 입력에서 전혀 안 쓰였다”고 판단되면
           학습이 중단될 수 있습니다.
           그래서 값에는 영향을 주지 않도록(0을 곱해서) 아주 약하게 연결해 둡니다.

        Args:
            batch_size (int):
                배치 크기 B.
            ref_encoding (torch.Tensor):
                road-safety 임베딩 텐서의 dtype/device를 맞추기 위한 기준 텐서.
                예: encoding_agents_chunk (B, N_agents_tok, H)
            ref_pos (torch.Tensor):
                road-safety 위치(pos) 텐서의 dtype/device를 맞추기 위한 기준 텐서.
                예: agents_chunk_pos (B, N_agents_tok, 9)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - encoding_road_safety: (B, M, H)  (값 0)
                - road_safety_mask:     (B, M)     (전부 True=패딩)
                - road_safety_pos:      (B, M, 9)  (값 0, type_onehot의 road_safety 위치만 1로 설정 가능)

                여기서
                - B = batch_size
                - M = self.road_safety_encoder.num_seeds
                - H = self.hidden_dim
        """
        B: int = int(batch_size)
        M: int = int(getattr(self.road_safety_encoder, "num_seeds"))
        H: int = int(self.hidden_dim)

        # ----- (A) road-safety 임베딩/마스크 -----
        # encoding_road_safety: (B, M, H)
        encoding_road_safety: torch.Tensor = torch.zeros(
            (B, M, H),
            device=ref_encoding.device,
            dtype=ref_encoding.dtype,
        )
        # road_safety_mask: (B, M)  True=pad
        road_safety_mask: torch.Tensor = torch.ones(
            (B, M),
            device=ref_encoding.device,
            dtype=torch.bool,
        )

        # ----- (B) road-safety pos -----
        # road_safety_pos: (B, M, 9) = [x,y,cos,sin] + type_onehot(5)
        # 여기서는 "빈 토큰"이므로 좌표는 0, 타입은 road_safety 인덱스만 1로 둡니다.
        road_safety_pos: torch.Tensor = torch.zeros(
            (B, M, 9),
            device=ref_pos.device,
            dtype=ref_pos.dtype,
        )
        if M > 0:
            # type_onehot(5)의 마지막(road_safety) 위치 = index 8
            road_safety_pos[:, :, 8] = 1.0

        # ----- (C) 파라미터를 0계수로 아주 약하게 연결(값은 그대로 0) -----
        # touch_f32: shape ()
        touch_f32: torch.Tensor = torch.zeros((),
                                              device=ref_encoding.device,
                                              dtype=torch.float32)
        for p in self.road_safety_encoder.parameters():
            touch_f32 = touch_f32 + p.view(-1)[:1].sum().float()

        # encoding dtype에 맞춰 캐스팅 후 0을 곱해 더함 (값 변화 없음)
        touch: torch.Tensor = touch_f32.to(dtype=ref_encoding.dtype)
        encoding_road_safety = encoding_road_safety + touch * 0.0

        return encoding_road_safety, road_safety_mask, road_safety_pos

    def _sample_uniform_prefix_lengths(self, batch_size: int,
                                       max_future_len: int,
                                       device: torch.device) -> torch.Tensor:
        """무작위 길이 M_i를 각 배치별로 균일 분포에서 샘플링합니다.

        Args:
            batch_size (int): 배치 크기 B.
            max_future_len (int): 전체 미래 길이 N(예: 80).
            device (torch.device): 결과 텐서를 올릴 디바이스.

        Returns:
            torch.Tensor: [B] 형태의 정수 텐서. 각 값은 M_i ∈ {0,1,...,N}.
        """
        # shape: (B,)
        # 낮은 값 포함(0), 높은 값은 제외 → high = N+1 로 설정해 {0..N} 범위
        prefix_lengths: torch.Tensor = torch.randint(low=0,
                                                     high=max_future_len + 1,
                                                     size=(batch_size,),
                                                     device=device)
        return prefix_lengths

    def _build_known_mask_from_lengths(self, prefix_lengths: torch.Tensor,
                                       max_future_len: int) -> torch.Tensor:
        """길이 M_i로부터 '알려진 구간(조건 제공)' 마스크를 만듭니다.

        정의:
            known_mask[b, t] = True  ⇔  t < M_b
            (여기서 t는 0..N-1 인덱스이며, 시간은 0.1s,1.0s 등과 매핑 가능)

        Args:
            prefix_lengths (torch.Tensor): [B] 각 배치의 M_i.
            max_future_len (int): 전체 미래 길이 N.

        Returns:
            torch.Tensor: [B, N]의 bool 텐서. True=알려진(조건), False=미제공.
        """
        # time_index: [1, N] → [0..N-1] 인덱스
        time_index = torch.arange(max_future_len,
                                  device=prefix_lengths.device)  # (N,)
        # 브로드캐스트 비교 → (B, N)
        known_mask = time_index.unsqueeze(0) < prefix_lengths.unsqueeze(1)
        return known_mask  # (B, N), bool

    def _truncate_and_pad_ego_future_for_encoder(
            self,
            planner_future_11_dim: torch.Tensor,  # (B, future_len, 11)
            known_mask: torch.Tensor) -> torch.Tensor:
        """미래 궤적을 [처음 M_i 스텝 유지 + 나머지 0패딩]으로 변환합니다.

        주의:
        - 에이전트 인코더의 assert(길이 고정)를 만족시키기 위해 **길이는 그대로 N 유지**합니다.
        - 마스크 판단은 에이전서 내부에서 첫 8채널이 0인지로 이루어지므로,
          M_i 이후 프레임은 **앞 8채널을 0**으로 채웁니다.
        - 마지막 3채널(type)은 항상 ego를 의미하도록 **[1,0,0]**(또는 입력의 ego one-hot)을 유지합니다.

        Args:
            planner_future_11_dim (torch.Tensor):
                형태: [B, future_len, 11]
                채널: [x, y, cos, sin, vx, vy, w, l, type(3)]
            known_mask (torch.Tensor):
                형태: [B, future_len], bool
                True=조건 제공(유지), False=미제공(패딩)

        Returns:
            torch.Tensor:
                형태: [B, future_len, 11]
                처음 M_i는 원본 유지, 나머지는 앞 8채널 0, type 3채널은 ego one-hot 유지.
        """
        B, future_len, D = planner_future_11_dim.shape  # (B, future_len, 11)
        assert D == 11, "ego_future_full의 마지막 차원은 11이어야 합니다."

        # (B, 3) ego 타입 벡터를 첫 프레임에서 추출(이미 one-hot이라 가정)
        ego_type = planner_future_11_dim[:, 0, 8:11].clone()  # (B, 3)

        # 기본 패딩 텐서: 앞 8채널=0, type 3채널=ego one-hot 반복
        padded_default = planner_future_11_dim.new_zeros(
            B, future_len, D)  # (B, future_len, 11)
        padded_default[:, :,
                       8:11] = ego_type.unsqueeze(1).expand(-1, future_len, -1)

        # keep: (B, future_len, 1)  True=원본 유지
        keep = known_mask.unsqueeze(-1)

        # 최종: 알려진 구간은 원본, 나머지는 기본 패딩
        truncated_ego_future = torch.where(
            keep, planner_future_11_dim, padded_default)  # (B, future_len, 11)
        return truncated_ego_future

    def _prepare_encoder_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
            Optional[torch.
                     Tensor],  # ego_agent_past: (B, 1, time_len, 11) or None
            Optional[
                torch.
                Tensor],  # planner_future_11_dim: (B, future_len, 11) or None
            Optional[
                torch.
                Tensor],  # neighbor_agents_past: (B, A, time_len, 11) or None
            Optional[
                torch.
                Tensor],  # non_near_agents_past: (B, non_near_A, time_len, 11) or None
            Optional[torch.Tensor],  # static_objects: (B, P, D_static) or None
            Optional[torch.Tensor],  # lanes: (B, L, lane_len, D_lane) or None
            Optional[torch.Tensor],  # lanes_speed_limit: (B, L, 1) or None
            Optional[torch.Tensor],  # lanes_has_speed_limit: (B, L, 1) or None
            Optional[torch.
                     Tensor],  # agent_route_lane_order: (B, Pnn, L) or None
            Optional[torch.Tensor],  # lane_type: (B, L, 4) or None
            Optional[torch.Tensor],  # left_line_type: (B, L, 13) or None
            Optional[torch.Tensor],  # right_line_type: (B, L, 13) or None
            Optional[
                torch.
                Tensor],  # stop_sign_points: (B, Ns, safety_len, 2) or None
            Optional[torch.Tensor],  # stop_sign_is_valid: (B, Ns) bool or None
            Optional[
                torch.
                Tensor],  # crosswalk_points: (B, Nc, safety_len, 2) or None
            Optional[torch.Tensor],  # crosswalk_is_valid: (B, Nc) bool or None
            Optional[
                torch.
                Tensor],  # speed_bump_points: (B, Nb, safety_len, 2) or None
            Optional[torch.Tensor],  # speed_bump_is_valid: (B, Nb) bool or None
            Optional[torch.
                     Tensor],  # driveway_points: (B, Nd, safety_len, 2) or None
            Optional[torch.Tensor],  # driveway_is_valid: (B, Nd) bool or None
            Optional[torch.Tensor],  # road_edge: (B, E, safety_len, 2) or None
            Optional[torch.Tensor],  # road_edge_is_valid: (B, E) bool or None
            Optional[torch.Tensor],  # road_edge_type: (B, E, 3) or None
            int,  # B
    ]:
        """입력 dict에서 인코더가 쓸 텐서들을 꺼내고 모양을 맞춥니다.

        요구사항 반영
        ------------
        - **모든 값**을 `inputs.get("key", None)` 방식으로 가져옵니다.
          즉, 키가 없으면 해당 출력은 `None` 이 됩니다.

        추가 동작
        --------
        1) ego 과거 궤적이 존재하면 (B, T, 11) → (B, 1, T, 11)로 바꿉니다.
        2) 속도 입력을 쓰지 않는 설정이면, 존재하는 텐서에 한해 vx, vy 채널([4:6])을 0으로 만듭니다.
        3) 평가 모드에서 route encoding을 강제로 무시하도록 켜면,
           존재하는 `agent_route_lane_order`에 한해 전부 -1로 채웁니다.

        Args:
            inputs (Dict[str, torch.Tensor]):
                키가 없을 수 있으므로, 모든 키는 optional로 취급합니다.
                예:
                    - "ego_agent_past":          (B, time_len, 11)
                    - "planner_future_11_dim":   (B, future_len, 11)
                    - "neighbor_agents_past":    (B, A, time_len, 11)
                    - "static_objects":          (B, P, D_static)
                    - "lanes":                   (B, L, lane_len, D_lane)
                    - "lanes_speed_limit":       (B, L, 1)
                    - "lanes_has_speed_limit":   (B, L, 1)
                    - "agent_route_lane_order":  (B, Pnn, L)
                    - "lane_type":               (B, L, 4)
                    - "left_line_type":          (B, L, 13)
                    - "right_line_type":         (B, L, 13)
                    - "stop_sign_points":        (B, Ns, safety_len, 2)
                    - "crosswalk_points":        (B, Nc, safety_len, 2)
                    - "speed_bump_points":       (B, Nb, safety_len, 2)
                    - "driveway_points":         (B, Nd, safety_len, 2)
                    - "road_edge":               (B, E, safety_len, 2)
                    - "road_edge_type":          (B, E, 3)

        Returns:
            Tuple[...]:
                각 키에 해당하는 텐서(없으면 None) + (B, future_len)

                - B:
                    우선순위로 배치 크기를 추정합니다.
                    1) neighbor_agents_past.shape[0]
                    2) ego_agent_past(before unsqueeze).shape[0]
                    3) planner_future_11_dim.shape[0]
                    4) 그 외 전부 None이면 0
                - future_len:
                    planner_future_11_dim이 있으면 shape[1], 없으면 0
        """
        # --- 1) 전부 get(...) 로 가져오기 ---
        ego_agent_past: Optional[torch.Tensor] = inputs.get(
            "ego_agent_past", None)  # (B, T, 11) or None
        planner_future_11_dim: Optional[torch.Tensor] = inputs.get(
            "planner_future_11_dim", None)  # (B, Tf, 11) or None
        neighbor_agents_past: Optional[torch.Tensor] = inputs.get(
            "neighbor_agents_past", None)  # (B, A, T, 11) or None
        non_near_agents_past: Optional[torch.Tensor] = inputs.get(
            "non_near_agents_past", None)  # (B, non_near_A, T, 11) or None

        static_objects: Optional[torch.Tensor] = inputs.get(
            "static_objects", None)  # (B, P, D_static) or None
        lanes: Optional[torch.Tensor] = inputs.get(
            "lanes", None)  # (B, L, lane_len, D_lane) or None
        lanes_speed_limit: Optional[torch.Tensor] = inputs.get(
            "lanes_speed_limit", None)  # (B, L, 1) or None
        lanes_has_speed_limit: Optional[torch.Tensor] = inputs.get(
            "lanes_has_speed_limit", None)  # (B, L, 1) or None
        agent_route_lane_order: Optional[torch.Tensor] = inputs.get(
            "agent_route_lane_order", None)  # (B, Pnn, L) or None

        lane_type: Optional[torch.Tensor] = inputs.get(
            "lane_type", None)  # (B, L, 4) or None
        left_line_type: Optional[torch.Tensor] = inputs.get(
            "left_line_type", None)  # (B, L, 13) or None
        right_line_type: Optional[torch.Tensor] = inputs.get(
            "right_line_type", None)  # (B, L, 13) or None

        stop_sign_points: Optional[torch.Tensor] = inputs.get(
            "stop_sign_points", None)  # (B, Ns, safety_len, 2) or None
        stop_sign_is_valid: Optional[torch.Tensor] = inputs.get(
            "stop_sign_is_valid", None)  # (B, Ns) bool or None
        crosswalk_points: Optional[torch.Tensor] = inputs.get(
            "crosswalk_points", None)  # (B, Nc, safety_len, 2) or None
        crosswalk_is_valid: Optional[torch.Tensor] = inputs.get(
            "crosswalk_is_valid", None)  # (B, Nc) bool or None
        speed_bump_points: Optional[torch.Tensor] = inputs.get(
            "speed_bump_points", None)  # (B, Nb, safety_len, 2) or None
        speed_bump_is_valid: Optional[torch.Tensor] = inputs.get(
            "speed_bump_is_valid", None)  # (B, Nb) bool or None
        driveway_points: Optional[torch.Tensor] = inputs.get(
            "driveway_points", None)  # (B, Nd, safety_len, 2) or None
        driveway_is_valid: Optional[torch.Tensor] = inputs.get(
            "driveway_is_valid", None)  # (B, Nd) bool or None

        road_edge: Optional[torch.Tensor] = inputs.get(
            "road_edge", None)  # (B, E, safety_len, 2) or None
        road_edge_is_valid: Optional[torch.Tensor] = inputs.get(
            "road_edge_is_valid", None)  # (B, E) bool or None
        road_edge_type: Optional[torch.Tensor] = inputs.get(
            "road_edge_type", None)  # (B, E, 3) or None

        # --- 2) ego/neighbor 속도 채널 제거 + ego 차원 맞추기 ---
        if ego_agent_past is not None:
            if not self.config.use_vel_input:
                ego_agent_past[:, :, 4:6] = 0.0  # vx, vy
            ego_agent_past = ego_agent_past.unsqueeze(1)  # (B, 1, T, 11)

        if neighbor_agents_past is not None:
            if not self.config.use_vel_input:
                neighbor_agents_past[:, :, :, 4:6] = 0.0  # vx, vy
        if non_near_agents_past is not None:
            if not self.config.use_vel_input:
                non_near_agents_past[:, :, :, 4:6] = 0.0  # vx, vy

        B: int = 0
        if neighbor_agents_past is not None:
            B = int(neighbor_agents_past.shape[0])
        elif ego_agent_past is not None:
            B = int(ego_agent_past.shape[0])
        elif planner_future_11_dim is not None:
            B = int(planner_future_11_dim.shape[0])

        # --- 4) (옵션) 평가 모드에서 route encoding 강제 무시 ---
        self.neglect_route_encoding = False
        if (not self.training) and bool(self.neglect_route_encoding):
            if agent_route_lane_order is not None:
                agent_route_lane_order = torch.full_like(agent_route_lane_order,
                                                         fill_value=-1)

        return (
            ego_agent_past,
            planner_future_11_dim,
            neighbor_agents_past,
            non_near_agents_past,
            static_objects,
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            agent_route_lane_order,
            lane_type,
            left_line_type,
            right_line_type,
            stop_sign_points,
            stop_sign_is_valid,
            crosswalk_points,
            crosswalk_is_valid,
            speed_bump_points,
            speed_bump_is_valid,
            driveway_points,
            driveway_is_valid,
            road_edge,
            road_edge_is_valid,
            road_edge_type,
            B,
        )

    def _ensure_static_objects_tensor(
        self,
        static_objects: Optional[torch.Tensor],
        batch_size: int,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """static_objects가 None이어도 정적 물체 인코더가 항상 동작하도록 입력 텐서를 보장합니다.

        학습 데이터에서 정적 물체가 하나도 없는 장면은 `static_objects=None`으로 들어올 수 있습니다.
        그런데 StaticFusionEncoder는 내부에서 `static_objects.shape`를 바로 사용하므로,
        None이면 즉시 에러가 납니다.

        이 함수는 None인 경우에도 **정적 물체 인코더가 실제로 실행되도록**
        (B, P, D_static) 형태의 텐서를 만들어 반환합니다.

        동작 방식(쉽게 설명):
            1) static_objects가 텐서면:
               - (B, P, D_static) 모양인지 확인하고,
               - device가 다르면 ref_tensor의 device로만 옮겨서 그대로 반환합니다.

            2) static_objects가 None이거나, P==0(정말 비어있는 텐서)면:
               - (B, P, D_static) 텐서를 새로 만듭니다.
               - P는 가능하면 config.static_num(또는 config.static_objects_num)을 사용하고,
                 둘 다 없으면 1로 둡니다.
               - 그중 **첫 번째 물체(인덱스 0)** 를 “정적 물체가 없음”을 나타내는 자리로 사용합니다.
                 이 토큰이 StaticFusionEncoder 내부 마스크에서 **유효(mask=False)** 로 잡히도록,
                 [x, y, cos, sin] 중 cos 채널(인덱스 2)에 1.0을 넣습니다.
                 (나머지 값은 0이라서, 실제 물체와 겹치기 어려운 형태가 됩니다.)

        Args:
            static_objects (Optional[torch.Tensor]):
                - shape: (B, P, D_static) 또는 None
                - D_static은 보통 10이며, 앞 4채널은 [x, y, cos, sin]이라고 가정합니다.
            batch_size (int):
                - 배치 크기 B
            ref_tensor (torch.Tensor):
                - device/dtype 기준 텐서
                - 예: encoding_agents_chunk (B, N_agents_tok, H)

        Returns:
            torch.Tensor:
                - shape: (B, P, D_static)
                - static_objects가 None이어도 StaticFusionEncoder에 바로 넣을 수 있는 텐서
        """
        B: int = int(batch_size)
        if B <= 0:
            raise ValueError(f"batch_size must be > 0. got {B}")

        # D_static: 정적 물체 feature 차원
        D_static: int = int(getattr(self.config, "static_objects_state_dim"))
        if D_static < 10:
            raise ValueError(
                f"static_objects_state_dim must be >= 10. got {D_static}")
        static_objects_num = 1
        # # P(정적 물체 최대 개수): 가능하면 config에서 가져오고, 없으면 1
        # if hasattr(self.config, "static_num"):
        #     static_objects_num: int = int(getattr(self.config, "static_num"))
        # elif hasattr(self.config, "static_objects_num"):
        #     static_objects_num = int(getattr(self.config, "static_objects_num"))
        # else:
        #     static_objects_num = 1
        #
        # static_objects_num = max(1, static_objects_num)

        # -------------------------
        # (1) 입력이 이미 텐서인 경우
        # -------------------------
        if static_objects is not None:
            if static_objects.dim() != 3:
                raise ValueError(
                    f"static_objects must be (B, P, D_static). got {tuple(static_objects.shape)}"
                )
            if int(static_objects.shape[0]) != B:
                raise ValueError(
                    f"static_objects batch size mismatch. expected B={B}, got {int(static_objects.shape[0])}"
                )
            if int(static_objects.shape[2]) != D_static:
                raise ValueError(
                    f"static_objects last dim mismatch. expected D_static={D_static}, got {int(static_objects.shape[2])}"
                )

            # (B, P, D_static)
            if static_objects.device != ref_tensor.device:
                static_objects = static_objects.to(device=ref_tensor.device)
            return static_objects

        # -------------------------
        # (2) None(또는 사실상 비어있음)인 경우: "없음 표시" 입력 생성
        # -------------------------
        # placeholder_static_objects: (B, P, D_static)
        placeholder_static_objects: torch.Tensor = ref_tensor.new_zeros(
            (B, static_objects_num, D_static))

        # 첫 번째 토큰을 "없음 표시"로 사용하되, 마스크가 False(유효)로 잡히도록 cos=1
        # x=0, y=0, cos=1, sin=0  (B, )
        # placeholder_static_objects[:, 0, 2] = 1.0

        return placeholder_static_objects

    def _encode_agents_static_lanes(
        self,
        static_objects: torch.Tensor,  # (B, P, D_static)
        lanes: torch.Tensor,  # (B, L, lane_len, D_lane)
        lanes_speed_limit: torch.Tensor,  # (B, L, 1)
        lanes_has_speed_limit: torch.Tensor,  # (B, L, 1)
        lane_type: Optional[torch.Tensor],  # (B, L, 4)
        left_line_type: Optional[torch.Tensor],  # (B, L, 13)
        right_line_type: Optional[torch.Tensor],  # (B, L, 13)
        stop_sign_points: Optional[
            torch.Tensor] = None,  # (B, Ns, safety_len, 2)
        stop_sign_is_valid: Optional[torch.Tensor] = None,  # (B, Ns)
        crosswalk_points: Optional[
            torch.Tensor] = None,  # (B, Nc, safety_len, 2)
        crosswalk_is_valid: Optional[torch.Tensor] = None,  # (B, Nc)
        speed_bump_points: Optional[
            torch.Tensor] = None,  # (B, Nb, safety_len, 2)
        speed_bump_is_valid: Optional[torch.Tensor] = None,  # (B, Nb)
        driveway_points: Optional[
            torch.Tensor] = None,  # (B, Nd, safety_len, 2)
        driveway_is_valid: Optional[torch.Tensor] = None,  # (B, Nd)
        road_edge: Optional[torch.Tensor] = None,  # (B, E, safety_len, 2)
        road_edge_is_valid: Optional[torch.Tensor] = None,  # (B, E)
        road_edge_type: Optional[torch.Tensor] = None,  # (B, E, 3)
    ) -> Tuple[
            torch.Tensor,  # encoding_static:       (B, N_static, H)
            torch.Tensor,  # static_mask:           (B, N_static)
            torch.Tensor,  # static_pos:            (B, N_static, 9)
            torch.Tensor,  # encoding_lanes:        (B, N_lanes, H)
            torch.Tensor,  # lanes_mask:            (B, N_lanes)
            torch.Tensor,  # lane_pos:              (B, N_lanes, 9)
            torch.Tensor,  # encoding_road_safety:  (B, N_road_safety, H)
            torch.Tensor,  # road_safety_mask:      (B, N_road_safety)
            torch.Tensor,  # road_safety_pos:       (B, N_road_safety, 9)
    ]:
        """에이전트 / 정적 객체 / 차선 인코더를 한 번에 호출한다.
        Returns:
            위 각 인코더에서 나오는 토큰/마스크/좌표 텐서들.
        """
        # --- static encoder ---
        """
        encoding_static: (B, static_objects_num, hidden_dim)
        static_mask:     (B, static_objects_num)
        static_pos:      (B, static_objects_num, 9)
        """
        # static_objects가 None일 수도 있으므로, 항상 텐서로 보장해서 넣는다.
        static_objects_tensor: torch.Tensor = self._ensure_static_objects_tensor(
            static_objects=static_objects,  # (B, P, D_static) or None
            batch_size=int(lanes.shape[0]),  # B
            ref_tensor=lanes,  # (B, N_agents_tok, H)  device/dtype 기준
        )  # (B, P, D_static)
        encoding_static, static_mask, static_pos = self.static_encoder(
            static_objects_tensor)

        # --- lane encoder ---
        """
        encoding_lanes: (B, lane_num, hidden_dim)
        lanes_mask:     (B, lane_num)
        lane_pos:       (B, lane_num, 9)
        """
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit, lane_type,
            left_line_type, right_line_type)
        # --- road safety encoder (입력이 전부 None이면 "빈 토큰"으로 대체) ---
        """
        encoding_road_safety : (B, road_safety_num, hidden_dim)
        road_safety_mask     : (B, road_safety_num)
        road_safety_pos      : (B, road_safety_num, 9)
        """
        if self._road_safety_inputs_are_all_none(
                stop_sign_points,
                stop_sign_is_valid,
                crosswalk_points,
                crosswalk_is_valid,
                speed_bump_points,
                speed_bump_is_valid,
                driveway_points,
                driveway_is_valid,
                road_edge,
                road_edge_is_valid,
                road_edge_type,
        ):
            (encoding_road_safety, road_safety_mask,
             road_safety_pos) = self._build_empty_road_safety_tokens(
                 batch_size=int(lanes.shape[0]),
                 ref_encoding=encoding_lanes,
                 ref_pos=lane_pos)
        else:
            (encoding_road_safety, road_safety_mask,
             road_safety_pos) = self.road_safety_encoder(
                 stop_sign_points,
                 stop_sign_is_valid,
                 crosswalk_points,
                 crosswalk_is_valid,
                 speed_bump_points,
                 speed_bump_is_valid,
                 driveway_points,
                 driveway_is_valid,
                 road_edge,
                 road_edge_is_valid,
                 road_edge_type,
             )

        return (encoding_static, static_mask, static_pos, encoding_lanes,
                lanes_mask, lane_pos, encoding_road_safety, road_safety_mask,
                road_safety_pos)

    def _add_pos_embedding_to_tokens(
            self,
            token_embeddings: torch.Tensor,  # (B, N, H)
            token_pos: torch.Tensor,  # (B, N, 9)
            token_mask: torch.Tensor,  # (B, N) True=pad
    ) -> torch.Tensor:
        """토큰 임베딩에 위치 정보를 더해줍니다.

        이 함수는 다음을 보장합니다.
        - 패딩 토큰(mask=True)은 위치 임베딩을 더하지 않습니다.
        - 유효 토큰(mask=False)만 골라 pos_emb(Linear(9→H))를 적용합니다.
        - pos_scale(스칼라)을 곱한 뒤 token_embeddings에 더합니다.

        Args:
            token_embeddings:
                shape: (B, N, H)
            token_pos:
                shape: (B, N, 9)
                [x,y,cos,sin] + type_onehot(5)
            token_mask:
                shape: (B, N)  True=패딩(무효)

        Returns:
            torch.Tensor:
                shape: (B, N, H)
                위치 임베딩이 더해진 토큰 텐서
        """
        if token_mask.dtype != torch.bool:
            token_mask = token_mask.to(torch.bool)

        if token_embeddings.dim() != 3 or token_pos.dim(
        ) != 3 or token_mask.dim() != 2:
            raise ValueError(
                f"Expected token_embeddings (B,N,H), token_pos (B,N,9), token_mask (B,N). "
                f"got {tuple(token_embeddings.shape)}, {tuple(token_pos.shape)}, {tuple(token_mask.shape)}"
            )

        B, N, H = token_embeddings.shape
        if token_pos.shape != (B, N, 9):
            raise ValueError(
                f"token_pos must be (B, N, 9). got {tuple(token_pos.shape)}")
        if token_mask.shape != (B, N):
            raise ValueError(
                f"token_mask must be (B, N). got {tuple(token_mask.shape)}")

        # (B*N,)
        mask_flat: torch.Tensor = token_mask.reshape(-1)
        # (B*N, 9)
        pos_flat: torch.Tensor = token_pos.reshape(B * N, 9)

        # 유효 토큰만 pos_emb 적용
        if (~mask_flat).any().item():
            pos_valid: torch.Tensor = self.pos_emb(
                pos_flat[~mask_flat])  # (N_valid, H)
            pos_valid = pos_valid.to(dtype=token_embeddings.dtype,
                                     device=token_embeddings.device)

            scale: torch.Tensor = self.pos_scale.to(
                dtype=token_embeddings.dtype, device=token_embeddings.device)
            pos_valid = scale * pos_valid  # (N_valid, H)

            pos_result_flat: torch.Tensor = torch.zeros(
                (B * N, H),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            )
            pos_result_flat[~mask_flat] = pos_valid  # (B*N, H)
            pos_result = pos_result_flat.view(B, N, H)  # (B,N,H)
        else:
            # 유효 토큰이 하나도 없으면 pos_result는 0
            pos_result = torch.zeros((B, N, H),
                                     device=token_embeddings.device,
                                     dtype=token_embeddings.dtype)

        return token_embeddings + pos_result

    def _build_fusion_inputs(
        self,
        encoding_static: torch.Tensor,  # (B, N_static, H)
        static_mask: torch.Tensor,  # (B, N_static)
        static_pos: torch.Tensor,  # (B, N_static, 9)
        encoding_lanes: torch.Tensor,  # (B, N_lanes, H)
        lanes_mask: torch.Tensor,  # (B, N_lanes)
        lane_pos: torch.Tensor,  # (B, N_lanes, 9)
        encoding_road_safety: torch.Tensor,  # (B, N_road_safety, H)
        road_safety_mask: torch.Tensor,  # (B, N_road_safety)
        road_safety_pos: torch.Tensor,  # (B, N_road_safety, 9)
    ) -> Tuple[
            torch.Tensor,  # encoding_input_with_pos: (B, token_num, H)
            torch.Tensor,  # encoding_mask_2d:        (B, token_num)
    ]:
        """agents/static/lanes/road_safety 토큰을 한 줄로 이어 붙이고, 위치 임베딩을 더합니다.

        변경점(핵심)
        ----------
        - FusionEncoder에는 lane 토큰 전체(N_lanes개)를 넣지 않고,
          lane_summary_num개(M개)의 요약 토큰을 만들어 넣을 수 있습니다.
        - route-lane 인코더(NearAgentsRouteLaneEncoder)는 기존처럼
          "전체 lane 토큰(원본 N_lanes개)"을 그대로 사용합니다.
          (agent_route_lane_order가 lane 인덱스를 그대로 쓰기 때문)

        Args:
            encoding_lanes / lanes_mask / lane_pos:
                - 전체 lane 토큰(원본) 정보입니다.
                - Fusion 입력에서는 필요하면 요약 토큰으로 바꿉니다.

        Returns:
            encoding_input_with_pos:
                (B, token_num, H)  FusionEncoder로 들어갈 토큰(요약 lane 포함 가능)
            encoding_mask_2d:
                (B, token_num)     FusionEncoder용 마스크(True=pad)
        """

        # ------------------------------------------------------------------
        # (1) route-lane용 "전체 lane 토큰 + pos" 준비
        #     - lane_summary를 쓰는 경우에만 별도로 계산이 필요합니다.
        #     - lane_summary를 안 쓰면, 아래에서 Fusion 입력을 만들 때 lane slice로 얻습니다.
        # ------------------------------------------------------------------
        use_lane_summary: bool = (self.lane_summary_pooler is not None)
        # ------------------------------------------------------------------
        # (2) Fusion에 넣을 lane 토큰 결정
        #     - use_lane_summary=True: (B,M,H) 요약 토큰
        #     - use_lane_summary=False: (B,N_lanes,H) 원본 lane 토큰
        # ------------------------------------------------------------------
        if use_lane_summary:
            (
                encoding_lanes_for_fusion,  # (B, M, H)
                lanes_mask_for_fusion,  # (B, M)
                lane_pos_for_fusion,  # (B, M, 9)
            ) = self.lane_summary_pooler(
                lane_embeddings=encoding_lanes,
                lane_mask=lanes_mask,
                lane_pos=lane_pos,
            )
        else:
            encoding_lanes_for_fusion = encoding_lanes
            lanes_mask_for_fusion = lanes_mask
            lane_pos_for_fusion = lane_pos

        # ------------------------------------------------------------------
        # (3) 토큰/마스크/포지션 concat (Fusion용)
        # ------------------------------------------------------------------
        encoding_input: torch.Tensor = torch.cat(
            [encoding_static, encoding_lanes_for_fusion, encoding_road_safety],
            dim=1,
        )  # (B, token_num, H)

        encoding_mask_2d: torch.Tensor = torch.cat(
            [static_mask, lanes_mask_for_fusion, road_safety_mask],
            dim=1,
        )  # (B, token_num)

        encoding_pos_2d: torch.Tensor = torch.cat(
            [static_pos, lane_pos_for_fusion, road_safety_pos],
            dim=1,
        )  # (B, token_num, 9)

        # ------------------------------------------------------------------
        # (4) pos_emb 더하기
        # ------------------------------------------------------------------
        encoding_input_with_pos: torch.Tensor = self._add_pos_embedding_to_tokens(
            token_embeddings=encoding_input,  # (B, token_num, H)
            token_pos=encoding_pos_2d,  # (B, token_num, 9)
            token_mask=encoding_mask_2d,  # (B, token_num)
        )

        return encoding_input_with_pos, encoding_mask_2d

    def _run_fusion_and_route_encoder(
            self,
            encoding_input_with_pos: torch.Tensor,  # (B, token_num, H)
            encoding_mask_2d: torch.Tensor,  # (B, token_num)
    ) -> Dict[str, torch.Tensor]:
        """FusionEncoder + route-lane 인코더까지 실행해 출력 dict를 만든다.

        Args:
            encoding_input_with_pos: (B, token_num, H)  위치 임베딩까지 포함된 입력 토큰.
            encoding_mask_2d:        (B, token_num)     True=pad.

        Returns:
            encoder_outputs:
                - "encoding":                  (B, token_num, H)
                - "encoding_mask":             (B, token_num)
                - "near_agents_route_lane_emb":(B, Pnn, H)
                - "route_known_mask":          (B, Pnn)  True=유효 route 보유 에이전트
        """
        encoder_outputs: Dict[str, torch.Tensor] = {}

        # 1) FusionEncoder 통과
        encoding_tokens, fused_mask = self.fusion(
            encoding_input_with_pos,  # (B, token_num, H)
            encoding_mask_2d,  # (B, token_num)
        )  # encoding_tokens: (B, token_num, H), fused_mask: (B, token_num)

        encoder_outputs["encoding"] = encoding_tokens
        encoder_outputs["encoding_mask"] = fused_mask
        return encoder_outputs

    # --------------------------------------------------------------------- #
    # 리팩토링된 Encoder.forward
    # --------------------------------------------------------------------- #

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인코더 전방 패스(무작위 길이 M만큼 ego 미래를 조건으로 사용하는 버전).

        입력 딕셔너리
            ego_agent_past : (B, time_len, 11) #
            ego_future_gt_3_dim : (B, future_len, 3)
            neighbor_agents_past : (B, agent_num, time_len, 11) #
            lanes : (B, lane_num, lane_len, 12) #
            lanes_speed_limit : (B, lane_num, 1) #
            lanes_has_speed_limit : (B, lane_num, 1) #
            route_lanes : (B, route_num, route_len, 12)
            route_lanes_speed_limit : (B, route_num, 1)
            route_lanes_has_speed_limit : (B, route_num, 1)
            static_objects : (B, static_num, 10) #
            near_future_gt_3_dim: (B, Pnn, future_len, 3)
            planner_future_11_dim: (B, future_len, 11) #
            agent_route_lane_order: (B, agent_num, lane_num)

            near_future_valid: (B, Pnn, future_len) 미래 유효 마스크.
            near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4) 현재+미래 x_T.
            batch_diffusion_time: (B,) diffusion 시간.
            cond_last_pos_norm: (B, Pnn, 4) cond 용 마지막 위치.

        처리 흐름:
            1) 입력 텐서 정리 및 속도 채널(vx, vy) 0 세팅(옵션).
            2) 학습 모드에서는 배치별로 ego 미래를 앞쪽 M 스텝만 남기고 나머지는 0으로 패딩.
            3) 에이전트/정적/차선 인코더를 통과시켜 토큰/마스크/위치 좌표를 만든다.
            4) 세 토큰을 한 줄로 이어 붙이고, 위치 임베딩(pos_emb)을 더해 FusionEncoder에 넣는다.
            5) lane 토큰으로부터 route-lane 요약 벡터를 뽑고, 최종 encoder 출력 dict를 만든다.

        Returns:
            Dict[str, torch.Tensor]:
                - "encoding":                  (B, token_num, hidden_dim)
                - "encoding_mask":             (B, token_num)
                - "ego_fut_global":            (B, hidden_dim)
                - "near_agents_route_lane_emb":(B, Pnn, hidden_dim)
                - "route_known_mask":          (B, Pnn)
        """
        # 로컬 인코더가 고정돼 있으면, 학습 모드에서도 로컬만 eval로 내려 출력 흔들림을 막는다.
        self._sync_encoder_local_train_eval_mode()
        device_type: str = inputs["ego_agent_past"].device.type

        with profile_block(
                "encoder.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            # ---- (1) 입력 텐서들 준비 ----
            (
                ego_agent_past,  # (B, 1, time_len, 11)
                planner_future_11_dim,  # (B, future_len, 11)
                neighbor_agents_past,  # (B, A, time_len, 11)
                non_near_agents_past,  # (B, non_near_A, time_len, 11)
                static_objects,  # (B, P, D_static)
                lanes,  # (B, L, lane_len, D_lane)
                lanes_speed_limit,  # (B, L, 1)
                lanes_has_speed_limit,  # (B, L, 1)
                agent_route_lane_order,  # (B, Pnn, L)
                lane_type,  # (B, L, 4) or None
                left_line_type,  # (B, L, 13) or None
                right_line_type,  # (B, L, 13) or None
                stop_sign_points,  # (B, stop_sign_num, safety_len, 2) or None
                stop_sign_is_valid,  # (B, stop_sign_num) or None
                crosswalk_points,  #  (B, crosswalk_num, safety_len, 2) or None
                crosswalk_is_valid,  # (B, crosswalk_num) or None
                speed_bump_points,  #  (B, speed_bump_num, safety_len, 2) or None
                speed_bump_is_valid,  # (B, speed_bump_num) or None
                driveway_points,  # (B, driveway_num, safety_len, 2) or None
                driveway_is_valid,  # (B, driveway_num) or None
                road_edge,  # (B, road_edge_num, safety_len, 2) or None
                road_edge_is_valid,  # (B, road_edge_num) or None
                road_edge_type,  # (B, road_edge_num, 3) or None
                B,
            ) = self._prepare_encoder_inputs(inputs)
            future_len = self.config.future_len

            # ---- (3) agents/static/lanes 인코딩 ----
            (
                encoding_static,  # (B, N_static, H)
                static_mask,  # (B, N_static)
                static_pos,  # (B, N_static, 9)
                encoding_lanes,  # (B, N_lanes, H)
                lanes_mask,  # (B, N_lanes)
                lane_pos,  # (B, N_lanes, 9),
                encoding_road_safety,  # (B, N_road_safety, H)
                road_safety_mask,  # (B, N_road_safety)
                road_safety_pos,  # (B, N_road_safety, 9)
            ) = self._encode_agents_static_lanes(
                static_objects=static_objects,
                lanes=lanes,
                lanes_speed_limit=lanes_speed_limit,
                lanes_has_speed_limit=lanes_has_speed_limit,
                lane_type=lane_type,
                left_line_type=left_line_type,
                right_line_type=right_line_type,
                stop_sign_points=stop_sign_points,
                stop_sign_is_valid=stop_sign_is_valid,
                crosswalk_points=crosswalk_points,
                crosswalk_is_valid=crosswalk_is_valid,
                speed_bump_points=speed_bump_points,
                speed_bump_is_valid=speed_bump_is_valid,
                driveway_points=driveway_points,
                driveway_is_valid=driveway_is_valid,
                road_edge=road_edge,
                road_edge_is_valid=road_edge_is_valid,
                road_edge_type=road_edge_type,
            )

            # ---- (4) Fusion 입력 토큰 + 위치 임베딩 구성 ----
            (
                encoding_input_with_pos,  # (B, token_num, H)
                encoding_mask_2d  # (B, token_num)
            ) = self._build_fusion_inputs(
                encoding_static=encoding_static,
                static_mask=static_mask,
                static_pos=static_pos,
                encoding_lanes=encoding_lanes,
                lanes_mask=lanes_mask,
                lane_pos=lane_pos,
                encoding_road_safety=encoding_road_safety,
                road_safety_mask=road_safety_mask,
                road_safety_pos=road_safety_pos,
            )

            # ---- (5) FusionEncoder + route-lane 인코더 실행 ----
            encoder_outputs: Dict[
                str, torch.Tensor] = self._run_fusion_and_route_encoder(
                    encoding_input_with_pos=encoding_input_with_pos,
                    encoding_mask_2d=encoding_mask_2d,
                )
            return encoder_outputs


class SelfAttentionBlock(nn.Module):

    def __init__(
            self,
            dim=192,
            heads=8,
            attn_drop_p: float = 0.0,  # 어텐션 드롭아웃
            ffn_drop_p: float = 0.0,  # FFN 드롭아웃
            drop_path_p: float = 0.0,  # Stochastic Depth
            mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        # 원본의 self.attn(nn.MultiheadAttention)은 폴백 경로에서만 사용
        self.use_fallback_mha = not _FA2_AVAILABLE
        if self.use_fallback_mha:
            self.attn = nn.MultiheadAttention(dim,
                                              heads,
                                              attn_drop_p,
                                              batch_first=True)
        else:
            self.attn = None
        self.attn_out_drop = nn.Dropout(attn_drop_p)
        self.drop_path = DropPath(
            drop_path_p) if drop_path_p > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU,
                       drop=ffn_drop_p)

        # === FlashAttention‑2용 QKV/출력 프로젝션 ===
        self.num_heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, f"dim({dim}) must be divisible by heads({heads})"
        if (self.head_dim % 8) != 0:
            # 권장: 8 배수
            print(f"[Warning] head_dim={self.head_dim} (not multiple of 8). "
                  "FlashAttention‑2 성능이 저하될 수 있습니다.")

        # 유효 토큰에만 적용하는 선형 레이어
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # FlashAttention dropout 확률(학습 시에만 사용)
        self._attn_dropout_p = attn_drop_p

    # ------------------------------------------------------------------
    # 아래 유틸리티 함수들은 varlen 커널 구동을 위한 핵심 로직입니다.
    # 모두 타입 힌트와 Google Style 한글 docstring, 자주 쓰이는 shape 주석 포함.
    # ------------------------------------------------------------------

    @staticmethod
    def _get_compute_dtype(x: torch.Tensor) -> torch.dtype:
        """연산 dtype을 선택합니다(BF16/FP16 우선).

        1) autocast가 켜져 있으면 해당 dtype을 우선 사용합니다.
        2) 입력 텐서 `x`가 FP16/BF16이면 그대로 사용합니다.
        3) 그 외에는 GPU 아키텍처에 따라 BF16(>=SM80) 또는 FP16을 반환합니다.

        Args:
            x (torch.Tensor): 임의 텐서. (shape 무관)

        Returns:
            torch.dtype: torch.float16 또는 torch.bfloat16
        """
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                pass

        if x.dtype in (torch.float16, torch.bfloat16):
            return x.dtype

        if x.is_cuda and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(x.device)
            return torch.bfloat16 if major >= 8 else torch.float16

        return torch.float16

    def _self_attn_flash_varlen(
            self,
            x: torch.Tensor,  # (B, L, D)
            mask: torch.Tensor,  # (B, L) True=pad
    ) -> torch.Tensor:
        """FlashAttention‑2(varlen) Self‑Attention을 수행합니다.

        1) `mask`를 이용해 유효 토큰만 언패드 → 2) QKV/어텐션/출력 프로젝션을 유효 토큰에만 수행
        → 3) pad back으로 (B, L, D) 복원합니다. 패딩 토큰은 연산 경로에 참여하지 않으므로
        연산량과 메모리 사용량이 유효 길이에 비례합니다.

        Args:
            x (torch.Tensor): 입력(쿼리=키=값), 모양 (B, L, D).
            mask (torch.Tensor): 키 패딩 마스크(True=pad), 모양 (B, L).

        Returns:
            torch.Tensor: Self‑Attention 출력, 모양 (B, L, D).
        """
        if not _FA2_AVAILABLE:
            raise RuntimeError("FlashAttention‑2(varlen) 모듈을 불러오지 못했습니다. "
                               "pip install flash-attn>=2.3 등으로 설치 후 다시 시도하세요. "
                               f"(원인: {_FA2_IMPORT_ERR})")

            # TODO: cpu 만 사용가능할 때, PyTorch SDPA(패딩 포함) 사용하는 옵션 추가 (아래 주석 해제)
            # # FlashAttention‑2가 없으면 원래 경로로 폴백
            # y = self.attn(x, x, x, key_padding_mask=mask,
            #               need_weights=False)[0]  # (B, L, D)
            return y

        B, L, D = x.shape

        # (1) 언패드
        """
        x_unpad: (T, D) 유효 토큰만 이어붙인 텐서. T = sum(seqlens). [학습 input]
        indices: (T,)  원래 (B*L) 평탄화 인덱스에서 유효 토큰의 위치. [pad back용]
        cu_seqlens: (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0. [학습 input]
        max_len= max_seqlen: int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1). [학습 input]
        seqlens = seqlens: (B,) int32  배치별 유효 토큰 개수.
        """
        # x_unpad, indices, cu_seqlens, max_seqlen = self._unpad_from_mask(
        #     x, mask)  # x_unpad: (T, D)
        attention_mask = (~mask).to(torch.bool)  # True=유효

        res = unpad_input(x, attention_mask)

        # v2.7 이하: 4개 / v2.8.x: 5개
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        T = x_unpad.shape[0]
        if T == 0 or max_seqlen == 0:
            touch = (self.qkv_proj.weight.view(-1)[:1].sum() +
                     (self.qkv_proj.bias.view(-1)[:1].sum()
                      if self.qkv_proj.bias is not None else 0) +
                     self.out_proj.weight.view(-1)[:1].sum() +
                     (self.out_proj.bias.view(-1)[:1].sum()
                      if self.out_proj.bias is not None else 0)) * 0.0
            return torch.zeros_like(x) + touch

        # (2) QKV 프로젝션 (유효 토큰만)
        qkv = self.qkv_proj(x_unpad)  # (T, 3*D)
        qkv = qkv.view(T, 3, self.num_heads, self.head_dim)  # (T, 3, H, Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        # (3) FlashAttention‑2 varlen
        """
        # 원래 input x: (B, L, D) → 언패드 x_unpad: (T, D) → QKV: (T, 3, H, Hd)
        batch 내에서, 각 L 토큰들끼리만 서로 어텐션을 수행합니다.
        """
        # out: (T, H, Hd)
        out = flash_attn_varlen_qkvpacked_func(
            qkv,  # (T, 3, H, Hd)
            cu_seqlens=cu_seqlens.to(
                torch.int32),  # (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0.
            max_seqlen=max_seqlen,  # int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1).
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (T, H, Hd)

        # (4) 출력 프로젝션 + pad back
        out = out.reshape(T, self.num_heads * self.head_dim)  # (T, D)
        out = self.out_proj(out.to(x.dtype))  # (T, D) -> in dtype
        out = out.to(x.dtype)
        # out = self._pad_to_batch(out, indices, B, L, D, x.device)  # (B, L, D)
        out = pad_input(out, indices, B, L)  # (B, L, D)
        return out

    # ------------------------------------------------------------------

    def forward(self, x, mask):
        """
        x:  [on_B, 1 + token_num, H]
        mask: [on_B, 1 + token_num]  # True=pad

        Note:
            - 입력을 먼저 마스크로 0 클램프(원본 유지)한 뒤 LN → varlen Self‑Attention.
            - pad back된 출력은 마스크 위치가 0이며, 이후 MLP/DropPath를 통과하면서도
              마스크는 다시 0으로 클램프합니다(원본 동작과 정합).
        """
        # (on_B, L, H)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x_norm = self.norm1(x)  # (on_B, L, H)

        # === FlashAttention‑2(varlen) 또는 폴백 ===
        y = self._self_attn_flash_varlen(x_norm, mask)  # (on_B, L, H)

        y = self.attn_out_drop(y)
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class StaticFusionEncoder(nn.Module):

    def __init__(self,
                 dim,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 device="cuda",
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self._hidden_dim = hidden_dim
        self.projection = Mlp(in_features=dim,
                              hidden_features=hidden_dim,
                              out_features=hidden_dim,
                              act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, static_objects):
        """
        static_objects: B, static_objects_num, D_10 (x, y, cos, sin, w, l, type(4))

        returns:
            static_encoding: (B, static_objects_num, hidden_dim)
            mask_p: (B, static_objects_num)
            static_feature: (B, static_objects_num, 4 + 5)
        """
        B, static_objects_num, _ = static_objects.shape
        static_xyyaw = static_objects[:, :, :4].clone(
        )  # (B, static_objects_num, 4)
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = self._get_static_feature(static_xyyaw)
        # [FIX] 오토캐스트 환경이면 버퍼 dtype을 현재 GPU autocast dtype으로
        out_dtype = (torch.get_autocast_gpu_dtype()
                     if torch.is_autocast_enabled() and static_objects.is_cuda
                     else static_objects.dtype)
        static_encoding = torch.zeros(
            (B * static_objects_num, self._hidden_dim),
            device=static_objects.device,
            dtype=out_dtype)

        mask_p = torch.sum(torch.ne(static_objects[..., :10], 0),
                           dim=-1).to(static_objects.device) == 0

        valid_indices = ~mask_p.reshape(-1)

        if valid_indices.sum().item() > 0:
            static_objects = static_objects.reshape(B * static_objects_num, -1)
            static_objects = static_objects[valid_indices]
            static_objects = self.projection(static_objects)
            static_objects = static_objects.to(
                dtype=static_encoding.dtype)  # [FIX] 좌변 dtype 일치화
            static_encoding[valid_indices] = static_objects
        else:
            # projection 파라미터를 0-스케일로 터치
            touch = static_encoding.new_zeros(())
            for p in self.projection.parameters():
                touch = touch + p.view(-1)[:1].sum()
            static_encoding = static_encoding + touch * 0.0
        hidden_dim = static_encoding.shape[-1]
        static_encoding = static_encoding.reshape(
            B, static_objects_num,
            hidden_dim)  # (B, static_objects_num, hidden_dim)
        mask_p = mask_p.reshape(B,
                                static_objects_num)  # (B, static_objects_num)
        return static_encoding, mask_p, static_feature

    def _get_static_feature(
        self,
        static_xyyaw: torch.Tensor  # (B, static_objects_num, 4)
    ) -> torch.Tensor:
        B, static_objects_num, _ = static_xyyaw.shape
        # static_type: (B, static_objects_num, 4) # 4:
        static_type = torch.zeros(
            (B, static_objects_num, 5),
            device=static_xyyaw.device,
            dtype=static_xyyaw.dtype,
        )
        static_type[:, :, 2] = 1.0  # type
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = torch.cat([static_xyyaw, static_type], dim=-1)
        assert static_feature.shape == (B, static_objects_num, 9), \
            f"Expected static_feature shape (B, static_objects_num, 9), got {static_feature.shape}"
        return static_feature


class RoadSafetyFusionEncoder(nn.Module):
    """Road-safety(정지표지/횡단보도/과속방지/진출입로/도로경계) 요소를
    작은 개수의 토큰으로 압축해 반환하는 인코더입니다.

    설계 목표
    ----------
    - 점(포인트)들을 그대로 토큰화하지 않고,
      1) 요소(폴리곤/폴리라인) 단위로 먼저 요약(1요소=1벡터),
      2) 전체 요소 집합을 다시 num_seeds개 요약 토큰으로 압축합니다.
    - 카테고리별 토큰 상한을 따로 두지 않습니다.
    - 가까운 것만 K개 선택 같은 휴리스틱을 사용하지 않습니다.
    - is_valid 입력을 그대로 신뢰해 마스크를 만들며,
      "점이 전부 0이면 무조건 invalid" 같은 추가 휴리스틱을 넣지 않습니다.

    반환
    ----------
    - encoding_road_safety: (B, num_seeds, hidden_dim)
    - road_safety_mask:     (B, num_seeds)  True=pad(무효)
    - road_safety_pos:      (B, num_seeds, 9) = [x,y,cos,sin] + type_onehot(5)
      여기서 type_onehot은 마지막 인덱스(4)를 road_safety로 둡니다.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        num_seeds: int = 8,
        ffn_ratio: float = 2.0,
        attn_drop_p: float = 0.0,
        out_drop_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim: int = int(hidden_dim)
        self.num_seeds: int = int(num_seeds)
        self.attn_drop_p: float = float(attn_drop_p)

        # (cx, cy, dir_x, dir_y, extent_x, extent_y, length, closed_flag)
        self._numeric_in_dim: int = 8

        self.numeric_mlp = Mlp(
            in_features=self._numeric_in_dim,
            hidden_features=max(32, int(self.hidden_dim * float(ffn_ratio))),
            out_features=self.hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # 카테고리(one-hot 5) 임베딩 + 스칼라 게이트(작게 시작)
        self.category_emb = nn.Linear(5, self.hidden_dim, bias=True)
        self.category_scale = nn.Parameter(torch.tensor(0.01))

        # road_edge_type(3) 임베딩 + 스칼라 게이트(작게 시작)
        self.road_edge_type_emb = nn.Linear(3, self.hidden_dim, bias=True)
        self.road_edge_type_scale = nn.Parameter(torch.tensor(0.01))

        # 요소 임베딩 정규화
        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # 요약 토큰(학습되는 쿼리)
        self.seeds = nn.Parameter(
            torch.zeros(1, self.num_seeds, self.hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)

        # 요약 토큰 쪽 잔차 FFN(게이트 0 시작)
        ffn_hidden = max(32, int(self.hidden_dim * float(ffn_ratio)))
        self.seed_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, self.hidden_dim, bias=True),
        )
        self.seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # DeepSets(mean/max) 잔차 게이트(0 시작)
        self.ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        # 출력 정규화/드롭
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_drop = nn.Dropout(
            out_drop_p) if out_drop_p > 0 else nn.Identity()

    # ------------------------- 유틸 함수들 ------------------------- #

    @staticmethod
    def _infer_batch_size(tensors: List[Optional[torch.Tensor]]) -> int:
        """입력 텐서들 중 하나에서 배치 크기(B)를 추정합니다.

        Args:
            tensors: None일 수도 있는 텐서들의 리스트.

        Returns:
            int: 배치 크기 B.
        """
        for t in tensors:
            if t is not None:
                return int(t.shape[0])
        raise ValueError("RoadSafetyFusionEncoder: 배치 크기(B)를 추정할 수 없습니다.")

    @staticmethod
    def _get_output_dtype(ref: torch.Tensor) -> torch.dtype:
        """AMP 환경이면 autocast dtype을, 아니면 ref dtype을 사용합니다."""
        if torch.is_autocast_enabled() and ref.is_cuda:
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                pass
        return ref.dtype

    def _touch_self_parameters(self, ref: torch.Tensor) -> torch.Tensor:
        """ref 텐서에 self 파라미터들을 0계수로 연결해 unused param 문제를 줄입니다.

        Args:
            ref: 기준 텐서. (shape 무관)

        Returns:
            ref와 같은 shape의 텐서(값은 같고, 그래프만 살짝 연결됨)
        """
        touch = ref.new_zeros(())
        for p in self.parameters():
            touch = touch + p.view(-1)[:1].sum().to(ref.dtype)
        return ref + touch * 0.0

    @staticmethod
    def _build_element_mask(
            points: torch.Tensor,  # (B, N, S, 2)
            is_valid: Optional[torch.Tensor],  # (B, N) or None
    ) -> torch.Tensor:
        """요소 단위 패딩 마스크(True=무효)를 만듭니다.

        주의:
            - is_valid가 주어지면 그것만 신뢰합니다.
            - is_valid가 None이면 "추가 휴리스틱"을 쓰지 않고, 전부 유효로 간주합니다.

        Args:
            points: (B, N, S, 2)
            is_valid: (B, N) 또는 None

        Returns:
            (B, N) bool 텐서. True=pad(무효)
        """
        B, N = int(points.shape[0]), int(points.shape[1])
        if is_valid is None:
            return torch.zeros((B, N), device=points.device, dtype=torch.bool)
        return (~is_valid.to(torch.bool)).reshape(B, N)

    @staticmethod
    def _principal_axis_dir(points_xy_f32: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
        """점 구름이 '가장 길게 늘어난 방향'을 (cos, sin)으로 반환합니다.

        Args:
            points_xy_f32: (M, S, 2) float32
            eps: 수치 안정용 작은 값

        Returns:
            (M, 2) float32, [cos, sin]
        """
        mean = points_xy_f32.mean(dim=1)  # (M, 2)
        centered = points_xy_f32 - mean.unsqueeze(1)  # (M, S, 2)
        x = centered[..., 0]  # (M, S)
        y = centered[..., 1]  # (M, S)

        cov_xx = (x * x).mean(dim=1)  # (M,)
        cov_yy = (y * y).mean(dim=1)  # (M,)
        cov_xy = (x * y).mean(dim=1)  # (M,)

        theta = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy + eps)  # (M,)
        return torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=-1)  # (M, 2)

    @staticmethod
    def _compute_geometry_features(
        points_valid: torch.Tensor,  # (M, S, 2)
        closed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """요소(폴리곤/폴리라인)의 점 집합에서 간단한 기하 요약을 만듭니다.

        Args:
            points_valid: (M, S, 2)  유효 요소만 모은 점들
            closed: True면 폴리곤(닫힘), False면 폴리라인(열림)

        Returns:
            numeric_feat: (M, 8) float32
                [cx, cy, dir_x, dir_y, extent_x, extent_y, length, closed_flag]
            pos4: (M, 4) float32
                [cx, cy, dir_x, dir_y]  (Fusion에서 pos_emb용)
        """
        pts = points_valid.to(torch.float32)  # (M, S, 2)

        center = pts.mean(dim=1)  # (M, 2)
        dir_vec = RoadSafetyFusionEncoder._principal_axis_dir(pts)  # (M, 2)

        min_xy = pts.amin(dim=1)  # (M, 2)
        max_xy = pts.amax(dim=1)  # (M, 2)
        extent = max_xy - min_xy  # (M, 2)

        # 길이: 인접 점 거리 합
        diffs = pts[:, 1:, :] - pts[:, :-1, :]  # (M, S-1, 2)
        seglen = torch.sqrt((diffs * diffs).sum(dim=-1) + 1e-6).sum(
            dim=1, keepdim=True)  # (M, 1)

        # 폴리곤이면 마지막-처음도 더해 "둘레" 쪽으로
        if closed and pts.shape[1] >= 2:
            last = pts[:, -1, :]  # (M, 2)
            first = pts[:, 0, :]  # (M, 2)
            close_len = torch.sqrt(((last - first)**2).sum(dim=-1) +
                                   1e-6).unsqueeze(-1)  # (M, 1)
            seglen = seglen + close_len  # (M, 1)

        closed_flag = torch.full(
            (pts.shape[0], 1),
            1.0 if closed else 0.0,
            device=pts.device,
            dtype=torch.float32,
        )  # (M, 1)

        numeric_feat = torch.cat([center, dir_vec, extent, seglen, closed_flag],
                                 dim=-1)  # (M, 8)
        pos4 = torch.cat([center, dir_vec], dim=-1)  # (M, 4)
        return numeric_feat, pos4

    def _encode_category(
        self,
        points: torch.Tensor,  # (B, N, S, 2)
        is_valid: Optional[torch.Tensor],  # (B, N) or None
        category_index: int,
        closed: bool,
        road_edge_type: Optional[
            torch.Tensor] = None,  # (B, N, 3) road_edge만 사용
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 종류의 요소들을 (B, N, hidden_dim) 임베딩으로 바꿉니다.

        Args:
            points: (B, N, S, 2)
            is_valid: (B, N) 또는 None
            category_index: 0..4 (stop/cross/speed/driveway/edge)
            closed: 폴리곤이면 True, 폴리라인이면 False
            road_edge_type: (B, N, 3) 또는 None

        Returns:
            emb:  (B, N, H)
            mask: (B, N) True=pad(무효)
            pos4: (B, N, 4) [x,y,cos,sin]
        """
        B, N, S, _ = points.shape
        device = points.device
        out_dtype = self._get_output_dtype(points)

        element_mask = self._build_element_mask(points,
                                                is_valid)  # (B, N) True=pad
        valid_flat = (~element_mask).reshape(B * N)  # (B*N,)
        num_valid: int = int(valid_flat.sum().item())

        emb_flat = torch.zeros((B * N, self.hidden_dim),
                               device=device,
                               dtype=out_dtype)  # (B*N, H)
        pos4_flat = torch.zeros((B * N, 4), device=device,
                                dtype=out_dtype)  # (B*N, 4)

        if num_valid > 0:
            points_flat = points.reshape(B * N, S, 2)  # (B*N, S, 2)
            points_valid = points_flat[valid_flat]  # (M, S, 2)

            numeric_feat_f32, pos4_f32 = self._compute_geometry_features(
                points_valid, closed=closed)  # (M,8),(M,4)
            numeric_feat = numeric_feat_f32.to(device=device,
                                               dtype=out_dtype)  # (M,8)

            emb_valid = self.numeric_mlp(numeric_feat)  # (M, H)

            # 카테고리 임베딩(게이트로 작게 시작)
            onehot = torch.zeros((1, 5), device=device,
                                 dtype=emb_valid.dtype)  # (1,5)
            onehot[0, category_index] = 1.0
            cat_emb = self.category_emb(onehot).expand(num_valid, -1)  # (M, H)
            emb_valid = emb_valid + self.category_scale.to(
                emb_valid.dtype) * cat_emb

            # road_edge_type 임베딩(road_edge에서만)
            if road_edge_type is not None:
                if road_edge_type.shape[:2] != (
                        B, N) or road_edge_type.shape[-1] != 3:
                    raise ValueError(
                        f"road_edge_type must be (B, N, 3). got {tuple(road_edge_type.shape)}"
                    )
                edge_type_valid = road_edge_type.reshape(
                    B * N, 3)[valid_flat].to(device=device,
                                             dtype=emb_valid.dtype)  # (M,3)
                emb_valid = emb_valid + self.road_edge_type_scale.to(
                    emb_valid.dtype) * self.road_edge_type_emb(edge_type_valid)

            emb_valid = self.in_norm(emb_valid)  # (M, H)

            emb_flat[valid_flat] = emb_valid.to(out_dtype)
            pos4_flat[valid_flat] = pos4_f32.to(device=device, dtype=out_dtype)

        emb = emb_flat.reshape(B, N, self.hidden_dim)  # (B, N, H)
        pos4 = pos4_flat.reshape(B, N, 4)  # (B, N, 4)

        # pad 위치는 0 고정
        emb = emb.masked_fill(element_mask.unsqueeze(-1), 0.0)
        pos4 = pos4.masked_fill(element_mask.unsqueeze(-1), 0.0)

        return emb, element_mask, pos4

    @staticmethod
    def _concat_parts(
        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """카테고리별 결과를 dim=1(요소 축)으로 합칩니다."""
        embs = [p[0] for p in parts]
        masks = [p[1] for p in parts]
        pos4s = [p[2] for p in parts]
        return torch.cat(embs, dim=1), torch.cat(masks, dim=1), torch.cat(pos4s,
                                                                          dim=1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x:(B,N,H), mask:(B,N) True=pad -> (B,H)"""
        valid = (~mask).to(torch.float32)  # (B,N)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x:(B,N,H), mask:(B,N) True=pad -> (B,H), 전부 pad인 배치는 0"""
        very_neg = torch.finfo(
            x.dtype).min if x.dtype in (torch.float16,
                                        torch.bfloat16) else -1e30
        x2 = x.masked_fill(mask.unsqueeze(-1), very_neg)
        mx = x2.amax(dim=1)  # (B,H)
        all_off = mask.all(dim=1)  # (B,)
        return torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)

    @staticmethod
    def _normalize_dir(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """v:(...,2)를 길이 1로 정규화합니다."""
        n = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps)
        return v / n

    def _seed_pool(
        self,
        elements_emb: torch.Tensor,  # (B, N_total, H)
        elements_pos4: torch.Tensor,  # (B, N_total, 4)
        elements_mask: torch.Tensor,  # (B, N_total) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """요소 집합을 num_seeds개의 '요약 토큰'으로 압축합니다.

        Returns:
            seed_emb: (B, M, H)
            seed_mask:(B, M) True=pad
            seed_pos4:(B, M, 4)
        """
        B, N_total, H = elements_emb.shape
        M = self.num_seeds
        device = elements_emb.device
        dtype = elements_emb.dtype

        if N_total == 0:
            seed_emb = torch.zeros((B, M, H), device=device, dtype=dtype)
            seed_pos4 = torch.zeros((B, M, 4), device=device, dtype=dtype)
            seed_mask = torch.ones((B, M), device=device, dtype=torch.bool)
            seed_emb = self._touch_self_parameters(seed_emb)
            return seed_emb, seed_mask, seed_pos4

        seeds = self.seeds.to(dtype).expand(B, M, H)  # (B,M,H)
        logits = torch.einsum("bmh,bnh->bmn", seeds, elements_emb) / math.sqrt(
            max(1.0, float(H)))  # (B,M,N)

        if elements_mask.any().item():
            logits = logits.masked_fill(elements_mask.unsqueeze(1),
                                        float("-inf"))

        all_off = elements_mask.all(dim=1)  # (B,)
        if all_off.any().item():
            logits = logits.clone()
            logits[all_off] = 0.0  # softmax NaN 방지

        attn = F.softmax(logits, dim=-1)  # (B,M,N)
        if self.attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop_p)

        seed_emb = torch.einsum("bmn,bnh->bmh", attn, elements_emb)  # (B,M,H)
        seed_emb = seed_emb + self.seed_ffn_alpha.to(dtype) * self.seed_ffn(
            seed_emb)

        # 집합 통계(mean/max)를 요약 토큰에 잔차로 추가 (게이트 0 시작)
        ds_mean = self._masked_mean(elements_emb, elements_mask)  # (B,H)
        ds_max = self._masked_max(elements_emb, elements_mask)  # (B,H)
        seed_emb = seed_emb + self.ds_mean_alpha.to(dtype) * ds_mean.unsqueeze(
            1) + self.ds_max_alpha.to(dtype) * ds_max.unsqueeze(1)

        seed_emb = self.out_drop(self.out_norm(seed_emb))  # (B,M,H)

        # pos4도 같은 attn으로 가중합 (방향은 정규화)
        pos_xy = torch.einsum("bmn,bnd->bmd", attn,
                              elements_pos4[..., :2])  # (B,M,2)
        pos_dir = torch.einsum("bmn,bnd->bmd", attn,
                               elements_pos4[..., 2:])  # (B,M,2)
        pos_dir = self._normalize_dir(pos_dir)
        seed_pos4 = torch.cat([pos_xy, pos_dir], dim=-1)  # (B,M,4)

        # batch마다 유효 요소 수에 따라 앞쪽 K개 seed만 활성화
        num_valid = (~elements_mask).sum(dim=1).to(torch.long)  # (B,)
        active_k = torch.clamp(num_valid, max=M)  # (B,)
        rank = torch.arange(M, device=device).unsqueeze(0)  # (1,M)
        seed_active = rank < active_k.unsqueeze(1)  # (B,M)
        seed_mask = ~seed_active  # True=pad

        seed_emb = seed_emb.masked_fill(seed_mask.unsqueeze(-1), 0.0)
        seed_pos4 = seed_pos4.masked_fill(seed_mask.unsqueeze(-1), 0.0)
        return seed_emb, seed_mask, seed_pos4

    @staticmethod
    def _build_pos9_from_pos4(seed_pos4: torch.Tensor) -> torch.Tensor:
        """pos4([x,y,cos,sin])에 type_onehot(5)을 붙여 (B,M,9)을 만듭니다."""
        B, M, _ = seed_pos4.shape
        token_type = torch.zeros((B, M, 5),
                                 device=seed_pos4.device,
                                 dtype=seed_pos4.dtype)
        token_type[:, :, 4] = 1.0  # road_safety
        return torch.cat([seed_pos4, token_type], dim=-1)  # (B,M,9)

    # ------------------------- forward ------------------------- #

    def forward(
            self,
            stop_sign_points: Optional[torch.Tensor],  # (B, Ns, S, 2)
            stop_sign_is_valid: Optional[torch.Tensor],  # (B, Ns)
            crosswalk_points: Optional[torch.Tensor],  # (B, Nc, S, 2)
            crosswalk_is_valid: Optional[torch.Tensor],  # (B, Nc)
            speed_bump_points: Optional[torch.Tensor],  # (B, Nb, S, 2)
            speed_bump_is_valid: Optional[torch.Tensor],  # (B, Nb)
            driveway_points: Optional[torch.Tensor],  # (B, Nd, S, 2)
            driveway_is_valid: Optional[torch.Tensor],  # (B, Nd)
            road_edge: Optional[torch.Tensor],  # (B, Ne, S, 2)
            road_edge_is_valid: Optional[torch.Tensor],  # (B, Ne)
            road_edge_type: Optional[torch.Tensor],  # (B, Ne, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Road-safety 입력을 요약 토큰으로 변환합니다.

        Returns:
            encoding_road_safety: (B, num_seeds, hidden_dim)
            road_safety_mask:     (B, num_seeds)  True=pad
            road_safety_pos:      (B, num_seeds, 9)
        """
        B = self._infer_batch_size([
            stop_sign_points, stop_sign_is_valid, crosswalk_points,
            crosswalk_is_valid, speed_bump_points, speed_bump_is_valid,
            driveway_points, driveway_is_valid, road_edge, road_edge_is_valid,
            road_edge_type
        ])

        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # 카테고리 인덱스: 0 stop, 1 crosswalk, 2 speed_bump, 3 driveway, 4 road_edge
        if stop_sign_points is not None:
            parts.append(
                self._encode_category(stop_sign_points,
                                      stop_sign_is_valid,
                                      0,
                                      closed=True))
        if crosswalk_points is not None:
            parts.append(
                self._encode_category(crosswalk_points,
                                      crosswalk_is_valid,
                                      1,
                                      closed=True))
        if speed_bump_points is not None:
            parts.append(
                self._encode_category(speed_bump_points,
                                      speed_bump_is_valid,
                                      2,
                                      closed=True))
        if driveway_points is not None:
            parts.append(
                self._encode_category(driveway_points,
                                      driveway_is_valid,
                                      3,
                                      closed=True))
        if road_edge is not None:
            parts.append(
                self._encode_category(road_edge,
                                      road_edge_is_valid,
                                      4,
                                      closed=False,
                                      road_edge_type=road_edge_type))

        if len(parts) == 0:
            raise ValueError(
                "RoadSafetyFusionEncoder: 모든 입력이 None이라 처리할 수 없습니다.")

        elements_emb, elements_mask, elements_pos4 = self._concat_parts(
            parts)  # (B,N,H),(B,N),(B,N,4)
        total_valid = int((~elements_mask).sum().item())

        # 유효 요소가 하나도 없으면: 전부 pad 토큰 반환 + unused param 방지 터치
        if total_valid == 0:
            ref = elements_emb
            seed_emb = torch.zeros((B, self.num_seeds, self.hidden_dim),
                                   device=ref.device,
                                   dtype=ref.dtype)
            seed_mask = torch.ones((B, self.num_seeds),
                                   device=ref.device,
                                   dtype=torch.bool)
            seed_pos4 = torch.zeros((B, self.num_seeds, 4),
                                    device=ref.device,
                                    dtype=ref.dtype)
            seed_emb = self._touch_self_parameters(seed_emb)
            road_safety_pos = self._build_pos9_from_pos4(seed_pos4)
            return seed_emb, seed_mask, road_safety_pos

        seed_emb, seed_mask, seed_pos4 = self._seed_pool(
            elements_emb, elements_pos4, elements_mask)
        road_safety_pos = self._build_pos9_from_pos4(seed_pos4)
        return seed_emb, seed_mask, road_safety_pos


class LaneSummaryTokenPooler(nn.Module):
    """많은 lane 토큰을 작은 개수의 '요약 토큰'으로 압축하는 모듈입니다.

    목적
    ----
    FusionEncoder는 토큰 수가 많아질수록 계산량이 매우 크게 늘어납니다.
    lane 토큰은 개수가 큰 편이라 Fusion 전체 비용을 크게 만드는 원인이 될 수 있습니다.

    이 모듈은 lane 토큰(L개)을 직접 Fusion에 넣는 대신,
    M개(num_seeds)의 요약 토큰으로 압축해서 Fusion에 넣을 수 있게 해줍니다.

    입력 / 출력 모양
    --------------
    입력:
        lane_embeddings: (B, L, H)
        lane_mask:       (B, L)     True=패딩(무효)
        lane_pos:        (B, L, 9)  [x,y,cos,sin] + type_onehot(5)

    출력:
        summary_embeddings: (B, M, H)
        summary_mask:       (B, M)     True=패딩(무효)
        summary_pos:        (B, M, 9)

    동작(쉽게 설명)
    --------------
    - 학습되는 M개의 "요약 기준 벡터"가 있고,
      각 요약 기준 벡터가 L개의 lane 토큰을 얼마나 참고할지 가중치를 만든 뒤,
      가중치로 lane 토큰들을 섞어서 요약 토큰을 만듭니다.
    - pos(위치/방향)도 같은 가중치로 함께 섞어 만들어 줍니다.
    - lane이 너무 적으면(유효 lane 수 < M) 뒤쪽 요약 토큰은 패딩 처리합니다.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int,
        ffn_ratio: float = 3.,
        attn_drop_p: float = 0.0,
        out_drop_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim: int = int(hidden_dim)
        self.num_seeds: int = int(num_seeds)
        self.attn_drop_p: float = float(attn_drop_p)

        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # (1, M, H) 학습되는 요약 기준 벡터
        self.seeds = nn.Parameter(
            torch.zeros(1, self.num_seeds, self.hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)

        # 요약 토큰에 붙는 작은 FFN (게이트 0에서 시작)
        ffn_hidden = max(32, int(self.hidden_dim * float(ffn_ratio)))
        self.seed_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, self.hidden_dim, bias=True),
        )
        self.seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # 집합 통계(mean/max) 잔차 게이트(0에서 시작)
        self.ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_drop = nn.Dropout(
            out_drop_p) if out_drop_p > 0 else nn.Identity()

    @staticmethod
    def _normalize_dir(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """방향 벡터를 길이 1로 정규화합니다.

        Args:
            v: (..., 2)
            eps: 0으로 나누기 방지용 작은 값

        Returns:
            (..., 2)  길이 1로 정규화된 방향 벡터
        """
        n = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps)
        return v / n

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """패딩을 제외한 평균을 계산합니다.

        Args:
            x: (B, L, H)
            mask: (B, L) True=패딩(무효)

        Returns:
            (B, H)
        """
        valid = (~mask).to(torch.float32)  # (B,L)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """패딩을 제외한 최댓값을 계산합니다. 전부 패딩이면 0을 반환합니다.

        Args:
            x: (B, L, H)
            mask: (B, L) True=패딩(무효)

        Returns:
            (B, H)
        """
        very_neg = (torch.finfo(x.dtype).min
                    if x.dtype in (torch.float16, torch.bfloat16) else -1e30)
        x2 = x.masked_fill(mask.unsqueeze(-1), very_neg)  # (B,L,H)
        mx = x2.amax(dim=1)  # (B,H)
        all_off = mask.all(dim=1)  # (B,)
        return torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)

    def _touch_self_parameters(self, ref: torch.Tensor) -> torch.Tensor:
        """값은 그대로 두고, 파라미터들을 0계수로 살짝 연결합니다(학습 안정용).

        Args:
            ref: 기준 텐서 (shape 무관)

        Returns:
            ref와 같은 shape 텐서(값은 같고 그래프만 연결됨)
        """
        touch = ref.new_zeros(())
        for p in self.parameters():
            touch = touch + p.view(-1)[:1].sum().to(ref.dtype)
        return ref + touch * 0.0

    def forward(
            self,
            lane_embeddings: torch.Tensor,  # (B, L, H)
            lane_mask: torch.Tensor,  # (B, L) True=pad
            lane_pos: torch.Tensor,  # (B, L, 9)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """lane 토큰을 M개 요약 토큰으로 압축합니다.

        Args:
            lane_embeddings:
                shape: (B, L, H)
            lane_mask:
                shape: (B, L)  True=패딩(무효)
            lane_pos:
                shape: (B, L, 9)
                [x,y,cos,sin] + type_onehot(5)

        Returns:
            summary_embeddings:
                shape: (B, M, H)
            summary_mask:
                shape: (B, M)  True=패딩(무효)
            summary_pos:
                shape: (B, M, 9)
        """
        if lane_mask.dtype != torch.bool:
            lane_mask = lane_mask.to(torch.bool)

        B, L, H = lane_embeddings.shape
        if H != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch. got {H}, expected {self.hidden_dim}")
        if lane_pos.shape != (B, L, 9):
            raise ValueError(
                f"lane_pos must be (B, L, 9). got {tuple(lane_pos.shape)}")

        M: int = int(self.num_seeds)
        device = lane_embeddings.device
        dtype = lane_embeddings.dtype

        # (A) M==0이면 빈 출력
        if M <= 0:
            empty_emb = lane_embeddings.new_zeros(B, 0, H)  # (B,0,H)
            empty_mask = torch.ones((B, 0), device=device,
                                    dtype=torch.bool)  # (B,0)
            empty_pos = lane_embeddings.new_zeros(B, 0, 9)  # (B,0,9)
            return empty_emb, empty_mask, empty_pos

        # (B) L==0이면 요약 불가 → 전부 패딩
        if L == 0:
            summary_embeddings = lane_embeddings.new_zeros(B, M, H)  # (B,M,H)
            summary_mask = torch.ones((B, M), device=device,
                                      dtype=torch.bool)  # (B,M)
            summary_pos = lane_embeddings.new_zeros(B, M, 9)  # (B,M,9)
            summary_embeddings = self._touch_self_parameters(summary_embeddings)
            return summary_embeddings, summary_mask, summary_pos

        # (C) 입력 정리: 패딩은 0으로 고정하고, 정규화 후 다시 0으로 고정
        lane_embeddings = lane_embeddings.masked_fill(lane_mask.unsqueeze(-1),
                                                      0.0)  # (B,L,H)
        lane_embeddings = self.in_norm(lane_embeddings)  # (B,L,H)
        lane_embeddings = lane_embeddings.masked_fill(lane_mask.unsqueeze(-1),
                                                      0.0)  # (B,L,H)

        # (D) 요약 가중치 계산: (B,M,L)
        seeds = self.seeds.to(dtype).expand(B, M, H)  # (B,M,H)
        logits = torch.einsum("bmh,blh->bml", seeds.float(),
                              lane_embeddings.float())
        logits = logits / math.sqrt(max(1.0, float(H)))  # (B,M,L)

        if lane_mask.any().item():
            logits = logits.masked_fill(lane_mask.unsqueeze(1), float("-inf"))

        # 전부 패딩인 배치 softmax NaN 방지
        all_off = lane_mask.all(dim=1)  # (B,)
        if all_off.any().item():
            logits = logits.clone()
            logits[all_off] = 0.0

        attn = F.softmax(logits, dim=-1)  # (B,M,L) float32
        if self.attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop_p)

        # (E) 요약 토큰 생성: (B,M,H)
        summary_embeddings = torch.einsum("bml,blh->bmh", attn,
                                          lane_embeddings.float()).to(
                                              dtype)  # (B,M,H)

        summary_embeddings = summary_embeddings + self.seed_ffn_alpha.to(
            dtype) * self.seed_ffn(summary_embeddings)

        # 집합 통계 잔차(게이트 0에서 시작)
        ds_mean = self._masked_mean(lane_embeddings,
                                    lane_mask).to(dtype)  # (B,H)
        ds_max = self._masked_max(lane_embeddings, lane_mask).to(dtype)  # (B,H)
        summary_embeddings = summary_embeddings \
            + self.ds_mean_alpha.to(dtype) * ds_mean.unsqueeze(1) \
            + self.ds_max_alpha.to(dtype) * ds_max.unsqueeze(1)  # (B,M,H)

        summary_embeddings = self.out_drop(
            self.out_norm(summary_embeddings))  # (B,M,H)

        # (F) pos(대표 위치/방향)도 같은 가중치로 계산
        lane_pos4 = lane_pos[..., :4].to(torch.float32)  # (B,L,4)
        pos_xy = torch.einsum("bml,bld->bmd", attn,
                              lane_pos4[..., :2])  # (B,M,2)
        pos_dir = torch.einsum("bml,bld->bmd", attn, lane_pos4[...,
                                                               2:4])  # (B,M,2)
        pos_dir = self._normalize_dir(pos_dir)  # (B,M,2)
        pos4 = torch.cat([pos_xy, pos_dir], dim=-1).to(dtype)  # (B,M,4)

        # type_onehot(5)에서 lane 인덱스(3)만 1로
        token_type = torch.zeros((B, M, 5), device=device,
                                 dtype=dtype)  # (B,M,5)
        token_type[:, :, 3] = 1.0
        summary_pos = torch.cat([pos4, token_type], dim=-1)  # (B,M,9)

        # (G) 유효 lane 수가 적으면 뒤쪽 seed는 패딩 처리
        num_valid = (~lane_mask).sum(dim=1).to(torch.long)  # (B,)
        active_k = torch.clamp(num_valid, max=M)  # (B,)
        rank = torch.arange(M, device=device).unsqueeze(0)  # (1,M)
        seed_active = rank < active_k.unsqueeze(1)  # (B,M)
        summary_mask = ~seed_active  # (B,M) True=pad

        summary_embeddings = summary_embeddings.masked_fill(
            summary_mask.unsqueeze(-1), 0.0)
        summary_pos = summary_pos.masked_fill(summary_mask.unsqueeze(-1), 0.0)

        return summary_embeddings, summary_mask, summary_pos


class LaneFusionEncoder(nn.Module):

    def __init__(self,
                 lane_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=192,
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self._lane_len = lane_len
        self.num_fourier_frequencies = num_fourier_frequencies
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        nn.init.normal_(self.speed_limit_emb.weight, std=0.02)

        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)
        nn.init.normal_(self.traffic_emb.weight, std=0.02)
        self.lane_type_emb = nn.Linear(4, channels_mlp_dim)
        nn.init.normal_(self.lane_type_emb.weight, std=0.02)
        self.left_line_type_emb = nn.Linear(13, channels_mlp_dim)
        nn.init.normal_(self.left_line_type_emb.weight, std=0.02)
        self.right_line_type_emb = nn.Linear(13, channels_mlp_dim)
        nn.init.normal_(self.right_line_type_emb.weight, std=0.02)
        # lane_type / line_type 임베딩 스칼라 게이트 (초기 0)
        # - 초반에는 영향이 거의 없고(0에 가까움),
        # - 학습이 "쓸만하다"고 판단하면 자동으로 커지도록 유도
        self.lane_type_alpha = nn.Parameter(torch.tensor(0.0))
        self.left_line_type_alpha = nn.Parameter(torch.tensor(0.0))
        self.right_line_type_alpha = nn.Parameter(torch.tensor(0.0))

        self.channel_pre_project = Mlp(in_features=10,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    @staticmethod
    def _apply_scalar_gate_to_embedding(
            embedding: torch.Tensor,  # (N, C)
            gate: torch.Tensor,  # ()
    ) -> torch.Tensor:  # (N, C)
        """임베딩에 '하나의 숫자(스칼라) 게이트'를 곱해 반영 정도를 조절합니다.

        이 함수는 lane_type / left_line_type / right_line_type처럼
        데이터에 따라 비어 있거나(정보가 없음), 노이즈가 있을 수 있는 입력의 영향을
        **학습이 알아서 켜고/끄도록** 만드는 용도입니다.

        동작:
            - gate는 학습되는 파라미터이며 시작값이 0입니다.
            - 시작값이 0이면 embedding을 더해도 영향이 거의 없어 초반 학습이 안정적입니다.
            - 학습이 진행되면서 필요하다고 판단되면 gate 값이 커져 해당 embedding의 영향이 커집니다.
            - 필요 없으면 gate가 0 근처에 머물러 영향이 계속 작게 유지됩니다.

        Args:
            embedding (torch.Tensor):
                모양: (N, C)
                - N: 유효 lane 개수(num_valid)
                - C: channel 차원(self._channel)
            gate (torch.Tensor):
                모양: () (스칼라)
                - embedding에 곱해질 하나의 숫자입니다.

        Returns:
            torch.Tensor:
                모양: (N, C)
                - gate가 반영된 embedding 입니다.
        """
        gate_value: torch.Tensor = gate.to(dtype=embedding.dtype,
                                           device=embedding.device)  # ()
        return embedding * gate_value  # (N, C)

    @staticmethod
    def _compute_boundary_valid_flags(
            lanes_xy_and_offsets: torch.Tensor,  # (B, lane_num, lane_len, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """좌/우 경계 정보가 있는지 여부를 점 단위로 계산합니다.

        Args:
            lanes_xy_and_offsets (torch.Tensor):
                모양: (B, lane_num, lane_len, 8)
                마지막 채널 의미(예시):
                    0: x
                    1: y
                    2: 진행방향(또는 변화량) x
                    3: 진행방향(또는 변화량) y
                    4: 왼쪽 경계 변화량 x
                    5: 왼쪽 경계 변화량 y
                    6: 오른쪽 경계 변화량 x
                    7: 오른쪽 경계 변화량 y

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - left_is_valid:  (B, lane_num, lane_len) bool
                - right_is_valid: (B, lane_num, lane_len) bool

        Note:
            - 값이 0이면 "해당 경계 정보가 없다"라고 가정하고 True/False를 만듭니다.
        """
        if lanes_xy_and_offsets.dim(
        ) != 4 or lanes_xy_and_offsets.shape[-1] < 8:
            raise ValueError(
                f"lanes_xy_and_offsets must be (B, lane_num, lane_len, 8+). got {tuple(lanes_xy_and_offsets.shape)}"
            )

        # left: (B, lane_num, lane_len)
        left_is_valid: torch.Tensor = (lanes_xy_and_offsets[..., 4:6]
                                       != 0).any(dim=-1)
        # right: (B, lane_num, lane_len)
        right_is_valid: torch.Tensor = (lanes_xy_and_offsets[..., 6:8]
                                        != 0).any(dim=-1)

        return left_is_valid, right_is_valid

    def _embed_optional_lane_attribute(
        self,
        lane_attribute: Optional[torch.Tensor],  # (B, lane_num, F) or None
        embedding_layer: nn.Linear,  # F -> channel
        valid_indices: torch.Tensor,  # (B * lane_num,) bool
        batch_size: int,
        lane_num: int,
        out_dtype: torch.dtype,
        out_device: torch.device,
    ) -> torch.Tensor:
        """lane 속성(one-hot 또는 float 벡터)을 임베딩으로 바꿔서 valid lane에만 맞춰 반환합니다.

        Args:
            lane_attribute (Optional[torch.Tensor]):
                - None 이거나, 모양이 (B, lane_num, F) 인 텐서입니다.
                - None이면 "추가 정보 없음"으로 보고 0 임베딩을 반환합니다.
            embedding_layer (nn.Linear):
                - 입력 F를 channel 차원으로 바꾸는 선형 레이어입니다.
            valid_indices (torch.Tensor):
                - 모양: (B * lane_num,)
                - True인 위치가 "유효 lane"입니다.
            batch_size (int):
                - B
            lane_num (int):
                - lane_num
            out_dtype (torch.dtype):
                - 반환 임베딩의 dtype (보통 lanes.dtype)
            out_device (torch.device):
                - 반환 임베딩의 device (보통 lanes.device)

        Returns:
            torch.Tensor:
                - 모양: (num_valid, channel)
                - num_valid = valid_indices.sum()

        Note:
            - 입력이 None이거나, 유효 lane이 0개인 경우에도 학습이 안전하게 돌아가도록,
              값은 0이지만 레이어 파라미터를 아주 약하게 연결해 둡니다.
        """
        if valid_indices.dtype != torch.bool:
            valid_indices = valid_indices.to(torch.bool)

        num_valid: int = int(valid_indices.sum().item())
        # base: (num_valid, channel)
        base_embedding: torch.Tensor = torch.zeros(
            (num_valid, self._channel),
            device=out_device,
            dtype=out_dtype,
        )

        # 유효 lane이 없거나 입력이 None이면: 0 임베딩 + (레이어 파라미터 살짝 연결)
        if (lane_attribute is None) or (num_valid == 0):
            touch: torch.Tensor = base_embedding.new_zeros(())
            for p in embedding_layer.parameters():
                touch = touch + p.view(-1)[:1].sum().to(out_dtype)
            return base_embedding + touch * 0.0

        # ---- 입력 shape 점검 ----
        if lane_attribute.dim() != 3:
            raise ValueError(
                f"lane_attribute must be (B, lane_num, F). got {tuple(lane_attribute.shape)}"
            )
        if int(lane_attribute.shape[0]) != int(batch_size) or int(
                lane_attribute.shape[1]) != int(lane_num):
            raise ValueError(
                f"lane_attribute shape mismatch. expected (B={batch_size}, lane_num={lane_num}, F), "
                f"got {tuple(lane_attribute.shape)}")

        # ---- flatten 후 valid lane만 선택 ----
        lane_attribute = lane_attribute.to(device=out_device,
                                           dtype=out_dtype)  # (B, lane_num, F)
        lane_attribute_flat: torch.Tensor = lane_attribute.reshape(
            batch_size * lane_num, -1)  # (B*lane_num, F)

        if lane_attribute_flat.shape[0] != valid_indices.shape[0]:
            raise ValueError(
                f"valid_indices length mismatch. valid_indices={valid_indices.shape[0]}, "
                f"lane_attribute_flat={lane_attribute_flat.shape[0]}")

        lane_attribute_valid: torch.Tensor = lane_attribute_flat[
            valid_indices]  # (num_valid, F)
        lane_attribute_embedding: torch.Tensor = embedding_layer(
            lane_attribute_valid)  # (num_valid, channel)
        return lane_attribute_embedding.to(dtype=out_dtype)

    def _get_lane_feature(
        self, lane_xyyaw: torch.Tensor
        # (B, lane_num, 4)
    ) -> torch.Tensor:  # (B, lane_num, 4 + 5)
        B, lane_num, _ = lane_xyyaw.shape
        # lane_type: (B, lane_num, 4)
        lane_type = torch.zeros(
            (B, lane_num, 5),
            device=lane_xyyaw.device,
            dtype=lane_xyyaw.dtype,
        )
        lane_type[:, :, 3] = 1.0  # type
        # static_feature: (B, lane_num, 4 + 5)
        lane_feature = torch.cat([lane_xyyaw, lane_type], dim=-1)
        return lane_feature

    def forward(
        self,
        lanes: torch.Tensor,  # (B, lane_num, lane_len, D_lane)
        lanes_speed_limit: torch.Tensor,  # (B, lane_num, 1)
        lanes_has_speed_limit: torch.Tensor,  # (B, lane_num, 1)
        lane_type: Optional[torch.Tensor],  # (B, lane_num, 4) or None
        left_line_type: Optional[torch.Tensor],  # (B, lane_num, 13) or None
        right_line_type: Optional[torch.Tensor],  # (B, lane_num, 13) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """차선 정보를 lane 단위 임베딩으로 바꿔 반환합니다.

        Args:
            lanes (torch.Tensor):
                모양: (B, lane_num, lane_len, D_lane)
                예시 채널 구성:
                    - 앞 8개 채널: 차선 중심/경계 관련 값
                    - 뒤 4개 채널: 신호등(traffic) 관련 값
            lanes_speed_limit (torch.Tensor):
                모양: (B, lane_num, 1)
            lanes_has_speed_limit (torch.Tensor):
                모양: (B, lane_num, 1)  (0/1 또는 bool)
            lane_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 4) 또는 None
            left_line_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 13) 또는 None
            right_line_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 13) 또는 None

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - lane_embedding: (B, lane_num, hidden_dim)
                - mask_p:         (B, lane_num)  True=해당 lane이 비어있음
                - lane_feature:   (B, lane_num, 8)  위치 임베딩용 [x,y,dir(2), type(4)]
        """
        # traffic: (B, lane_num, 4)
        traffic: torch.Tensor = lanes[:, :, 0, 8:]

        # lanes_8: (B, lane_num, lane_len, 8)
        lanes_8: torch.Tensor = lanes[..., :8]

        # ---- (1) 좌/우 경계 유효 플래그 추가 ----
        left_is_valid, right_is_valid = self._compute_boundary_valid_flags(
            lanes_8)  # (B, lane_num, lane_len), (B, lane_num, lane_len)

        # lanes_10: (B, lane_num, lane_len, 10)
        lanes_10: torch.Tensor = torch.cat(
            [
                lanes_8,
                left_is_valid.unsqueeze(-1).to(lanes_8.dtype),
                right_is_valid.unsqueeze(-1).to(lanes_8.dtype),
            ],
            dim=-1,
        )

        # ---- (2) lane 위치 특징(lane_feature) 생성 ----
        # lane_pos: (B, lane_num, 4)
        lane_pos: torch.Tensor = lanes_10[:, :,
                                          int(self._lane_len / 2), :4].clone()
        lane_feature: torch.Tensor = self._get_lane_feature(
            lane_pos)  # (B, lane_num, 8)

        # ---- (3) lane 유효 마스크 ----
        B, lane_num, lane_len, _ = lanes_10.shape

        # mask_v: (B, lane_num, lane_len)  True=해당 점이 비어있음
        mask_v: torch.Tensor = torch.sum(torch.ne(lanes_10[..., :8], 0),
                                         dim=-1).to(lanes_10.device) == 0
        # mask_p: (B, lane_num) True=해당 lane이 전부 비어있음
        mask_p: torch.Tensor = torch.sum(~mask_v, dim=-1) == 0

        # flatten
        lanes_flat: torch.Tensor = lanes_10.reshape(
            B * lane_num, lane_len, -1)  # (B*lane_num, lane_len, 10)
        valid_indices: torch.Tensor = ~mask_p.reshape(-1)  # (B*lane_num,) bool
        num_valid: int = int(valid_indices.sum().item())

        # ---- (4) 유효 lane이 하나도 없으면 바로 종료(0 반환) ----
        if num_valid == 0:
            H: int = int(self.emb_project.fc2.out_features)

            out_dtype = (torch.get_autocast_gpu_dtype()
                         if torch.is_autocast_enabled() and lanes_flat.is_cuda
                         else lanes_flat.dtype)
            lane_embedding = torch.zeros((B, lane_num, H),
                                         device=lanes_flat.device,
                                         dtype=out_dtype)

            # 여러 장치 학습에서도 안전하게 돌아가도록, 이번 경로에서 쓰이지 않은 레이어도 "살짝 연결"
            touch = lane_embedding.new_zeros(())
            touch_modules = [
                self.channel_pre_project,
                self.token_pre_project,
                self.emb_project,
                self.speed_limit_emb,
                self.unknown_speed_emb,
                self.traffic_emb,
                self.lane_type_emb,
                self.left_line_type_emb,
                self.right_line_type_emb,
                *self.blocks,
            ]
            for mod in touch_modules:
                for p in mod.parameters():
                    touch = touch + p.view(-1)[:1].sum().to(out_dtype)
            touch = touch + self.lane_type_alpha.to(out_dtype)
            touch = touch + self.left_line_type_alpha.to(out_dtype)
            touch = touch + self.right_line_type_alpha.to(out_dtype)
            return lane_embedding + touch * 0.0, mask_p.reshape(
                B, lane_num), lane_feature

        # ---- (5) 유효 lane만 인코딩 ----
        lanes_valid: torch.Tensor = lanes_flat[
            valid_indices]  # (num_valid, lane_len, 10)

        # (num_valid, lane_len, 10) -> (num_valid, lane_len, channel)
        lanes_valid = self.channel_pre_project(lanes_valid)
        lanes_valid = lanes_valid.permute(0, 2,
                                          1)  # (num_valid, channel, lane_len)
        lanes_valid = self.token_pre_project(
            lanes_valid)  # (num_valid, channel, tokens_mlp_dim)
        lanes_valid = lanes_valid.permute(
            0, 2, 1)  # (num_valid, tokens_mlp_dim, channel)

        for block in self.blocks:
            lanes_valid = block(lanes_valid)

        # (num_valid, tokens_mlp_dim, channel) -> (num_valid, channel)
        lanes_valid = lanes_valid.float().mean(dim=1).to(lanes_valid.dtype)

        # ---- (6) speed limit / traffic 임베딩 추가 ----
        lanes_speed_limit_flat: torch.Tensor = lanes_speed_limit.reshape(
            B * lane_num, 1)  # (B*lane_num, 1)
        lanes_has_speed_limit_flat: torch.Tensor = lanes_has_speed_limit.to(
            torch.bool).reshape(B * lane_num, 1)  # (B*lane_num, 1)
        traffic_flat: torch.Tensor = traffic.reshape(B * lane_num,
                                                     -1)  # (B*lane_num, 4)

        lanes_has_speed_limit_valid: torch.Tensor = lanes_has_speed_limit_flat[
            valid_indices].squeeze(-1)  # (num_valid,)
        lanes_speed_limit_valid: torch.Tensor = lanes_speed_limit_flat[
            valid_indices].squeeze(-1)  # (num_valid,)
        traffic_valid: torch.Tensor = traffic_flat[
            valid_indices]  # (num_valid, 4)

        # speed_limit_embedding: (num_valid, channel)
        speed_limit_embedding: torch.Tensor = torch.zeros(
            (num_valid, self._channel),
            device=lanes_valid.device,
            dtype=lanes_valid.dtype,
        )
        if lanes_has_speed_limit_valid.any().item():
            speed_limit_with_limit = self.speed_limit_emb(
                lanes_speed_limit_valid[lanes_has_speed_limit_valid].unsqueeze(
                    -1)).to(lanes_valid.dtype)
            speed_limit_embedding[
                lanes_has_speed_limit_valid] = speed_limit_with_limit

        if (~lanes_has_speed_limit_valid).any().item():
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                int((~lanes_has_speed_limit_valid).sum().item()),
                -1).to(lanes_valid.dtype)
            speed_limit_embedding[
                ~lanes_has_speed_limit_valid] = speed_limit_no_limit

        traffic_light_embedding: torch.Tensor = self.traffic_emb(
            traffic_valid).to(lanes_valid.dtype)  # (num_valid, channel)

        lanes_valid = lanes_valid + speed_limit_embedding + traffic_light_embedding  # (num_valid, channel)

        # ---- (7) (추가) lane_type / left_line_type / right_line_type 임베딩 (None 안전) ----
        lane_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=lane_type,
            embedding_layer=self.lane_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        left_line_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=left_line_type,
            embedding_layer=self.left_line_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        right_line_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=right_line_type,
            embedding_layer=self.right_line_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        # 변경 (게이트 적용)
        lanes_valid = (
            lanes_valid + self._apply_scalar_gate_to_embedding(
                lane_type_embedding, self.lane_type_alpha) +
            self._apply_scalar_gate_to_embedding(left_line_type_embedding,
                                                 self.left_line_type_alpha) +
            self._apply_scalar_gate_to_embedding(right_line_type_embedding,
                                                 self.right_line_type_alpha))
        # ---- (8) 최종 hidden_dim으로 투영 ----
        lanes_valid = self.emb_project(
            self.norm(lanes_valid))  # (num_valid, hidden_dim)

        # ---- (9) 원래 (B, lane_num) 자리로 복원 ----
        lane_embedding_flat: torch.Tensor = torch.zeros(
            (B * lane_num, lanes_valid.shape[-1]),
            device=lanes_valid.device,
            dtype=lanes_valid.dtype,
        )
        lane_embedding_flat[valid_indices] = lanes_valid

        lane_embedding: torch.Tensor = lane_embedding_flat.reshape(
            B, lane_num, -1)  # (B, lane_num, hidden_dim)
        return lane_embedding, mask_p.reshape(B, lane_num), lane_feature


class FusionEncoder(nn.Module):

    def __init__(
            self,
            hidden_dim=192,
            num_heads=8,
            drop_path_rate=0.2,
            depth=3,
            attn_drop_p: float = 0.025,  # 권장 0.0~0.1
            ffn_drop_p: float = 0.05,  # 권장 0.0~0.1
            device='cuda'):
        super().__init__()

        # 1) CLS/scene 토큰과 그 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        # 0→drop_path_rate 선형 스케줄
        dpr_list = torch.linspace(0.0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim,
                num_heads,
                attn_drop_p=attn_drop_p,
                ffn_drop_p=ffn_drop_p,
                drop_path_p=dpr_list[i],
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self, encoding_input: torch.Tensor,
            encoding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """장면 융합 전용 포워드(배치별 전부 패딩 샘플은 건너뜀).

        모든 토큰이 패딩(True)인 배치는 연산을 생략하고 0을 반환한다.
        유효 토큰이 하나라도 있는 배치만 CLS 토큰을 붙여 블록을 통과시킨 뒤,
        최종 출력에서 CLS를 제거하고 원래 위치에 복원한다.


        # encoding_input:
        # encoding_mask:
        Args:
            encoding_input (torch.Tensor): 입력 토큰 시퀀스.
                모양: [B, token_num, H] (B, token_num, hidden_dim)
                - B: 배치 크기
                - token_num: 토큰 길이
                - H: 히든 차원
            encoding_mask (torch.Tensor): 키 패딩 마스크(True=패딩으로 무시).
                모양: [B, token_num]  (B, token_num)

        Returns:
            torch.Tensor: CLS 제외 최종 시퀀스 임베딩.
                모양: [B, token_num, H]
        """
        B, token_num, H = encoding_input.shape  # B, token_num, H 스칼라

        # 1) 전체 패딩 배치 식별 및 결과 버퍼 준비
        is_invalid_batch = encoding_mask.all(dim=1)  # [B]
        is_valid_batch = ~is_invalid_batch  # [B]
        out_tokens = encoding_input.new_zeros(B, token_num,
                                              H)  # [B, token_num, H]

        # 2) 유효 배치만 선택
        if is_valid_batch.any().item():
            # [on_B, token_num, H]
            on_batch_tokens = encoding_input[is_valid_batch]
            # [on_B, token_num]
            on_batch_token_mask = encoding_mask[is_valid_batch]

            on_B: int = on_batch_tokens.size(0)

            # 3) CLS 부착 및 CLS 위치 임베딩 추가
            cls_tokens = self.cls_token.expand(on_B, 1, H).to(
                on_batch_tokens.dtype)  # [on_B, 1, H]
            cls_with_tokens = torch.cat([cls_tokens, on_batch_tokens],
                                        dim=1)  # [on_B, token_num+1, H]
            cls_pos = self.cls_pos.to(cls_with_tokens.dtype)
            cls_with_tokens[:, 0:
                            1, :] = cls_with_tokens[:, 0:
                                                    1, :] + cls_pos  # [on_B, 1, H] += pos

            # 4) 마스크에 CLS(False) 추가
            cls_false = torch.zeros(
                on_B, 1, dtype=torch.bool,
                device=on_batch_token_mask.device)  # [on_B,1]
            cls_with_token_mask = torch.cat([cls_false, on_batch_token_mask],
                                            dim=1)  # [on_B, token_num+1]

            # 5) 블록 통과
            # cls_with_tokens  # [on_B, token_num+1, H]
            for block in self.blocks:
                cls_with_tokens = block(
                    cls_with_tokens,  # [on_B, token_num+1, H]
                    cls_with_token_mask)  # [on_B, token_num+1]
            cls_with_tokens = self.norm(
                cls_with_tokens)  # [on_B, token_num+1, H]
            # >>> add this to keep padded tokens truly zero <<<
            cls_with_tokens = cls_with_tokens.masked_fill(
                cls_with_token_mask.unsqueeze(-1), 0.0)

            # 6) CLS 제거 후 원래 배치 위치에 복원
            fused_wo_cls = cls_with_tokens[:, 1:, :]  # [on_B, token_num, H]
            out_tokens[is_valid_batch] = fused_wo_cls.to(
                out_tokens.dtype)  # [B, token_num, H]
        else:
            # 모든 배치가 패딩이면, 블록 파라미터를 0-스케일로 터치해 DDP unused param 방지
            touch = (self.cls_token[..., :1].sum() +
                     self.cls_pos[..., :1].sum())
            for blk in self.blocks:
                for p in blk.parameters():
                    touch = touch + p.view(-1)[:1].sum()
            out_tokens = out_tokens + touch * 0.0
            # ★ 추가: final LayerNorm 파라미터도 터치
            for p in self.norm.parameters():
                touch = touch + p.view(-1)[:1].sum()
        # 전부 패딩 배치는 out_tokens의 0 유지
        # out_tokens [B, token_num, H]
        # encoding_mask: [B, token_num]  # True=패딩
        return out_tokens, encoding_mask
