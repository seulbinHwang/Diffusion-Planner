from __future__ import annotations
import torch.nn as nn

# from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.encoder_wosac import Encoder
# from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.decoder_wosac import Decoder
from typing import Iterator

from dataclasses import dataclass
from typing import List, Optional, Set, Iterable, Dict, Tuple

import torch
import torch.nn as nn


def count_parameters(params: Iterable[nn.Parameter]) -> int:
    """파라미터(학습으로 값이 바뀌는 숫자들)의 총 개수를 셉니다.

    Args:
        params (Iterable[nn.Parameter]): 모델 안에 있는 파라미터들의 모음입니다.

    Returns:
        int: 파라미터를 이루는 '숫자'의 총 개수입니다.

    Notes:
        - 각 파라미터는 보통 여러 칸짜리 숫자 묶음(예: 행렬, 벡터)입니다.
        - 이 함수는 그 안에 들어있는 숫자 칸(원소) 개수를 전부 더합니다.
    """
    total: int = 0
    for p in params:
        total += int(p.numel())
    return total


def count_trainable_parameters(params: Iterable[nn.Parameter]) -> int:
    """학습 중에 실제로 값이 바뀌는(학습되는) 파라미터의 총 개수를 셉니다.

    Args:
        params (Iterable[nn.Parameter]): 모델 안에 있는 파라미터들의 모음입니다.

    Returns:
        int: 학습되는 파라미터를 이루는 '숫자'의 총 개수입니다.
    """
    total: int = 0
    for p in params:
        if p.requires_grad:
            total += int(p.numel())
    return total


@dataclass(frozen=True)
class ParamCountResult:
    """파라미터 개수 결과 묶음입니다."""
    total: int
    trainable: int


def summarize_module_params(module: nn.Module) -> ParamCountResult:
    """특정 모듈(예: 인코더, 디코더)의 파라미터 개수를 요약합니다.

    Args:
        module (nn.Module): 파라미터를 세고 싶은 PyTorch 모듈입니다.

    Returns:
        ParamCountResult: 총 개수와 학습되는 개수를 담아 돌려줍니다.
    """
    total: int = count_parameters(module.parameters())
    trainable: int = count_trainable_parameters(module.parameters())
    return ParamCountResult(total=total, trainable=trainable)


def summarize_iter_params(
        params_iter: Iterator[nn.Parameter]) -> ParamCountResult:
    """이터레이터로 제공되는 파라미터들의 개수를 요약합니다.

    Args:
        params_iter (Iterator[nn.Parameter]): 파라미터 이터레이터입니다.

    Returns:
        ParamCountResult: 총 개수와 학습되는 개수를 담아 돌려줍니다.
    """
    params_list = list(params_iter)  # 이터레이터는 1번 쓰면 끝나서 리스트로 고정
    total: int = count_parameters(params_list)
    trainable: int = count_trainable_parameters(params_list)
    return ParamCountResult(total=total, trainable=trainable)


def summarize_params(params: Iterable[nn.Parameter]) -> ParamCountResult:
    """파라미터 묶음의 개수를 요약합니다.

    Args:
        params (Iterable[nn.Parameter]):
            파라미터들의 모음입니다.
            각 파라미터는 여러 칸짜리 숫자 묶음(텐서)이며 shape은 제각각입니다.
            예: (out_dim, in_dim), (dim,), (C_out, C_in, kH, kW) 등.

    Returns:
        ParamCountResult:
            - total: 파라미터 숫자 칸 총합 (shape: ())
            - trainable: 학습되는 파라미터 숫자 칸 총합 (shape: ())
    """
    params_list: List[nn.Parameter] = list(params)
    total: int = count_parameters(params_list)
    trainable: int = count_trainable_parameters(params_list)
    return ParamCountResult(total=total, trainable=trainable)


def _collect_param_id_set(params: Iterable[nn.Parameter]) -> Set[int]:
    """파라미터들의 '고유 식별자(id)' 집합을 만듭니다.

    Args:
        params (Iterable[nn.Parameter]):
            파라미터들의 모음입니다.
            각 파라미터는 텐서이며 shape은 제각각입니다. (고정 shape 없음)

    Returns:
        Set[int]:
            파라미터 객체 id의 집합입니다. (shape: (N_params,))
    """
    out: Set[int] = set()
    for p in params:
        out.add(id(p))
    return out


def _filter_params_by_id(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    keep_ids: Set[int],
) -> List[nn.Parameter]:
    """named_parameters() 결과에서, 지정한 id에 해당하는 파라미터만 뽑습니다.

    Args:
        named_params (Iterable[Tuple[str, nn.Parameter]]):
            (이름, 파라미터) 튜플들의 모음입니다.
            파라미터 텐서의 shape은 제각각입니다.
        keep_ids (Set[int]):
            남길 파라미터 id 집합입니다. (shape: (N_keep,))

    Returns:
        List[nn.Parameter]:
            keep_ids에 해당하는 파라미터 리스트입니다. (length: N_keep)
    """
    kept: List[nn.Parameter] = []
    for _, p in named_params:
        if id(p) in keep_ids:
            kept.append(p)
    return kept


def _get_decoder_param_id_sets_by_category(
        decoder_module: nn.Module) -> Dict[str, Set[int]]:
    """Decoder 내부 파라미터를 4개 카테고리로 '겹치지 않게' 나눕니다.

    카테고리 규칙(겹치지 않게)
    -------------------------
    1) FeasibleProjector:
       - decoder_module.dit.feasible_projector 아래 파라미터(있을 때만)
    2) PRAM-v2:
       - DiT 내부 named_parameters 이름에 "pram_v2_"가 들어간 파라미터
         (예: pram_v2_composer, pram_v2_time_mod, pram_v2_* 등 + 관련 스칼라 파라미터)
    3) DiT(기본):
       - DiT 전체 파라미터에서 (FeasibleProjector + PRAM-v2)를 뺀 나머지
    4) 기타:
       - Decoder 전체 파라미터에서 (위 3개)를 뺀 나머지

    Args:
        decoder_module (nn.Module):
            보통 diffusion_planner.decoder.decoder(= Decoder 인스턴스) 입니다.
            텐서 입력 shape(B, Pnn, ...)과 무관하며, 파라미터만 봅니다.

    Returns:
        Dict[str, Set[int]]:
            각 카테고리별 파라미터 id 집합.
            - "dit_core"
            - "feasible_projector"
            - "pram_v2"
            - "others"
    """
    all_ids: Set[int] = _collect_param_id_set(decoder_module.parameters())

    dit = getattr(decoder_module, "dit", None)
    if not isinstance(dit, nn.Module):
        # DiT가 없으면 전부 기타로 처리
        return {
            "dit_core": set(),
            "feasible_projector": set(),
            "pram_v2": set(),
            "others": set(all_ids),
        }

    # 1) FeasibleProjector
    feasible_ids: Set[int] = set()
    feasible_module = getattr(dit, "feasible_projector", None)
    if isinstance(feasible_module, nn.Module):
        feasible_ids = _collect_param_id_set(feasible_module.parameters())

    # 2) PRAM-v2 (이름 규칙: "pram_v2_" 포함)
    pram_ids: Set[int] = set()
    for name, p in dit.named_parameters(recurse=True):
        # 예: "pram_v2_composer....", "pram_v2_final_scale_scalar" 등
        if "pram_v2_" in str(name):
            pram_ids.add(id(p))

    # 3) DiT(기본) = DiT 전체 - feasible - pram
    dit_all_ids: Set[int] = _collect_param_id_set(dit.parameters())
    dit_core_ids: Set[int] = dit_all_ids - feasible_ids - pram_ids

    # 4) 기타 = Decoder 전체 - (위 3개)
    others_ids: Set[int] = all_ids - feasible_ids - pram_ids - dit_core_ids

    return {
        "dit_core": dit_core_ids,
        "feasible_projector": feasible_ids,
        "pram_v2": pram_ids,
        "others": others_ids,
    }


def summarize_decoder_param_categories(
        decoder_module: nn.Module) -> Dict[str, ParamCountResult]:
    """Decoder 내부 파라미터를 4개 카테고리로 나눠 각각 개수를 요약합니다.

    Args:
        decoder_module (nn.Module):
            Decoder 인스턴스(보통 model.decoder.decoder) 입니다.

    Returns:
        Dict[str, ParamCountResult]:
            키는 아래 4개입니다.
            - "dit_core"
            - "feasible_projector"
            - "pram_v2"
            - "others"
    """
    id_sets = _get_decoder_param_id_sets_by_category(decoder_module)
    named_params = list(decoder_module.named_parameters(recurse=True))

    out: Dict[str, ParamCountResult] = {}
    for key, ids in id_sets.items():
        params = _filter_params_by_id(named_params, ids)
        out[key] = summarize_params(params)
    return out


def _collect_self_attention_block_param_ids(root_module: nn.Module) -> Set[int]:
    """root_module 내부에서 SelfAttentionBlock에 속한 파라미터 id들을 모읍니다.

    Args:
        root_module (nn.Module):
            보통 Encoder 인스턴스입니다.
            입력 텐서 shape(B, ...)와 무관하고, 모듈 구조만 봅니다.

    Returns:
        Set[int]:
            SelfAttentionBlock에 속한 파라미터 객체 id 집합입니다. (shape: (N_params,))
    """
    ids: Set[int] = set()
    for m in root_module.modules():
        # import 없이도 동작하도록 "클래스 이름 문자열"로만 판정합니다.
        if m.__class__.__name__ == "SelfAttentionBlock":
            for p in m.parameters():
                ids.add(id(p))
    return ids


def _get_encoder_param_id_sets_by_category(
        encoder_module: nn.Module) -> Dict[str, Set[int]]:
    """Encoder 내부 파라미터를 8개 카테고리로 '겹치지 않게' 나눕니다.

    카테고리(겹치지 않게)
    ---------------------
    1) AgentFusionEncoder          : encoder_module.agents_encoder
    2) StaticFusionEncoder         : encoder_module.static_encoder
    3) RoadSafetyFusionEncoder     : encoder_module.road_safety_encoder
    4) LaneFusionEncoder           : encoder_module.lane_encoder
    5) LaneSummaryTokenPooler      : encoder_module.lane_summary_pooler (없으면 0)
    6) SelfAttentionBlock          : Encoder 내부에 존재하는 SelfAttentionBlock 전부
    7) FusionEncoder(나머지)        : encoder_module.fusion 중에서 (SelfAttentionBlock 제외) 나머지
    8) 기타                        : Encoder 전체에서 위 7개를 뺀 나머지

    Args:
        encoder_module (nn.Module):
            보통 model.encoder.encoder(= Encoder 인스턴스) 입니다.
            입력 텐서 shape(B, ...)과 무관하며, 파라미터만 봅니다.

    Returns:
        Dict[str, Set[int]]:
            각 카테고리별 파라미터 id 집합입니다.
            - "agent_fusion_encoder"
            - "static_fusion_encoder"
            - "road_safety_fusion_encoder"
            - "lane_fusion_encoder"
            - "lane_summary_token_pooler"
            - "fusion_encoder"
            - "self_attention_block"
            - "others"
    """
    all_ids: Set[int] = _collect_param_id_set(encoder_module.parameters())

    agents_encoder = getattr(encoder_module, "agents_encoder", None)
    static_encoder = getattr(encoder_module, "static_encoder", None)
    road_safety_encoder = getattr(encoder_module, "road_safety_encoder", None)
    lane_encoder = getattr(encoder_module, "lane_encoder", None)
    lane_summary_pooler = getattr(encoder_module, "lane_summary_pooler", None)
    fusion = getattr(encoder_module, "fusion", None)

    agent_ids: Set[int] = set()
    if isinstance(agents_encoder, nn.Module):
        agent_ids = _collect_param_id_set(agents_encoder.parameters())

    static_ids: Set[int] = set()
    if isinstance(static_encoder, nn.Module):
        static_ids = _collect_param_id_set(static_encoder.parameters())

    road_safety_ids: Set[int] = set()
    if isinstance(road_safety_encoder, nn.Module):
        road_safety_ids = _collect_param_id_set(
            road_safety_encoder.parameters())

    lane_ids: Set[int] = set()
    if isinstance(lane_encoder, nn.Module):
        lane_ids = _collect_param_id_set(lane_encoder.parameters())

    lane_summary_ids: Set[int] = set()
    if isinstance(lane_summary_pooler, nn.Module):
        lane_summary_ids = _collect_param_id_set(
            lane_summary_pooler.parameters())

    # SelfAttentionBlock (FusionEncoder 내부 블록 포함)
    self_attn_ids: Set[int] = _collect_self_attention_block_param_ids(
        encoder_module)

    # FusionEncoder "나머지" (SelfAttentionBlock 제외)
    fusion_ids: Set[int] = set()
    if isinstance(fusion, nn.Module):
        fusion_all_ids: Set[int] = _collect_param_id_set(fusion.parameters())
        fusion_ids = fusion_all_ids - self_attn_ids

    used_ids: Set[int] = set().union(
        agent_ids,
        static_ids,
        road_safety_ids,
        lane_ids,
        lane_summary_ids,
        self_attn_ids,
        fusion_ids,
    )

    others_ids: Set[int] = all_ids - used_ids

    return {
        "agent_fusion_encoder": agent_ids,
        "static_fusion_encoder": static_ids,
        "road_safety_fusion_encoder": road_safety_ids,
        "lane_fusion_encoder": lane_ids,
        "lane_summary_token_pooler": lane_summary_ids,
        "fusion_encoder": fusion_ids,
        "self_attention_block": self_attn_ids,
        "others": others_ids,
    }


def summarize_encoder_param_categories(
        encoder_module: nn.Module) -> Dict[str, ParamCountResult]:
    """Encoder 내부 파라미터를 8개 카테고리로 나눠 각각 개수를 요약합니다.

    Args:
        encoder_module (nn.Module):
            Encoder 인스턴스(보통 model.encoder.encoder) 입니다.

    Returns:
        Dict[str, ParamCountResult]:
            키는 아래 8개입니다.
            - "agent_fusion_encoder"
            - "static_fusion_encoder"
            - "road_safety_fusion_encoder"
            - "lane_fusion_encoder"
            - "lane_summary_token_pooler"
            - "fusion_encoder"
            - "self_attention_block"
            - "others"
    """
    id_sets = _get_encoder_param_id_sets_by_category(encoder_module)
    named_params = list(encoder_module.named_parameters(recurse=True))

    out: Dict[str, ParamCountResult] = {}
    for key, ids in id_sets.items():
        params = _filter_params_by_id(named_params, ids)
        out[key] = summarize_params(params)
    return out


def _sort_param_count_results(
    groups: Dict[str, ParamCountResult],) -> List[Tuple[str, ParamCountResult]]:
    """그룹별 파라미터 개수를 '큰 순서'로 정렬합니다.

    Args:
        groups (Dict[str, ParamCountResult]):
            그룹 이름 -> 개수 요약 결과.
            텐서 입력 shape(B, ...)과 무관합니다.

    Returns:
        List[Tuple[str, ParamCountResult]]:
            (그룹 이름, 결과) 목록이며, total이 큰 순서로 정렬됩니다.
    """
    return sorted(groups.items(), key=lambda kv: kv[1].total, reverse=True)


def summarize_pram_v2_param_groups(
    dit_module: nn.Module,
    base_prefix: Optional[str],
    group_depth: int,
) -> Dict[str, ParamCountResult]:
    """PRAM-v2 관련 파라미터를 이름 기준으로 묶어서(그룹화) 개수를 셉니다.

    이 함수가 하는 일(쉽게 설명)
    ---------------------------
    - DiT 안의 파라미터들 중에서, 이름이 "pram_v2_"로 시작하는 것만 고릅니다.
    - 그리고 이름의 앞부분을 기준으로 묶습니다.
      예)
        group_depth=1이면:
          "pram_v2_composer. ...", "pram_v2_out_proj. ..." 처럼 큰 덩어리로 묶임
        group_depth=2이면:
          "pram_v2_composer.adapt_S", "pram_v2_out_proj.0" 처럼 더 잘게 묶임
    - base_prefix를 주면, 그 하위만 더 잘게 쪼갭니다.
      예) base_prefix="pram_v2_composer", group_depth=2

    Args:
        dit_module (nn.Module):
            보통 Decoder 안의 DiT 모듈입니다. (예: model.decoder.decoder.dit)
            입력 텐서 shape(B, ...)과 무관합니다.
        base_prefix (Optional[str]):
            - None이면 PRAM-v2 전체를 대상으로 합니다. (name이 "pram_v2_"로 시작)
            - 문자열이면 해당 prefix 아래만 대상으로 합니다.
              예: "pram_v2_composer" 또는 "pram_v2_out_proj"
        group_depth (int):
            이름을 '.'으로 나눈 뒤, 앞에서 몇 조각을 합쳐 그룹 이름으로 쓸지.
            - 1 또는 2를 권장합니다.

    Returns:
        Dict[str, ParamCountResult]:
            그룹 이름 -> (총 개수, 학습되는 개수)
    """
    if int(group_depth) < 1:
        raise ValueError(f"group_depth must be >= 1. got {group_depth}")

    named_params: List[Tuple[str, nn.Parameter]] = list(
        dit_module.named_parameters(recurse=True))

    # group_key -> param_id_set
    group_to_ids: Dict[str, Set[int]] = {}

    for name, p in named_params:
        # p: 파라미터 텐서 (shape은 각 파라미터마다 다름)
        if base_prefix is None:
            if not str(name).startswith("pram_v2_"):
                continue
        else:
            if str(name) != base_prefix and not str(name).startswith(
                    base_prefix + "."):
                continue

        parts: List[str] = str(name).split(".")
        key = ".".join(parts[:int(group_depth)]) if len(parts) >= int(
            group_depth) else str(name)

        if key not in group_to_ids:
            group_to_ids[key] = set()
        group_to_ids[key].add(id(p))

    out: Dict[str, ParamCountResult] = {}
    for key, ids in group_to_ids.items():
        params = _filter_params_by_id(named_params, ids)
        out[key] = summarize_params(params)

    return out


def _is_dit_core_param_name(param_name: str) -> bool:
    """DiT 안에서 '기본 DiT'에 해당하는 학습 숫자(파라미터)인지 이름으로 판정합니다.

    여기서 "기본 DiT"는 아래를 제외한 나머지를 뜻합니다.
      - PRAM-v2 관련(이름이 "pram_v2_"로 시작)
      - feasible projector 관련(이름이 "feasible_projector"로 시작)

    Args:
        param_name (str):
            파라미터 이름 문자열입니다. (shape: ())

    Returns:
        bool:
            - True  : 기본 DiT에 포함
            - False : PRAM-v2 또는 feasible projector 쪽
    """
    name = str(param_name)
    if name.startswith("pram_v2_"):
        return False
    if name == "feasible_projector" or name.startswith("feasible_projector."):
        return False
    return True


def summarize_dit_core_param_groups(
    dit_module: nn.Module,
    base_prefix: Optional[str],
    group_depth: int,
) -> Dict[str, ParamCountResult]:
    """DiT(Feasible/PRAM 제외) 파라미터를 이름 기준으로 묶어 개수를 셉니다.

    동작(쉽게 설명)
    --------------
    1) DiT 안의 파라미터 중에서
       - "pram_v2_"로 시작하는 것(PRAM-v2)
       - "feasible_projector"로 시작하는 것(FeasibleProjector)
       을 제외합니다.
    2) 남은 파라미터(= DiT core)를 이름의 앞부분 기준으로 묶습니다.
       - group_depth=1: "blocks", "preproj", "t_embedder" 같은 큰 묶음
       - group_depth=2: "preproj.fc1", "preproj.fc2" 같은 중간 묶음
       - group_depth=3: "blocks.0.mlp1" 같은 더 세부 묶음

    Args:
        dit_module (nn.Module):
            Decoder 안의 DiT 모듈입니다. 입력 텐서 shape(B, ...)과 무관합니다.
        base_prefix (Optional[str]):
            - None이면 DiT core 전체를 대상으로 합니다.
            - 문자열이면 그 prefix 아래만 대상으로 합니다.
              예: "blocks", "preproj", "t_embedder"
        group_depth (int):
            이름을 '.'으로 나눈 뒤, 앞에서 몇 조각을 합쳐 그룹 이름으로 쓸지.

    Returns:
        Dict[str, ParamCountResult]:
            그룹 이름 -> (총 개수, 학습되는 개수)
    """
    if int(group_depth) < 1:
        raise ValueError(f"group_depth must be >= 1. got {group_depth}")

    named_params: List[Tuple[str, nn.Parameter]] = list(
        dit_module.named_parameters(recurse=True))

    group_to_ids: Dict[str, Set[int]] = {}

    for name, p in named_params:
        # p: 파라미터 텐서 (shape은 각 파라미터마다 다름)
        if not _is_dit_core_param_name(name):
            continue

        if base_prefix is not None:
            if str(name) != base_prefix and not str(name).startswith(
                    base_prefix + "."):
                continue

        parts: List[str] = str(name).split(".")
        key = ".".join(parts[:int(group_depth)]) if len(parts) >= int(
            group_depth) else str(name)

        if key not in group_to_ids:
            group_to_ids[key] = set()
        group_to_ids[key].add(id(p))

    out: Dict[str, ParamCountResult] = {}
    for key, ids in group_to_ids.items():
        params = _filter_params_by_id(named_params, ids)
        out[key] = summarize_params(params)

    return out


def _merge_dit_block_groups_across_indices(
    block_groups: Dict[str, ParamCountResult],) -> Dict[str, ParamCountResult]:
    """blocks.<번호>.<이름> 형태를 blocks.<이름> 형태로 합쳐서 더 큰 흐름을 보여줍니다.

    예)
      - blocks.0.mlp1, blocks.1.mlp1, ...  -> blocks.mlp1
      - blocks.0.qkv_proj, blocks.2.qkv_proj -> blocks.self_attn_proj (묶어서)
      - blocks.0.q_proj_cross, blocks.1.kv_proj_cross -> blocks.cross_attn_proj (묶어서)
      - blocks.0.norm1 ~ norm4 -> blocks.norm (묶어서)

    Args:
        block_groups (Dict[str, ParamCountResult]):
            summarize_dit_core_param_groups(..., base_prefix="blocks", group_depth=3)
            같은 결과를 넣는 것을 기대합니다.

    Returns:
        Dict[str, ParamCountResult]:
            합쳐진 그룹 결과입니다.
    """
    merged: Dict[str, ParamCountResult] = {}

    for key, res in block_groups.items():
        parts = str(key).split(".")  # 예: ["blocks","0","mlp1"]
        if len(parts) >= 3 and parts[0] == "blocks":
            base = f"blocks.{parts[2]}"
        else:
            base = str(key)

        # 더 보기 좋게 추가 묶음 규칙 적용
        if base in ("blocks.norm1", "blocks.norm2", "blocks.norm3",
                    "blocks.norm4"):
            out_key = "blocks.norm"
        elif base in ("blocks.qkv_proj", "blocks.out_proj"):
            out_key = "blocks.self_attn_proj"
        elif base in ("blocks.q_proj_cross", "blocks.kv_proj_cross",
                      "blocks.out_proj_cross"):
            out_key = "blocks.cross_attn_proj"
        elif base == "blocks.gate_mlp2":
            out_key = "blocks.gate"
        else:
            out_key = base

        if out_key not in merged:
            merged[out_key] = ParamCountResult(total=0, trainable=0)

        merged[out_key] = ParamCountResult(
            total=int(merged[out_key].total + res.total),
            trainable=int(merged[out_key].trainable + res.trainable),
        )

    return merged


def print_param_report(model: nn.Module) -> None:
    """Diffusion_Planner 모델의 인코더/디코더/전체 및 그룹별(A/B/C) 파라미터 개수를 출력합니다."""
    # 1) 전체 / 인코더 / 디코더 (모듈 기준)
    whole = summarize_module_params(model)
    encoder = summarize_module_params(model.encoder)
    decoder = summarize_module_params(model.decoder)

    print("[모듈 기준 파라미터 개수]")
    print(f"- 전체: total={whole.total:,} / trainable={whole.trainable:,}")
    print(f"- 인코더: total={encoder.total:,} / trainable={encoder.trainable:,}")
    print(f"- 디코더: total={decoder.total:,} / trainable={decoder.trainable:,}")

    # 2) 그룹 기준
    has_group_iters: bool = all(
        hasattr(model, name) for name in [
            "iter_group_encoder_local_parameters",
            "iter_group_encoder_global_parameters",
            "iter_group_decoder_parameters",
        ])

    if has_group_iters:
        group_a = summarize_iter_params(
            model.iter_group_encoder_local_parameters())
        group_b = summarize_iter_params(
            model.iter_group_encoder_global_parameters())
        group_c = summarize_iter_params(model.iter_group_decoder_parameters())

        print("\n[그룹 기준 파라미터 개수 (A/B/C iterator 기준)]")
        print(
            f"- Group A(로컬 인코더): total={group_a.total:,} / trainable={group_a.trainable:,}"
        )
        print(
            f"- Group B(글로벌 인코더): total={group_b.total:,} / trainable={group_b.trainable:,}"
        )
        print(
            f"- Group C(디코더): total={group_c.total:,} / trainable={group_c.trainable:,}"
        )

        group_sum_total: int = group_a.total + group_b.total + group_c.total
        group_sum_trainable: int = group_a.trainable + group_b.trainable + group_c.trainable
        print(
            f"- A+B+C 합: total={group_sum_total:,} / trainable={group_sum_trainable:,}"
        )

    # ------------------------------------------------------------
    # ✅ (추가) 디코더 내부 4카테고리별 파라미터 개수
    # ------------------------------------------------------------
    decoder_core = getattr(model.decoder, "decoder", None)
    if isinstance(decoder_core, nn.Module):
        dec_module = decoder_core
    else:
        dec_module = model.decoder  # fallback

    dec_total = summarize_module_params(dec_module)
    dec_cats = summarize_decoder_param_categories(dec_module)

    dit_core = dec_cats["dit_core"]
    feasible = dec_cats["feasible_projector"]
    pram = dec_cats["pram_v2"]
    others = dec_cats["others"]

    sum_total = dit_core.total + feasible.total + pram.total + others.total
    sum_trainable = dit_core.trainable + feasible.trainable + pram.trainable + others.trainable

    print("\n[디코더 내부 카테고리별 파라미터 개수]")
    print(
        f"- DiT(Feasible/PRAM 제외): total={dit_core.total:,} / trainable={dit_core.trainable:,}"
    )
    print(
        f"- FeasibleProjector: total={feasible.total:,} / trainable={feasible.trainable:,}"
    )
    print(f"- PRAM-v2: total={pram.total:,} / trainable={pram.trainable:,}")
    print(f"- 기타: total={others.total:,} / trainable={others.trainable:,}")
    print(f"- (4개 합): total={sum_total:,} / trainable={sum_trainable:,}")

    if sum_total != dec_total.total:
        print(
            f"  [주의] (4개 합)과 디코더 total이 다릅니다: decoder_total={dec_total.total:,}"
        )
        # ------------------------------------------------------------
        # ✅ (추가) 인코더 내부 8카테고리별 파라미터 개수
        # ------------------------------------------------------------
    encoder_core = getattr(model.encoder, "encoder", None)
    if isinstance(encoder_core, nn.Module):
        enc_module = encoder_core
    else:
        enc_module = model.encoder  # fallback

    enc_total = summarize_module_params(enc_module)
    enc_cats = summarize_encoder_param_categories(enc_module)

    agent = enc_cats["agent_fusion_encoder"]
    static = enc_cats["static_fusion_encoder"]
    road_safety = enc_cats["road_safety_fusion_encoder"]
    lane = enc_cats["lane_fusion_encoder"]
    lane_summary = enc_cats["lane_summary_token_pooler"]
    fusion_core = enc_cats["fusion_encoder"]
    self_attn = enc_cats["self_attention_block"]
    others = enc_cats["others"]

    sum_total = (agent.total + static.total + road_safety.total + lane.total +
                 lane_summary.total + fusion_core.total + self_attn.total +
                 others.total)
    sum_trainable = (agent.trainable + static.trainable +
                     road_safety.trainable + lane.trainable +
                     lane_summary.trainable + fusion_core.trainable +
                     self_attn.trainable + others.trainable)

    print("\n[인코더 내부 카테고리별 파라미터 개수]")
    print(
        f"- AgentFusionEncoder: total={agent.total:,} / trainable={agent.trainable:,}"
    )
    print(
        f"- StaticFusionEncoder: total={static.total:,} / trainable={static.trainable:,}"
    )
    print(
        f"- RoadSafetyFusionEncoder: total={road_safety.total:,} / trainable={road_safety.trainable:,}"
    )
    print(
        f"- LaneFusionEncoder: total={lane.total:,} / trainable={lane.trainable:,}"
    )
    print(
        f"- LaneSummaryTokenPooler: total={lane_summary.total:,} / trainable={lane_summary.trainable:,}"
    )
    print(
        f"- FusionEncoder(SelfAttentionBlock 제외): total={fusion_core.total:,} / trainable={fusion_core.trainable:,}"
    )
    print(
        f"- SelfAttentionBlock: total={self_attn.total:,} / trainable={self_attn.trainable:,}"
    )
    print(f"- 기타: total={others.total:,} / trainable={others.trainable:,}")
    print(f"- (8개 합): total={sum_total:,} / trainable={sum_trainable:,}")

    if sum_total != enc_total.total:
        print(
            f"  [주의] (8개 합)과 인코더 total이 다릅니다: encoder_total={enc_total.total:,}"
        )

    # ------------------------------------------------------------
    # ✅ (추가) PRAM-v2 내부에서 "어느 부분이 큰지" 더 자세히 출력
    # ------------------------------------------------------------
    dit_module = getattr(dec_module, "dit", None)
    if isinstance(dit_module, nn.Module) and pram.total > 0:
        # (A) PRAM-v2 큰 덩어리(pram_v2_XXX)별
        pram_top_groups = summarize_pram_v2_param_groups(
            dit_module=dit_module,
            base_prefix=None,
            group_depth=1,
        )
        pram_top_sorted = _sort_param_count_results(pram_top_groups)

        pram_top_sum_total: int = sum(r.total for _, r in pram_top_sorted)
        pram_top_sum_trainable: int = sum(
            r.trainable for _, r in pram_top_sorted)

        print("\n[PRAM-v2 내부 파라미터(큰 순서)]")
        for key, res in pram_top_sorted:
            print(f"- {key}: total={res.total:,} / trainable={res.trainable:,}")

        if pram_top_sum_total != pram.total:
            print(
                f"  [주의] PRAM-v2 합이 다릅니다: pram_total={pram.total:,}, grouped={pram_top_sum_total:,}"
            )

        # (B) 자주 큰 덩어리 3~5개를 더 잘게 쪼개서 확인
        detail_targets: List[str] = [
            "pram_v2_out_proj",
            "pram_v2_composer",
            "pram_v2_time_mod",
            "pram_v2_state_token_encoder",
            "pram_v2_block_path_scalars",
        ]

        for base in detail_targets:
            if base not in pram_top_groups:
                continue

            sub_groups = summarize_pram_v2_param_groups(
                dit_module=dit_module,
                base_prefix=base,
                group_depth=2,
            )
            sub_sorted = _sort_param_count_results(sub_groups)

            # 의미 있는 분해가 없으면(1개뿐이면) 출력 생략
            if len(sub_sorted) <= 1:
                continue

            base_total: int = int(pram_top_groups[base].total)
            sub_sum_total: int = sum(r.total for _, r in sub_sorted)

            print(f"\n[PRAM-v2 상세: {base}]")
            for full_key, res in sub_sorted:
                # "pram_v2_composer.adapt_S" -> "adapt_S" 처럼 짧게 표시
                short_key: str = full_key
                if full_key.startswith(base + "."):
                    short_key = full_key[len(base) + 1:]
                print(
                    f"  - {short_key}: total={res.total:,} / trainable={res.trainable:,}"
                )

            if sub_sum_total != base_total:
                print(
                    f"  [주의] {base} 합이 다릅니다: base_total={base_total:,}, grouped={sub_sum_total:,}"
                )
    # ------------------------------------------------------------
    # ✅ (추가) DiT(Feasible/PRAM 제외) 내부에서 "어느 부분이 큰지" 자세히 출력
    # ------------------------------------------------------------
    dit_module = getattr(dec_module, "dit", None)
    if isinstance(dit_module, nn.Module) and dit_core.total > 0:
        # (A) DiT core 큰 덩어리: blocks / preproj / t_embedder ...
        dit_top_groups = summarize_dit_core_param_groups(
            dit_module=dit_module,
            base_prefix=None,
            group_depth=1,
        )
        dit_top_sorted = _sort_param_count_results(dit_top_groups)

        dit_top_sum_total: int = sum(r.total for _, r in dit_top_sorted)
        dit_top_sum_trainable: int = sum(r.trainable for _, r in dit_top_sorted)

        print("\n[DiT(Feasible/PRAM 제외) 내부 파라미터(큰 순서)]")
        for key, res in dit_top_sorted:
            print(f"- {key}: total={res.total:,} / trainable={res.trainable:,}")

        if dit_top_sum_total != dit_core.total:
            print(
                f"  [주의] DiT core 합이 다릅니다: dit_core_total={dit_core.total:,}, grouped={dit_top_sum_total:,}"
            )

        # (B) preproj 상세(fc1 vs fc2)
        if "preproj" in dit_top_groups:
            preproj_groups = summarize_dit_core_param_groups(
                dit_module=dit_module,
                base_prefix="preproj",
                group_depth=2,
            )
            preproj_sorted = _sort_param_count_results(preproj_groups)
            preproj_sum_total: int = sum(r.total for _, r in preproj_sorted)
            preproj_total: int = int(dit_top_groups["preproj"].total)

            if len(preproj_sorted) > 1:
                print("\n[DiT 상세: preproj]")
                for full_key, res in preproj_sorted:
                    short_key = full_key[len("preproj."
                                            ):] if full_key.startswith(
                                                "preproj.") else full_key
                    print(
                        f"  - {short_key}: total={res.total:,} / trainable={res.trainable:,}"
                    )
                if preproj_sum_total != preproj_total:
                    print(
                        f"  [주의] preproj 합이 다릅니다: base_total={preproj_total:,}, grouped={preproj_sum_total:,}"
                    )

        # (C) t_embedder 상세(mlp.0 vs mlp.2)
        if "t_embedder" in dit_top_groups:
            t_groups = summarize_dit_core_param_groups(
                dit_module=dit_module,
                base_prefix="t_embedder",
                group_depth=3,
            )
            t_sorted = _sort_param_count_results(t_groups)
            t_sum_total: int = sum(r.total for _, r in t_sorted)
            t_total: int = int(dit_top_groups["t_embedder"].total)

            if len(t_sorted) > 1:
                print("\n[DiT 상세: t_embedder]")
                for full_key, res in t_sorted:
                    short_key = full_key[len("t_embedder."
                                            ):] if full_key.startswith(
                                                "t_embedder.") else full_key
                    print(
                        f"  - {short_key}: total={res.total:,} / trainable={res.trainable:,}"
                    )
                if t_sum_total != t_total:
                    print(
                        f"  [주의] t_embedder 합이 다릅니다: base_total={t_total:,}, grouped={t_sum_total:,}"
                    )

        # (D) blocks 상세(블록 번호를 없애고 묶어서 출력)
        if "blocks" in dit_top_groups:
            blocks_raw = summarize_dit_core_param_groups(
                dit_module=dit_module,
                base_prefix="blocks",
                group_depth=3,
            )
            blocks_merged = _merge_dit_block_groups_across_indices(blocks_raw)
            blocks_sorted = _sort_param_count_results(blocks_merged)

            blocks_sum_total: int = sum(r.total for _, r in blocks_sorted)
            blocks_total: int = int(dit_top_groups["blocks"].total)

            print("\n[DiT 상세: blocks(묶음, 큰 순서)]")
            for key, res in blocks_sorted:
                print(
                    f"- {key}: total={res.total:,} / trainable={res.trainable:,}"
                )

            if blocks_sum_total != blocks_total:
                print(
                    f"  [주의] blocks 합이 다릅니다: base_total={blocks_total:,}, grouped={blocks_sum_total:,}"
                )


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _EncoderIntentionalInitSnapshot:
    """Encoder 내부에서 '의도적으로 초기화된 값'을 보존하기 위한 스냅샷입니다.

    - pos_emb: Encoder._build_pos_embedding_module()에서 std=0.02 등으로 초기화한 값이
      전역 초기화(xavier)로 덮이지 않도록, state_dict 전체를 저장합니다.
    - lane 임베딩 Linear들: LaneFusionEncoder.__init__()에서 weight만 std=0.02로 초기화한
      모듈들에 대해, weight만 저장합니다. (bias는 전역 초기화에서 0으로 유지)
    """
    pos_emb_state: Optional[Dict[str, torch.Tensor]]
    lane_linear_weight: Dict[str, torch.Tensor]


class Diffusion_Planner(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)

    def iter_group_encoder_local_parameters(self) -> Iterator[nn.Parameter]:
        """로컬 인코더(Group A) 파라미터 이터레이터를 돌려줍니다."""
        encoder_core: Encoder = self.encoder.encoder
        if hasattr(encoder_core, "iter_encoder_local_parameters"):
            for param in encoder_core.iter_encoder_local_parameters():
                # param: (out_dim, in_dim) 또는 (dim,)
                yield param

    def iter_group_encoder_global_parameters(self) -> Iterator[nn.Parameter]:
        """글로벌 인코더(Group B) 파라미터 이터레이터를 돌려줍니다."""
        encoder_core: Encoder = self.encoder.encoder
        if hasattr(encoder_core, "iter_encoder_global_parameters"):
            for param in encoder_core.iter_encoder_global_parameters():
                # param: (out_dim, in_dim) 또는 (dim,)
                yield param

    def iter_group_decoder_parameters(self) -> Iterator[nn.Parameter]:
        """디코더(Group C) 파라미터 이터레이터를 돌려줍니다.

        Decoder 전체(DiT + PRAM + Feasible projector)의 파라미터가 포함됩니다.
        """
        for param in self.decoder.decoder.parameters():
            # param: (out_dim, in_dim) 또는 (dim,)
            yield param

    @property
    def sde(self):
        return self.decoder.decoder.sde #VPSDE_linear 를 쓰고 있음

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs


class Diffusion_Planner_Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    @staticmethod
    def _lane_encoder_intentional_linear_names() -> Tuple[str, ...]:
        """LaneFusionEncoder에서 weight를 의도적으로 초기화한 Linear 이름 목록.

        Returns:
            Tuple[str, ...]: LaneFusionEncoder 내부 attribute 이름들.
        """
        return (
            "speed_limit_emb",
            "traffic_emb",
            "lane_type_emb",
            "left_line_type_emb",
            "right_line_type_emb",
        )

    def _snapshot_encoder_intentional_init(
            self) -> _EncoderIntentionalInitSnapshot:
        """전역 초기화 전에, Encoder가 '의도적으로 초기화'한 값만 저장합니다.

        저장 대상
        - pos_emb: state_dict 전체 저장
        - lane_encoder의 특정 Linear들: weight만 저장
          (weight shape 예: (C, in_dim) where C=channels_mlp_dim)

        Returns:
            _EncoderIntentionalInitSnapshot: 복원에 필요한 텐서 스냅샷
        """
        pos_emb_state: Optional[Dict[str, torch.Tensor]] = None
        lane_linear_weight: Dict[str, torch.Tensor] = {}

        # (1) pos_emb: (Linear 또는 timm.Mlp) 내부 fc1/fc2까지 포함해서 통째로 보존
        pos_emb = getattr(self.encoder, "pos_emb", None)
        if isinstance(pos_emb, nn.Module):
            # state_dict 텐서들을 clone()으로 복사해 저장 (값만 저장, RNG 추가 사용 없음)
            state = pos_emb.state_dict()
            pos_emb_state = {k: v.detach().clone() for k, v in state.items()}

        # (2) lane_encoder의 일부 Linear: weight만 보존 (bias는 전역 init으로 0 유지)
        lane_encoder = getattr(self.encoder, "lane_encoder", None)
        if isinstance(lane_encoder, nn.Module):
            for attr_name in self._lane_encoder_intentional_linear_names():
                layer = getattr(lane_encoder, attr_name, None)
                if isinstance(layer, nn.Linear):
                    lane_linear_weight[attr_name] = layer.weight.detach().clone(
                    )

        return _EncoderIntentionalInitSnapshot(
            pos_emb_state=pos_emb_state,
            lane_linear_weight=lane_linear_weight,
        )

    def _restore_encoder_intentional_init(
        self,
        snapshot: _EncoderIntentionalInitSnapshot,
    ) -> None:
        """전역 초기화 이후, '의도적으로 초기화된 값'을 다시 복원합니다.

        - pos_emb: 저장해 둔 state_dict로 전체 복원
        - lane 임베딩 Linear들: weight만 복원 (bias는 전역 초기화 결과 유지)

        Args:
            snapshot: _snapshot_encoder_intentional_init()에서 만든 스냅샷
        """
        # (1) pos_emb 전체 복원
        if snapshot.pos_emb_state is not None:
            pos_emb = getattr(self.encoder, "pos_emb", None)
            if isinstance(pos_emb, nn.Module):
                # load_state_dict는 내부적으로 copy를 수행합니다.
                pos_emb.load_state_dict(snapshot.pos_emb_state, strict=True)

        # (2) lane 임베딩 Linear weight만 복원
        lane_encoder = getattr(self.encoder, "lane_encoder", None)
        if isinstance(lane_encoder, nn.Module) and len(
                snapshot.lane_linear_weight) > 0:
            with torch.no_grad():
                for attr_name, w_saved in snapshot.lane_linear_weight.items():
                    layer = getattr(lane_encoder, attr_name, None)
                    if not isinstance(layer, nn.Linear):
                        continue
                    if layer.weight.shape != w_saved.shape:
                        raise ValueError(
                            f"lane_encoder.{attr_name}.weight shape mismatch: "
                            f"current={tuple(layer.weight.shape)}, saved={tuple(w_saved.shape)}"
                        )
                    layer.weight.copy_(
                        w_saved.to(device=layer.weight.device,
                                   dtype=layer.weight.dtype))

    def initialize_weights(self):
        """전역 초기화를 하되, Encoder가 의도적으로 초기화한 일부는 최종적으로 유지합니다."""
        # (A) Encoder 내부의 '의도적 초기화' 값 스냅샷 저장
        snapshot = self._snapshot_encoder_intentional_init()

        # (B) 기존 전역 초기화 로직 유지
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # (C) 전역 초기화로 덮일 수 있는 '의도적 초기화' 값만 다시 복원
        self._restore_encoder_intentional_init(snapshot)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        return encoder_outputs


class Diffusion_Planner_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        return

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return decoder_outputs
