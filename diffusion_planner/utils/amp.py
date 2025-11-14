from __future__ import annotations
import contextlib
import torch
from typing import Dict, Any, Iterator


def amp_context_for_infer() -> Iterator[None]:
    """장비에 맞춰 추론용 autocast를 자동 선택합니다.

    - A100/RTX30/40 등 BF16 지원 GPU: BF16
    - 그 외 CUDA GPU: FP16
    - CPU: autocast 없음(FP32)
    """
    if torch.cuda.is_available():
        # 암페어 이상이면 True (A100/RTX30/40 등)
        if torch.cuda.is_bf16_supported():
            return_ = torch.autocast("cuda", dtype=torch.bfloat16)
            print("[amp_context_for_infer] Using bfloat16 autocast for inference.")
            return return_
        else:
            return_ = torch.autocast("cuda", dtype=torch.float16)
            print("[amp_context_for_infer] Using float16 autocast for inference.")
            return return_
    return_ = contextlib.nullcontext()
    print("[amp_context_for_infer] Using no autocast for inference (CPU mode).")
    return return_
