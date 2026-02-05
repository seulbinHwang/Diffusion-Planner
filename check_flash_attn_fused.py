# check_flash_attn_fused_v2.py
import importlib
import inspect

def _try(mod: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        if attr:
            v = getattr(m, attr)
            # 함수면 시그니처까지 같이 확인
            if callable(v):
                try:
                    sig = str(inspect.signature(v))
                    return True, f"OK (callable) sig={sig}"
                except Exception:
                    return True, "OK (callable)"
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main() -> int:
    checks = [
        ("flash_attn", None),

        # 네가 이미 성공한 fused MLP
        ("flash_attn.ops.fused_dense", None),
        ("flash_attn.ops.fused_dense", "FusedMLP"),

        # ✅ LayerNorm은 보통 여기서 “함수/다른 클래스 이름”으로 존재
        ("flash_attn.ops.layer_norm", None),
        ("flash_attn.ops.layer_norm", "dropout_add_layer_norm"),
        ("flash_attn.ops.layer_norm", "dropout_add_rms_norm"),
        ("flash_attn.ops.layer_norm", "DropoutAddLayerNorm"),
        ("flash_attn.ops.layer_norm", "DropoutAddRMSNorm"),

        # (옵션) Triton 구현 경로도 같이 확인
        ("flash_attn.ops.triton.layer_norm", None),
        ("flash_attn.ops.triton.layer_norm", "layer_norm_fn"),
        ("flash_attn.ops.triton.layer_norm", "RMSNorm"),
    ]

    print("== flash_attn fused availability check (v2) ==")
    for mod, attr in checks:
        ok, msg = _try(mod, attr)
        target = f"{mod}" + (f".{attr}" if attr else "")
        print(f"- {target:55s} : {'YES' if ok else 'NO'} | {msg}")

    fused_ln_ok = any(
        _try("flash_attn.ops.layer_norm", a)[0]
        for a in ["dropout_add_layer_norm", "DropoutAddLayerNorm", "layer_norm_fn"]
    ) or _try("flash_attn.ops.triton.layer_norm", "layer_norm_fn")[0]

    fused_mlp_ok = _try("flash_attn.ops.fused_dense", "FusedMLP")[0]

    print("\n== summary ==")
    print(f"- fused LayerNorm usable : {'YES' if fused_ln_ok else 'NO'}")
    print(f"- fused MLP usable       : {'YES' if fused_mlp_ok else 'NO'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
