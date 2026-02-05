# check_flash_attn_fused.py
import importlib
import sys

def _try_import(mod: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        if attr is not None:
            getattr(m, attr)
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main() -> int:
    checks = [
        ("flash_attn", None),
        ("flash_attn.ops.layer_norm", None),
        ("flash_attn.ops.layer_norm", "LayerNorm"),
        ("flash_attn.ops.layer_norm", "RMSNorm"),
        ("flash_attn.ops.fused_dense", None),
        ("flash_attn.ops.fused_dense", "FusedMLP"),
        ("flash_attn.ops.fused_dense", "FusedDense"),
    ]

    print("== flash_attn fused availability check ==")
    for mod, attr in checks:
        ok, msg = _try_import(mod, attr)
        target = f"{mod}" + (f".{attr}" if attr else "")
        print(f"- {target:40s} : {'YES' if ok else 'NO'} | {msg}")

    # 버전도 같이 출력
    ok, msg = _try_import("flash_attn", "__version__")
    if ok:
        import flash_attn
        print(f"\nflash_attn.__version__ = {getattr(flash_attn, '__version__', 'unknown')}")
    else:
        print(f"\nflash_attn version read failed: {msg}")

    # 핵심 결론
    fused_ln_ok, _ = _try_import("flash_attn.ops.layer_norm", "LayerNorm")
    fused_mlp_ok, _ = _try_import("flash_attn.ops.fused_dense", "FusedMLP")

    print("\n== summary ==")
    print(f"- fused LayerNorm usable : {'YES' if fused_ln_ok else 'NO'}")
    print(f"- fused MLP usable       : {'YES' if fused_mlp_ok else 'NO'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
