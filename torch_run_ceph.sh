#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# (ADD) OOM 상황에서 "학습 런처"가 먼저 죽기 쉽게(best-effort)
# - 권한/환경에 따라 실패할 수 있으니 실패해도 무시합니다.
# ============================================================
if [[ -w /proc/self/oom_score_adj ]]; then
  ( echo 500 > /proc/self/oom_score_adj ) 2>/dev/null || true
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export WANDB_DEBUG=0
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTHONWARNINGS="ignore:nvfuser is no longer supported in torch script:UserWarning"

RUN_ID=$(date +%Y%m%d-%H%M%S)

# 사용자가 이 값만 0/1로 바꿔서 모드 선택
DEBUG_LOG=0   # 1: 상세 디버그, 0: 빠른 학습(파일 로그 최소)

# -------------------------
# 로그/디버그 옵션 분기
# -------------------------
TORCHRUN_LOG_ARGS=()
PY_ARGS=()

if (( DEBUG_LOG )); then
  # ✅ 디버그 모드: 파일 로그 허용 (기존 의도 유지)
  LOG_DIR="/mnt/nuplan/logs/$RUN_ID"
  mkdir -p "$LOG_DIR"

  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT

  # 파이썬 디버그(에러 때 도움)
  export TORCH_SHOW_CPP_STACKTRACES=1
  export PYTHONFAULTHANDLER=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_DISABLE_ADDR2LINE=1

  # torchelastic 에러 파일(디버그 모드에서는 Ceph에 저장)
  export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"

  # 출력이 많아도 “바로바로” 보이게(디버그 편의)
  export PYTHONUNBUFFERED=1
  PY_ARGS=(-u -X faulthandler)

  # torchrun이 rank별 stdout/stderr를 파일로 저장 + 콘솔에도 출력(디버그 편의)
  TORCHRUN_LOG_ARGS=(--log_dir "$LOG_DIR" --redirects 3 --tee 3)

else
  # ✅ 빠른 학습 모드: "학습 중 파일 로그"는 끔
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT

  unset TORCH_SHOW_CPP_STACKTRACES || true
  unset PYTHONFAULTHANDLER || true
  unset TORCH_NCCL_ASYNC_ERROR_HANDLING || true
  unset TORCH_DISABLE_ADDR2LINE || true

  export TORCHELASTIC_ERROR_FILE="/tmp/torchelastic_error_${RUN_ID}.json"

  unset PYTHONUNBUFFERED || true
  PY_ARGS=()

  TORCHRUN_LOG_ARGS=()
fi

printf "[ENV] %-28s %s\n" "DEBUG_LOG:" "$DEBUG_LOG"
printf "[ENV] %-28s %s\n" "TORCHELASTIC_ERROR_FILE:" "${TORCHELASTIC_ERROR_FILE-<unset>}"
printf "[ENV] %-28s %s\n" "CUDA_DEVICE_MAX_CONNECTIONS:" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"

# -------------------------
# User Configuration
# -------------------------
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"

# (원격 CephRBD) 원본
REMOTE_TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"
REMOTE_TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"

# (로컬) 복사본
LOCAL_TRAIN_SET_PATH="/workspace/local_shards_v1"
LOCAL_TRAIN_SET_LIST_PATH="/workspace/local_shards_v1/diffusion_planner_training.json"

# ---------------------------------------
# (NEW) force_rebuild 인자 파싱
# 사용 예:
#   ./torch_run_ceph.sh
#   ./torch_run_ceph.sh --force_rebuild true
#   ./torch_run_ceph.sh --force_rebuild=1
# ---------------------------------------
FORCE_REBUILD="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force_rebuild)
      if [[ $# -ge 2 ]]; then
        FORCE_REBUILD="$2"
        shift 2
      else
        FORCE_REBUILD="true"
        shift 1
      fi
      ;;
    --force_rebuild=*)
      FORCE_REBUILD="${1#*=}"
      shift 1
      ;;
    *)
      shift 1
      ;;
  esac
done

printf "[DATA] %-28s %s\n" "REMOTE_TRAIN_SET_PATH:" "$REMOTE_TRAIN_SET_PATH"
printf "[DATA] %-28s %s\n" "REMOTE_TRAIN_SET_LIST_PATH:" "$REMOTE_TRAIN_SET_LIST_PATH"
printf "[DATA] %-28s %s\n" "LOCAL_TRAIN_SET_PATH:" "$LOCAL_TRAIN_SET_PATH"
printf "[DATA] %-28s %s\n" "LOCAL_TRAIN_SET_LIST_PATH:" "$LOCAL_TRAIN_SET_LIST_PATH"
printf "[DATA] %-28s %s\n" "force_rebuild:" "$FORCE_REBUILD"

# ============================================================
# (NEW) 원격 -> 로컬 데이터 복사 + 로컬 list(json) 생성
# - 이미 준비돼 있으면 1번만 수행
# - force_rebuild=true면 기존 로컬을 지우고 다시 복사
# ============================================================
export DP_REMOTE_TRAIN_SET_PATH="$REMOTE_TRAIN_SET_PATH"
export DP_REMOTE_TRAIN_SET_LIST_PATH="$REMOTE_TRAIN_SET_LIST_PATH"
export DP_LOCAL_TRAIN_SET_PATH="$LOCAL_TRAIN_SET_PATH"
export DP_LOCAL_TRAIN_SET_LIST_PATH="$LOCAL_TRAIN_SET_LIST_PATH"
export DP_FORCE_REBUILD_LOCAL_DATASET="$FORCE_REBUILD"

"$RUN_PYTHON_PATH" - <<'PY'
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

def _parse_bool(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def _common_is_under_root(path: str, root: str) -> bool:
    try:
        ap = os.path.abspath(path)
        ar = os.path.abspath(root)
        return os.path.commonpath([ap, ar]) == ar
    except Exception:
        return False

def _collect_npz_paths_from_json(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, list):
        for x in obj:
            out.extend(_collect_npz_paths_from_json(x))
    elif isinstance(obj, dict):
        # key는 건드리지 않고 value만 훑습니다.
        for v in obj.values():
            out.extend(_collect_npz_paths_from_json(v))
    elif isinstance(obj, str):
        if obj.lower().endswith(".npz"):
            out.append(obj)
    return out

def _rewrite_npz_paths_to_local(obj: Any, remote_root: str, local_root: str) -> Any:
    # 원본 JSON 구조는 유지하되,
    # "절대경로(remote_root 아래)"로 적힌 npz 경로만 "로컬 절대경로"로 바꿉니다.
    if isinstance(obj, list):
        return [_rewrite_npz_paths_to_local(x, remote_root, local_root) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_npz_paths_to_local(v, remote_root, local_root) for k, v in obj.items()}
    if isinstance(obj, str) and obj.lower().endswith(".npz") and os.path.isabs(obj):
        if _common_is_under_root(obj, remote_root):
            rel = os.path.relpath(obj, remote_root)
            return os.path.join(local_root, rel)
    return obj

def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

remote_root = os.path.normpath(os.environ.get("DP_REMOTE_TRAIN_SET_PATH", ""))
remote_list = os.environ.get("DP_REMOTE_TRAIN_SET_LIST_PATH", "")
local_root = os.path.normpath(os.environ.get("DP_LOCAL_TRAIN_SET_PATH", ""))
local_list = os.environ.get("DP_LOCAL_TRAIN_SET_LIST_PATH", "")
force_rebuild = _parse_bool(os.environ.get("DP_FORCE_REBUILD_LOCAL_DATASET", "false"))

if not remote_root or not local_root or not local_list:
    raise ValueError("remote/local 경로가 비었습니다. 환경변수를 확인하세요.")

marker_path = os.path.join(local_root, ".local_dataset_sync_done.json")

# force_rebuild면 로컬 폴더를 지우고 새로 만듭니다.
if force_rebuild:
    if os.path.isdir(local_root):
        shutil.rmtree(local_root, ignore_errors=True)
    elif os.path.exists(local_root):
        try:
            os.remove(local_root)
        except Exception:
            pass

# 이미 준비돼 있으면 스킵
if (not force_rebuild) and os.path.isfile(marker_path) and os.path.isfile(local_list):
    print(f"[LOCAL-DATA] already prepared -> skip (marker={marker_path})")
    raise SystemExit(0)

os.makedirs(local_root, exist_ok=True)

# (1) 원본 list(json) 읽기
data: Optional[Any] = None
if remote_list and os.path.isfile(remote_list):
    with open(remote_list, "r", encoding="utf-8") as f:
        data = json.load(f)

npz_refs: List[str] = _collect_npz_paths_from_json(data) if data is not None else []

# (2) list가 없거나, list 안에서 npz를 못 찾으면: 원본 폴더를 스캔해서 npz 목록을 만듭니다.
if not npz_refs:
    npz_refs = []
    for root, _, files in os.walk(remote_root):
        for fn in files:
            if fn.lower().endswith(".npz"):
                abs_p = os.path.join(root, fn)
                rel_p = os.path.relpath(abs_p, remote_root)
                npz_refs.append(rel_p)
    npz_refs.sort()
    data = npz_refs
    print(f"[LOCAL-DATA] remote list missing/empty -> scanned npz: {len(npz_refs)} files")

# (3) npz 복사
copied = 0
skipped = 0
missing = 0

t0 = time.time()
for ref in npz_refs:
    if not isinstance(ref, str) or (not ref.lower().endswith(".npz")):
        continue

    if os.path.isabs(ref):
        src = ref
        if _common_is_under_root(src, remote_root):
            rel = os.path.relpath(src, remote_root)
        else:
            # 예상 밖 케이스: 일단 파일명만으로 로컬에 둡니다.
            rel = os.path.basename(src)
    else:
        rel = ref
        src = os.path.join(remote_root, rel)

    dst = os.path.join(local_root, rel)

    if not os.path.isfile(src):
        missing += 1
        continue

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # force_rebuild가 아니면, "크기가 같으면" 복사 생략
    if (not force_rebuild) and os.path.isfile(dst):
        try:
            if os.path.getsize(dst) == os.path.getsize(src):
                skipped += 1
                continue
        except Exception:
            pass

    shutil.copy2(src, dst)
    copied += 1

elapsed = time.time() - t0

# (4) 로컬 list(json) 생성
# - 원본 JSON 구조는 유지
# - 절대경로(remote_root 아래)로 적힌 npz 경로만 로컬 절대경로로 변환
new_data = _rewrite_npz_paths_to_local(data, remote_root, local_root)
_atomic_write_json(local_list, new_data)

# (5) marker 저장 (복사 완료 표시)
meta: Dict[str, Any] = {
    "remote_root": remote_root,
    "remote_list": remote_list,
    "local_root": local_root,
    "local_list": local_list,
    "npz_count": int(len(npz_refs)),
    "copied": int(copied),
    "skipped": int(skipped),
    "missing": int(missing),
    "force_rebuild": bool(force_rebuild),
    "created_at_unix": float(time.time()),
    "copy_elapsed_sec": float(elapsed),
}
_atomic_write_json(marker_path, meta)

print(f"[LOCAL-DATA] sync done. npz={len(npz_refs)} copied={copied} skipped={skipped} missing={missing} elapsed_sec={elapsed:.1f}")
print(f"[LOCAL-DATA] local_root={local_root}")
print(f"[LOCAL-DATA] local_list={local_list}")
print(f"[LOCAL-DATA] marker={marker_path}")
PY

# -------------------------
# ✅ 이제부터는 로컬 데이터로 학습
# -------------------------
"$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 6 --standalone \
  "${TORCHRUN_LOG_ARGS[@]}" \
  train_predictor.py \
    --train_set "$LOCAL_TRAIN_SET_PATH"/ \
    --train_set_list "$LOCAL_TRAIN_SET_LIST_PATH" \
    --name "final_48_synchronize" \
    --batch_size 1536 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-6 \
    --profile_feasible False \
    --use_feasible False \
    --use_feasible_dl False \
    --use_feasible_filter False \
    --feasible_stride_dt 0.1
