#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 고정값 (요구사항 반영)
###############################################################################
ENV_NAME="diffusion_planner"
MINIFORGE_HOST="/media/user/E/miniforge"

IMAGE_NAME="easy_dp_gpu_img"

CPU_SET="0-31,64-95"
CONTAINER_NAME="hsb_container"
GPU_DEVICE="${GPU_DEVICE:-0}"          # GPU 1개면 보통 0
FORCE_RECREATE="${FORCE_RECREATE:-0}"  # 1이면 컨테이너 삭제 후 재생성

# 0이면 miniforge를 "읽기 전용"으로 마운트(환경 보호)
# 1이면 miniforge를 "읽기/쓰기"로 마운트(컨테이너에서 pip/conda 설치하면 호스트도 같이 바뀜)
MINIFORGE_RW="${MINIFORGE_RW:-0}"

###############################################################################
# 경로
###############################################################################
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$PROJECT_DIR/docker_easy"
mkdir -p "$DOCKER_DIR"

###############################################################################
# docker 권한 처리
###############################################################################
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker가 설치되어 있지 않습니다."
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

###############################################################################
# 1) 사전 체크: miniforge/conda/env 존재 확인
###############################################################################
echo "[1/6] 호스트 miniforge + conda 환경 확인"

if [ ! -d "$MINIFORGE_HOST" ]; then
  echo "ERROR: miniforge 경로가 없습니다: $MINIFORGE_HOST"
  exit 1
fi

if [ ! -f "$MINIFORGE_HOST/etc/profile.d/conda.sh" ]; then
  echo "ERROR: conda 초기화 파일이 없습니다:"
  echo "       $MINIFORGE_HOST/etc/profile.d/conda.sh"
  exit 1
fi

CONDA_BIN="$MINIFORGE_HOST/bin/conda"
if [ ! -x "$CONDA_BIN" ]; then
  echo "ERROR: miniforge의 conda 실행 파일이 없습니다:"
  echo "       $CONDA_BIN"
  echo "       (MINIFORGE_HOST 경로가 맞는지 확인하세요)"
  exit 1
fi

# env 확인(디렉토리로 1차 확인 + conda env list로 2차 확인)
if [ ! -d "$MINIFORGE_HOST/envs/$ENV_NAME" ]; then
  echo "WARN: $MINIFORGE_HOST/envs/$ENV_NAME 경로가 안 보입니다. conda env list로 재확인합니다."
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda 환경 '$ENV_NAME' 을(를) 찾지 못했습니다."
  echo "       ($CONDA_BIN env list 결과에 '$ENV_NAME' 이 있어야 합니다.)"
  exit 1
fi

###############################################################################
# 2) 사전 체크: NVIDIA GPU + 도커 GPU 전달 가능 여부
###############################################################################
echo "[2/6] NVIDIA GPU 확인"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi가 없습니다. (NVIDIA 드라이버/GPU 상태를 먼저 확인하세요)"
  exit 1
fi
nvidia-smi -L || { echo "ERROR: nvidia-smi 실패. GPU/드라이버 상태를 먼저 해결하세요."; exit 1; }

echo "[3/6] 도커에서 GPU 접근 테스트"
if ! "${DOCKER[@]}" run --rm --gpus "device=${GPU_DEVICE}" ubuntu:24.04 bash -lc 'ls -l /dev/nvidia* >/dev/null 2>&1'; then
  echo "ERROR: 도커 컨테이너에서 GPU 장치를 못 잡았습니다."
  echo "       (보통 도커가 GPU를 쓰도록 준비가 안 된 상태일 때 이렇게 됩니다.)"
  exit 1
fi

###############################################################################
# 3) /dev/shm 정보 출력 (참고)
###############################################################################
echo "[4/6] 호스트 /dev/shm 크기(참고)"
df -h /dev/shm || true
HOST_SHM_BYTES="$(df -B1 --output=size /dev/shm 2>/dev/null | tail -n 1 | tr -d ' ')"

###############################################################################
# 4) 컨테이너 내부에서 사용할 tmux/쉘 스크립트 + Dockerfile 생성
###############################################################################
echo "[5/6] 도커 파일 생성 (환경 재설치 없음: 호스트 miniforge를 그대로 사용)"

cat > "$DOCKER_DIR/tmux.conf" << 'EOF'
set -g mouse on
set -g history-limit 100000
setw -g mode-keys vi
set -g default-terminal "screen-256color"
set -s set-clipboard on

# 새 pane이 열릴 때마다 conda env 자동 활성화된 쉘로 시작
set -g default-shell "/bin/bash"
set -g default-command "/usr/local/bin/dp_shell"

# 복사: Ctrl+b -> [  -> v 선택 시작 -> y 복사
# 붙여넣기: Ctrl+b -> ]
bind-key -T copy-mode-vi v send -X begin-selection
bind-key -T copy-mode-vi y send -X copy-selection-and-cancel
EOF

cat > "$DOCKER_DIR/dp_shell" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/media/user/E/miniforge}"
ENV_NAME="${ENV_NAME:-diffusion_planner}"
WORKDIR="${WORKDIR:-/media/user/E/projects/Diffusion-Planner}"

if [ ! -f "${MINIFORGE_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "ERROR: conda 초기화 파일이 없습니다:"
  echo "       ${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
  echo "       (호스트 miniforge가 컨테이너에 같은 경로로 마운트됐는지 확인하세요)"
  exec bash -i
fi

# 여기서부터: 새 bash가 시작될 때도 conda 초기화 + env 활성화 되게 rcfile을 만들어 사용
RCFILE="/tmp/dp_bashrc"

cat > "$RCFILE" <<RC_EOF
source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
export CONDA_CHANGEPS1=true
conda activate "${ENV_NAME}" 2>/dev/null || echo "WARN: conda activate 실패 (env 이름/경로 확인 필요)"

cd "${WORKDIR}" 2>/dev/null || true
export PYTHONPATH="${WORKDIR}:\${PYTHONPATH-}"
RC_EOF

exec bash --rcfile "$RCFILE" -i


# 프로젝트 폴더로 이동 + PYTHONPATH
cd "${WORKDIR}" 2>/dev/null || true
export PYTHONPATH="${WORKDIR}:${PYTHONPATH-}"

exec bash -i
EOF
chmod +x "$DOCKER_DIR/dp_shell"

cat > "$DOCKER_DIR/into_tmux" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
SESSION="dp"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux가 없습니다"
  exit 1
fi

# 세션이 없으면 만들기: 윈도우 2개, 각 윈도우 좌우 2 pane
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n "win0"
  tmux split-window -h -t "$SESSION:0"
  tmux select-layout -t "$SESSION:0" even-horizontal

  tmux new-window -t "$SESSION" -n "win1"
  tmux split-window -h -t "$SESSION:1"
  tmux select-layout -t "$SESSION:1" even-horizontal
fi

exec tmux attach -t "$SESSION"
EOF
chmod +x "$DOCKER_DIR/into_tmux"

cat > "$DOCKER_DIR/Dockerfile" << 'EOF'
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates bash git tmux \
    procps htop less vim gcc g++ make \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

COPY tmux.conf /etc/tmux.conf
COPY dp_shell /usr/local/bin/dp_shell
COPY into_tmux /usr/local/bin/into_tmux

RUN chmod +x /usr/local/bin/dp_shell /usr/local/bin/into_tmux

CMD ["bash", "-lc", "sleep infinity"]
EOF

###############################################################################
# 5) 이미지 빌드 (가볍게: 환경 재설치 없음)
###############################################################################
"${DOCKER[@]}" build -t "$IMAGE_NAME" -f "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR"

###############################################################################
# 6) 컨테이너 생성/시작 + tmux로 바로 접속
###############################################################################
echo "[6/6] 컨테이너 생성/시작 + tmux 접속"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# miniforge 마운트 옵션
MINIFORGE_MOUNT_OPT="ro"
if [[ "$MINIFORGE_RW" == "1" ]]; then
  MINIFORGE_MOUNT_OPT="rw"
fi

# 기본 마운트: /media/user/E 전체를 같은 경로로 마운트(절대경로 그대로 쓰기 위함)
# 그리고 miniforge는 기본 "읽기 전용"으로 한 번 더 덮어씌움(환경 보호)
MOUNTS=(-v "/media/user:/media/user")
MOUNTS+=(-v "$MINIFORGE_HOST:$MINIFORGE_HOST:${MINIFORGE_MOUNT_OPT}")
MOUNTS+=(-v "/usr/local/cuda:/usr/local/cuda:ro")
MOUNTS+=(-v "/usr/local/cuda-12.*/:/usr/local/cuda-12.*:ro")


# 작업 폴더(스크립트 위치와 같은 경로를 컨테이너에서도 그대로 사용)
WORKDIR_IN_CONTAINER="$PROJECT_DIR"

if [[ "$FORCE_RECREATE" == "1" ]]; then
  "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

if "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if ! "${DOCKER[@]}" ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    "${DOCKER[@]}" start "$CONTAINER_NAME" >/dev/null
  fi
else
  RUN_ARGS=(--name "$CONTAINER_NAME" -d
            --cpuset-cpus "$CPU_SET"
            --gpus "device=${GPU_DEVICE}"
            --user "${HOST_UID}:${HOST_GID}"
            -e HOME=/tmp
            -e MINIFORGE_ROOT="$MINIFORGE_HOST"
            -e ENV_NAME="$ENV_NAME"
            -e WORKDIR="$WORKDIR_IN_CONTAINER"
            -w "$WORKDIR_IN_CONTAINER")

  # /dev/shm 제약 최소화: 우선 --ipc=host 시도, 실패 시 호스트 /dev/shm 크기로 shm-size
  if ! "${DOCKER[@]}" run "${RUN_ARGS[@]}" "${MOUNTS[@]}" --ipc=host "$IMAGE_NAME" >/dev/null; then
    echo "WARN: --ipc=host 실패. --shm-size로 대체 시도합니다."
    if [[ -z "${HOST_SHM_BYTES}" ]]; then
      echo "ERROR: 호스트 /dev/shm 크기를 못 구했습니다. 수동 설정이 필요합니다."
      exit 1
    fi
    "${DOCKER[@]}" run "${RUN_ARGS[@]}" "${MOUNTS[@]}" --shm-size "${HOST_SHM_BYTES}" "$IMAGE_NAME" >/dev/null
  fi
fi

echo
echo "OK. 지금 tmux로 들어갑니다."
echo "다음부터 재접속:"
echo "  ${DOCKER[*]} exec -it $CONTAINER_NAME /usr/local/bin/into_tmux"
echo
echo "컨테이너 정지:"
echo "  ${DOCKER[*]} stop $CONTAINER_NAME"
echo

"${DOCKER[@]}" exec -it "$CONTAINER_NAME" /usr/local/bin/into_tmux
