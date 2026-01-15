#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 고정/기본값 (요구사항 반영)
###############################################################################
ENV_NAME="diffusion_planner"

IMAGE_NAME="dp_diffusion_planner_gpu_img"
CONTAINER_NAME="dp_diffusion_planner_gpu_cpu0_63"

CPU_SET="0-63"              # 확정
GPU_DEVICE="${GPU_DEVICE:-0}" # GPU 1개면 보통 0
FORCE_RECREATE="${FORCE_RECREATE:-0}"  # 1이면 컨테이너 지우고 새로 만듦

###############################################################################
# 경로
###############################################################################
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$PROJECT_DIR/docker_dp"
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
# 1) 사전 체크: conda 환경 존재
###############################################################################
echo "[1/7] conda 환경 확인"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda 명령이 안 잡혀요. (miniforge가 활성화된 쉘에서 실행하세요)"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ERROR: conda 환경 '$ENV_NAME' 을(를) 찾지 못했습니다."
  echo "       conda env list 로 환경 이름을 확인하세요."
  exit 1
fi

###############################################################################
# 2) 사전 체크: NVIDIA GPU + 도커 GPU 전달 가능 여부
###############################################################################
echo "[2/7] NVIDIA GPU 확인"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi가 없습니다. (NVIDIA 드라이버가 없거나 NVIDIA GPU가 아닐 수 있어요)"
  exit 1
fi
nvidia-smi -L || { echo "ERROR: nvidia-smi 실패. GPU/드라이버 상태를 먼저 해결하세요."; exit 1; }

echo "[3/7] 도커에서 GPU 접근 테스트"
if ! "${DOCKER[@]}" run --rm --gpus "device=${GPU_DEVICE}" ubuntu:24.04 bash -lc 'ls -l /dev/nvidia* >/dev/null 2>&1'; then
  echo "ERROR: 도커 컨테이너에서 GPU 장치를 못 잡았습니다."
  echo "       (보통 '도커에서 GPU 쓰는 준비'가 안 된 상태일 때 이렇게 됩니다.)"
  exit 1
fi

###############################################################################
# 3) /dev/shm 정보 출력 (참고)
###############################################################################
echo "[4/7] 호스트 /dev/shm 크기(참고)"
df -h /dev/shm || true
HOST_SHM_BYTES="$(df -B1 --output=size /dev/shm 2>/dev/null | tail -n 1 | tr -d ' ')"

###############################################################################
# 4) conda 환경을 docker용 yml로 내보내기
###############################################################################
echo "[5/7] conda 환경 내보내기 (docker용 environment.yml 생성)"
RAW_YML="$DOCKER_DIR/environment.raw.yml"
DOCKER_YML="$DOCKER_DIR/environment.docker.yml"

conda env export -n "$ENV_NAME" --no-builds > "$RAW_YML"

# prefix(호스트 절대경로) 제거 + 로컬경로/편집설치(-e, file://) 제거 + name 고정
awk -v newname="$ENV_NAME" '
  /^name:/ { print "name: " newname; next }
  /^prefix:/ { next }
  /^[[:space:]]*-[[:space:]]*-e[[:space:]]/ { next }
  /^[[:space:]]*-[[:space:]].*@ file:\/\// { next }
  /^[[:space:]]*-[[:space:]].*file:\/\// { next }
  { print }
' "$RAW_YML" > "$DOCKER_YML"

###############################################################################
# 5) tmux 설정 + 레이아웃 스크립트
###############################################################################
echo "[6/7] tmux 설정 생성"
cat > "$DOCKER_DIR/tmux.conf" << 'EOF'
set -g mouse on
set -g history-limit 100000
setw -g mode-keys vi
set -g default-terminal "screen-256color"

# 터미널이 지원하면 복사 내용을 클립보드로 보내도록 시도
set -s set-clipboard on

# 복사: Ctrl+b -> [  -> v 선택 시작 -> y 복사
# 붙여넣기: Ctrl+b -> ]
bind-key -T copy-mode-vi v send -X begin-selection
bind-key -T copy-mode-vi y send -X copy-selection-and-cancel
EOF

cat > "$DOCKER_DIR/into_tmux" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
SESSION="dp"

# 세션이 없으면 만들고: 윈도우 2개, 각 윈도우 좌우 2 pane
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

###############################################################################
# 6) Dockerfile 생성 (miniforge + conda env + 자동 tmux/conda)
###############################################################################
cat > "$DOCKER_DIR/Dockerfile" << 'EOF'
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-lc"]

# 기본 도구 + tmux + 자주 필요한 라이브러리 몇 개
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tmux \
    procps htop less vim \
    build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Miniforge 설치
ENV CONDA_DIR=/opt/conda
RUN curl -L -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
  && rm -f /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# mamba 설치(환경 생성이 더 잘 되는 편)
RUN conda install -n base -c conda-forge -y mamba && conda clean -a -y

# 사용자 생성 (호스트와 UID/GID 맞추기)
RUN if ! getent group $USER_GID >/dev/null; then groupadd --gid $USER_GID $USERNAME; fi \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# conda 환경 생성
COPY environment.docker.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
 && conda clean -a -y \
 && rm -f /tmp/environment.yml

# 작업 폴더
RUN mkdir -p /workspace/Diffusion-Planner
RUN chown -R $USERNAME:$USERNAME /workspace $CONDA_DIR

# tmux 설정 + 진입 스크립트
COPY tmux.conf /home/$USERNAME/.tmux.conf
COPY into_tmux /usr/local/bin/into_tmux
RUN chmod +x /usr/local/bin/into_tmux \
 && chown -R $USERNAME:$USERNAME /home/$USERNAME

# 접속하면 자동으로:
# 1) conda diffusion_planner 활성화
# 2) 프로젝트 폴더로 이동
# 3) tmux 자동 진입 (이미 tmux 안이면 다시 안 들어감)
RUN cat >> /home/$USERNAME/.bashrc << 'BASHRC_EOF'

if [[ $- == *i* ]]; then
  # conda 활성화
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate diffusion_planner >/dev/null 2>&1 || true
  fi

  # 프로젝트 경로
  cd /workspace/Diffusion-Planner 2>/dev/null || true
  export PYTHONPATH="/workspace/Diffusion-Planner:${PYTHONPATH-}"

  # tmux 자동 진입
  if command -v tmux >/dev/null 2>&1 && [ -z "${TMUX-}" ]; then
    exec /usr/local/bin/into_tmux
  fi
fi
BASHRC_EOF

WORKDIR /workspace/Diffusion-Planner
USER $USERNAME

# 컨테이너가 계속 켜져있도록 유지
CMD ["bash", "-lc", "sleep infinity"]
EOF

###############################################################################
# 7) 이미지 빌드
###############################################################################
echo "[7/7] 도커 이미지 빌드"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"

"${DOCKER[@]}" build \
  --build-arg USERNAME="$HOST_USER" \
  --build-arg USER_UID="$HOST_UID" \
  --build-arg USER_GID="$HOST_GID" \
  -t "$IMAGE_NAME" \
  -f "$DOCKER_DIR/Dockerfile" \
  "$DOCKER_DIR"

###############################################################################
# 컨테이너 생성/시작
###############################################################################
echo "[RUN] 컨테이너 생성/시작"

if [[ "$FORCE_RECREATE" == "1" ]]; then
  "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

if "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if ! "${DOCKER[@]}" ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    "${DOCKER[@]}" start "$CONTAINER_NAME" >/dev/null
  fi
else
  # /dev/shm “컨테이너 제약 없이” 사용: --ipc=host
  # (만약 정책/모드 때문에 실패하면, 호스트 /dev/shm 크기만큼 --shm-size로 대체)
  RUN_ARGS=(--name "$CONTAINER_NAME" -d
            --cpuset-cpus "$CPU_SET"
            --gpus "device=${GPU_DEVICE}"
            -v "$PROJECT_DIR:/workspace/Diffusion-Planner"
            -w "/workspace/Diffusion-Planner")

  if ! "${DOCKER[@]}" run "${RUN_ARGS[@]}" --ipc=host "$IMAGE_NAME" >/dev/null; then
    echo "WARN: --ipc=host 실패. --shm-size로 대체 시도합니다."
    if [[ -z "${HOST_SHM_BYTES}" ]]; then
      echo "ERROR: 호스트 /dev/shm 크기를 못 구했습니다. 수동으로 --shm-size 값을 정해야 합니다."
      exit 1
    fi
    "${DOCKER[@]}" run "${RUN_ARGS[@]}" --shm-size "${HOST_SHM_BYTES}" "$IMAGE_NAME" >/dev/null
  fi
fi

echo
echo "OK. 이제 컨테이너에 접속합니다."
echo "접속하면: conda 활성화 + tmux 자동 진입"
echo
echo "다음부터 재접속:"
echo "  ${DOCKER[*]} exec -it $CONTAINER_NAME bash"
echo
"${DOCKER[@]}" exec -it "$CONTAINER_NAME" bash
