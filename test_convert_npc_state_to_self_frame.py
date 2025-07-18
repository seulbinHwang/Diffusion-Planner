# ================== TEST & VISUALIZE  ==================
"""
* Batch (B) = 2
* agent_num = 2  (= 주변 차량 2대)
* Pnn = 1       (= 예측‑대상 1대)
* augment_prob = 0.5
────────────────────────────────────────────────────────
‑ 각 배치(b=0,1) 별로 **별도 창** 을 띄운다.
‑ 모든 차량(2 대)을 그리되,
  · **augment =True** 로 선택된 차량만 *보정 궤적* 추가로 표시
  · 나머지 차량은 원본 궤적만 표시
‑ VIS_MODE 로 “past / future / both” 선택 가능
"""
import os, sys, types, time
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation

# ─────────────────── CONFIG ───────────────────
VIS_MODE   = "future"        # "past" | "future" | "both"
B, N, Pnn  = 2, 2, 1
T_past, T_fut, dt = 21, 80, 0.1               # 2 s 과거, 8 s 미래

# ───────── reproducibility ─────────
os.environ["OMP_NUM_THREADS"] = "1"
seed = int(time.time() * 1000) % 10000
torch.manual_seed(seed);  np.random.seed(seed)

# ───────── 0. 더미 데이터 생성 ─────────
def make_dummy_batch(y_offset):
    """batch 한 개(2 agents) 의 dummy past/future 궤적"""
    # 과거 (T_past,11)
    t_p = np.arange(-(T_past-1), 1) * dt           # -2.0 … 0.0
    past = np.zeros((N, T_past, 11), np.float32)
    for a in range(N):
        past[a, :, 0] = 4.0 * t_p + 8.0 * a        # x
        past[a, :, 1] = y_offset + 6.0 * a         # y (agent간 분리)
        past[a, :, 2] = 1.0                        # cosθ=1
        past[a, :, 4] = 4.0                        # vx
        past[a, :, 6] = 2.0;   past[a, :, 7] = 5.0 # W,L
        past[a, :, 8] = 1.0                        # class‑one‑hot

    # 미래 (T_fut,3)  – heading=0
    t_f = np.arange(1, T_fut + 1) * dt
    future = np.zeros((N, T_fut, 3), np.float32)
    for a in range(N):
        future[a, :, 0] = 4.0 * t_f + 8.0 * a
        future[a, :, 1] = y_offset + 6.0 * a

    return (torch.tensor(past).unsqueeze(0),         # (1,N,Tp,11)
            torch.tensor(future).unsqueeze(0))       # (1,N,Tf,3)

batch_past  = []
batch_future= []
for b in range(B):
    p, f = make_dummy_batch(y_offset=5.0 + 15.0*b)   # 배치별 y 오프셋
    batch_past.append(p);  batch_future.append(f)

neighbor_agents_past = torch.cat(batch_past,  dim=0) # (B,N,Tp,11)
neighbors_future_all = torch.cat(batch_future, dim=0) # (B,N,Tf,3)

inputs = {"neighbor_agents_past": neighbor_agents_past.clone()}
args   = SimpleNamespace(predicted_neighbor_num=Pnn)

# ───────── 1. Augmentation (prob=0.5) ─────────
augmentor = NPCStatePerturbation(augment_prob=0.6, device="cpu")
orig_past   = inputs["neighbor_agents_past"].clone()          # (B,N,Tp,11)
orig_future = neighbors_future_all.clone()                    # (B,N,Tf,3)
inputs_after, neighbors_future = augmentor(inputs,
                                           neighbors_future_all,
                                           args)
past_after   = inputs_after["neighbor_agents_past"]           # (B,N,Tp,11)
# neighbors_future shape (B,Pnn,Tf,3) – 첫 Pnn 차량만 반환
future_after = neighbors_future

# ───────── 2. 보정 수행 여부(mask) 계산 ─────────
#  → 각 (B,N) 위치별로 past 시퀀스가 변했는지 비교
diff = (past_after - orig_past).abs().sum(dim=(2,3))         # (B,N)
aug_mask = diff > 1e-4                                        # (B,N) bool

# ───────── 3. Helper (draw) ─────────
def draw_car(ax, x, y, yaw, L=4.8, W=2.0,
             color='k', alpha=1.0, z=3):
    trans = Affine2D().rotate_around(0,0, yaw).translate(x,y) + ax.transData
    rect  = patches.Rectangle((-L/2,-W/2), L,W,
                              linewidth=1.2, edgecolor=color,
                              facecolor='none', alpha=alpha,
                              transform=trans, zorder=z)
    ax.add_patch(rect)
    dx, dy = (L/2)*np.cos(yaw), (L/2)*np.sin(yaw)
    ax.arrow(x,y, dx,dy, head_width=0.35, head_length=0.6,
             fc=color, ec=color, linewidth=1.0, alpha=alpha,
             length_includes_head=True, zorder=z)

def draw_heading(ax,x,y,yaw,len_=0.5,color='k',alpha=0.8,z=2):
    dx,dy = len_*np.cos(yaw), len_*np.sin(yaw)
    ax.arrow(x,y,dx,dy, head_width=0.15, head_length=0.22,
             fc=color, ec=color, linewidth=0.8, alpha=alpha,
             length_includes_head=True, zorder=z)

# ───────── 4. Batch‑별 시각화 ─────────
colors = ['blue','green']                 # agent 0,1 색상
for b in range(B):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_title(f'Batch {b} – {VIS_MODE.upper()}  '
                 '(Blue/Green = original,  Red = refined)')
    for a in range(N):
        c = colors[a % len(colors)]
        # --- past 궤적 ---
        if VIS_MODE in ('past','both'):
            ax.plot(orig_past[b,a,:,0], orig_past[b,a,:,1],
                    marker='o', color=c, label=f'past‑orig a{a}' if b==0 else "",
                    linewidth=1.5)
            # 헤딩 화살표
            yaws = np.arctan2(orig_past[b,a,:,3], orig_past[b,a,:,2])
            for xy,yaw in zip(orig_past[b,a,:,:2], yaws):
                draw_heading(ax, xy[0], xy[1], yaw, color=c, alpha=0.6)
        # --- future 궤적 ---
        if VIS_MODE in ('future','both'):
            ax.plot(orig_future[b,a,:,0], orig_future[b,a,:,1],
                    linestyle='--', marker='^', color=c,
                    label=f'fut‑orig a{a}' if b==0 else "",
                    linewidth=1.2)

        # --- refine(augment =True) 이면 빨강으로 추가 ---
        if aug_mask[b,a]:
            if VIS_MODE in ('past','both'):
                ax.plot(past_after[b,a,:,0], past_after[b,a,:,1],
                        'r-', marker='o',
                        label='past‑refined' if (b==0 and a==0) else "")
            if VIS_MODE in ('future','both') and a < Pnn:
                ax.plot(future_after[b,0,:,0], future_after[b,0,:,1],
                        'r--', marker='^',
                        label='fut‑refined' if (b==0 and a==0) else "")
            # 차체 직사각형 (현재프레임)
            x,y = past_after[b,a,-1,:2]
            yaw = np.arctan2(past_after[b,a,-1,3], past_after[b,a,-1,2])
            draw_car(ax,x,y,yaw,L=5.0,W=2.0,color='red',z=4)

        else:
            # non‑aug 차량도 현재프레임 직사각형 (원본)
            x,y = orig_past[b,a,-1,:2]
            yaw = np.arctan2(orig_past[b,a,-1,3], orig_past[b,a,-1,2])
            draw_car(ax,x,y,yaw,L=5.0,W=2.0,color=c,alpha=0.5,z=3)

    ax.set_xlabel('x [m]');  ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True);  ax.set_aspect('equal','box')
    plt.tight_layout()

plt.show()
