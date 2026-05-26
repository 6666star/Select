# -*- coding: utf-8 -*-
"""
推荐风机特性曲线综合绘图模块
文件：plot_res.py

自动调用选型模块，对推荐机型绘制以下三组曲线：

  ┌─────────────────────────────┬───────────────┐
  │                             │  ② η-Q 曲线   │
  │  ① H-Q 曲线（主图）         │  效率 vs 风量  │
  │  额定 + 两个工况转速曲线     ├───────────────┤
  │  + 管道阻力线 + 工作点       │  ③ N-Q 曲线   │
  │                             │  轴功率 vs 风量│
  └─────────────────────────────┴───────────────┘

运行：python plot_res.py
"""

import matplotlib
import font_config                          # 统一中文字体配置

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import brentq

from calc_engine_range import DualPeriodParams, run_dual
from calc_engine import SelectionCoeffs
from selector import run_selector
from fan_db import build_fan_interpolated


# ─────────────────────────────────────────────
# 相似律曲线计算工具
# ─────────────────────────────────────────────

def _fan_HQ_at_ratio(fi, n_ratio, n_pts=300):
    """
    相似律：在转速比 r = n/n₀ 下生成 H-Q 数据。
    Q' = Q × r，H' = H × r²
    poly1d 直接接受数组，无需逐点调用。
    """
    Q0 = np.linspace(fi.Q_min, fi.Q_max, n_pts)
    H0 = np.maximum(fi.H_spline(Q0), 0.0)
    return Q0 * n_ratio, H0 * n_ratio ** 2


def _fan_eta_at_ratio(fi, n_ratio, n_pts=300):
    """
    相似律：效率曲线 η 值不变，Q 轴按 r 缩放。
    """
    Q0  = np.linspace(fi.Q_min, fi.Q_max, n_pts)
    eta = np.clip(fi.eta_spline(Q0), 0.0, 1.0)
    return Q0 * n_ratio, eta


def _fan_shaft_power_at_ratio(fi, n_ratio, eta_t=0.95, n_pts=300):
    """
    相似律：轴功率 N ∝ n³。
    N₀ = Q·H/(1000·η·η_t)  →  N = N₀ × r³，Q = Q₀ × r
    """
    Q0  = np.linspace(fi.Q_min, fi.Q_max, n_pts)
    H0  = np.maximum(fi.H_spline(Q0), 0.01)
    e0  = np.clip(fi.eta_spline(Q0), 0.01, 1.0)
    N0  = Q0 * H0 / (1000.0 * e0 * eta_t)
    return Q0 * n_ratio, N0 * n_ratio ** 3


def _solve_base_point(fi, R):
    """
    在额定 H-Q 曲线上求满足 H_fan(u) = R·u² 的基准点 u。
    返回 u 或 None（无交点时）。
    """
    def f(Q):
        return float(fi.H_spline(Q)) - R * Q ** 2

    fa, fb = f(fi.Q_min), f(fi.Q_max)
    if fa * fb > 0:
        return None
    return brentq(f, fi.Q_min, fi.Q_max, xtol=1e-4)


# ─────────────────────────────────────────────
# 主绘图函数
# ─────────────────────────────────────────────

def plot_fan_result(best, dual_result, eta_t=0.95, eta_min=0.70, save_path=None):
    """
    对推荐风机绘制完整特性曲线综合图。

    参数：
        best         : SelectionCandidate（selector.run_selector 输出的最优机型）
        dual_result  : DualPeriodResult（双工况引擎输出）
        eta_t        : 机械传动效率，默认 0.95
        save_path    : 保存路径（None 则弹窗显示）
    """
    fan     = best.fan
    fi      = build_fan_interpolated(fan)
    n0      = fan.rated_rpm
    n_easy  = best.n_easy_rpm
    n_hard  = best.n_hard_rpm
    r_easy  = n_easy / n0
    r_hard  = n_hard / n0

    Re = dual_result.easy.R
    Rh = dual_result.hard.R
    Qe, He = best.Q_op_easy, best.H_op_easy   # 容易时期设计工作点
    Qh, Hh = best.Q_op_hard, best.H_op_hard   # 困难时期设计工作点

    # 额定曲线上的基准点 u（用于标注）
    u_easy = _solve_base_point(fi, Re)
    u_hard = _solve_base_point(fi, Rh)
    H_u_easy = Re * u_easy ** 2 if u_easy else None
    H_u_hard = Rh * u_hard ** 2 if u_hard else None

    # ── 曲线数据生成 ──────────────────────────────────
    # 风机 H-Q
    Q_rated, H_rated   = _fan_HQ_at_ratio(fi, 1.0)
    Q_at_easy, H_at_easy = _fan_HQ_at_ratio(fi, r_easy)
    Q_at_hard, H_at_hard = _fan_HQ_at_ratio(fi, r_hard)

    # 变频曲线族（60%~90% 额定，用于背景参考）
    family_ratios = np.arange(0.6, 0.91, 0.1)

    # 效率 η-Q
    Q_eta_r,    eta_r    = _fan_eta_at_ratio(fi, 1.0)
    Q_eta_easy, eta_easy_curve = _fan_eta_at_ratio(fi, r_easy)
    Q_eta_hard, eta_hard_curve = _fan_eta_at_ratio(fi, r_hard)

    # 轴功率 N-Q
    Q_N_r,    N_r    = _fan_shaft_power_at_ratio(fi, 1.0,    eta_t)
    Q_N_easy, N_easy = _fan_shaft_power_at_ratio(fi, r_easy, eta_t)
    Q_N_hard, N_hard = _fan_shaft_power_at_ratio(fi, r_hard, eta_t)

    # 管道阻力线范围
    Q_plot_max = fi.Q_max * 1.15
    Q_pipe = np.linspace(0, Q_plot_max, 300)
    H_pipe_easy = Re * Q_pipe ** 2
    H_pipe_hard = Rh * Q_pipe ** 2

    # ── 画布布局 ──────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"推荐风机特性曲线 — {fan.model_id}  "
        f"（{fan.motor_count}×{fan.motor_kw} kW，额定 {n0} r/min）",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.55, 1],
                           hspace=0.38, wspace=0.28,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax_hq  = fig.add_subplot(gs[:, 0])    # 左列：H-Q 主图（占满两行）
    ax_eta = fig.add_subplot(gs[0, 1])    # 右上：η-Q
    ax_nq  = fig.add_subplot(gs[1, 1])    # 右下：N-Q

    # ══════════════════════════════════════════════════
    # ① H-Q 主图
    # ══════════════════════════════════════════════════

    # 变频曲线族（背景参考，浅灰）
    for r in family_ratios:
        Qf, Hf = _fan_HQ_at_ratio(fi, r)
        ax_hq.plot(Qf, Hf, color='#cccccc', linewidth=0.9,
                   linestyle='--', zorder=1)
        # 在曲线右端标注转速
        ax_hq.text(Qf[-1] * 1.005, Hf[-1],
                   f'{r*n0:.0f}',
                   fontsize=7, color='#aaaaaa', va='center')

    # 额定转速 H-Q 曲线
    ax_hq.plot(Q_rated, H_rated,
               color='#222222', linewidth=2.5, label=f'额定转速  {n0} r/min', zorder=4)

    # 容易时期转速曲线
    ax_hq.plot(Q_at_easy, H_at_easy,
               color='steelblue', linewidth=2.0, linestyle='-.',
               label=f'容易时期  {n_easy:.0f} r/min  ({r_easy:.3f}×额定)', zorder=4)

    # 困难时期转速曲线
    ax_hq.plot(Q_at_hard, H_at_hard,
               color='darkorange', linewidth=2.0, linestyle='-.',
               label=f'困难时期  {n_hard:.0f} r/min  ({r_hard:.3f}×额定)', zorder=4)

    # 管道阻力线
    ax_hq.plot(Q_pipe, H_pipe_easy,
               color='steelblue', linewidth=1.5, linestyle=':',
               label=f'管道阻力（容易）  R={Re:.5f}', zorder=3)
    ax_hq.plot(Q_pipe, H_pipe_hard,
               color='darkorange', linewidth=1.5, linestyle=':',
               label=f'管道阻力（困难）  R={Rh:.5f}', zorder=3)

    # 额定曲线上的基准点 u（相似律出发点）
    if u_easy:
        ax_hq.scatter([u_easy], [H_u_easy],
                      marker='^', s=90, color='steelblue',
                      zorder=6, label=f'基准点 u_easy={u_easy:.2f} m³/s')
        ax_hq.annotate(
            f'u_e={u_easy:.1f}',
            xy=(u_easy, H_u_easy),
            xytext=(u_easy - 12, H_u_easy + 180),
            fontsize=8, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.9)
        )
    if u_hard:
        ax_hq.scatter([u_hard], [H_u_hard],
                      marker='^', s=90, color='darkorange',
                      zorder=6, label=f'基准点 u_hard={u_hard:.2f} m³/s')
        ax_hq.annotate(
            f'u_h={u_hard:.1f}',
            xy=(u_hard, H_u_hard),
            xytext=(u_hard + 3, H_u_hard + 180),
            fontsize=8, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.9)
        )

    # 实际工作点（设计目标，VFD 精确交付）
    ax_hq.scatter([Qe], [He], color='steelblue', s=130,
                  marker='o', zorder=7)
    ax_hq.scatter([Qh], [Hh], color='darkorange', s=130,
                  marker='o', zorder=7)

    # 工作点标注
    ax_hq.annotate(
        f'容易时期工作点\n({Qe:.1f} m³/s, {He:.0f} Pa)\nη={best.eta_easy:.2%}',
        xy=(Qe, He),
        xytext=(Qe - 30, He - 650),
        fontsize=8.5, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4fd', edgecolor='steelblue', alpha=0.92)
    )
    ax_hq.annotate(
        f'困难时期工作点\n({Qh:.1f} m³/s, {Hh:.0f} Pa)\nη={best.eta_hard:.2%}',
        xy=(Qh, Hh),
        xytext=(Qh + 6, Hh + 350),
        fontsize=8.5, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff4e6', edgecolor='darkorange', alpha=0.92)
    )

    # 相似抛物线（两工作点所在）
    Q_sim = np.linspace(80, max(u_easy or Qe, u_hard or Qh) * 1.05, 200)
    R_sim_easy = He / Qe ** 2
    R_sim_hard = Hh / Qh ** 2
    ax_hq.plot(Q_sim, R_sim_easy * Q_sim ** 2,
               color='steelblue', linewidth=0.8, linestyle='--', alpha=0.45, zorder=2)
    ax_hq.plot(Q_sim, R_sim_hard * Q_sim ** 2,
               color='darkorange', linewidth=0.8, linestyle='--', alpha=0.45, zorder=2)

    # 坐标轴投影虚线
    for Q_val, H_val, col in [(Qe, He, 'steelblue'), (Qh, Hh, 'darkorange')]:
        ax_hq.axvline(Q_val, color=col, linestyle=':', linewidth=0.9, alpha=0.5)
        ax_hq.axhline(H_val, color=col, linestyle=':', linewidth=0.9, alpha=0.5)

    ax_hq.set_xlabel('风量 Q（m³/s）', fontsize=11)
    ax_hq.set_ylabel('全压 H（Pa）', fontsize=11)
    ax_hq.set_title('H-Q 特性曲线（额定 + 变频工况 + 管道阻力线）', fontsize=11)
    ax_hq.legend(fontsize=7.5, loc='upper right', ncol=1,
                 framealpha=0.9, edgecolor='#cccccc')
    ax_hq.grid(True, linestyle='--', alpha=0.3)
    ax_hq.set_xlim(max(0, fi.Q_min * r_easy * 0.85), Q_plot_max)
    ax_hq.set_ylim(0, fi.Q_max and max(H_rated) * 1.12)

    # 右上角信息框
    info = (
        f"转速比（困难/容易）：{best.n_ratio:.4f}\n"
        f"容易时期：{n_easy:.0f} r/min（{r_easy:.3f}×额定）\n"
        f"困难时期：{n_hard:.0f} r/min（{r_hard:.3f}×额定）\n"
        f"相似律误差 ε：{best.vfd_error:.2f}%"
    )
    ax_hq.text(0.02, 0.98, info, transform=ax_hq.transAxes, fontsize=8.2,
               va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.92))

    # ══════════════════════════════════════════════════
    # ② η-Q 曲线（右上）
    # ══════════════════════════════════════════════════

    # 背景曲线族（效率不随转速变化，Q 轴缩放）
    for r in family_ratios:
        Qf, ef = _fan_eta_at_ratio(fi, r)
        ax_eta.plot(Qf, ef * 100, color='#dddddd', linewidth=0.8, zorder=1)

    ax_eta.plot(Q_eta_r,    eta_r    * 100,
                color='#222222', linewidth=2.2, label=f'{n0} r/min（额定）', zorder=4)
    ax_eta.plot(Q_eta_easy, eta_easy_curve * 100,
                color='steelblue', linewidth=1.8, linestyle='-.',
                label=f'{n_easy:.0f} r/min（容易）', zorder=4)
    ax_eta.plot(Q_eta_hard, eta_hard_curve * 100,
                color='darkorange', linewidth=1.8, linestyle='-.',
                label=f'{n_hard:.0f} r/min（困难）', zorder=4)

    # η_min 线
    eta_min_pct = eta_min * 100.0
    ax_eta.axhline(eta_min_pct, color='crimson', linewidth=1.2,
                   linestyle='--', label=f'η_min = {eta_min_pct:.0f}%', alpha=0.8)

    # 工作点效率标注
    ax_eta.scatter([Qe], [best.eta_easy * 100],
                   color='steelblue', s=90, zorder=6)
    ax_eta.scatter([Qh], [best.eta_hard * 100],
                   color='darkorange', s=90, zorder=6)
    ax_eta.annotate(f'{best.eta_easy*100:.1f}%',
                    xy=(Qe, best.eta_easy * 100),
                    xytext=(Qe - 8, best.eta_easy * 100 - 4),
                    fontsize=8, color='steelblue')
    ax_eta.annotate(f'{best.eta_hard*100:.1f}%',
                    xy=(Qh, best.eta_hard * 100),
                    xytext=(Qh + 2, best.eta_hard * 100 - 4),
                    fontsize=8, color='darkorange')

    ax_eta.set_xlabel('风量 Q（m³/s）', fontsize=10)
    ax_eta.set_ylabel('全压效率 η（%）', fontsize=10)
    ax_eta.set_title('η-Q 效率特性曲线', fontsize=10)
    ax_eta.legend(fontsize=7.5, loc='lower left')
    ax_eta.grid(True, linestyle='--', alpha=0.3)
    ax_eta.set_xlim(max(0, fi.Q_min * r_easy * 0.85), fi.Q_max * 1.05)
    ax_eta.set_ylim(0, 100)

    # ══════════════════════════════════════════════════
    # ③ N-Q 轴功率曲线（右下）
    # ══════════════════════════════════════════════════

    # 曲线族
    for r in family_ratios:
        Qf, Nf = _fan_shaft_power_at_ratio(fi, r, eta_t)
        ax_nq.plot(Qf, Nf, color='#dddddd', linewidth=0.8, zorder=1)

    ax_nq.plot(Q_N_r,    N_r,
               color='#222222', linewidth=2.2, label=f'{n0} r/min（额定）', zorder=4)
    ax_nq.plot(Q_N_easy, N_easy,
               color='steelblue', linewidth=1.8, linestyle='-.',
               label=f'{n_easy:.0f} r/min（容易）', zorder=4)
    ax_nq.plot(Q_N_hard, N_hard,
               color='darkorange', linewidth=1.8, linestyle='-.',
               label=f'{n_hard:.0f} r/min（困难）', zorder=4)

    # 装机功率线
    N_installed = fan.total_motor_kw
    ax_nq.axhline(N_installed, color='crimson', linewidth=1.3,
                  linestyle='--', label=f'装机功率 {N_installed:.0f} kW', alpha=0.8)

    # 工作点轴功率
    ax_nq.scatter([Qe], [best.N_shaft_easy],
                  color='steelblue', s=90, zorder=6)
    ax_nq.scatter([Qh], [best.N_shaft_hard],
                  color='darkorange', s=90, zorder=6)
    ax_nq.annotate(f'{best.N_shaft_easy:.0f} kW',
                   xy=(Qe, best.N_shaft_easy),
                   xytext=(Qe - 12, best.N_shaft_easy + N_installed * 0.04),
                   fontsize=8, color='steelblue')
    ax_nq.annotate(f'{best.N_shaft_hard:.0f} kW',
                   xy=(Qh, best.N_shaft_hard),
                   xytext=(Qh + 2, best.N_shaft_hard + N_installed * 0.04),
                   fontsize=8, color='darkorange')

    ax_nq.set_xlabel('风量 Q（m³/s）', fontsize=10)
    ax_nq.set_ylabel('轴功率 N（kW）', fontsize=10)
    ax_nq.set_title('N-Q 轴功率特性曲线', fontsize=10)
    ax_nq.legend(fontsize=7.5, loc='upper right')
    ax_nq.grid(True, linestyle='--', alpha=0.3)
    ax_nq.set_xlim(max(0, fi.Q_min * r_easy * 0.85), fi.Q_max * 1.05)
    ax_nq.set_ylim(0, N_installed * 1.35)

    # ── 保存或显示 ────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存：{save_path}")
    elif matplotlib.get_backend().lower() != 'agg':
        plt.show()

    return fig


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # ── 第一步：双工况引擎 ────────────────────────────
    dp = DualPeriodParams(
        Q1_easy  = 114.5 * 60,
        h_f_easy = 2276.0,
        Q1_hard  = 129.5 * 60,
        h_f_hard = 3050.2,
    )
    dual = run_dual(dp, SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0))

    # ── 第二步：选型 ──────────────────────────────────
    result = run_selector(dual, eta_min=0.70)

    if result.best is None:
        print("未找到合格机型，无法绘图。")
        raise SystemExit(1)

    best = result.best
    print(f"推荐机型：{best.fan.model_id}")
    print(f"容易时期：Q={best.Q_op_easy:.2f} m³/s  H={best.H_op_easy:.1f} Pa  "
          f"η={best.eta_easy:.2%}  n={best.n_easy_rpm:.1f} r/min")
    print(f"困难时期：Q={best.Q_op_hard:.2f} m³/s  H={best.H_op_hard:.1f} Pa  "
          f"η={best.eta_hard:.2%}  n={best.n_hard_rpm:.1f} r/min")
    print(f"轴功率：容易={best.N_shaft_easy:.1f} kW  困难={best.N_shaft_hard:.1f} kW")
    print(f"变频转速比：{best.n_ratio:.4f}  相似律误差 ε={best.vfd_error:.2f}%")

    # ── 第三步：绘图 ──────────────────────────────────
    save = r'C:\Users\star\Desktop\Cluade\seclect\ref\fan_result_curves.png'
    plot_fan_result(best, dual, save_path=save)
