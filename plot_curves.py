# -*- coding: utf-8 -*-
"""
管道特性曲线绘图模块
运行：python plot_curves.py

包含两个绘图函数：
  plot_pipeline()      单工况管道特性曲线（原有）
  plot_dual_period()   双工况管道特性曲线 + 工作点矩形区域（新增）
"""

import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from calc_engine import MineParams, SelectionCoeffs, run
from calc_engine_range import DualPeriodParams, run_dual


# ─────────────────────────────────────────────
# 单工况绘图（保留原有功能）
# ─────────────────────────────────────────────

def plot_pipeline(result, params, coeffs, save_path=None):
    """绘制单工况管道特性曲线与工作点"""
    Q_f   = result.Q_f_ms
    H_f   = result.H_f
    R     = result.R
    Q_max = Q_f * 1.6
    Q_arr = np.linspace(0, Q_max, 300)
    H_arr = R * Q_arr ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Q_arr, H_arr, color='steelblue', linewidth=2,
            label=f'管道特性曲线  H = {R:.4f}·Q²')
    ax.scatter([Q_f], [H_f], color='crimson', s=80, zorder=5,
               label=f'工作点  ({Q_f:.3f} m³/s, {H_f:.1f} Pa)')
    ax.axvline(Q_f, color='crimson', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(H_f, color='crimson', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.annotate(
        f'  Q_f = {Q_f:.3f} m³/s\n  H_f = {H_f:.1f} Pa',
        xy=(Q_f, H_f), xytext=(Q_f + Q_max * 0.05, H_f + H_f * 0.08),
        fontsize=9, color='crimson',
        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2)
    )
    info = (f"Q₁ = {params.Q1} m³/min\nh_f = {params.h_f} Pa\n"
            f"K_Q = {coeffs.K_Q}   K_H = {coeffs.K_H}\nR = {R:.4f} N·s²/m⁸")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    ax.set_xlabel('风量 Q（m³/s）', fontsize=11)
    ax.set_ylabel('风压 H（Pa）', fontsize=11)
    ax.set_title('矿井通风网路管道特性曲线', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, Q_max)
    ax.set_ylim(0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存：{save_path}")
    elif matplotlib.get_backend().lower() != 'agg':
        plt.show()
    return fig


# ─────────────────────────────────────────────
# 双工况绘图（新增）
# ─────────────────────────────────────────────

def plot_dual_period(res, save_path=None):
    """
    绘制容易时期与困难时期两条管道特性曲线，
    并将两个工作点围成的矩形区域标注在图上。

    参数：
        res       : run_dual() 返回的 DualPeriodResult
        save_path : 若指定则保存为 PNG，否则弹窗显示
    """
    # ── 从结果中取数据 ────────────────────────────────
    Qe, He = res.easy.Q_f_ms, res.easy.H_f   # 容易时期工作点
    Qh, Hh = res.hard.Q_f_ms, res.hard.H_f   # 困难时期工作点
    Re, Rh = res.easy.R, res.hard.R

    Q_easy_arr = np.array(res.easy_curve[0])
    H_easy_arr = np.array(res.easy_curve[1])
    Q_hard_arr = np.array(res.hard_curve[0])
    H_hard_arr = np.array(res.hard_curve[1])

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── 两条管道特性曲线 ──────────────────────────────
    ax.plot(Q_easy_arr, H_easy_arr,
            color='steelblue', linewidth=2.2, label=f'容易时期  H = {Re:.4f}·Q²')
    ax.plot(Q_hard_arr, H_hard_arr,
            color='darkorange', linewidth=2.2, label=f'困难时期  H = {Rh:.4f}·Q²')

    # ── 工作点矩形区域 ────────────────────────────────
    # 矩形左下角 = (Qe, He)，右上角 = (Qh, Hh)
    rect_x = Qe
    rect_y = He
    rect_w = Qh - Qe
    rect_h = Hh - He
    rect = mpatches.FancyArrowPatch   # 仅占位，用 Rectangle 绘制
    ax.add_patch(mpatches.FancyBboxPatch(
        (rect_x, rect_y), rect_w, rect_h,
        boxstyle="square,pad=0",
        linewidth=1.8, linestyle='--',
        edgecolor='crimson', facecolor='crimson', alpha=0.08,
        zorder=3, label='工作范围矩形'
    ))
    # 矩形四条边的虚线投影到坐标轴
    for Q_val, color in [(Qe, 'steelblue'), (Qh, 'darkorange')]:
        ax.axvline(Q_val, color=color, linestyle=':', linewidth=1.0, alpha=0.7)
    for H_val, color in [(He, 'steelblue'), (Hh, 'darkorange')]:
        ax.axhline(H_val, color=color, linestyle=':', linewidth=1.0, alpha=0.7)

    # ── 两个工作点 ────────────────────────────────────
    ax.scatter([Qe], [He], color='steelblue', s=90, zorder=6)
    ax.scatter([Qh], [Hh], color='darkorange', s=90, zorder=6)

    # 容易时期工作点标注（左下）
    ax.annotate(
        f'容易时期\n({Qe:.2f}, {He:.1f})',
        xy=(Qe, He),
        xytext=(Qe - (Qh - Qe) * 0.55, He - (Hh - He) * 0.38),
        fontsize=9, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='steelblue', alpha=0.9)
    )
    # 困难时期工作点标注（右上）
    ax.annotate(
        f'困难时期\n({Qh:.2f}, {Hh:.1f})',
        xy=(Qh, Hh),
        xytext=(Qh + (Qh - Qe) * 0.12, Hh + (Hh - He) * 0.15),
        fontsize=9, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkorange', alpha=0.9)
    )

    # ── 矩形尺寸标注 ──────────────────────────────────
    mid_Q = (Qe + Qh) / 2
    mid_H = (He + Hh) / 2
    ax.annotate(
        '', xy=(Qh, He * 0.97), xytext=(Qe, He * 0.97),
        arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.2)
    )
    ax.text(mid_Q, He * 0.93,
            f'ΔQ = {Qh - Qe:.2f} m³/s', ha='center', fontsize=8.5,
            color='crimson')
    ax.annotate(
        '', xy=(Qh * 1.015, Hh), xytext=(Qh * 1.015, He),
        arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.2)
    )
    ax.text(Qh * 1.022, mid_H,
            f'ΔH = {Hh - He:.1f} Pa', ha='left', fontsize=8.5,
            color='crimson', rotation=90, va='center')

    # ── 参数信息框 ────────────────────────────────────
    info = (
        f"容易时期  Q={Qe:.2f} m³/s  H={He:.1f} Pa  R={Re:.6f}\n"
        f"困难时期  Q={Qh:.2f} m³/s  H={Hh:.1f} Pa  R={Rh:.6f}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    # ── 轴与图例 ─────────────────────────────────────
    ax.set_xlabel('风量 Q（m³/s）', fontsize=11)
    ax.set_ylabel('风压 H（Pa）', fontsize=11)
    ax.set_title('矿井通风网路管道特性曲线（容易时期 vs 困难时期）', fontsize=13)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.88))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlim(0, max(Q_hard_arr) * 1.05)
    ax.set_ylim(0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存：{save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # ── 双工况图 ──────────────────────────────────────
    dp = DualPeriodParams(
        Q1_easy  = 114.5 * 60,
        h_f_easy = 2276.0,
        Q1_hard  = 129.5 * 60,
        h_f_hard = 3050.2,
    )
    coeffs = SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0)
    res = run_dual(dp, coeffs)

    plot_dual_period(res,
        save_path=r'C:\Users\star\Desktop\Cluade\seclect\ref\dual_period_curve.png')

    # ── 单工况图（原有，保留备用） ────────────────────
    params = MineParams(Q1=1250, h_f=56)
    coeffs2 = SelectionCoeffs(K=1.0, K_Q=1.15, K_H=1.10)
    result2 = run(params, coeffs2)
    plot_pipeline(result2, params, coeffs2,
        save_path=r'C:\Users\star\Desktop\Cluade\seclect\ref\pipeline_curve.png')
