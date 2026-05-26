# -*- coding: utf-8 -*-
"""
矿井通风机选型计算引擎 —— 双工况版
文件：calc_engine_range.py

适用场景：分别计算矿井容易时期与困难时期两个工况的
          管道特性曲线及选型指标。

核心逻辑：对两组参数各调用一次 calc_engine.run()，
          汇总结果并生成两条管道特性曲线数据。

依赖：calc_engine.py
"""

from dataclasses import dataclass, field
from typing import Optional
from calc_engine import MineParams, SelectionCoeffs, EngineResult, run


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class DualPeriodParams:
    """容易时期与困难时期的双工况输入参数"""
    # 容易时期
    Q1_easy:   float          # 容易时期需风量 (m³/min)
    h_f_easy:  float          # 容易时期网路总阻力 (Pa)
    # 困难时期
    Q1_hard:   float          # 困难时期需风量 (m³/min)
    h_f_hard:  float          # 困难时期网路总阻力 (Pa)
    # 固定参数（两个时期共用）
    Q2:        float = 0.0
    h_vc:      float = 0.0
    h_e:       float = 0.0
    altitude_m: float = 0.0


@dataclass
class DualPeriodResult:
    """双工况计算结果"""
    easy: EngineResult         # 容易时期单点结果
    hard: EngineResult         # 困难时期单点结果

    # 两条管道特性曲线数据（供绘图）
    easy_curve: tuple = field(default_factory=tuple)   # (Q_list, H_list)
    hard_curve: tuple = field(default_factory=tuple)   # (Q_list, H_list)

    # 全程计算步骤
    steps: list = field(default_factory=list)


# ─────────────────────────────────────────────
# 步骤函数
# ─────────────────────────────────────────────

def calc_period(
    label: str,
    Q1: float,
    h_f: float,
    params_base: DualPeriodParams,
    coeffs: SelectionCoeffs
) -> tuple[EngineResult, list]:
    """
    计算单个时期的选型结果，返回 (EngineResult, steps)。
    内部直接调用 calc_engine.run()，不重复任何计算逻辑。
    """
    steps = []
    params = MineParams(
        Q1=Q1, h_f=h_f,
        Q2=params_base.Q2,
        h_vc=params_base.h_vc,
        h_e=params_base.h_e,
        altitude_m=params_base.altitude_m,
    )
    result = run(params, coeffs)

    steps.append(f"[{label}] Q1={Q1} m³/min，h_f={h_f} Pa")
    steps.extend(f"[{label}] {s}" for s in result.steps)
    return result, steps


def calc_pipeline_curve(
    label: str,
    R: float,
    Q_max: float,
    n: int = 200
) -> tuple[tuple, list]:
    """
    生成一条管道特性曲线 H = R·Q² 的数据点，返回 ((Q_list, H_list), steps)。
    """
    steps = []
    Q_list = [Q_max * i / (n - 1) for i in range(n)]
    H_list = [R * q ** 2 for q in Q_list]
    steps.append(
        f"[{label}管道曲线] H = {R:.6f} × Q²，"
        f"Q 范围 0 → {Q_max:.4f} m³/s，共 {n} 个数据点"
    )
    return (Q_list, H_list), steps


# ─────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────

def run_dual(
    dp: DualPeriodParams,
    coeffs: Optional[SelectionCoeffs] = None,
    curve_n: int = 200
) -> DualPeriodResult:
    """
    双工况通风机选型计算，返回 DualPeriodResult。

    编排：
        1. calc_period("容易时期", ...)   → easy EngineResult
        2. calc_period("困难时期", ...)   → hard EngineResult
        3. calc_pipeline_curve(easy)      → easy 管道特性曲线
        4. calc_pipeline_curve(hard)      → hard 管道特性曲线

    示例：
        dp = DualPeriodParams(
            Q1_easy=6870, h_f_easy=2276.0,
            Q1_hard=7770, h_f_hard=3050.2,
        )
        result = run_dual(dp, SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0))
    """
    if coeffs is None:
        coeffs = SelectionCoeffs()

    all_steps = []

    # 1 & 2. 两个时期的选型计算
    easy, s = calc_period("容易时期", dp.Q1_easy, dp.h_f_easy, dp, coeffs)
    all_steps.extend(s)

    hard, s = calc_period("困难时期", dp.Q1_hard, dp.h_f_hard, dp, coeffs)
    all_steps.extend(s)

    # 3 & 4. 管道特性曲线（X 轴上限取困难时期风量的 1.3 倍）
    Q_max = hard.Q_f_ms * 1.3

    easy_curve, s = calc_pipeline_curve("容易时期", easy.R, Q_max, curve_n)
    all_steps.extend(s)

    hard_curve, s = calc_pipeline_curve("困难时期", hard.R, Q_max, curve_n)
    all_steps.extend(s)

    return DualPeriodResult(
        easy=easy,
        hard=hard,
        easy_curve=easy_curve,
        hard_curve=hard_curve,
        steps=all_steps,
    )


# ─────────────────────────────────────────────
# 命令行验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("双工况通风机选型引擎 —— 论文铁矿实测数据验证")
    print("=" * 60)

    dp = DualPeriodParams(
        Q1_easy  = 114.5 * 60,   # m³/s → m³/min
        h_f_easy = 2276.0,
        Q1_hard  = 129.5 * 60,
        h_f_hard = 3050.2,
    )
    coeffs = SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0)
    res = run_dual(dp, coeffs)

    # ── 对比表 ─────────────────────────────────────────
    print(f"\n{'指标':<20} {'容易时期':>14} {'困难时期':>14}")
    print("-" * 50)
    print(f"{'工作风量 Q_f (m³/s)':<20} {res.easy.Q_f_ms:>14.4f} {res.hard.Q_f_ms:>14.4f}")
    print(f"{'工作风压 H_f (Pa)':<20} {res.easy.H_f:>14.2f} {res.hard.H_f:>14.2f}")
    print(f"{'阻力系数 R':<20} {res.easy.R:>14.6f} {res.hard.R:>14.6f}")
    print(f"{'估算轴功率 (kW)':<20} {res.easy.N_shaft:>14.2f} {res.hard.N_shaft:>14.2f}")
    print(f"{'配套电机功率 (kW)':<20} {res.easy.N_motor:>14.2f} {res.hard.N_motor:>14.2f}")
    print("-" * 50)
    print(f"{'风机选型按困难时期':>50}")

    # ── 管道特性曲线数据预览 ────────────────────────────
    print(f"\n管道特性曲线数据预览（各取 5 点）：")
    print(f"{'Q (m³/s)':>12}  {'容易时期 H (Pa)':>16}  {'困难时期 H (Pa)':>16}")
    print("-" * 50)
    idx = [0, 25, 50, 75, 99]
    for i in idx:
        q  = res.easy_curve[0][i]
        he = res.easy_curve[1][i]
        hh = res.hard_curve[1][i]
        print(f"{q:>12.3f}  {he:>16.2f}  {hh:>16.2f}")

    # ── 完整 steps ──────────────────────────────────────
    print(f"\n── 完整计算过程 ──")
    for step in res.steps:
        print(f"  {step}")
