# -*- coding: utf-8 -*-
"""
风机初步筛选模块
文件：fan_screen.py

功能：根据双工况引擎输出的 (Q_easy, H_easy) 和 (Q_hard, H_hard)，
      对风机数据库进行初步范围筛选，支持变频调速工况校核。

架构风格：与 calc_engine.py 一致，每步独立函数，返回 (结果, steps)
依赖：calc_engine_range.py → DualPeriodResult
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class FanSpec:
    """
    风机基本规格（来自厂家样本，仅含范围数据）
    用于初步筛选，尚不含 H-Q 曲线数据点。
    """
    model_id:      str      # 型号，如 "FBCDZ(C)-12-No32"
    rated_rpm:     float    # 额定转速 (r/min)
    motor_kw:      float    # 单台电机功率 (kW)
    motor_count:   int      # 电机台数
    Q_min:         float    # 额定转速下最小风量 (m³/s)
    Q_max:         float    # 额定转速下最大风量 (m³/s)
    H_min:         float    # 对应最小全压 (Pa)
    H_max:         float    # 对应最大全压 (Pa)

    @property
    def total_motor_kw(self) -> float:
        """总装机功率 (kW)"""
        return self.motor_kw * self.motor_count


@dataclass
class ScreenResult:
    """单台风机的筛选结果"""
    fan:            FanSpec
    passed:         bool        # 是否通过筛选
    hard_Q_ok:      bool        # 困难时期风量是否覆盖
    hard_H_ok:      bool        # 困难时期风压是否覆盖
    easy_Q_ok:      bool        # 容易时期风量是否可达（变频降速后）
    easy_H_ok:      bool        # 容易时期风压是否在范围内
    n_ratio:        float       # 容易/困难时期转速比（变频时）
    n_easy_rpm:     float       # 容易时期所需转速 (r/min)
    motor_ok:       bool        # 总装机功率是否满足需求
    reject_reason:  str         # 淘汰原因（passed=False 时说明）
    steps:          list = field(default_factory=list)


@dataclass
class ScreenSummary:
    """全部风机的筛选汇总"""
    candidates:     list        # List[ScreenResult]，通过筛选的型号
    rejected:       list        # List[ScreenResult]，被淘汰的型号
    steps:          list = field(default_factory=list)


# ─────────────────────────────────────────────
# 内置数据库（来自厂家样本）
# ─────────────────────────────────────────────

BUILTIN_FANS = [
    FanSpec("FBCDZ(B)-6-No20",  980, 220, 2,  55,  119, 1170, 4100),
    FanSpec("FBCDZ(B)-8-No19",  740,  90, 2,  35,   77,  600, 2100),
    FanSpec("FBCDZ(B)-8-No22",  740, 160, 2,  57,  120,  800, 2800),
    FanSpec("FBCDZ(C)-10-No34", 580, 710, 2, 129,  270, 1100, 5200),
    FanSpec("FBCDZ(C)-12-No32", 480, 355, 2,  86,  203,  780, 3400),
]


# ─────────────────────────────────────────────
# 步骤函数
# ─────────────────────────────────────────────

def check_hard_period(
    fan: FanSpec,
    Q_hard: float,
    H_hard: float
) -> tuple[bool, bool, list]:
    """
    步骤一：校验困难时期（额定转速）工况是否在风机范围内。

    条件：
        Q_hard ≤ Q_max   风机最大风量需覆盖困难时期需求
        H_hard ≤ H_max   风机最大风压需覆盖困难时期需求
        Q_hard ≥ Q_min   困难时期风量不低于风机最小风量

    返回：(Q_ok, H_ok, steps)
    """
    steps = []
    Q_ok = fan.Q_min <= Q_hard <= fan.Q_max
    H_ok = fan.H_min <= H_hard <= fan.H_max

    steps.append(
        f"  [困难时期·风量] 要求 {Q_hard} m³/s，风机范围 {fan.Q_min}~{fan.Q_max}  "
        + ("✅" if Q_ok else "❌")
    )
    steps.append(
        f"  [困难时期·风压] 要求 {H_hard} Pa，风机范围 {fan.H_min}~{fan.H_max}  "
        + ("✅" if H_ok else "❌")
    )
    return Q_ok, H_ok, steps


def check_easy_period_vfd(
    fan: FanSpec,
    Q_easy: float,
    H_easy: float,
    Q_hard: float,
    H_hard: float
) -> tuple[bool, bool, float, float, list]:
    """
    步骤二：变频调速下容易时期工况校核。

    原理（相似律）：
        n_easy / n_hard = Q_easy / Q_hard
        H_easy_predict  = H_hard × (Q_easy / Q_hard)²

    校验：
        · 预测风压 H_easy_predict 与实际需求 H_easy 偏差 < 15%（相似律近似误差范围）
        · 降速后风量 Q_easy 仍在风机允许的调速下限以上（取 Q_min × 0.7 为保守估计）

    返回：(Q_ok, H_ok, n_ratio, n_easy_rpm, steps)
    """
    steps = []
    n_ratio   = Q_easy / Q_hard                        # 转速比（容易/困难）
    n_easy_rpm = fan.rated_rpm * n_ratio               # 容易时期所需转速

    # 相似律预测容易时期风压
    H_easy_predict = H_hard * (n_ratio ** 2)
    H_err_pct = abs(H_easy_predict - H_easy) / H_easy * 100

    # 风量：降速后最小可达风量 = Q_min × n_ratio（相似律）
    Q_min_vfd = fan.Q_min * n_ratio
    Q_ok = Q_easy >= Q_min_vfd

    # 风压：相似律预测误差 < 15% 视为可接受
    H_ok = H_err_pct < 15.0

    steps.append(
        f"  [容易时期·转速比] n_easy/n_hard = {Q_easy}/{Q_hard} = {n_ratio:.4f}  "
        f"→ 所需转速 {n_easy_rpm:.1f} r/min"
    )
    steps.append(
        f"  [容易时期·风量]  降速后最小风量 {Q_min_vfd:.2f} m³/s，需求 {Q_easy} m³/s  "
        + ("✅" if Q_ok else "❌")
    )
    steps.append(
        f"  [容易时期·风压]  相似律预测 {H_easy_predict:.1f} Pa，实际需求 {H_easy} Pa，"
        f"偏差 {H_err_pct:.1f}%  " + ("✅" if H_ok else "❌（偏差过大）")
    )
    return Q_ok, H_ok, n_ratio, n_easy_rpm, steps


def check_motor_power(
    fan: FanSpec,
    Q_hard: float,
    H_hard: float,
    eta_min: float = 0.70,
    eta_t: float = 0.95,
    K_N: float = 1.20
) -> tuple[bool, float, list]:
    """
    步骤三：电机功率是否满足困难时期需求（最大功率工况）。

    N_required = Q_hard × H_hard / (1000 × eta_min × eta_t) × K_N

    返回：(ok, N_required, steps)
    """
    steps = []
    N_required = Q_hard * H_hard / (1000.0 * eta_min * eta_t) * K_N
    ok = fan.total_motor_kw >= N_required

    steps.append(
        f"  [电机功率]  需求 ≥ {N_required:.1f} kW，"
        f"装机 {fan.total_motor_kw:.0f} kW ({fan.motor_count}×{fan.motor_kw} kW)  "
        + ("✅" if ok else "❌")
    )
    return ok, N_required, steps


# ─────────────────────────────────────────────
# 单台风机完整筛选
# ─────────────────────────────────────────────

def screen_one(
    fan: FanSpec,
    Q_easy: float, H_easy: float,
    Q_hard: float, H_hard: float,
    eta_min: float = 0.70
) -> ScreenResult:
    """
    对单台风机执行全部三步筛选，返回 ScreenResult。
    """
    all_steps = [f"── {fan.model_id}（{fan.motor_count}×{fan.motor_kw}kW，"
                 f"n={fan.rated_rpm}rpm）──"]

    # 步骤一：困难时期范围校验
    hard_Q_ok, hard_H_ok, s = check_hard_period(fan, Q_hard, H_hard)
    all_steps.extend(s)

    # 步骤二：变频容易时期校验
    easy_Q_ok, easy_H_ok, n_ratio, n_easy_rpm, s = check_easy_period_vfd(
        fan, Q_easy, H_easy, Q_hard, H_hard
    )
    all_steps.extend(s)

    # 步骤三：电机功率校验
    motor_ok, N_req, s = check_motor_power(fan, Q_hard, H_hard, eta_min)
    all_steps.extend(s)

    # 综合判断
    passed = hard_Q_ok and hard_H_ok and easy_Q_ok and easy_H_ok and motor_ok

    # 淘汰原因汇总
    reasons = []
    if not hard_Q_ok: reasons.append(f"困难时期风量 {Q_hard} 超出范围 {fan.Q_min}~{fan.Q_max}")
    if not hard_H_ok: reasons.append(f"困难时期风压 {H_hard} 超出范围 {fan.H_min}~{fan.H_max}")
    if not easy_Q_ok: reasons.append("变频降速后风量不足")
    if not easy_H_ok: reasons.append("相似律风压预测偏差 ≥ 15%")
    if not motor_ok:  reasons.append(f"装机功率 {fan.total_motor_kw} kW < 需求 {N_req:.1f} kW")

    verdict = "✅ 通过" if passed else f"❌ 淘汰：{'；'.join(reasons)}"
    all_steps.append(f"  [结论] {verdict}")

    return ScreenResult(
        fan=fan,
        passed=passed,
        hard_Q_ok=hard_Q_ok, hard_H_ok=hard_H_ok,
        easy_Q_ok=easy_Q_ok, easy_H_ok=easy_H_ok,
        n_ratio=n_ratio, n_easy_rpm=n_easy_rpm,
        motor_ok=motor_ok,
        reject_reason="" if passed else "；".join(reasons),
        steps=all_steps,
    )


# ─────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────

def run_screen(
    Q_easy: float, H_easy: float,
    Q_hard: float, H_hard: float,
    fans: Optional[list] = None,
    eta_min: float = 0.70
) -> ScreenSummary:
    """
    对全部风机执行初步筛选，返回 ScreenSummary。

    参数：
        Q_easy / H_easy : 容易时期工作点 (m³/s, Pa)
        Q_hard / H_hard : 困难时期工作点 (m³/s, Pa)
        fans            : 风机列表，默认使用内置 BUILTIN_FANS
        eta_min         : 效率下限（用于功率估算），默认 0.70

    示例：
        result = run_screen(114.5, 2276.0, 129.5, 3050.2)
    """
    if fans is None:
        fans = BUILTIN_FANS

    all_steps = [
        f"[筛选条件] 容易时期 Q={Q_easy} m³/s，H={H_easy} Pa",
        f"[筛选条件] 困难时期 Q={Q_hard} m³/s，H={H_hard} Pa",
        f"[筛选条件] 效率下限 η_min={eta_min}，共 {len(fans)} 款待筛选",
        "",
    ]

    candidates, rejected = [], []
    for fan in fans:
        sr = screen_one(fan, Q_easy, H_easy, Q_hard, H_hard, eta_min)
        all_steps.extend(sr.steps)
        all_steps.append("")
        (candidates if sr.passed else rejected).append(sr)

    all_steps.append(
        f"[汇总] 通过 {len(candidates)} 款 / 淘汰 {len(rejected)} 款"
    )
    return ScreenSummary(candidates=candidates, rejected=rejected, steps=all_steps)


# ─────────────────────────────────────────────
# 命令行验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    Q_e, H_e = 114.5, 2276.0
    Q_h, H_h = 129.5, 3050.2

    print("=" * 62)
    print("矿井风机初步筛选 —— 论文铁矿数据")
    print("=" * 62)

    result = run_screen(Q_e, H_e, Q_h, H_h)

    # ── 逐台打印计算过程 ─────────────────────────────
    for step in result.steps:
        print(step)

    # ── 汇总表 ───────────────────────────────────────
    print("=" * 62)
    print(f"{'型号':<24} {'结论':^6} {'装机(kW)':>9} {'转速比':>7} {'容易转速':>9} {'淘汰原因'}")
    print("-" * 62)
    for sr in result.candidates + result.rejected:
        verdict = "✅通过" if sr.passed else "❌淘汰"
        reason  = "" if sr.passed else sr.reject_reason[:20]
        print(f"{sr.fan.model_id:<24} {verdict:^6} "
              f"{sr.fan.total_motor_kw:>9.0f} "
              f"{sr.n_ratio:>7.4f} "
              f"{sr.n_easy_rpm:>7.1f}rpm  "
              f"{reason}")

    print("=" * 62)
    print(f"推荐候选（共 {len(result.candidates)} 款）：")
    for sr in result.candidates:
        print(f"  ▶ {sr.fan.model_id}")
        print(f"    装机：{sr.fan.motor_count}×{sr.fan.motor_kw} kW = {sr.fan.total_motor_kw:.0f} kW")
        print(f"    困难时期转速：{sr.fan.rated_rpm} r/min")
        print(f"    容易时期转速：{sr.n_easy_rpm:.1f} r/min（降速比 {sr.n_ratio:.4f}）")
