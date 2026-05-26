# -*- coding: utf-8 -*-
"""
矿井风机选型核心模块
文件：selector.py

功能：
  接收双工况引擎输出（DualPeriodResult），对风机数据库中的候选型号
  依次执行 5 步选型算法，最终按综合评分排序输出推荐结果。

选型流程：
  ① screen_by_range    —— 按 Q_max / H_max / 电机功率范围快速初筛
  ② solve_working_point —— 管道特性曲线 ∩ 风机 H-Q 曲线（brentq 求根）
  ③ verify_efficiency  —— 两个工况的效率 η ≥ η_min
  ④ verify_vfd_speed   —— 变频转速比 + 相似律误差 ε
  ⑤ score_candidate    —— 综合评分（加权）

依赖：fan_db.py（FanModel / FanInterpolated / BUILTIN_DB）
      calc_engine_range.py（DualPeriodResult）
      scipy.optimize.brentq
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from fan_db import FanModel, FanInterpolated, build_fan_interpolated, FAN_DB, BUILTIN_DB


# ─────────────────────────────────────────────
# 数据结构（来自 selector_plan.md §5）
# ─────────────────────────────────────────────

@dataclass
class SelectionCandidate:
    """单个候选风机的完整选型结果"""
    fan:           FanModel   # 风机型号信息

    # ── 工作点（在额定转速曲线上的交点，对应各时期管道阻力线）──
    Q_op_easy:    float        # 容易时期工作风量 (m³/s)，≈ Q_easy 要求值
    H_op_easy:    float        # 容易时期工作风压 (Pa)，= R_easy × Q_op_easy²
    Q_op_hard:    float        # 困难时期工作风量 (m³/s)，≈ Q_hard 要求值
    H_op_hard:    float        # 困难时期工作风压 (Pa)，= R_hard × Q_op_hard²

    # ── 效率 ──
    eta_easy:     float        # 容易时期工作点效率（相似律保持）
    eta_hard:     float        # 困难时期工作点效率

    # ── 功率 ──
    N_shaft_easy: float        # 容易时期轴功率 (kW)
    N_shaft_hard: float        # 困难时期轴功率 (kW)
    N_motor_req:  float        # 建议最小电机功率 (kW)（含裕量系数 K_N=1.2）

    # ── 变频参数 ──
    n_ratio:      float        # 转速比 n_hard / n_easy = Q_op_hard / Q_op_easy
    n_easy_rpm:   float        # 容易时期所需转速 (r/min)
    n_hard_rpm:   float        # 困难时期所需转速 (r/min)（= 额定转速）
    vfd_error:    float        # 相似律误差 ε (%)，越小越好

    # ── 综合评分 ──
    score:        float        # 0~1，越大越优

    # ── 计算过程 ──
    steps:        list = field(default_factory=list)


@dataclass
class SelectorResult:
    """选型模块完整输出"""
    candidates:   list         # List[SelectionCandidate]，按 score 降序
    best:         object       # candidates[0]（最优推荐），无合格机型时为 None
    screened_out: int          # 初筛淘汰的型号数量
    rejected_eff: int          # 效率不达标的型号数量
    steps:        list = field(default_factory=list)


# ─────────────────────────────────────────────
# 步骤一：按范围初筛
# ─────────────────────────────────────────────

def screen_by_range(
    fans:       list,
    Q_hard:     float,
    H_hard:     float,
    N_motor_est: float,
    power_margin_lo: float = 0.5,   # 电机功率下限：估算值的 50%
    power_margin_hi: float = 3.0,   # 电机功率上限：估算值的 300%
) -> tuple[list, list, list]:
    """
    步骤一：快速范围过滤，排除明显不适合的型号。

    筛选条件（全部必须满足）：
        ① Q_max  ≥ Q_hard        风机最大额定风量 ≥ 困难时期需求
        ② H_max  ≥ H_hard        风机最大额定风压 ≥ 困难时期需求
        ③ Q_min  ≤ Q_hard        困难时期工作点在风机覆盖范围内
        ④ 总装机功率在合理范围内  避免功率严重不匹配

    参数：
        fans         : List[FanModel] 待筛数据库
        Q_hard       : 困难时期风量需求 (m³/s)
        H_hard       : 困难时期风压需求 (Pa)
        N_motor_est  : 估算电机功率 (kW)（来自 DualPeriodResult.hard.N_motor）
        power_margin_lo / hi : 可接受功率范围比例

    返回：(passed_fans, rejected_fans, steps)
    """
    steps = [
        f"[初筛条件] Q_hard={Q_hard:.2f} m³/s，H_hard={H_hard:.1f} Pa",
        f"[初筛条件] 估算装机功率={N_motor_est:.1f} kW，"
        f"接受范围 {N_motor_est*power_margin_lo:.0f}~{N_motor_est*power_margin_hi:.0f} kW",
        f"[初筛条件] 待筛型号共 {len(fans)} 款",
        "",
    ]

    passed, rejected = [], []
    for fan in fans:
        reasons = []

        if fan.Q_max < Q_hard:
            reasons.append(f"Q_max={fan.Q_max} < 需求 {Q_hard:.1f} m³/s")
        if fan.H_max < H_hard:
            reasons.append(f"H_max={fan.H_max} < 需求 {H_hard:.1f} Pa")
        if fan.Q_min > Q_hard:
            reasons.append(f"Q_min={fan.Q_min} > 需求 {Q_hard:.1f} m³/s（工作点超出范围）")

        N_total = fan.total_motor_kw
        if N_total < N_motor_est * power_margin_lo:
            reasons.append(f"装机 {N_total:.0f} kW 偏小（< 估算 {N_motor_est:.0f} 的 {power_margin_lo*100:.0f}%）")
        if N_total > N_motor_est * power_margin_hi:
            reasons.append(f"装机 {N_total:.0f} kW 偏大（> 估算 {N_motor_est:.0f} 的 {power_margin_hi*100:.0f}%）")

        if reasons:
            steps.append(f"  ❌ {fan.model_id}：{'；'.join(reasons)}")
            rejected.append(fan)
        else:
            steps.append(
                f"  ✅ {fan.model_id}：Q {fan.Q_min}~{fan.Q_max} m³/s，"
                f"H {fan.H_min}~{fan.H_max} Pa，"
                f"装机 {N_total:.0f} kW"
            )
            passed.append(fan)

    steps.append("")
    steps.append(f"[初筛结果] 通过 {len(passed)} 款，淘汰 {len(rejected)} 款")
    return passed, rejected, steps


# ─────────────────────────────────────────────
# 步骤二：求工作点
# ─────────────────────────────────────────────

def solve_working_point(
    fi:      FanInterpolated,
    R:       float,
) -> tuple[Optional[float], Optional[float], list]:
    """
    步骤二：在额定转速 H-Q 曲线上，求与管道特性曲线 H = R·Q² 的交点。

    方法：令 f(Q) = H_fan(Q) - R·Q²，用 brentq 二分法求零点。

    物理意义：
        对于变频风机，此交点 (u, H_fan(u)) 是"基准点"——
        通过调节转速 n = n_0 × Q_required / u，
        可精确在系统阻力 R 下交付所需风量（相似律保证效率不变）。

    返回：(Q_op, H_op, steps)
        Q_op = None 表示当前管道阻力曲线与风机曲线无交点（风机不适用）
    """
    from scipy.optimize import brentq

    steps = []
    Q_lo, Q_hi = fi.Q_min, fi.Q_max

    def f(Q):
        h_fan = float(fi.H_spline(Q))
        h_pipe = R * Q**2
        return h_fan - h_pipe

    f_lo = f(Q_lo)
    f_hi = f(Q_hi)

    steps.append(
        f"    求根：f(Q_min={Q_lo:.2f})={f_lo:.1f}，"
        f"f(Q_max={Q_hi:.2f})={f_hi:.1f}"
    )

    # 检查区间内是否有根（同号 → 无交点）
    if f_lo * f_hi > 0:
        steps.append("    ❌ 无交点：管道阻力曲线与风机 H-Q 曲线不相交（风机不适用）")
        return None, None, steps

    try:
        Q_op = brentq(f, Q_lo, Q_hi, xtol=1e-4, rtol=1e-6)
        H_op = R * Q_op**2
        steps.append(f"    ✅ 交点 Q_op={Q_op:.3f} m³/s，H_op={H_op:.1f} Pa")
        return Q_op, H_op, steps
    except ValueError as e:
        steps.append(f"    ❌ brentq 求根失败：{e}")
        return None, None, steps


# ─────────────────────────────────────────────
# 步骤三：效率验证
# ─────────────────────────────────────────────

def verify_efficiency(
    fi:           FanInterpolated,
    u_easy:       float,          # 额定曲线基准点（用于 η 查取）
    H_op_easy:    float,          # 容易时期实际工作点压头（用于功率计算）
    u_hard:       float,          # 额定曲线基准点（用于 η 查取）
    H_op_hard:    float,          # 困难时期实际工作点压头（用于功率计算）
    Q_easy:       float = None,   # 容易时期实际工作点流量（用于功率，默认等于 u_easy）
    Q_hard:       float = None,   # 困难时期实际工作点流量（用于功率，默认等于 u_hard）
    eta_min:      float = 0.70,
    eta_t:        float = 0.95,
    K_N:          float = 1.15,   # 矿用主通风机持续工况，GB 50215 取 1.10~1.20，大型取 1.15
) -> tuple[float, float, float, float, float, bool, list]:
    """
    步骤三：在两个工作点处查取效率，验证是否达到最低要求。

    参数说明（VFD 与定速风机均适用）：
        u_easy / u_hard  ：额定曲线上的基准点 Q 值，用于查取效率
                           · VFD 风机：u = brentq 所得交点（相似律保证与实际工作点效率相同）
                           · 定速风机：u = 额定速下的实际工作点，与 Q_easy/Q_hard 相等
        H_op_easy/hard   ：实际工作点压头（VFD 情况下 = 设计目标 H）
        Q_easy / Q_hard  ：实际工作点流量（VFD 情况下 = 设计目标 Q；未指定则用 u 代替）

    轴功率公式：N_shaft = Q_actual × H_actual / (1000 × η × η_t)

    返回：(eta_easy, eta_hard, N_shaft_easy, N_shaft_hard, N_motor_req, passed, steps)
    """
    # 功率计算用实际工作点流量；若未指定则退化为基准点（定速风机情况）
    Q_pwr_easy = Q_easy if Q_easy is not None else u_easy
    Q_pwr_hard = Q_hard if Q_hard is not None else u_hard

    steps = []

    # 从额定曲线基准点读取效率
    eta_easy = float(fi.eta_spline(u_easy))
    eta_hard = float(fi.eta_spline(u_hard))
    eta_easy = max(0.01, min(1.0, eta_easy))
    eta_hard = max(0.01, min(1.0, eta_hard))

    # 轴功率（用实际工作点 Q 和 H）
    N_shaft_easy = Q_pwr_easy * H_op_easy / (1000.0 * eta_easy * eta_t)
    N_shaft_hard = Q_pwr_hard * H_op_hard / (1000.0 * eta_hard * eta_t)
    N_motor_req  = max(N_shaft_easy, N_shaft_hard) * K_N

    passed = (eta_easy >= eta_min) and (eta_hard >= eta_min)

    steps.append(
        f"    容易时期：η={eta_easy:.4f}  "
        + ("✅" if eta_easy >= eta_min else f"❌（< {eta_min}）")
        + f"  N_轴={N_shaft_easy:.1f} kW  （Q={Q_pwr_easy:.2f} m³/s，H={H_op_easy:.1f} Pa）"
    )
    steps.append(
        f"    困难时期：η={eta_hard:.4f}  "
        + ("✅" if eta_hard >= eta_min else f"❌（< {eta_min}）")
        + f"  N_轴={N_shaft_hard:.1f} kW  （Q={Q_pwr_hard:.2f} m³/s，H={H_op_hard:.1f} Pa）"
    )
    steps.append(
        f"    建议电机功率 ≥ {N_motor_req:.1f} kW  "
        + ("✅" if fi.fan.total_motor_kw >= N_motor_req else
           f"❌（装机 {fi.fan.total_motor_kw:.0f} kW 不足）")
    )
    return eta_easy, eta_hard, N_shaft_easy, N_shaft_hard, N_motor_req, passed, steps


# ─────────────────────────────────────────────
# 步骤四：变频转速比验证
# ─────────────────────────────────────────────

def verify_vfd_speed(
    fan:        FanModel,
    n_easy_rpm: float,
    n_hard_rpm: float,
    H_op_easy:  float,
    H_op_hard:  float,
) -> tuple[float, float, list]:
    """
    步骤四：校验变频转速是否在允许范围内，并计算相似律误差 ε。

    输入的 n_easy_rpm / n_hard_rpm 由 evaluate_one 根据基准点计算得出：
        n_period = n_rated × Q_required / u_base

    相似律误差（两目标工作点是否接近同一相似抛物线）：
        n_ratio = n_hard / n_easy
        ε = |H_op_hard/H_op_easy - n_ratio²| / n_ratio² × 100%

    ε 越小说明两个工况的效率越接近（变频节能效果越好）。

    返回：(n_ratio, vfd_error_pct, steps)
    """
    steps = []

    n_ratio   = n_hard_rpm / n_easy_rpm
    n_min_allowed, n_max_allowed = fan.rpm_range

    rpm_easy_ok = fan.rpm_range[0] <= n_easy_rpm <= fan.rpm_range[1]
    rpm_hard_ok = fan.rpm_range[0] <= n_hard_rpm <= fan.rpm_range[1]

    # 相似律误差（使用两个目标工作点的实际 H 值）
    H_ratio_actual  = H_op_hard / H_op_easy
    H_ratio_predict = n_ratio ** 2
    vfd_error = abs(H_ratio_actual - H_ratio_predict) / H_ratio_predict * 100.0

    if vfd_error < 3.0:
        vfd_grade = "✅ 优秀"
    elif vfd_error < 8.0:
        vfd_grade = "⚠️ 良好"
    else:
        vfd_grade = "❌ 较差，建议复核"

    steps.append(
        f"    转速比 n_hard/n_easy = {n_hard_rpm:.1f}/{n_easy_rpm:.1f} = {n_ratio:.4f}"
    )
    steps.append(
        f"    容易时期 {n_easy_rpm:.1f} r/min  "
        + ("✅" if rpm_easy_ok else f"❌（不在允许范围 {n_min_allowed}~{n_max_allowed}）")
    )
    steps.append(
        f"    困难时期 {n_hard_rpm:.1f} r/min  "
        + ("✅" if rpm_hard_ok else f"❌（不在允许范围 {n_min_allowed}~{n_max_allowed}）")
    )
    steps.append(f"    相似律误差 ε={vfd_error:.2f}%  {vfd_grade}")

    return n_ratio, vfd_error, steps


# ─────────────────────────────────────────────
# 步骤五：综合评分
# ─────────────────────────────────────────────

def score_candidate(
    eta_easy:      float,
    eta_hard:      float,
    vfd_error:     float,
    N_shaft_hard:  float,
    N_motor_inst:  float,
    eta_ref:       float = 0.85,    # 参考最高效率（归一化基准）
    vfd_error_max: float = 8.0,     # 误差满分上限 (%)
) -> tuple[float, list]:
    """
    步骤五：对通过全部验证的候选机型打分（0~1），用于排序推荐。

    评分公式（来自 selector_plan.md §4）：
        score = 0.4·η̃_hard + 0.3·η̃_easy + 0.2·VFD̃ + 0.1·motor̃

    各项：
        η̃_hard = min(η_hard / η_ref, 1.0)
        η̃_easy = min(η_easy / η_ref, 1.0)
        VFD̃    = max(0, 1 - ε / ε_max)
        motor̃  = 1 - |利用率 - 0.80| / 0.80    （利用率=N_shaft_hard/N_motor_inst，0.8为最优）
                  负值截断为 0

    返回：(score, steps)
    """
    steps = []

    s_hard  = min(eta_hard / eta_ref, 1.0)
    s_easy  = min(eta_easy / eta_ref, 1.0)
    s_vfd   = max(0.0, 1.0 - vfd_error / vfd_error_max)

    utilization = N_shaft_hard / N_motor_inst if N_motor_inst > 0 else 0
    s_motor = max(0.0, 1.0 - abs(utilization - 0.80) / 0.80)

    score = 0.4 * s_hard + 0.3 * s_easy + 0.2 * s_vfd + 0.1 * s_motor

    steps.append(
        f"    η̃_hard={s_hard:.3f}(×0.4)  η̃_easy={s_easy:.3f}(×0.3)  "
        f"VFD̃={s_vfd:.3f}(×0.2)  motor̃={s_motor:.3f}(×0.1)"
    )
    steps.append(f"    综合评分 = {score:.4f}")

    return score, steps


# ─────────────────────────────────────────────
# 单台风机完整评估
# ─────────────────────────────────────────────

def evaluate_one(
    fan:       FanModel,
    R_easy:    float,
    Q_easy:    float,
    H_easy:    float,
    R_hard:    float,
    Q_hard:    float,
    H_hard:    float,
    eta_min:   float = 0.70,
    eta_t:     float = 0.95,
) -> tuple[Optional[SelectionCandidate], list]:
    """
    对单台风机依次执行步骤②③④⑤，返回 (SelectionCandidate | None, steps)。
    返回 None 表示该机型不满足要求（被淘汰）。

    变频工作点逻辑（核心）：
        ① 在额定转速曲线上，用 brentq 求解 H_fan(u) = R·u² 得到"基准点" u
        ② 变频转速 n = n_rated × Q_required / u，使风机精确交付目标工作点
        ③ 相似律保证：η 在基准点 u 与目标工作点处完全相同
        ④ 因此：效率从 η_fan(u) 读取，实际工作点坐标用设计目标值 (Q_req, H_req)
    """
    all_steps = [f"── {fan.model_id}（{fan.motor_count}×{fan.motor_kw}kW，n={fan.rated_rpm}rpm）──"]

    # 构建插值对象
    fi = build_fan_interpolated(fan)

    # ── 步骤②：求额定曲线上的基准点 u ─────────────────
    all_steps.append("  [步骤②] 基准点求解（H_fan(u) = R·u²）")

    u_easy, _, s = solve_working_point(fi, R_easy)
    all_steps.extend(f"  容易时期 {x}" for x in s)

    u_hard, _, s = solve_working_point(fi, R_hard)
    all_steps.extend(f"  困难时期 {x}" for x in s)

    if u_easy is None or u_hard is None:
        all_steps.append("  [结论] ❌ 淘汰：额定曲线与管道阻力线无交点（额定压头不足或过大）")
        return None, all_steps

    # 计算变频转速（将基准点缩放到目标工作点）
    n_easy_rpm = fan.rated_rpm * Q_easy / u_easy
    n_hard_rpm = fan.rated_rpm * Q_hard / u_hard

    all_steps.append(
        f"  [步骤②] 容易时期：基准点 u={u_easy:.3f} m³/s → "
        f"变频转速 {fan.rated_rpm}×{Q_easy:.1f}/{u_easy:.3f} = {n_easy_rpm:.1f} r/min"
    )
    all_steps.append(
        f"  [步骤②] 困难时期：基准点 u={u_hard:.3f} m³/s → "
        f"变频转速 {fan.rated_rpm}×{Q_hard:.1f}/{u_hard:.3f} = {n_hard_rpm:.1f} r/min"
    )

    # 检查转速是否在允许范围
    n_min, n_max = fan.rpm_range
    if not (n_min <= n_easy_rpm <= n_max):
        all_steps.append(
            f"  [结论] ❌ 淘汰：容易时期转速 {n_easy_rpm:.1f} r/min 不在允许范围 {n_min}~{n_max}"
        )
        return None, all_steps
    if not (n_min <= n_hard_rpm <= n_max):
        all_steps.append(
            f"  [结论] ❌ 淘汰：困难时期转速 {n_hard_rpm:.1f} r/min 不在允许范围 {n_min}~{n_max}"
        )
        return None, all_steps

    # 实际工作点 = 设计目标（VFD 精确控制保证）
    Q_op_easy, H_op_easy = Q_easy, H_easy
    Q_op_hard, H_op_hard = Q_hard, H_hard

    # ── 步骤③：效率验证 ────────────────────────────────
    # 效率从基准点读取（相似律：与目标工作点效率完全相同）
    all_steps.append("  [步骤③] 效率验证（效率在基准点 u 处读取，相似律保证等效）")
    (eta_easy, eta_hard, N_shaft_easy, N_shaft_hard,
     N_motor_req, eff_ok, s) = verify_efficiency(
        fi,
        u_easy, H_op_easy,          # 基准点 u 查效率，目标 H 算功率
        u_hard, H_op_hard,
        Q_easy=Q_easy,              # 实际工作流量（VFD 精确交付）
        Q_hard=Q_hard,
        eta_min=eta_min, eta_t=eta_t
    )
    all_steps.extend(f"  {x}" for x in s)

    if not eff_ok:
        all_steps.append(f"  [结论] ❌ 淘汰：效率不达标（η_min={eta_min}）")
        return None, all_steps

    # 电机功率校验
    if fan.total_motor_kw < N_motor_req:
        all_steps.append(
            f"  [结论] ❌ 淘汰：装机功率 {fan.total_motor_kw:.0f} kW < 需求 {N_motor_req:.1f} kW"
        )
        return None, all_steps

    # ── 步骤④：变频转速比验证 ──────────────────────────
    all_steps.append("  [步骤④] 变频转速比")
    n_ratio, vfd_error, s = verify_vfd_speed(
        fan, n_easy_rpm, n_hard_rpm, H_op_easy, H_op_hard
    )
    all_steps.extend(f"  {x}" for x in s)

    # ── 步骤⑤：综合评分 ────────────────────────────────
    all_steps.append("  [步骤⑤] 综合评分")
    score, s = score_candidate(
        eta_easy, eta_hard, vfd_error,
        N_shaft_hard, fan.total_motor_kw
    )
    all_steps.extend(f"  {x}" for x in s)

    all_steps.append(f"  [结论] ✅ 通过，综合评分 = {score:.4f}")

    candidate = SelectionCandidate(
        fan=fan,
        Q_op_easy=Q_op_easy,   H_op_easy=H_op_easy,
        Q_op_hard=Q_op_hard,   H_op_hard=H_op_hard,
        eta_easy=eta_easy,     eta_hard=eta_hard,
        N_shaft_easy=N_shaft_easy, N_shaft_hard=N_shaft_hard,
        N_motor_req=N_motor_req,
        n_ratio=n_ratio,       n_easy_rpm=n_easy_rpm,
        n_hard_rpm=n_hard_rpm, vfd_error=vfd_error,
        score=score,
        steps=all_steps,
    )
    return candidate, all_steps


# ─────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────

def run_selector(
    dual_result,                         # DualPeriodResult
    eta_min:   float = 0.70,
    eta_t:     float = 0.95,
    fans:      Optional[list] = None,    # List[FanModel]，默认使用内置库
    power_margin_lo: float = 0.5,
    power_margin_hi: float = 3.0,
) -> SelectorResult:
    """
    矿井风机选型统一入口：编排全部 5 个步骤，返回 SelectorResult。

    参数：
        dual_result      : 双工况引擎输出（DualPeriodResult）
        eta_min          : 最低全压效率要求，默认 0.70
        eta_t            : 机械传动效率，默认 0.95
        fans             : 待筛风机列表，默认使用 fan_db.BUILTIN_DB
        power_margin_lo/hi : 初筛功率范围比例（相对估算值）

    使用示例：
        from calc_engine_range import DualPeriodParams, run_dual
        from calc_engine import SelectionCoeffs
        from selector import run_selector

        dp = DualPeriodParams(
            Q1_easy=114.5*60, h_f_easy=2276.0,
            Q1_hard=129.5*60, h_f_hard=3050.2,
        )
        dual = run_dual(dp, SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0))
        result = run_selector(dual)
        print(result.best.fan.model_id)
    """
    if fans is None:
        fans = FAN_DB

    # 从 DualPeriodResult 提取关键参数
    Q_easy = dual_result.easy.Q_f_ms
    H_easy = dual_result.easy.H_f
    R_easy = dual_result.easy.R

    Q_hard = dual_result.hard.Q_f_ms
    H_hard = dual_result.hard.H_f
    R_hard = dual_result.hard.R

    N_motor_est = dual_result.hard.N_motor   # 按困难时期估算

    all_steps = [
        "=" * 60,
        "矿井风机选型计算",
        "=" * 60,
        f"容易时期设计工作点：Q={Q_easy:.3f} m³/s，H={H_easy:.1f} Pa，R={R_easy:.6f}",
        f"困难时期设计工作点：Q={Q_hard:.3f} m³/s，H={H_hard:.1f} Pa，R={R_hard:.6f}",
        f"最低效率要求：η_min={eta_min}，传动效率 η_t={eta_t}",
        f"待选机型：{len(fans)} 款",
        "",
    ]

    # ── 步骤①：范围初筛 ────────────────────────────────
    all_steps.append("[步骤①] 按范围初筛")
    passed_fans, rejected_fans, s = screen_by_range(
        fans, Q_hard, H_hard, N_motor_est,
        power_margin_lo, power_margin_hi
    )
    all_steps.extend(s)
    all_steps.append("")

    screened_out = len(rejected_fans)

    # ── 步骤②③④⑤：逐台精细评估 ────────────────────────
    all_steps.append(f"[步骤②～⑤] 对 {len(passed_fans)} 款机型进行精细评估")
    all_steps.append("")

    candidates = []
    rejected_eff_count = 0

    for fan in passed_fans:
        cand, s = evaluate_one(
            fan,
            R_easy, Q_easy, H_easy,
            R_hard, Q_hard, H_hard,
            eta_min=eta_min, eta_t=eta_t,
        )
        all_steps.extend(s)
        all_steps.append("")

        if cand is not None:
            candidates.append(cand)
        else:
            rejected_eff_count += 1

    # ── 按综合评分排序 ──────────────────────────────────
    candidates.sort(key=lambda c: c.score, reverse=True)

    best = candidates[0] if candidates else None

    all_steps.append("=" * 60)
    all_steps.append(
        f"[汇总] 数据库 {len(fans)} 款  →  "
        f"初筛通过 {len(passed_fans)} 款  →  "
        f"效率/功率淘汰 {rejected_eff_count} 款  →  "
        f"最终候选 {len(candidates)} 款"
    )

    return SelectorResult(
        candidates=candidates,
        best=best,
        screened_out=screened_out,
        rejected_eff=rejected_eff_count,
        steps=all_steps,
    )


# ─────────────────────────────────────────────
# 格式化输出工具
# ─────────────────────────────────────────────

def print_selector_result(result: SelectorResult) -> None:
    """将选型结果打印为可读报告（供命令行调试）"""
    for step in result.steps:
        print(step)

    if not result.candidates:
        print("\n⚠️  没有符合要求的机型，请检查输入参数或扩大数据库。")
        return

    print()
    print("=" * 72)
    print(f"{'排名':<4} {'型号':<22} {'η_easy':>7} {'η_hard':>7} "
          f"{'N_motor':>9} {'转速比':>7} {'ε(%)':>7} {'评分':>7}")
    print("-" * 72)

    for rank, c in enumerate(result.candidates, start=1):
        print(
            f"{rank:<4} {c.fan.model_id:<22} "
            f"{c.eta_easy:>7.2%} {c.eta_hard:>7.2%} "
            f"{c.fan.total_motor_kw:>7.0f}kW "
            f"{c.n_ratio:>7.4f} "
            f"{c.vfd_error:>7.2f} "
            f"{c.score:>7.4f}"
        )

    print("=" * 72)

    best = result.best
    print(f"\n★ 推荐机型：{best.fan.model_id}")
    print(f"   系列：{best.fan.series}  类型：{best.fan.fan_type}")
    print(f"   装机：{best.fan.motor_count}×{best.fan.motor_kw} kW = "
          f"{best.fan.total_motor_kw:.0f} kW（双电机对旋）")
    print(f"   困难时期：转速 {best.n_hard_rpm:.0f} r/min，"
          f"Q={best.Q_op_hard:.2f} m³/s，H={best.H_op_hard:.1f} Pa，"
          f"η={best.eta_hard:.2%}，轴功率 {best.N_shaft_hard:.1f} kW")
    print(f"   容易时期：转速 {best.n_easy_rpm:.1f} r/min（降速比 {1/best.n_ratio:.4f}），"
          f"Q={best.Q_op_easy:.2f} m³/s，H={best.H_op_easy:.1f} Pa，"
          f"η={best.eta_easy:.2%}，轴功率 {best.N_shaft_easy:.1f} kW")
    print(f"   转速比（困难/容易）：{best.n_ratio:.4f}")
    print(f"   相似律误差 ε：{best.vfd_error:.2f}%")
    print(f"   建议最小电机功率：{best.N_motor_req:.1f} kW")
    print(f"   综合评分：{best.score:.4f}")


# ─────────────────────────────────────────────
# 命令行验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # 导入双工况引擎
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from calc_engine_range import DualPeriodParams, run_dual
    from calc_engine import SelectionCoeffs

    # ── 论文铁矿实测数据 ──────────────────────────────
    dp = DualPeriodParams(
        Q1_easy  = 114.5 * 60,   # m³/s → m³/min
        h_f_easy = 2276.0,
        Q1_hard  = 129.5 * 60,
        h_f_hard = 3050.2,
    )
    dual = run_dual(dp, SelectionCoeffs(K=1.0, K_Q=1.0, K_H=1.0))

    print("双工况引擎输出：")
    print(f"  容易时期：Q={dual.easy.Q_f_ms:.3f} m³/s，H={dual.easy.H_f:.1f} Pa，R={dual.easy.R:.6f}")
    print(f"  困难时期：Q={dual.hard.Q_f_ms:.3f} m³/s，H={dual.hard.H_f:.1f} Pa，R={dual.hard.R:.6f}")
    print()

    # ── 执行选型 ─────────────────────────────────────
    result = run_selector(dual, eta_min=0.70)

    print_selector_result(result)
