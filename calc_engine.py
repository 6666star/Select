

from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class MineParams:
    """矿井通风基础参数"""
    Q1: float               # 各采掘工作面总需风量 (m³/min)
    h_f: float              # 通风网路总阻力 (Pa)
    Q2: float = 0.0         # 其他用风地点风量 (m³/min)，默认 0
    h_vc: float = 0.0       # 出口速度压力损失 (Pa)，默认 0
    h_e: float = 0.0        # 附加损失 (Pa)，默认 0
    altitude_m: float = 0.0 # 矿井海拔高度 (m)，用于空气密度修正

@dataclass
class SelectionCoeffs:
    """选型系数（可按矿井实际情况调整）"""
    K: float = 1.20         # 需风量备用系数（采掘面）
    K_Q: float = 1.15       # 通风机工作风量系数
    K_H: float = 1.10       # 通风机工作风压系数
    K_N: float = 1.20       # 电机功率裕量系数
    eta_f: float = 0.75     # 通风机全压效率（估算用，实际由性能曲线确定）
    eta_t: float = 0.95     # 传动效率（直联取 1.0，皮带取 0.95）

@dataclass
class EngineResult:
    """计算引擎输出结果"""
    # 需风量计算
    Q_total: float = 0.0        # 矿井总需风量 (m³/min)
    Q_f_min: float = 0.0        # 通风机最小工作风量 (m³/min)
    Q_f_ms: float = 0.0         # 通风机工作风量 (m³/s)

    # 风压计算
    H_total: float = 0.0        # 通风网路总压力 (Pa)
    H_f: float = 0.0            # 通风机工作风压 (Pa)

    # 阻力系数
    R: float = 0.0              # 通风网路阻力系数 (N·s²/m⁸)

    # 功率计算
    N_shaft: float = 0.0        # 估算轴功率 (kW)
    N_motor: float = 0.0        # 配套电机最小功率 (kW)

    # 空气密度修正系数
    rho_ratio: float = 1.0      # 实际密度 / 标准密度

    # 调试信息
    steps: list = field(default_factory=list)


# ─────────────────────────────────────────────
# 核心计算函数
# ─────────────────────────────────────────────

def air_density_ratio(altitude_m: float) -> float:
    """
    海拔高度对空气密度的修正系数（国际标准大气近似）
    每升高 1000m，密度降低约 11%
    """
    return math.exp(-altitude_m / 8500.0)


def calc_Q_total(params: MineParams, coeffs: SelectionCoeffs) -> tuple[float, list]:
    """
    计算矿井总需风量 (m³/min)

    公式：Q_总 = K × (Q1 + Q2)
    """
    steps = []
    Q_total = coeffs.K * (params.Q1 + params.Q2)
    steps.append(
        f"[风量] Q_总 = K × (Q1 + Q2) = {coeffs.K} × ({params.Q1} + {params.Q2}) "
        f"= {Q_total:.2f} m³/min"
    )
    return Q_total, steps


def calc_Q_fan(Q_total: float, coeffs: SelectionCoeffs) -> tuple[float, float, list]:
    """
    计算通风机工作风量

    公式：Q_f = K_Q × Q_总
    返回：(m³/min, m³/s, steps)
    """
    steps = []
    Q_f_min = coeffs.K_Q * Q_total
    Q_f_ms = Q_f_min / 60.0
    steps.append(
        f"[风量] Q_f = K_Q × Q_总 = {coeffs.K_Q} × {Q_total:.2f} "
        f"= {Q_f_min:.2f} m³/min = {Q_f_ms:.4f} m³/s"
    )
    return Q_f_min, Q_f_ms, steps


def calc_H_total(params: MineParams) -> tuple[float, list]:
    """
    计算通风网路总压力 (Pa)

    公式：H_总 = h_f + h_vc + h_e
    """
    steps = []
    H_total = params.h_f + params.h_vc + params.h_e
    steps.append(
        f"[风压] H_总 = h_f + h_vc + h_e = {params.h_f} + {params.h_vc} + {params.h_e} "
        f"= {H_total:.2f} Pa"
    )
    return H_total, steps


def calc_H_fan(H_total: float, coeffs: SelectionCoeffs, rho_ratio: float = 1.0) -> tuple[float, list]:
    """
    计算通风机工作风压 (Pa)

    公式：H_f = K_H × H_总 × ρ_ratio
    高海拔时需乘以密度修正系数（风压与密度成正比）
    """
    steps = []
    H_f = coeffs.K_H * H_total * rho_ratio
    steps.append(
        f"[风压] H_f = K_H × H_总 × ρ_ratio = {coeffs.K_H} × {H_total:.2f} × {rho_ratio:.4f} "
        f"= {H_f:.2f} Pa"
    )
    return H_f, steps


def calc_resistance(H_f: float, Q_f_ms: float) -> tuple[float, list]:
    """
    计算通风网路阻力系数 R (N·s²/m⁸)

    公式：R = H_f / Q_f²
    管道特性曲线：H = R × Q²
    """
    steps = []
    if Q_f_ms <= 0:
        return 0.0, ["[阻力] Q_f = 0，无法计算 R"]
    R = H_f / (Q_f_ms ** 2)
    steps.append(
        f"[阻力] R = H_f / Q_f² = {H_f:.2f} / {Q_f_ms:.4f}² = {R:.4f} N·s²/m⁸"
    )
    return R, steps


def calc_power(Q_f_ms: float, H_f: float, coeffs: SelectionCoeffs) -> tuple[float, float, list]:
    """
    估算轴功率与配套电机功率

    轴功率：N = Q_f × H_f / (1000 × η_f × η_t)   (kW)
    电机功率：N_motor = K_N × N
    """
    steps = []
    N_shaft = (Q_f_ms * H_f) / (1000.0 * coeffs.eta_f * coeffs.eta_t)
    N_motor = coeffs.K_N * N_shaft
    steps.append(
        f"[功率] N_轴 = Q_f × H_f / (1000 × η_f × η_t) "
        f"= {Q_f_ms:.4f} × {H_f:.2f} / (1000 × {coeffs.eta_f} × {coeffs.eta_t}) "
        f"= {N_shaft:.2f} kW"
    )
    steps.append(
        f"[功率] N_电机 ≥ K_N × N_轴 = {coeffs.K_N} × {N_shaft:.2f} = {N_motor:.2f} kW"
    )
    return N_shaft, N_motor, steps


# ─────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────

def run(params: MineParams, coeffs: Optional[SelectionCoeffs] = None) -> EngineResult:
    """
    执行完整选型计算流程，返回 EngineResult

    示例：
        params = MineParams(Q1=1250, h_f=56)
        result = run(params)
    """
    if coeffs is None:
        coeffs = SelectionCoeffs()

    result = EngineResult()
    all_steps = []

    # 1. 海拔修正
    rho_ratio = air_density_ratio(params.altitude_m)
    result.rho_ratio = rho_ratio
    if params.altitude_m > 0:
        all_steps.append(
            f"[修正] 海拔 {params.altitude_m} m，空气密度比 ρ/ρ₀ = {rho_ratio:.4f}"
        )

    # 2. 需风量
    Q_total, s = calc_Q_total(params, coeffs)
    result.Q_total = Q_total
    all_steps.extend(s)

    Q_f_min, Q_f_ms, s = calc_Q_fan(Q_total, coeffs)
    result.Q_f_min = Q_f_min
    result.Q_f_ms = Q_f_ms
    all_steps.extend(s)

    # 3. 风压
    H_total, s = calc_H_total(params)
    result.H_total = H_total
    all_steps.extend(s)

    H_f, s = calc_H_fan(H_total, coeffs, rho_ratio)
    result.H_f = H_f
    all_steps.extend(s)

    # 4. 阻力系数
    R, s = calc_resistance(H_f, Q_f_ms)
    result.R = R
    all_steps.extend(s)

    # 5. 功率
    N_shaft, N_motor, s = calc_power(Q_f_ms, H_f, coeffs)
    result.N_shaft = N_shaft
    result.N_motor = N_motor
    all_steps.extend(s)

    result.steps = all_steps
    return result


def pipeline_curve(R: float, Q_range: tuple[float, float], n: int = 100) -> tuple[list, list]:
    """
    生成管道特性曲线数据点（用于绘图）

    H = R × Q²，Q 单位 m³/s，H 单位 Pa
    """
    Q_list = [Q_range[0] + i * (Q_range[1] - Q_range[0]) / (n - 1) for i in range(n)]
    H_list = [R * q ** 2 for q in Q_list]
    return Q_list, H_list


# ─────────────────────────────────────────────
# 命令行测试（教材例题 5-2）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("矿井通风机选型计算引擎 — 验证用例（教材例题 5-2）")
    print("=" * 60)

    # ── 用例 1：教材例 5-2 基本输入 ─────────────────
    print("\n【用例 1】教材例 5-2")
    print("  输入：Q1=1250 m3/min，h_f=56 Pa，K_Q=1.15，K_H=1.10")

    params = MineParams(Q1=1250, h_f=56)
    coeffs = SelectionCoeffs(K=1.0, K_Q=1.15, K_H=1.10)
    # 注：教材简化公式中 K 已含在 Q1 内，故 K 取 1.0

    result = run(params, coeffs)

    print("\n  计算过程：")
    for step in result.steps:
        print(f"    {step}")

    print(f"\n  ── 结果汇总 ──")
    print(f"    通风机工作风量  Q_f = {result.Q_f_min:.2f} m³/min  ({result.Q_f_ms:.4f} m³/s)")
    print(f"    通风机工作风压  H_f = {result.H_f:.2f} Pa")
    print(f"    通风网路阻力系数 R  = {result.R:.4f} N·s²/m⁸")
    print(f"    估算轴功率      N   = {result.N_shaft:.2f} kW")
    print(f"    配套电机功率    N_m ≥ {result.N_motor:.2f} kW")

    # 教材参考值
    print(f"\n  ── 教材参考值对照 ──")
    print(f"    Q_f ~= 23.96 m3/s  ->  本引擎: {result.Q_f_ms:.4f} m3/s")
    print(f"    H_f ~= 61.6  Pa    ->  本引擎: {result.H_f:.2f} Pa")
    err_Q = abs(result.Q_f_ms - 23.96) / 23.96 * 100
    err_H = abs(result.H_f - 61.6) / 61.6 * 100
    print(f"    风量误差: {err_Q:.2f}%   风压误差: {err_H:.2f}%")

    # ── 用例 2：高海拔矿井 ──────────────────────────
    print("\n\n【用例 2】高海拔矿井（海拔 2000m）")
    params2 = MineParams(Q1=1250, h_f=56, altitude_m=2000)
    result2 = run(params2, coeffs)
    print("\n  计算过程：")
    for step in result2.steps:
        print(f"    {step}")
    print(f"\n  ── 结果汇总 ──")
    print(f"    空气密度比       ρ/ρ₀ = {result2.rho_ratio:.4f}")
    print(f"    通风机工作风量   Q_f  = {result2.Q_f_ms:.4f} m³/s（与平原相同）")
    print(f"    通风机工作风压   H_f  = {result2.H_f:.2f} Pa（低于平原，风压需求下降）")
    print(f"    估算轴功率       N    = {result2.N_shaft:.2f} kW")
    print(f"    配套电机功率     N_m  ≥ {result2.N_motor:.2f} kW")

    # ── 用例 3：含速度压和附加损失 ─────────────────
    print("\n\n【用例 3】含出口速度压和附加损失")
    params3 = MineParams(Q1=1250, h_f=56, h_vc=5, h_e=3)
    result3 = run(params3, coeffs)
    print(f"\n  通风机工作风压   H_f = {result3.H_f:.2f} Pa（含 h_vc=5, h_e=3）")
    print(f"  估算轴功率       N   = {result3.N_shaft:.2f} kW")

    # ── 管道特性曲线数据预览 ───────────────────────
    print("\n\n【管道特性曲线】（R = {:.4f} N·s²/m⁸）".format(result.R))
    Q_curve, H_curve = pipeline_curve(result.R, (0, result.Q_f_ms * 1.5), n=6)
    print(f"  {'Q (m³/s)':>12}  {'H (Pa)':>10}")
    for q, h in zip(Q_curve, H_curve):
        print(f"  {q:>12.3f}  {h:>10.2f}")

    print("\n" + "=" * 60)
    print("验证完成。")
