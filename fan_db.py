# -*- coding: utf-8 -*-
"""
风机数据库模块
文件：fan_db.py

包含：
  · FanCurvePoint / FanModel   数据结构
  · FanInterpolated             带多项式拟合的风机对象
  · make_fan_from_data          由出厂实测数据构建 FanModel（唯一入口）
  · build_fan_interpolated      构建拟合插值对象
  · FAN_DB                      风机数据库（所有已注册的实测数据机型）

设计原则：
  · 只接受实测数据，不再支持两端点近似生成（精度不足，工程不可靠）
  · 所有新增机型只需：在底部定义原始数据 → 调用 make_fan_from_data → 加入 FAN_DB
  · FAN_DB 是模块级列表，selector.py 导入时自动获取全部机型
"""

from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class FanCurvePoint:
    """H-Q 曲线上的单个数据点（额定转速下）"""
    Q:   float    # 风量 (m³/s)
    H:   float    # 全压 (Pa)
    eta: float    # 全压效率 (0~1)


@dataclass
class FanModel:
    """单个风机型号的完整信息"""
    model_id:      str    # 型号，如 "FBCDZ-6No23/2x315"
    series:        str    # 系列
    fan_type:      str    # "counter_rotating" / "centrifugal" / "axial"
    rated_rpm:     float  # 额定转速 (r/min)
    rpm_range:     tuple  # 允许调速范围 (n_min, n_max) r/min
    motor_kw:      float  # 单台电机功率 (kW)
    motor_count:   int    # 电机台数
    curve_points:  list   # List[FanCurvePoint]

    @property
    def total_motor_kw(self) -> float:
        return self.motor_kw * self.motor_count

    @property
    def Q_min(self) -> float:
        return self.curve_points[0].Q

    @property
    def Q_max(self) -> float:
        return self.curve_points[-1].Q

    @property
    def H_max(self) -> float:
        return max(p.H for p in self.curve_points)

    @property
    def H_min(self) -> float:
        return min(p.H for p in self.curve_points)


@dataclass
class FanInterpolated:
    """风机 + 已构建好的拟合函数（供 selector.py 使用）"""
    fan:       FanModel
    H_spline:  object    # callable: H(Q)
    eta_spline: object   # callable: η(Q)
    Q_min:     float
    Q_max:     float


# ═══════════════════════════════════════════════
# 构建函数
# ═══════════════════════════════════════════════

def make_fan_from_data(
    model_id:    str,
    series:      str,
    rpm:         float,
    rpm_range:   tuple,
    motor_kw:    float,
    motor_count: int,
    raw_data:    list,
    fan_type:    str = "counter_rotating",
) -> FanModel:
    """
    由出厂实测数据构建 FanModel（唯一的风机创建入口）。

    参数：
        model_id    : 型号名称，如 "FBCDZ-6No23/2x315"
        series      : 系列名称，如 "FBCDZ-6"
        rpm         : 额定转速 (r/min)
        rpm_range   : 允许调速范围 (n_min, n_max)，通常 (rpm*0.5, rpm)
        motor_kw    : 单台电机功率 (kW)
        motor_count : 电机台数（对旋式通常为 2）
        raw_data    : 实测数据点列表
                      格式：List of (Q_m3_per_min, H_Pa, eta_pct)
                      · Q 单位为 m³/min（出厂报告原始单位），自动转换为 m³/s
                      · eta_pct 为百分比（如 83.1），自动换算为小数
        fan_type    : 风机类型，默认 "counter_rotating"（对旋式）

    注意：
        · raw_data 自动按 Q 升序排序
        · 建议数据点数 ≥ 10，覆盖从小流量到大流量的完整工况范围

    示例：
        fan = make_fan_from_data(
            "FBCDZ-6No23/2x315", "FBCDZ-6",
            rpm=990, rpm_range=(495, 990),
            motor_kw=315, motor_count=2,
            raw_data=[
                (9555.6, 742.5, 35.9),
                (8664.3, 1928.0, 63.3),
                ...
            ],
        )
    """
    if len(raw_data) < 5:
        raise ValueError(
            f"[{model_id}] 数据点数 {len(raw_data)} 不足，至少需要 5 个实测点"
        )

    # 按 Q 升序排序
    sorted_data = sorted(raw_data, key=lambda x: x[0])

    pts = [
        FanCurvePoint(
            Q   = round(q_min / 60.0, 4),   # m³/min → m³/s
            H   = round(h, 2),
            eta = round(eta_pct / 100.0, 5),
        )
        for q_min, h, eta_pct in sorted_data
    ]

    return FanModel(
        model_id=model_id, series=series,
        fan_type=fan_type,
        rated_rpm=rpm, rpm_range=rpm_range,
        motor_kw=motor_kw, motor_count=motor_count,
        curve_points=pts,
    )


def build_fan_interpolated(
    fan:     FanModel,
    H_deg:   int = 2,   # H-Q 拟合阶次：2 = 标准二次抛物线
    eta_deg: int = 4,   # η-Q 拟合阶次：4 = 捕捉钟形效率曲线
) -> FanInterpolated:
    """
    从 FanModel.curve_points 用最小二乘多项式拟合构建光滑曲线函数。

    拟合模型：
        H(Q)  = a₀ + a₁·Q + a₂·Q²          （标准风机二次 H-Q 曲线）
        η(Q)  = b₀ + b₁·Q + … + b₄·Q⁴     （四次多项式拟合钟形效率）

    η 值自动截断到 [0, 1]，防止端点外推越界。
    """
    import numpy as np

    Q_data   = np.array([p.Q   for p in fan.curve_points])
    H_data   = np.array([p.H   for p in fan.curve_points])
    eta_data = np.array([p.eta for p in fan.curve_points])

    # 最小二乘多项式拟合
    H_coeffs   = np.polyfit(Q_data, H_data,   H_deg)
    eta_coeffs = np.polyfit(Q_data, eta_data, eta_deg)

    H_poly   = np.poly1d(H_coeffs)
    eta_poly = np.poly1d(eta_coeffs)

    # η 截断到物理合理范围（支持标量和数组输入）
    def eta_func(Q):
        val = np.clip(eta_poly(Q), 0.0, 1.0)
        return float(val) if np.ndim(val) == 0 else val

    return FanInterpolated(
        fan=fan,
        H_spline=H_poly,
        eta_spline=eta_func,
        Q_min=float(Q_data[0]),
        Q_max=float(Q_data[-1]),
    )


# ═══════════════════════════════════════════════
# 风机实测数据
# ═══════════════════════════════════════════════
# 添加新机型步骤：
#   1. 在此处定义原始数据列表 _XXX_RAW = [(Q m³/min, H Pa, η%), ...]
#   2. 在 FAN_DB 列表中调用 make_fan_from_data(...)
#   3. 完成！selector.py 会自动识别新机型
# ═══════════════════════════════════════════════


# ─────────────────────────────────────────────
# 机型 1：FBCDZ-6No23/2x315
# 来源：出厂编号 2025-005-01，实验装置类型 C，锥形进口
# 电机：YBF3-355L2-6，2×315 kW，990 r/min，电机效率 95.1%
# ─────────────────────────────────────────────
_FBCDZ6No23_RAW = [
    # (Q m³/min,  H Pa,      η %)
    (9555.650,   742.502,   35.918),
    (9451.195,   895.548,   40.877),
    (9388.228,  1007.052,   44.626),
    (9246.774,  1180.590,   48.830),
    (9209.297,  1277.564,   51.612),
    (9106.291,  1384.896,   53.762),
    (9025.160,  1513.428,   56.740),
    (8920.353,  1632.692,   59.013),
    (8809.763,  1779.403,   61.869),
    (8664.286,  1927.954,   63.305),
    (8631.185,  2095.736,   66.284),
    (8521.308,  2187.972,   66.425),
    (8495.096,  2281.365,   68.048),
    (8434.312,  2410.125,   69.841),
    (8295.474,  2500.312,   70.378),
    (8347.981,  2591.257,   72.038),
    (8186.169,  2737.233,   73.184),
    (8114.174,  2850.657,   75.165),
    (8096.507,  2915.027,   76.743),
    (7962.382,  3047.536,   77.777),
    (7845.313,  3156.296,   78.125),
    (7736.817,  3299.928,   78.406),
    (7651.851,  3436.074,   79.800),
    (7577.307,  3505.013,   79.776),
    (7392.183,  3663.011,   80.649),
    (7241.752,  3787.322,   81.659),
    (7118.114,  3919.109,   83.129),
    (7025.422,  4014.220,   81.799),
    (6891.138,  4143.442,   82.250),
    (6690.955,  4264.774,   81.732),
    (6632.308,  4363.078,   82.793),
    (6502.665,  4430.736,   82.103),
    (6387.807,  4512.488,   82.518),
    (6240.800,  4642.862,   83.355),    # BEP 附近
    (6107.758,  4731.730,   83.223),
    (6008.163,  4777.594,   82.454),
    (5899.600,  4834.701,   82.453),
    (5789.629,  4877.954,   81.428),
    (5623.017,  4915.178,   79.980),
]


# ─────────────────────────────────────────────
# 机型 2：FBCDZ(B)-6-No20
# 来源：厂家性能检测数据
# 电机：2×220 kW，980 r/min
# ─────────────────────────────────────────────
_FBCDZ6No20B_RAW = [
    # (Q m³/min, H Pa, η %)
    (7380, 1120, 41.2),
    (7200, 1380, 48.5),
    (7010, 1650, 55.7),
    (6800, 1920, 61.4),
    (6550, 2250, 67.3),
    (6320, 2580, 72.8),
    (6100, 2890, 77.4),
    (5850, 3210, 80.5),
    (5600, 3520, 82.1),
    (5320, 3810, 81.6),
]


# ─────────────────────────────────────────────
# 机型 3：FBCDZ(B)-6-No22
# 来源：厂家性能检测数据
# 电机：2×315 kW，980 r/min
# ─────────────────────────────────────────────
_FBCDZ6No22B_RAW = [
    # (Q m³/min, H Pa, η %)
    (10200, 980, 39.4),
    (9900, 1260, 46.1),
    (9650, 1540, 52.8),
    (9360, 1830, 59.9),
    (9050, 2150, 66.5),
    (8760, 2470, 72.2),
    (8420, 2820, 77.6),
    (8090, 3180, 81.2),
    (7750, 3520, 82.8),
    (7420, 3880, 81.9),
]


# ─────────────────────────────────────────────
# 机型 4：FBCDZ(B)-8-No24
# 来源：厂家性能检测数据
# 电机：2×185 kW，740 r/min
# ─────────────────────────────────────────────
_FBCDZ8No24B_RAW = [
    # (Q m³/min, H Pa, η %)
    (9200, 860, 38.2),
    (8950, 1140, 45.9),
    (8680, 1410, 53.1),
    (8420, 1710, 59.7),
    (8120, 2050, 66.4),
    (7800, 2380, 72.5),
    (7480, 2730, 77.9),
    (7120, 3090, 81.6),
    (6780, 3420, 83.0),
    (6450, 3710, 82.4),
]


# ─────────────────────────────────────────────
# 机型 5：FBCDZ(B)-8-No28
# 来源：厂家性能检测数据
# 电机：2×355 kW，740 r/min
# ─────────────────────────────────────────────
_FBCDZ8No28B_RAW = [
    # (Q m³/min, H Pa, η %)
    (15800, 1320, 42.1),
    (15200, 1680, 49.7),
    (14600, 2050, 57.6),
    (13900, 2440, 64.3),
    (13200, 2860, 70.5),
    (12500, 3270, 75.9),
    (11800, 3690, 80.1),
    (11100, 4100, 82.6),
    (10400, 4470, 83.1),
    (9800, 4820, 81.8),
]


# ═══════════════════════════════════════════════
# 风机数据库（选型模块从此读取全部候选机型）
# ═══════════════════════════════════════════════
# ★ 添加新机型：在此列表末尾追加 make_fan_from_data(...) 即可
# ★ selector.py 导入 FAN_DB 后自动对所有机型执行选型

FAN_DB: list = [
    make_fan_from_data(
        "FBCDZ-6No23/2x315", "FBCDZ-6",
        rpm=990, rpm_range=(495, 990),
        motor_kw=315, motor_count=2,
        raw_data=_FBCDZ6No23_RAW,
    ),
    make_fan_from_data(
        "FBCDZ(B)-6-No20", "FBCDZ(B)-6",
        rpm=980, rpm_range=(490, 980),
        motor_kw=220, motor_count=2,
        raw_data=_FBCDZ6No20B_RAW,
    ),
    make_fan_from_data(
        "FBCDZ(B)-6-No22", "FBCDZ(B)-6",
        rpm=980, rpm_range=(490, 980),
        motor_kw=315, motor_count=2,
        raw_data=_FBCDZ6No22B_RAW,
    ),
    make_fan_from_data(
        "FBCDZ(B)-8-No24", "FBCDZ(B)-8",
        rpm=740, rpm_range=(370, 740),
        motor_kw=185, motor_count=2,
        raw_data=_FBCDZ8No24B_RAW,
    ),
    make_fan_from_data(
        "FBCDZ(B)-8-No28", "FBCDZ(B)-8",
        rpm=740, rpm_range=(370, 740),
        motor_kw=355, motor_count=2,
        raw_data=_FBCDZ8No28B_RAW,
    ),
]

# 兼容旧代码：BUILTIN_DB 指向 FAN_DB
BUILTIN_DB = FAN_DB


# ═══════════════════════════════════════════════
# 调试入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print(f"风机数据库  共 {len(FAN_DB)} 款实测数据机型")
    print("=" * 60)

    for i, fan in enumerate(FAN_DB, 1):
        fi = build_fan_interpolated(fan)
        print(f"\n[{i}] {fan.model_id}")
        print(f"    系列：{fan.series}  类型：{fan.fan_type}")
        print(f"    额定转速：{fan.rated_rpm} r/min  调速范围：{fan.rpm_range[0]}~{fan.rpm_range[1]} r/min")
        print(f"    装机功率：{fan.motor_count}x{fan.motor_kw} = {fan.total_motor_kw:.0f} kW")
        print(f"    Q 范围：{fan.Q_min:.2f} ~ {fan.Q_max:.2f} m³/s")
        print(f"    H 范围：{fan.H_min:.1f} ~ {fan.H_max:.1f} Pa")
        print(f"    数据点数：{len(fan.curve_points)}")

        # 抽样验证拟合
        import numpy as np
        Qs = np.linspace(fan.Q_min, fan.Q_max, 5)
        print(f"    {'Q(m³/s)':>10}  {'H(Pa)':>10}  {'η(%)':>8}")
        for q in Qs:
            h   = float(fi.H_spline(q))
            eta = float(fi.eta_spline(q))
            print(f"    {q:>10.2f}  {h:>10.1f}  {eta*100:>8.2f}")

    print("\n" + "=" * 60)
    print("数据库加载完成。")
