# -*- coding: utf-8 -*-
"""
矿井通风机选型系统 — Web UI
文件：app.py
运行：streamlit run app.py

基于 Streamlit 构建，调用后端模块完成全部计算。
"""

import sys
import os

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import matplotlib
matplotlib.use("Agg")                       # 非交互后端，适配 Streamlit
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np

from calc_engine import SelectionCoeffs
from calc_engine_range import DualPeriodParams, run_dual
from selector import run_selector
from fan_db import FAN_DB, build_fan_interpolated
from plot_res import plot_fan_result
from plot_curves import plot_dual_period


# ═══════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="矿井通风机选型系统",
    page_icon=":wind_blowing_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("矿井通风机选型系统")
st.caption("双工况变频风机自动选型 — 参数输入 → 计算 → 选型 → 特性曲线")


# ═══════════════════════════════════════════════
# 侧边栏：参数输入（对应页面①）
# ═══════════════════════════════════════════════

with st.sidebar:
    st.header("矿井工况参数")

    # ── 风量单位选择 ──
    unit = st.radio(
        "风量输入单位",
        ["m³/s（已含系数）", "m³/min（原始需风量）"],
        index=0,
        help="选择 m³/min 时系统自动除以 60 转换为 m³/s",
    )
    use_min = "m³/min" in unit
    q_label = "m³/min" if use_min else "m³/s"

    st.subheader("容易时期")
    col1, col2 = st.columns(2)
    with col1:
        q_easy_input = st.number_input(
            f"风量 ({q_label})", value=114.5 if not use_min else 6870.0,
            min_value=0.1, step=1.0, key="q_easy",
            help="容易时期（矿井开采初期）的需风量",
        )
    with col2:
        h_easy_input = st.number_input(
            "风压 (Pa)", value=2276.0, min_value=1.0, step=10.0, key="h_easy",
            help="容易时期通风网路总阻力",
        )

    st.subheader("困难时期")
    col3, col4 = st.columns(2)
    with col3:
        q_hard_input = st.number_input(
            f"风量 ({q_label})", value=129.5 if not use_min else 7770.0,
            min_value=0.1, step=1.0, key="q_hard",
        )
    with col4:
        h_hard_input = st.number_input(
            "风压 (Pa)", value=3050.2, min_value=1.0, step=10.0, key="h_hard",
        )

    # ── 附加参数 ──
    with st.expander("附加参数", expanded=False):
        altitude = st.number_input("海拔高度 (m)", value=0.0, min_value=0.0, step=100.0)
        h_vc = st.number_input("出口速度压损 (Pa)", value=0.0, min_value=0.0, step=1.0)
        h_e = st.number_input("附加损失 (Pa)", value=0.0, min_value=0.0, step=1.0)

    # ── 选型系数 ──
    with st.expander("选型系数", expanded=False):
        K = st.number_input("K — 需风量备用系数", value=1.00, min_value=0.5, max_value=2.0,
                            step=0.05, help="已含在风量中则填 1.0（GB 50215: 1.15~1.25）")
        K_Q = st.number_input("K_Q — 工作风量系数", value=1.00, min_value=0.5, max_value=2.0,
                              step=0.05, help="含漏风裕量（GB 50215: 1.10~1.15）")
        K_H = st.number_input("K_H — 工作风压系数", value=1.00, min_value=0.5, max_value=2.0,
                              step=0.05, help="含风压裕量（GB 50215: 1.05~1.10）")
        K_N = st.number_input("K_N — 电机功率裕量", value=1.15, min_value=1.0, max_value=1.5,
                              step=0.05, help="大型矿用主通风机取 1.10~1.20")
        eta_t = st.number_input("eta_t — 传动效率", value=0.95, min_value=0.80, max_value=1.0,
                                step=0.01, help="直联 1.0，皮带 0.95")
        eta_min = st.number_input("eta_min — 最低效率", value=0.70, min_value=0.50, max_value=0.90,
                                  step=0.01, help="全压效率要求下限")

    st.divider()

    # ── 开始计算 ──
    run_btn = st.button("开始计算并选型", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════
# 输入预处理与校验
# ═══════════════════════════════════════════════

# 统一转换为 m³/min（calc_engine 内部用 m³/min 输入）
if use_min:
    Q1_easy_min = q_easy_input
    Q1_hard_min = q_hard_input
else:
    Q1_easy_min = q_easy_input * 60.0
    Q1_hard_min = q_hard_input * 60.0


def validate_inputs():
    """校验输入参数，返回 (ok, error_msg)"""
    errors = []
    if Q1_easy_min <= 0 or Q1_hard_min <= 0:
        errors.append("风量必须为正数")
    if h_easy_input <= 0 or h_hard_input <= 0:
        errors.append("风压必须为正数")
    if Q1_hard_min < Q1_easy_min:
        errors.append("困难时期风量应 >= 容易时期风量（定义：困难时期通风需求更大）")
    if h_hard_input < h_easy_input:
        errors.append("困难时期风压应 >= 容易时期风压")
    return (len(errors) == 0, errors)


# ═══════════════════════════════════════════════
# 主计算流程（点击按钮后触发）
# ═══════════════════════════════════════════════

if run_btn:
    ok, errors = validate_inputs()
    if not ok:
        for e in errors:
            st.error(f"输入错误：{e}")
        st.stop()

    # ── 构建参数 ──
    dp = DualPeriodParams(
        Q1_easy=Q1_easy_min, h_f_easy=h_easy_input,
        Q1_hard=Q1_hard_min, h_f_hard=h_hard_input,
        Q2=0.0, h_vc=h_vc, h_e=h_e, altitude_m=altitude,
    )
    coeffs = SelectionCoeffs(
        K=K, K_Q=K_Q, K_H=K_H, K_N=K_N,
        eta_t=eta_t, eta_f=0.75,
    )

    # ── 第一步：双工况计算 ──
    with st.spinner("正在计算双工况参数..."):
        dual = run_dual(dp, coeffs)

    # ── 第二步：风机选型 ──
    with st.spinner("正在执行风机选型..."):
        result = run_selector(dual, eta_min=eta_min, eta_t=eta_t)

    # 保存到 session_state 供各 Tab 使用
    st.session_state["dual"] = dual
    st.session_state["result"] = result
    st.session_state["coeffs"] = coeffs
    st.session_state["eta_min"] = eta_min
    st.session_state["computed"] = True

    st.success("计算完成！")


# ═══════════════════════════════════════════════
# 结果展示（4 个 Tab，对应设计方案的 4 个页面）
# ═══════════════════════════════════════════════

if st.session_state.get("computed"):
    dual = st.session_state["dual"]
    result = st.session_state["result"]
    eta_min_val = st.session_state["eta_min"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "计算结果",
        "选型对比",
        "特性曲线",
        "计算过程",
    ])

    # ──────────────────────────────────────────
    # Tab 1：计算结果（对应设计方案页面②）
    # ──────────────────────────────────────────
    with tab1:
        st.subheader("双工况计算结果")

        col_e, col_h = st.columns(2)
        with col_e:
            st.markdown("**容易时期**")
            st.metric("工作风量 Q_f", f"{dual.easy.Q_f_ms:.3f} m³/s")
            st.metric("工作风压 H_f", f"{dual.easy.H_f:.1f} Pa")
            st.metric("阻力系数 R", f"{dual.easy.R:.6f}")
            st.metric("估算轴功率", f"{dual.easy.N_shaft:.1f} kW")
            st.metric("估算电机功率", f"{dual.easy.N_motor:.1f} kW")

        with col_h:
            st.markdown("**困难时期**")
            st.metric("工作风量 Q_f", f"{dual.hard.Q_f_ms:.3f} m³/s")
            st.metric("工作风压 H_f", f"{dual.hard.H_f:.1f} Pa")
            st.metric("阻力系数 R", f"{dual.hard.R:.6f}")
            st.metric("估算轴功率", f"{dual.hard.N_shaft:.1f} kW")
            st.metric("估算电机功率", f"{dual.hard.N_motor:.1f} kW")

        st.divider()
        st.subheader("管道特性曲线")

        fig_pipe = plot_dual_period(dual)
        st.pyplot(fig_pipe)
        plt.close(fig_pipe)

    # ──────────────────────────────────────────
    # Tab 2：选型对比（对应设计方案页面③）
    # ──────────────────────────────────────────
    with tab2:
        st.subheader("选型结果")

        # ── 漏斗统计 ──
        total = len(FAN_DB)
        passed_screen = total - result.screened_out
        passed_final = len(result.candidates)
        rejected_detail = result.rejected_eff

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("数据库机型", f"{total} 款")
        c2.metric("初筛通过", f"{passed_screen} 款", delta=f"-{result.screened_out} 淘汰")
        c3.metric("精细评估后", f"{passed_final} 款", delta=f"-{rejected_detail} 淘汰")
        c4.metric("最终候选", f"{passed_final} 款")

        st.divider()

        if not result.candidates:
            st.warning("没有符合要求的机型，请检查输入参数或扩大风机数据库。")
        else:
            # ── 候选排名表 ──
            st.subheader("候选机型排名")

            for rank, cand in enumerate(result.candidates, 1):
                is_best = (rank == 1)
                icon = "🏆" if is_best else f"#{rank}"

                with st.expander(
                    f"{icon}  {cand.fan.model_id}  |  "
                    f"评分 {cand.score:.4f}  |  "
                    f"η困难={cand.eta_hard:.1%}  η容易={cand.eta_easy:.1%}  |  "
                    f"装机 {cand.fan.total_motor_kw:.0f}kW  |  "
                    f"ε={cand.vfd_error:.2f}%",
                    expanded=is_best,
                ):
                    # 基本信息 + 变频参数
                    info_col, vfd_col = st.columns(2)

                    with info_col:
                        st.markdown("**基本信息**")
                        st.write(f"- 型号：{cand.fan.model_id}")
                        st.write(f"- 系列：{cand.fan.series}")
                        st.write(f"- 类型：{'对旋式' if cand.fan.fan_type == 'counter_rotating' else cand.fan.fan_type}")
                        st.write(f"- 额定转速：{cand.fan.rated_rpm} r/min")
                        st.write(f"- 装机功率：{cand.fan.motor_count}×{cand.fan.motor_kw} = {cand.fan.total_motor_kw:.0f} kW")
                        st.write(f"- 数据点数：{len(cand.fan.curve_points)}")

                    with vfd_col:
                        st.markdown("**变频参数**")
                        st.write(f"- 转速比 n困难/n容易：{cand.n_ratio:.4f}")
                        st.write(f"- 容易时期转速：{cand.n_easy_rpm:.1f} r/min")
                        st.write(f"- 困难时期转速：{cand.n_hard_rpm:.1f} r/min")
                        vfd_grade = "优秀" if cand.vfd_error < 3 else ("良好" if cand.vfd_error < 8 else "较差")
                        st.write(f"- 相似律误差 ε：{cand.vfd_error:.2f}%（{vfd_grade}）")

                    st.divider()

                    # 双工况对比
                    st.markdown("**双工况运行参数**")
                    op_data = {
                        "参数": ["风量 Q (m³/s)", "风压 H (Pa)", "效率 η",
                                 "轴功率 N (kW)", "转速 n (r/min)"],
                        "容易时期": [
                            f"{cand.Q_op_easy:.2f}",
                            f"{cand.H_op_easy:.1f}",
                            f"{cand.eta_easy:.2%}",
                            f"{cand.N_shaft_easy:.1f}",
                            f"{cand.n_easy_rpm:.1f}",
                        ],
                        "困难时期": [
                            f"{cand.Q_op_hard:.2f}",
                            f"{cand.H_op_hard:.1f}",
                            f"{cand.eta_hard:.2%}",
                            f"{cand.N_shaft_hard:.1f}",
                            f"{cand.n_hard_rpm:.1f}",
                        ],
                    }
                    st.table(op_data)

                    # 评分构成
                    st.markdown("**评分构成**")
                    eta_ref = 0.85
                    s_hard = min(cand.eta_hard / eta_ref, 1.0)
                    s_easy = min(cand.eta_easy / eta_ref, 1.0)
                    s_vfd = max(0.0, 1.0 - cand.vfd_error / 8.0)
                    util = cand.N_shaft_hard / cand.fan.total_motor_kw if cand.fan.total_motor_kw > 0 else 0
                    s_motor = max(0.0, 1.0 - abs(util - 0.80) / 0.80)

                    score_data = {
                        "分项": ["η困难时期", "η容易时期", "VFD匹配度", "电机利用率"],
                        "原始值": [f"{cand.eta_hard:.2%}", f"{cand.eta_easy:.2%}",
                                   f"ε={cand.vfd_error:.2f}%", f"{util:.1%}"],
                        "归一化": [f"{s_hard:.3f}", f"{s_easy:.3f}", f"{s_vfd:.3f}", f"{s_motor:.3f}"],
                        "权重": ["×0.4", "×0.3", "×0.2", "×0.1"],
                        "得分": [f"{s_hard*0.4:.3f}", f"{s_easy*0.3:.3f}",
                                 f"{s_vfd*0.2:.3f}", f"{s_motor*0.1:.3f}"],
                    }
                    st.table(score_data)
                    st.write(f"**综合评分 = {cand.score:.4f}**")

            # ── 淘汰机型追溯 ──
            st.divider()
            with st.expander(f"被淘汰的机型（{result.screened_out + result.rejected_eff} 款）", expanded=False):
                # 从 steps 中提取淘汰信息
                reject_lines = []
                for step in result.steps:
                    if "❌" in step:
                        reject_lines.append(step.strip())
                if reject_lines:
                    for line in reject_lines:
                        st.text(line)
                else:
                    st.info("所有机型均通过选型。")

    # ──────────────────────────────────────────
    # Tab 3：特性曲线（对应设计方案页面④）
    # ──────────────────────────────────────────
    with tab3:
        if not result.candidates:
            st.warning("没有候选机型，无法绘制特性曲线。")
        else:
            # 机型选择
            model_options = [
                f"{'🏆 ' if i == 0 else ''}{c.fan.model_id}（评分 {c.score:.4f}）"
                for i, c in enumerate(result.candidates)
            ]
            selected_idx = st.selectbox(
                "选择查看的机型",
                range(len(model_options)),
                format_func=lambda i: model_options[i],
            )

            selected_cand = result.candidates[selected_idx]
            fan = selected_cand.fan

            st.subheader(f"{fan.model_id} 特性曲线")

            # 参数摘要
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("装机功率", f"{fan.total_motor_kw:.0f} kW")
            sc2.metric("困难时期效率", f"{selected_cand.eta_hard:.1%}")
            sc3.metric("容易时期效率", f"{selected_cand.eta_easy:.1%}")
            sc4.metric("相似律误差", f"{selected_cand.vfd_error:.2f}%")

            st.divider()

            # 绘制综合特性曲线图
            fig_result = plot_fan_result(selected_cand, dual, eta_min=eta_min_val)
            st.pyplot(fig_result)
            plt.close(fig_result)

    # ──────────────────────────────────────────
    # Tab 4：详细计算过程
    # ──────────────────────────────────────────
    with tab4:
        st.subheader("完整计算过程记录")

        with st.expander("双工况引擎计算步骤", expanded=False):
            for step in dual.steps:
                st.text(step)

        with st.expander("选型计算步骤", expanded=True):
            for step in result.steps:
                st.text(step)

else:
    # ── 初始状态：未计算 ──
    st.info("请在左侧输入矿井工况参数，然后点击 **「开始计算并选型」** 按钮。")

    st.divider()
    st.subheader("当前风机数据库")
    st.caption(f"共 {len(FAN_DB)} 款实测数据机型")

    db_data = {
        "序号": [],
        "型号": [],
        "系列": [],
        "额定转速 (r/min)": [],
        "装机功率 (kW)": [],
        "Q范围 (m³/s)": [],
        "H范围 (Pa)": [],
        "数据点数": [],
    }
    for i, fan in enumerate(FAN_DB, 1):
        db_data["序号"].append(i)
        db_data["型号"].append(fan.model_id)
        db_data["系列"].append(fan.series)
        db_data["额定转速 (r/min)"].append(fan.rated_rpm)
        db_data["装机功率 (kW)"].append(f"{fan.motor_count}×{fan.motor_kw} = {fan.total_motor_kw:.0f}")
        db_data["Q范围 (m³/s)"].append(f"{fan.Q_min:.1f} ~ {fan.Q_max:.1f}")
        db_data["H范围 (Pa)"].append(f"{fan.H_min:.0f} ~ {fan.H_max:.0f}")
        db_data["数据点数"].append(len(fan.curve_points))

    st.table(db_data)
