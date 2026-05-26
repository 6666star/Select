# -*- coding: utf-8 -*-
"""
统一 matplotlib 中文字体配置。
优先使用项目自带字体（适配云端部署），否则回退系统字体。
"""
import os
import matplotlib
import matplotlib.font_manager as fm

def setup_chinese_font():
    """注册项目内中文字体并设置为 matplotlib 默认字体。"""
    font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'msyh.ttc')

    if os.path.exists(font_path):
        # 注册项目自带的微软雅黑
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.family'] = [font_name, 'DejaVu Sans']
    else:
        # 本地 Windows 环境回退
        matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

    matplotlib.rcParams['axes.unicode_minus'] = False

# 导入即生效
setup_chinese_font()
