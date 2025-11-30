# ui/app.py
"""
Streamlit应用主类 - RUNNING CC 版本

设计特色：
 统一集成：整合物理引擎、ML加速、可视化、性能监控
 现代化UI：基于时间的动态主题
 智能体验：实时反馈、渐进式加载、错误恢复
 多维分析：多视图、多维度数据探索
 模块化：组件化设计，易于维护扩展
"""
import streamlit as st
from typing import Any, Dict, TYPE_CHECKING
import numpy as np
import logging
import time
import traceback
import sys
import os
from datetime import datetime
import requests
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 类型注解导入（仅用于类型检查）
if TYPE_CHECKING:
    from core.engine import ComputationEngine
    from core.data_schema import FieldSolution, BEMSolution, VisualizationConfig, ModelParameters
    from visualization.backends import VisualizationBackend, DesignSystem
    from utils.performance import PerformanceMonitor, CacheManager
    from ml.interpolator import MLAccelerationEngine

# 运行时导入（带错误处理）
try:
    from core.engine import ComputationEngine, create_default_engine
    from core.data_schema import FieldSolution, BEMSolution, VisualizationConfig, ModelParameters
    from visualization.backends import VisualizationBackend, DesignSystem
    from utils.performance import PerformanceMonitor, CacheManager
    from ml.interpolator import MLAccelerationEngine

    # 标记导入成功
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None

except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


    # 定义回退类型，避免类型检查错误
    class ComputationEngine:
        """回退计算引擎类"""

        def __init__(self, enable_ml: bool = False):
            self.enable_ml = enable_ml

        def list_models(self):
            return ['point_charge', 'dipole']

        def compute(self, **kwargs):
            raise RuntimeError("计算引擎未正确导入")


    def create_default_engine(enable_ml: bool = False) -> ComputationEngine:
        """回退的默认引擎创建函数"""
        return ComputationEngine(enable_ml=enable_ml)


    class FieldSolution(dict):
        """回退场解类"""
        pass


    class BEMSolution(dict):
        """回退BEM解类"""
        pass


    class VisualizationConfig(dict):
        """回退可视化配置类"""
        pass


    class ModelParameters(dict):
        """回退模型参数类"""
        pass


    class VisualizationBackend:
        """回退可视化后端类"""

        @staticmethod
        def create(config):
            return MockBackend()


    class DesignSystem:
        """回退设计系统类"""
        pass


    class PerformanceMonitor:
        """回退性能监控类"""
        pass


    class CacheManager:
        """回退缓存管理类"""
        pass


    class MLAccelerationEngine:
        """回退ML加速引擎类"""

        def __init__(self, strategy: str = "idw"):
            self.strategy = strategy
            self.is_fitted = False

        def fit(self, solution):
            self.is_fitted = True

        def predict(self, query_points):
            return np.zeros((len(query_points), 3))

        def train(self, solution):
            self.fit(solution)


    class MockBackend:
        """模拟后端用于错误情况"""

        def plot_field(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "可视化后端未正确导入", ha='center', va='center', transform=ax.transAxes)
            return fig

        def plot_potential(self, *args, **kwargs):
            return self.plot_field(*args, **kwargs)

        def plot_field_lines(self, *args, **kwargs):
            return self.plot_field(*args, **kwargs)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置（必须在任何Streamlit代码之前）
st.set_page_config(
    page_title="RUNNING CC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ccstar99/ccPaper',
        'About': "# RUNNING CC v2.0\n基于物理优先的机器学习架构"
    }
)


# ============================================================================ #
# 主应用类 - RUNNING CC 设计
# ============================================================================ #

class ElectroFieldApp:
    """
    RUNNING CC 静电场仿真应用主控制器

    特色功能：
    -  统一模块集成
    -  动态主题
    -  实时性能监控
    -  多维度数据分析
    -  智能缓存管理
    """

    def __init__(self, enable_ml: bool = True, enable_cache: bool = True):
        """
        Args:
            enable_ml: 启用ML加速
            enable_cache: 启用智能缓存
        """
        # 检查导入状态
        if not IMPORT_SUCCESS:
            st.error(f"模块导入失败: {IMPORT_ERROR}")
            st.info("""
            **请确保以下模块已正确安装：**
            - core.engine: 计算引擎模块
            - core.data_schema: 数据契约模块  
            - visualization.backends: 可视化后端
            - utils.performance: 性能监控工具
            - ml.interpolator: ML加速模块

            **解决方法：**
            1. 检查项目结构是否正确
            2. 确保所有依赖包已安装
            3. 验证Python路径设置
            """)
            st.stop()

        self.enable_ml = enable_ml
        self.enable_cache = enable_cache

        # 初始化会话状态
        self._initialize_session_state()

        # 初始化组件
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager() if enable_cache else None

        logger.info("RUNNING CC 应用初始化完成")

    def _get_current_theme(self) -> str:
        """
        根据当前时间获取动态主题

        Returns:
            theme: 主题名称
        """
        now = datetime.now()
        hour = now.hour

        # 基于时间判断主题
        if 6 <= hour < 12:
            base_theme = "morning"
        elif 12 <= hour < 18:
            base_theme = "daylight"
        elif 18 <= hour < 22:
            base_theme = "evening"
        else:
            base_theme = "night"

        return base_theme

    def _initialize_session_state(self) -> None:
        """初始化现代化会话状态管理"""
        default_states = {
            # 核心引擎
            'engine': None,
            'ml_engine': None,

            # 计算结果
            'current_solution': None,
            'solution_history': [],

            # 性能数据
            'performance_history': [],
            'cache_stats': {'hits': 0, 'misses': 0, 'size': 0},

            # UI状态
            'ui_config': {
                'theme': self._get_current_theme(),
                'last_model': 'point_charge',
                'last_grid_size': 80,
                'ml_enabled': self.enable_ml,
                'auto_refresh': True,
                'current_view': 'main_viz'  # 当前视图
            },

            # 用户偏好
            'user_prefs': {
                'show_tutorial': True,
                'animation_speed': 1.0,
                'default_export_format': 'csv'
            },

            # 祝福语状态
            'current_blessing_index': 0,
            'last_blessing_update': time.time()
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _get_seasonal_blessings(self) -> list:
        """根据季节和时间生成祝福语列表"""
        now = datetime.now()
        month = now.month
        hour = now.hour

        # 季节判断
        if month in [12, 1, 2]:
            season = "冬"
        elif month in [3, 4, 5]:
            season = "春"
        elif month in [6, 7, 8]:
            season = "夏"
        else:
            season = "秋"

        # 时间段判断
        if 5 <= hour < 8:
            period = "清晨"
            blessings = [
                f"晨光熹微，{season}日静电场探索开始",
                f"{season}晨清爽，电场计算正当时",
                "朝霞映照，电磁奥秘待你发现",
                f"{season}晨微风，电场研究启新程",
                "清晨静谧，电磁场中寻真理",
                f"{season}日初升，电场计算伴晨光",
                "晨露未晞，静电场中探奥秘",
                f"{season}晨宁静，电磁研究正启航",
                "曙光初现，电场分布渐清晰",
                f"{season}晨美好，电磁世界任遨游"
            ]
        elif 8 <= hour < 12:
            period = "上午"
            blessings = [
                f"{season}日上午好，电场仿真之旅启程",
                f"阳光{season}日，静电场分析正当时",
                f"{season}日明媚，电磁世界任你遨游",
                f"{season}日温暖，电场计算更精准",
                "上午时光，电磁场研究正深入",
                f"{season}日晴好，电场分布显真章",
                "阳光正好，静电场探索不停歇",
                f"{season}日上午，电磁奥秘渐次开",
                "时光静好，电场计算正当时",
                f"{season}日灿烂，电磁研究更精彩"
            ]
        elif 12 <= hour < 14:
            period = "午间"
            blessings = [
                f"{season}日午安，电场计算伴你同行",
                "午间时光，探索电磁场的奇妙",
                f"{season}午静谧，电场分布显真章",
                "午安时分，静电场研究继续",
                f"{season}日正午，电磁场中寻真理",
                "午间小憩，电场计算不停步",
                f"{season}午温暖，电磁研究正深入",
                "午安时刻，电场分布渐清晰",
                f"{season}日午时，电磁奥秘待发现",
                "午间静谧，电场探索正当时"
            ]
        elif 14 <= hour < 18:
            period = "下午"
            blessings = [
                f"{season}日下午好，静电场研究继续",
                f"{season}日斜阳，电场分析正深入",
                "午后时光，电磁奥秘渐次展开",
                f"{season}日下午，电场计算更精准",
                "斜阳西照，电磁场研究正酣",
                f"{season}日午后，电场分布显规律",
                "下午时光，静电场探索不停歇",
                f"{season}日温暖，电磁研究正深入",
                "午后静谧，电场计算伴你行",
                f"{season}日下午，电磁世界任探索"
            ]
        elif 18 <= hour < 22:
            period = "傍晚"
            blessings = [
                f"{season}日傍晚，在暮色中研究电磁场",
                "暮色降临，电场计算更显深邃",
                f"{season}晚微风，静电场探索不停歇",
                "黄昏时分，电磁场中寻真理",
                f"{season}日傍晚，电场分布渐清晰",
                "暮色苍茫，电磁研究正当时",
                f"{season}晚霞美，电场计算伴晚风",
                "傍晚宁静，电磁奥秘待发现",
                f"{season}日黄昏，静电场中探规律",
                "暮色四合，电场探索正深入"
            ]
        else:
            period = "深夜"
            blessings = [
                "深夜静谧，在星空下研究电磁奥秘",
                "万籁俱寂，电场计算正当时",
                "夜色深沉，电磁场的秘密等待揭晓",
                "星空璀璨，静电场研究继续",
                "夜深人静，电磁场中寻真理",
                "月明星稀，电场分布渐清晰",
                "深夜时分，电磁奥秘待探索",
                "夜色温柔，电场计算伴星辰",
                "万籁无声，电磁研究正深入",
                "星空之下，静电场探索不停"
            ]

        # 特殊季节祝福语
        if season == "秋":
            autumn_blessings = [
                "秋风送爽，静电场研究正当时",
                "秋叶飘零，电磁场中寻规律",
                "秋高气爽，电场计算更精准",
                "秋意渐浓，电磁奥秘待发现",
                "秋日宁静，静电场探索深入",
                "秋月明净，电场分布显真章",
                "秋风轻拂，电磁研究正启航",
                "秋色宜人，电场计算伴秋光",
                "秋雨绵绵，电磁场中探真理",
                "秋日傍晚，暮色中研究电磁场"
            ]
            blessings.extend(autumn_blessings)

        # 通用祝福语
        general_blessings = [
            "静电场中探真理，电磁世界任遨游",
            "电场分布显规律，电磁研究正深入",
            "计算精准探奥秘，物理数学融一体",
            "电磁场中寻真理，科学研究无止境",
            "静电场探索继续，电磁奥秘待发现",
            "电场计算正当时，物理规律渐清晰",
            "电磁研究启新程，科学探索不停步",
            "静电场中求真知，电磁世界任探索",
            "电场分布显真章，物理规律待发现",
            "电磁奥秘渐次开，科学研究正深入"
        ]
        blessings.extend(general_blessings)

        return blessings

    def _get_current_blessing(self) -> str:
        """获取当前祝福语（10秒轮换）"""
        current_time = time.time()
        blessings = self._get_seasonal_blessings()

        # 每10秒更新一次祝福语
        if current_time - st.session_state['last_blessing_update'] > 10:
            st.session_state['current_blessing_index'] = (
                st.session_state['current_blessing_index'] + 1
            ) % len(blessings)
            st.session_state['last_blessing_update'] = current_time

        return blessings[st.session_state['current_blessing_index']]

    def _get_engine(self) -> "ComputationEngine":
        """获取计算引擎（智能初始化）"""
        if st.session_state['engine'] is None:
            with st.spinner("初始化计算引擎..."):
                try:
                    engine = create_default_engine(enable_ml=self.enable_ml)

                    # 预加载常用模型
                    available_models = engine.list_models()
                    logger.info(f"引擎初始化完成，可用模型: {available_models}")

                    st.session_state['engine'] = engine

                    # 初始化ML引擎
                    if self.enable_ml:
                        st.session_state['ml_engine'] = MLAccelerationEngine(strategy="idw")

                except Exception as e:
                    logger.error(f"引擎初始化失败: {e}")
                    st.error(f"计算引擎初始化失败: {e}")
                    raise

        return st.session_state['engine']

    def _build_model_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建模型参数

        Args:
            params: 来自侧边栏的参数

        Returns:
            模型参数字典
        """
        model_type = params.get('model_type', 'point_charge')

        # 基础参数
        model_params = {
            'model_type': model_type,
            'grid_size': params.get('grid_size', 80),
            'bbox': params.get('bbox', (-2, 2, -2, 2, -2, 2)),
            'timestamp': datetime.now().isoformat()
        }

        # 模型特定参数
        if model_type == 'point_charge':
            model_params.update({
                'charges': params.get('charges', []),
                'charge_count': len(params.get('charges', []))
            })
        elif model_type == 'bem_sphere':
            model_params.update({
                'radius': params.get('radius', 1.0),
                'voltage': params.get('voltage', 10.0),
                'divisions': params.get('divisions', 1),
                'mesh_resolution': params.get('resolution', 'medium')
            })
        elif model_type == 'dipole':
            model_params.update({
                'charges': params.get('charges', []),
                'separation': params.get('separation', 1.0),
                'orientation': params.get('orientation', 'horizontal')
            })
        elif model_type == 'line_charge':
            model_params.update({
                'charge_density': params.get('charge_density', 1e-9),
                'length': params.get('length', 2.0),
                'position': params.get('position', (0, 0, 0))
            })
        elif model_type == 'ring_charge':
            model_params.update({
                'charge': params.get('charge', 1e-9),
                'radius': params.get('radius', 1.0),
                'position': params.get('position', (0, 0, 0))
            })

        # 计算设置 - 默认启用ML加速
        model_params.update({
            'ml_enabled': True,  # 默认启用
            'validation_level': 'basic',
            'cache_enabled': True
        })

        logger.info(f"构建模型参数: {model_type}, 参数数量: {len(model_params)}")
        return model_params

    def render_sidebar(self) -> Dict[str, Any]:
        """
        渲染侧边栏控制面板

        Returns:
            参数字典，包含模型配置和UI设置
        """
        with st.sidebar:
            # 应用标题和主题选择
            self._render_sidebar_header()

            # 模型选择区域
            model_config = self._render_model_selection()

            # 参数配置区域
            params = self._render_parameter_panel(model_config['model_type'])

            # 计算设置区域
            compute_config = self._render_compute_settings()

            # 可视化设置区域
            viz_config = self._render_visualization_settings()

            # 操作按钮区域
            action_config = self._render_action_buttons()

            return {
                **model_config,
                **params,
                **compute_config,
                **viz_config,
                **action_config
            }

    def _render_sidebar_header(self):
        """渲染侧边栏头部"""
        # 显示应用标题
        st.markdown("""
        <style>
        .sidebar-title {
            font-size: 1.8rem;
            font-weight: bold;
            font-style: italic;
            background: linear-gradient(45deg, #8B5CF6, #6366F1, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .sidebar-subtitle {
            font-size: 0.9rem;
            font-style: italic;
            color: #6B7280;
            margin-bottom: 1rem;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">RUNNING CC</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">RUNNING ccElectrons • 基于物理数学方法</div>', unsafe_allow_html=True)

        # 显示当前时间
        now = datetime.now()

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 10px; 
                    border-radius: 10px; 
                    color: white; 
                    text-align: center;">
            <div>{now.strftime('%H:%M:%S')}</div>
            <div>{now.strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

    def _render_model_selection(self) -> Dict[str, Any]:
        """渲染模型选择区域"""
        st.subheader("物理模型")

        engine = self._get_engine()
        available_models = engine.list_models()

        model_descriptions = {
            'point_charge': '点电荷 - 基础静电学',
            'line_charge': '线电荷 - 无限长带电直线',
            'ring_charge': '带电圆环 - 轴对称场',
            'bem_sphere': '边界元法 - 导体球体',
            'dipole': '电偶极子 - 对称场分布'
        }

        model_type = st.selectbox(
            "选择仿真模型",
            options=available_models,
            format_func=lambda x: model_descriptions.get(x, x),
            index=available_models.index(st.session_state['ui_config']['last_model'])
            if st.session_state['ui_config']['last_model'] in available_models else 0,
            help="选择要仿真的物理模型"
        )

        st.session_state['ui_config']['last_model'] = model_type

        # 模型描述
        if model_type in model_descriptions:
            st.caption(f"{model_descriptions[model_type]}")

        return {'model_type': model_type}

    def _render_parameter_panel(self, model_type: str) -> Dict[str, Any]:
        """渲染参数配置面板"""
        st.subheader("模型参数")

        params = {'model_type': model_type}

        if model_type == "point_charge":
            params.update(self._render_point_charge_params())
        elif model_type == "bem_sphere":
            params.update(self._render_bem_sphere_params())
        elif model_type == "dipole":
            params.update(self._render_dipole_params())
        elif model_type == "line_charge":
            params.update(self._render_line_charge_params())
        elif model_type == "ring_charge":
            params.update(self._render_ring_charge_params())
        else:
            params.update(self._render_general_params())

        return params

    def _render_point_charge_params(self) -> Dict[str, Any]:
        """渲染点电荷参数"""
        # 电荷数量
        n_charges = st.slider(
            "电荷数量",
            min_value=1,
            max_value=5,
            value=2,
            help="设置仿真中的电荷数量"
        )

        charges = []
        for i in range(n_charges):
            st.markdown(f"**电荷 {i + 1}**")

            col1, col2 = st.columns([2, 1])

            with col1:
                # 位置设置
                x = st.slider(f"X (m)", -3.0, 3.0, -1.0 + i * 2.0, key=f"q{i}_x")
                y = st.slider(f"Y (m)", -3.0, 3.0, 0.0, key=f"q{i}_y")
                z = st.slider(f"Z (m)", -3.0, 3.0, 0.0, key=f"q{i}_z")

            with col2:
                # 电量设置
                q = st.number_input(
                    f"电量 (C)",
                    value=1e-9 * (-1 if i % 2 else 1),
                    format="%.2e",
                    key=f"q{i}_val"
                )

            charges.append({'position': (x, y, z), 'value': q})

        return {
            'charges': charges,
            'bbox': (-3, 3, -3, 3, -1, 1)
        }

    def _render_bem_sphere_params(self) -> Dict[str, Any]:
        """渲染BEM球体参数"""
        col1, col2 = st.columns(2)

        with col1:
            radius = st.slider("球体半径 (m)", 0.1, 2.0, 1.0)
            voltage = st.slider("球体电压 (V)", -100.0, 100.0, 10.0)

        with col2:
            resolution = st.select_slider(
                "网格分辨率",
                options=['低', '中', '高'],
                value='中'
            )
            res_map = {'低': 0, '中': 1, '高': 2}

        return {
            'radius': radius,
            'voltage': voltage,
            'divisions': res_map[resolution],
            'bbox': (-3, 3, -3, 3, -3, 3)
        }

    def _render_dipole_params(self) -> Dict[str, Any]:
        """渲染电偶极子参数"""
        col1, col2 = st.columns(2)

        with col1:
            separation = st.slider("偶极间距 (m)", 0.1, 2.0, 1.0)
            charge_magnitude = st.number_input("电荷大小 (C)", value=1e-9, format="%.2e")

        with col2:
            orientation = st.selectbox(
                "偶极方向",
                options=['horizontal', 'vertical', 'custom'],
                format_func=lambda x: {'horizontal': '水平', 'vertical': '垂直', 'custom': '自定义'}[x]
            )

        # 根据方向生成电荷
        if orientation == 'horizontal':
            charges = [
                {'position': (-separation / 2, 0, 0), 'value': charge_magnitude},
                {'position': (separation / 2, 0, 0), 'value': -charge_magnitude}
            ]
        elif orientation == 'vertical':
            charges = [
                {'position': (0, -separation / 2, 0), 'value': charge_magnitude},
                {'position': (0, separation / 2, 0), 'value': -charge_magnitude}
            ]
        else:
            # 自定义方向
            st.info("在高级设置中配置自定义电荷")
            charges = []

        return {
            'charges': charges,
            'separation': separation,
            'orientation': orientation,
            'bbox': (-2, 2, -2, 2, -1, 1)
        }

    def _render_line_charge_params(self) -> Dict[str, Any]:
        """渲染线电荷参数"""
        col1, col2 = st.columns(2)

        with col1:
            charge_density = st.number_input("线电荷密度 (C/m)", value=1e-9, format="%.2e")
            length = st.slider("线长度 (m)", 0.5, 5.0, 2.0)

        with col2:
            x = st.slider("位置 X (m)", -2.0, 2.0, 0.0)
            y = st.slider("位置 Y (m)", -2.0, 2.0, 0.0)

        return {
            'charge_density': charge_density,
            'length': length,
            'position': (x, y, 0),
            'bbox': (-3, 3, -3, 3, -1, 1)
        }

    def _render_ring_charge_params(self) -> Dict[str, Any]:
        """渲染圆环电荷参数"""
        col1, col2 = st.columns(2)

        with col1:
            charge = st.number_input("总电荷量 (C)", value=1e-9, format="%.2e")
            radius = st.slider("圆环半径 (m)", 0.5, 3.0, 1.0)

        with col2:
            x = st.slider("中心 X (m)", -2.0, 2.0, 0.0)
            y = st.slider("中心 Y (m)", -2.0, 2.0, 0.0)

        return {
            'charge': charge,
            'radius': radius,
            'position': (x, y, 0),
            'bbox': (-4, 4, -4, 4, -2, 2)
        }

    def _render_general_params(self) -> Dict[str, Any]:
        """渲染通用参数"""
        st.info("使用模型默认参数")
        return {'bbox': (-2, 2, -2, 2, -2, 2)}

    def _render_compute_settings(self) -> Dict[str, Any]:
        """渲染计算设置"""
        st.subheader("计算设置")

        grid_size = st.slider(
            "网格分辨率",
            min_value=20,
            max_value=200,
            value=st.session_state['ui_config']['last_grid_size'],
            step=10,
            help="更高的分辨率提供更精确的结果，但计算时间更长"
        )
        st.session_state['ui_config']['last_grid_size'] = grid_size

        # 默认启用ML加速，不显示选项
        return {
            'grid_size': grid_size,
            'ml_enabled': True  # 默认启用
        }

    def _render_visualization_settings(self) -> Dict[str, Any]:
        """渲染可视化设置"""
        st.subheader("可视化设置")

        # 后端选择
        backend = st.radio(
            "渲染引擎",
            options=["plotly", "matplotlib"],
            format_func=lambda x: "Plotly (交互式)" if x == "plotly" else "Matplotlib (高质量)",
            horizontal=True
        )

        col1, col2 = st.columns(2)

        with col1:
            show_vectors = st.checkbox("电场向量", value=True)
            show_contours = st.checkbox("等势面", value=True)

        with col2:
            show_charges = st.checkbox("显示电荷", value=True)
            show_field_lines = st.checkbox("电场线", value=True)

        # 自动颜色映射 - 宇宙风格
        current_theme = self._get_current_theme()
        if "night" in current_theme:
            colormap = "plasma"
        elif "evening" in current_theme:
            colormap = "hot"
        elif "morning" in current_theme:
            colormap = "viridis"
        else:
            colormap = "cool"

        viz_config = VisualizationConfig(
            backend=backend,
            style=current_theme,
            show_vectors=show_vectors,
            show_contours=show_contours,
            show_charges=show_charges,
            show_field_lines=show_field_lines,
            colormap=colormap,
            vector_scale=1.0
        )

        return {'viz_config': viz_config}

    def _render_action_buttons(self) -> Dict[str, Any]:
        """渲染操作按钮"""
        st.markdown("---")

        col1, col2 = st.columns([3, 1])

        with col1:
            calculate_btn = st.button(
                "开始计算",
                type="primary",
                use_container_width=True
            )

        with col2:
            if st.button("重置", use_container_width=True):
                self._reset_application()

        return {'calculate_requested': calculate_btn}

    def _reset_application(self):
        """重置应用状态"""
        keys_to_keep = ['user_prefs', 'ui_config']
        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]

        for key in keys_to_remove:
            del st.session_state[key]

        st.rerun()

    def _render_view_buttons(self):
        """渲染视图切换按钮"""
        st.markdown("""
        <style>
        .view-button {
            display: inline-block;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .view-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .view-button.active {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)

        views = [
            ('main_viz', '主可视化'),
            ('field_analysis', '场分析'),
            ('field_lines', '电场线'),
            ('data_stats', '数据统计')
        ]

        cols = st.columns(len(views))
        for idx, (view_key, view_name) in enumerate(views):
            with cols[idx]:
                if st.button(
                        view_name,
                        key=f"view_{view_key}",
                        use_container_width=True,
                        type="primary" if st.session_state.get('ui_config', {}).get('current_view',
                                                                                    'main_viz') == view_key else "secondary"
                ):
                    # 确保ui_config字典存在
                    if 'ui_config' not in st.session_state:
                        st.session_state['ui_config'] = {}
                    st.session_state['ui_config']['current_view'] = view_key
                    st.rerun()

    def render_main_content(self, params: Dict[str, Any]) -> None:
        """
        渲染主内容区域

        Args:
            params: 来自侧边栏的参数
        """
        # 应用标题
        st.markdown("""
        <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            background: linear-gradient(45deg, #8B5CF6, #6366F1, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main-subtitle {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .blessing-text {
            font-size: 0.9rem;
            font-style: italic;
            color: #8B5CF6;
            margin-bottom: 1rem;
            text-align: center;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        * {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)

        # 动态祝福语
        blessing = self._get_current_blessing()
        st.markdown(f'<div class="blessing-text">{blessing}</div>', unsafe_allow_html=True)

        # 欢迎界面或结果展示
        if st.session_state['current_solution'] is None:
            self._render_welcome_screen()
        else:
            self._render_results_dashboard(params)

    def _render_welcome_screen(self):
        """渲染欢迎界面"""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            欢迎使用 RUNNING CC 静电场仿真平台！

            快速开始:
            1. 在左侧面板选择物理模型
            2. 调整电荷参数和计算设置  
            3. 点击"开始计算"运行仿真
            4. 在结果面板探索可视化效果
            """)

    def _render_results_dashboard(self, params: Dict[str, Any]):
        """渲染结果仪表板"""
        solution = st.session_state['current_solution']

        # 渲染视图按钮
        self._render_view_buttons()
        st.markdown("---")

        # 根据当前视图渲染内容
        current_view = st.session_state['ui_config']['current_view']

        if current_view == 'main_viz':
            self._render_main_visualization(solution, params['viz_config'])
        elif current_view == 'field_analysis':
            self._render_field_analysis(solution, params['viz_config'])
        elif current_view == 'field_lines':
            self._render_field_lines(solution, params['viz_config'])
        elif current_view == 'data_stats':
            self._render_data_statistics(solution)

    def _render_main_visualization(self, solution: "FieldSolution", viz_config: "VisualizationConfig"):
        """渲染主可视化"""
        st.subheader("电场分布可视化")

        try:
            backend = VisualizationBackend.create(viz_config)
            fig = backend.plot_field(solution, viz_config)

            if viz_config['backend'] == 'matplotlib':
                st.pyplot(fig)
            else:
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        except Exception as e:
            st.error(f"可视化渲染失败: {e}")
            logger.error(f"主可视化错误: {e}", exc_info=True)

    def _render_field_analysis(self, solution: "FieldSolution", viz_config: "VisualizationConfig"):
        """渲染场分析"""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("电位分布")
            try:
                backend = VisualizationBackend.create(viz_config)
                fig = backend.plot_potential(solution, viz_config)

                if viz_config['backend'] == 'matplotlib':
                    st.pyplot(fig)
                else:
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"电位可视化失败: {e}")

        with col2:
            st.subheader("场强统计")
            self._render_field_statistics(solution)

    def _render_field_lines(self, solution: "FieldSolution", viz_config: "VisualizationConfig"):
        """渲染电场线"""
        st.subheader("电场线分析")

        col1, col2 = st.columns([3, 1])

        with col1:
            try:
                backend = VisualizationBackend.create(viz_config)

                # 检查是否为边界元法模型（BEM只能是3D）
                is_bem_model = solution.get('metadata', {}).get('model_name') == 'bem_sphere'

                if is_bem_model:
                    # 边界元法模型强制使用3D显示
                    is_3d = True
                    st.info("边界元法模型使用3D显示")
                else:
                    # 其他模型保留维度选择
                    dimension_option = st.radio(
                        "维度选择",
                        options=["2D", "3D"],
                        index=0,
                        horizontal=True
                    )
                    is_3d = dimension_option == "3D"

                # 电场线数量滑块
                default_num_lines = 15 if is_3d else 30
                n_lines = st.slider("电场线数量", 10, 100, default_num_lines, key="field_lines_slider")

                # 传递is_3d参数给后端
                fig = backend.plot_field_lines(solution, n_lines, viz_config, is_3d=is_3d)

                if viz_config['backend'] == 'matplotlib':
                    st.pyplot(fig)
                else:
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"电场线渲染失败: {e}")

        with col2:
            st.info("""
            电场线说明：
            - 从正电荷发出
            - 终止于负电荷
            - 密度表示场强大小
            - 切线方向为电场方向
            """)

    def _render_field_statistics(self, solution: "FieldSolution"):
        """渲染场统计信息"""
        vectors = solution['vectors']
        field_strength = np.linalg.norm(vectors, axis=1)

        metrics = {
            "最大场强": f"{np.max(field_strength):.3e} N/C",
            "平均场强": f"{np.mean(field_strength):.3e} N/C",
            "场强标准差": f"{np.std(field_strength):.3e}",
            "计算点数": len(vectors),
            "电荷数量": len(solution.get('charges', []))
        }

        for name, value in metrics.items():
            st.metric(name, value)

        # 场强分布直方图
        if len(field_strength) > 10:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(field_strength, bins=30, alpha=0.7, color='#6366F1', edgecolor='white')
            ax.set_xlabel('电场强度 (N/C)')
            ax.set_ylabel('频数')
            ax.grid(True, alpha=0.3)
            ax.set_title('场强分布直方图')

            st.pyplot(fig)

    def _render_data_statistics(self, solution: "FieldSolution"):
        """渲染数据统计"""
        st.subheader("数据概览")

        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                points = solution.get('points', np.array([]))
                point_count = len(points) if hasattr(points, '__len__') else 0
                st.metric("观察点数", point_count)

                if hasattr(points, 'shape') and len(points.shape) > 1:
                    dimensions = points.shape[1]
                else:
                    dimensions = 2
                st.metric("空间维度", f"{dimensions}D")
            except Exception as e:
                logger.error(f"数据维度统计错误: {e}")
                st.metric("观察点数", "N/A")
                st.metric("空间维度", "N/A")

        with col2:
            try:
                charges = solution.get('charges', [])
                if isinstance(charges, list):
                    total_charge = 0.0
                    valid_charges = 0
                    for c in charges:
                        if isinstance(c, dict) and 'value' in c:
                            try:
                                total_charge += float(c['value'])
                                valid_charges += 1
                            except (ValueError, TypeError):
                                continue
                    st.metric("总电荷量", f"{total_charge:.2e} C")
                    st.metric("电荷数量", valid_charges)
                else:
                    st.metric("总电荷量", "0.00e+00 C")
                    st.metric("电荷数量", 0)
            except Exception as e:
                logger.error(f"电荷统计错误: {e}")
                st.metric("总电荷量", "N/A")
                st.metric("电荷数量", "N/A")

        with col3:
            try:
                potentials = solution.get('potentials')
                if potentials is not None:
                    try:
                        max_potential = float(np.max(potentials))
                        min_potential = float(np.min(potentials))
                        st.metric("最大电位", f"{max_potential:.2f} V")
                        st.metric("最小电位", f"{min_potential:.2f} V")
                    except (ValueError, TypeError):
                        st.metric("最大电位", "N/A")
                        st.metric("最小电位", "N/A")
                else:
                    st.metric("电位数据", "未计算")
            except Exception as e:
                logger.error(f"电位统计错误: {e}")
                st.metric("电位数据", "错误")

        # 数据导出
        st.markdown("---")
        self._render_export_panel(solution)

    def _render_export_panel(self, solution: "FieldSolution"):
        """渲染数据导出面板"""
        st.subheader("数据导出")

        export_format = st.selectbox(
            "导出格式",
            options=['csv', 'json', 'npz', 'png'],
            format_func=lambda x: {
                'csv': 'CSV表格',
                'json': 'JSON数据',
                'npz': 'NumPy压缩',
                'png': '图像文件'
            }[x]
        )

        if st.button(f"导出{export_format.upper()}数据", use_container_width=True):
            self._export_data(solution, export_format)

    def _export_data(self, solution: "FieldSolution", format_type: str):
        """导出数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format_type == 'csv':
                self._export_to_csv(solution, timestamp)
            elif format_type == 'json':
                self._export_to_json(solution, timestamp)
            elif format_type == 'npz':
                self._export_to_npz(solution, timestamp)
            elif format_type == 'png':
                self._export_to_png(timestamp)

            st.success(f"数据已导出为 {format_type.upper()} 格式")

        except Exception as e:
            st.error(f"导出失败: {e}")
            logger.error(f"数据导出错误: {e}")

    def _export_to_csv(self, solution: "FieldSolution", timestamp: str):
        """导出为CSV格式"""
        import pandas as pd

        df = pd.DataFrame({
            'x': solution['points'][:, 0],
            'y': solution['points'][:, 1],
            'z': solution['points'][:, 2],
            'Ex': solution['vectors'][:, 0],
            'Ey': solution['vectors'][:, 1],
            'Ez': solution['vectors'][:, 2],
        })

        if solution['potentials'] is not None:
            df['potential'] = solution['potentials']

        csv_data = df.to_csv(index=False)

        st.download_button(
            label="下载CSV文件",
            data=csv_data,
            file_name=f"electrofield_data_{timestamp}.csv",
            mime="text/csv"
        )

    def _export_to_json(self, solution: "FieldSolution", timestamp: str):
        """导出为JSON格式"""
        import json

        json_data = {
            'points': solution['points'].tolist(),
            'vectors': solution['vectors'].tolist(),
            'potentials': solution['potentials'].tolist() if solution['potentials'] is not None else None,
            'charges': solution['charges'],
            'metadata': solution.get('metadata', {})
        }

        st.download_button(
            label="下载JSON文件",
            data=json.dumps(json_data, indent=2),
            file_name=f"electrofield_data_{timestamp}.json",
            mime="application/json"
        )

    def _export_to_npz(self, solution: "FieldSolution", timestamp: str):
        """导出为NPZ格式"""
        import io

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            points=solution['points'],
            vectors=solution['vectors'],
            potentials=solution['potentials'],
            charges=np.array([list(c['position']) + [c['value']] for c in solution['charges']])
        )
        buffer.seek(0)

        st.download_button(
            label="下载NPZ文件",
            data=buffer.getvalue(),
            file_name=f"electrofield_data_{timestamp}.npz",
            mime="application/octet-stream"
        )

    def _export_to_png(self, timestamp: str):
        """导出为PNG格式"""
        st.warning("PNG导出功能需要额外的截图库支持")

    def handle_computation(self, params: Dict[str, Any]) -> None:
        """处理计算请求"""
        if not params.get('calculate_requested'):
            return

        try:
            # 准备计算
            engine = self._get_engine()
            model_name = params['model_type']

            # 生成观察网格
            with st.spinner("生成观测网格..."):
                grid_size = params['grid_size']
                bbox = params.get('bbox', (-2, 2, -2, 2, -2, 2))

                # 创建2D观察平面（z=0）
                x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                x = np.linspace(x_min, x_max, grid_size)
                y = np.linspace(y_min, y_max, grid_size)
                X, Y = np.meshgrid(x, y)
                observation_points = np.column_stack([
                    X.ravel(), Y.ravel(), np.zeros_like(X.ravel())
                ])

                logger.info(f"生成观测网格: {len(observation_points)} 点")

            # 执行计算
            with st.spinner("计算电场分布..."):
                start_time = time.time()

                # 直接调用引擎的compute方法，让引擎处理模型初始化
                solution = engine.compute(
                    model_name=model_name,
                    charges=params.get('charges', []),
                    observation_points=observation_points,
                    parameters=self._build_model_parameters(params)
                )

                compute_time = time.time() - start_time

                # 记录性能数据
                performance_data = {
                    'timestamp': datetime.now(),
                    'model_type': model_name,
                    'compute_time': compute_time,
                    'grid_size': grid_size,
                    'points_count': len(observation_points)
                }
                st.session_state['performance_history'].append(performance_data)

                # 更新解决方案
                st.session_state['current_solution'] = solution
                st.session_state['solution_history'].append(solution)

                # ML训练（如果启用）
                if params.get('ml_enabled') and st.session_state.get('ml_engine'):
                    with st.spinner("训练ML加速模型..."):
                        st.session_state['ml_engine'].train(solution)

                st.success(f"计算完成！耗时: {compute_time:.2f}秒")

        except Exception as e:
            logger.error(f"计算失败: {e}", exc_info=True)
            st.error(f"计算失败: {str(e)}")

            # 显示模型初始化问题的特定建议
            if "unexpected keyword argument" in str(e) or "missing 1 required positional argument" in str(e):
                st.warning("""
                模型初始化参数不匹配
                这可能是由于模型类的构造函数签名不兼容导致的。
                """)

            # 显示详细错误信息
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())

            # 清理无效状态
            st.session_state['current_solution'] = None

    def render_knowledge_list(self, model_type: str):
        """渲染知识清单，显示当前模型的公式和理论说明"""
        st.markdown("---")
        st.subheader("知识清单")

        # 使用扩展器让内容可以折叠
        with st.expander(f"{model_type.upper()} 模型理论框架", expanded=True):
            self._render_knowledge_content(model_type)

    def _render_knowledge_content(self, model_type: str):
        """根据模型类型渲染相应的理论知识内容"""

        if model_type == 'bem_sphere':
            st.markdown("### 边界元法求解器实现")
            st.markdown("严格遵循李亚莎论文《三维静电场线性插值边界元中的解析积分方法》的理论框架。")

            st.markdown("**核心理论修正：**")

            st.markdown("**1. 边界积分方程：**")
            st.latex(
                r"c(\mathbf{r})\phi(\mathbf{r}) + \int_{\Gamma} \phi \frac{\partial G}{\partial n}  dS = \int_{\Gamma} G \frac{\partial \phi}{\partial n} dS")
            st.markdown("离散形式：$[H]\\{\phi\\} = [G]\\{q\\}$")
            st.markdown("其中：")
            st.markdown("- $H_{ij} = \\int_{\\Gamma_j} \\frac{\\partial G}{\\partial n} dS$")
            st.markdown("- $G_{ij} = \\int_{\\Gamma_j} G dS$")

            st.markdown("**2. 对角线处理：**")
            st.latex(r"H_{ii} = c(\mathbf{r}_i) + \int_{\Gamma_i} \frac{\partial G}{\partial n} dS")
            st.markdown("其中 $c(\\mathbf{r}_i) = \\Omega_i / 4\\pi$，$\\Omega_i$ 是顶点i处的立体角")
            st.markdown("对于光滑边界，$c(\\mathbf{r}_i) = 0.5$")

            st.markdown("**3. 固体角计算：使用精确解析公式**")
            st.latex(r"\Omega = 2\pi - \sum \theta_k")
            st.markdown("其中$\\theta_k$是边界边在顶点处的内角")

            st.markdown("**4. 奇异积分：**")
            st.markdown("- $G_{ii}$ 使用解析积分（非零）")
            st.markdown("- $H_{ii}$ 主值处理，不包含自相互作用")

            st.markdown("**参考文献：**")
            st.markdown("- 赵凯华《电磁学》")
            st.markdown("- 李亚莎《三维静电场线性插值边界元中的解析积分方法》")

        elif model_type == 'point_charge':
            st.markdown("### 点电荷场理论")

            st.markdown("**库仑定律：**")
            st.latex(r"\mathbf{E} = \frac{1}{4\pi\varepsilon_0} \frac{q}{|\mathbf{r}|^2} \hat{\mathbf{r}}")
            st.latex(r"\phi = \frac{1}{4\pi\varepsilon_0} \frac{q}{|\mathbf{r}|}")

            st.markdown("**多电荷系统叠加原理：**")
            st.latex(r"\mathbf{E}_{\text{total}} = \sum \mathbf{E}_i")
            st.latex(r"\phi_{\text{total}} = \sum \phi_i")

            st.markdown("**场强计算：**")
            st.latex(r"|\mathbf{E}| = \sqrt{E_x^2 + E_y^2 + E_z^2}")

            st.markdown("**参数说明：**")
            st.markdown("- $\\mathbf{E}$: 电场强度向量 (N/C)")
            st.markdown("- $\\phi$: 电势标量 (V)")
            st.markdown("- $q$: 电荷量 (C)")
            st.markdown("- $\\mathbf{r}$: 位置向量 (m)")
            st.markdown("- $\\varepsilon_0$: 真空介电常数 = $8.854 \\times 10^{-12}$ F/m")

            st.markdown("**参考文献：**")
            st.markdown("- 赵凯华《电磁学》第1章")

        elif model_type == 'dipole':
            st.markdown("### 电偶极子理论")

            st.markdown("**偶极矩定义：**")
            st.latex(r"\mathbf{p} = q \mathbf{d}")
            st.markdown("其中 $\\mathbf{d}$ 为从负电荷指向正电荷的位移向量")

            st.markdown("**电势分布：**")
            st.latex(r"\phi = \frac{1}{4\pi\varepsilon_0} \frac{\mathbf{p} \cdot \hat{\mathbf{r}}}{r^2}")

            st.markdown("**电场分布：**")
            st.latex(
                r"\mathbf{E} = \frac{1}{4\pi\varepsilon_0} \frac{3(\mathbf{p} \cdot \hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{p}}{r^3}")

            st.markdown("**轴上场强（偶极方向为z轴）：**")
            st.latex(r"E_z = \frac{1}{4\pi\varepsilon_0} \frac{2p}{r^3}")
            st.markdown("$E_x = E_y = 0$")

            st.markdown("**垂直平分线上场强：**")
            st.latex(r"E_x = \frac{1}{4\pi\varepsilon_0} \frac{-p}{r^3}")
            st.markdown("$E_z = E_y = 0$")

            st.markdown("**参考文献：**")
            st.markdown("- 赵凯华《电磁学》第2章")
            st.markdown("- Jackson《经典电动力学》")

        elif model_type == 'line_charge':
            st.markdown("### 线电荷场理论")

            st.markdown("**线电荷密度：**")
            st.latex(r"\lambda = \frac{dq}{dl}")

            st.markdown("**电场强度（无限长直线电荷）：**")
            st.latex(r"\mathbf{E} = \frac{\lambda}{2\pi\varepsilon_0 r} \hat{\mathbf{r}}")

            st.markdown("**电势分布：**")
            st.latex(r"\phi = \frac{\lambda}{2\pi\varepsilon_0} \ln\left(\frac{1}{r}\right) + C")

            st.markdown("**有限长线电荷场强分量：**")
            st.latex(r"E_x = \frac{\lambda}{4\pi\varepsilon_0} \frac{\cos\theta_1 - \cos\theta_2}{r}")
            st.latex(r"E_y = \frac{\lambda}{4\pi\varepsilon_0} \frac{\sin\theta_2 - \sin\theta_1}{r}")
            st.markdown("其中 $\\theta_1$, $\\theta_2$ 为端点到场点的角度")

            st.markdown("**参考文献：**")
            st.markdown("- 赵凯华《电磁学》第1章")

        elif model_type == 'ring_charge':
            st.markdown("### 圆环电荷场理论")

            st.markdown("**圆环参数：**")
            st.markdown("- 半径: $R$")
            st.markdown("- 总电荷: $Q$")
            st.markdown("- 线电荷密度: $\\lambda = Q/(2\\pi R)$")

            st.markdown("**对称轴上电势：**")
            st.latex(r"\phi(z) = \frac{1}{4\pi\varepsilon_0} \frac{Q}{\sqrt{R^2 + z^2}}")

            st.markdown("**对称轴上电场：**")
            st.latex(r"E(z) = \frac{1}{4\pi\varepsilon_0} \frac{Qz}{(R^2 + z^2)^{3/2}}")

            st.markdown("**任意点电势积分形式：**")
            st.latex(
                r"\phi(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \oint \frac{\lambda dl}{|\mathbf{r} - \mathbf{r}'|}")

            st.markdown("**最大场强位置：**")
            st.latex(r"z_{\text{max}} = \frac{R}{\sqrt{2}}")
            st.latex(r"E_{\text{max}} = \frac{1}{4\pi\varepsilon_0} \frac{2Q}{3\sqrt{3} R^2}")

            st.markdown("**参考文献：**")
            st.markdown("- 赵凯华《电磁学》第1章")

        else:
            st.markdown("### 通用静电学原理")

            st.markdown("**高斯定理：**")
            st.latex(r"\oint \mathbf{E} \cdot d\mathbf{A} = \frac{Q_{\text{enclosed}}}{\varepsilon_0}")

            st.markdown("**电势与电场关系：**")
            st.latex(r"\mathbf{E} = -\nabla \phi")

            st.markdown("**泊松方程：**")
            st.latex(r"\nabla^2 \phi = -\frac{\rho}{\varepsilon_0}")

            st.markdown("**拉普拉斯方程（无电荷区域）：**")
            st.latex(r"\nabla^2 \phi = 0")

            st.markdown("**叠加原理：**")
            st.markdown("电场和电势满足线性叠加原理，多个电荷产生的总场等于各电荷单独产生场的矢量和。")

            st.markdown("**边界条件：**")
            st.markdown("- 导体表面：等势面，电场垂直表面")
            st.markdown("- 介质界面：电位移法向分量连续，电场切向分量连续")

    def run(self):
        """运行应用主循环"""
        try:
            # 渲染侧边栏并获取参数
            params = self.render_sidebar()
            # 处理计算请求
            self.handle_computation(params)
            # 渲染主内容
            self.render_main_content(params)
            # 渲染知识清单（替换原来的页脚）
            model_type = params.get('model_type', 'point_charge')
            self.render_knowledge_list(model_type)

        except Exception as e:
            # 全局异常处理
            logger.critical(f"应用运行时错误: {e}", exc_info=True)

            st.error("""
            应用遇到严重错误!

            请尝试以下操作：
            1. 点击侧边栏的"重置"按钮
            2. 刷新页面重新加载
            3. 检查控制台错误信息

            如果问题持续存在，请联系技术支持。
            """)

            with st.expander("技术详情"):
                st.exception(e)


# ============================================================================ #
# 应用启动器
# ============================================================================ #

def main():
    """应用主入口点"""
    # 创建并运行应用
    app = ElectroFieldApp(
        enable_ml=True,
        enable_cache=True
    )

    app.run()


if __name__ == "__main__":
    # 命令行参数处理
    import argparse

    parser = argparse.ArgumentParser(description='RUNNING CC 静电场仿真平台')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    parser.add_argument('--no-ml', action='store_true', help='禁用ML加速')
    parser.add_argument('--theme', choices=['morning', 'daylight', 'evening', 'night'],
                        default=None, help='强制主题（默认自动）')

    args = parser.parse_args()

    # 设置主题（如果指定）
    if args.theme and 'ui_config' in st.session_state:
        st.session_state['ui_config']['theme'] = args.theme

    # 运行应用
    main()