# ui/app.py
"""
Streamlit应用主类 - 现代化集成版本

设计特色：
 统一集成：整合物理引擎、ML加速、可视化、性能监控
 现代化UI：基于时间和天气的动态主题
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
    page_title="🌌 智能静电场仿真平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ccstar99/ccPaper',
        'About': "# 🌟 智能静电场仿真平台 v2.0\n基于物理优先的机器学习架构"
    }
)


# ============================================================================ #
# 主应用类 - 现代化设计
# ============================================================================ #

class ElectroFieldApp:
    """
    智能静电场仿真应用主控制器

    特色功能：
    -  统一模块集成
    -  动态主题（基于时间和天气）
    -  实时性能监控
    -  多维度数据分析
    -  智能缓存管理
    -  上下文感知帮助
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

        logger.info("ElectroFieldApp 初始化完成")

    def _get_current_theme(self) -> str:
        """
        根据当前时间和天气获取动态主题

        Returns:
            theme: 主题名称
        """
        now = datetime.now()
        hour = now.hour

        # 基于时间判断主题
        if 6 <= hour < 12:
            base_theme = "morning"  # 清晨
        elif 12 <= hour < 18:
            base_theme = "daylight"  # 白天
        elif 18 <= hour < 22:
            base_theme = "evening"  # 傍晚
        else:
            base_theme = "night"  # 夜晚

        # 尝试获取天气信息（失败时使用时间主题）
        try:
            weather_theme = self._get_weather_theme()
            return f"{base_theme}_{weather_theme}"
        except:
            return base_theme

    def _get_weather_theme(self) -> str:
        """
        获取天气主题（简化版，实际使用时需要天气API）

        Returns:
            weather_type: 天气类型
        """
        # 这里简化实现，实际应该调用天气API
        # 例如：openweathermap.org
        weather_types = ["clear", "cloudy", "rainy", "stormy"]

        # 模拟根据月份和小时简单判断
        now = datetime.now()
        month = now.month
        hour = now.hour

        if month in [12, 1, 2]:  # 冬季
            if hour < 7 or hour > 18:
                return "clear"  # 冬季夜晚通常晴朗
            else:
                return "cloudy"
        elif month in [6, 7, 8]:  # 夏季
            if 14 <= hour <= 16:
                return "stormy"  # 夏季午后可能有雷雨
            else:
                return "clear"
        else:  # 春秋季
            return "clear"

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
                'theme': self._get_current_theme(),  # 动态主题
                'last_model': 'point_charge',
                'last_grid_size': 80,
                'ml_enabled': self.enable_ml,
                'auto_refresh': True,
                'expert_mode': False
            },

            # 用户偏好
            'user_prefs': {
                'show_tutorial': True,
                'animation_speed': 1.0,
                'default_export_format': 'csv'
            }
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

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

                    # 初始化ML引擎 - 修复参数问题
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

        # 计算设置
        model_params.update({
            'ml_enabled': params.get('ml_enabled', False),
            'validation_level': params.get('validation_level', 'basic'),
            'cache_enabled': params.get('cache_enabled', True)
        })

        logger.info(f"构建模型参数: {model_type}, 参数数量: {len(model_params)}")
        return model_params

    def render_sidebar(self) -> Dict[str, Any]:
        """
        渲染现代化侧边栏控制面板

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

            # 高级设置区域
            advanced_config = self._render_advanced_settings()

            # 操作按钮区域
            action_config = self._render_action_buttons()

            return {
                **model_config,
                **params,
                **compute_config,
                **viz_config,
                **advanced_config,
                **action_config
            }

    def _render_sidebar_header(self):
        """渲染侧边栏头部"""
        st.title("🌌 ElectroField")
        st.markdown("### 智能静电场仿真")

        # 显示当前时间和主题信息
        now = datetime.now()
        current_theme = self._get_current_theme()

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 10px; 
                    border-radius: 10px; 
                    color: white; 
                    text-align: center;">
            <div>🕐 {now.strftime('%H:%M:%S')}</div>
            <div>📅 {now.strftime('%Y-%m-%d')}</div>
            <div>🌤️ {self._get_theme_display_name(current_theme)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

    def _get_theme_display_name(self, theme: str) -> str:
        """获取主题的显示名称"""
        theme_names = {
            "morning_clear": "🌅 晴朗清晨",
            "morning_cloudy": "🌥️ 多云清晨",
            "daylight_clear": "☀️ 晴朗白天",
            "daylight_cloudy": "⛅ 多云白天",
            "daylight_rainy": "🌧️ 雨天白天",
            "evening_clear": "🌇 晴朗傍晚",
            "evening_stormy": "⛈️ 雷雨傍晚",
            "night_clear": "🌙 晴朗夜晚",
            "night_cloudy": "☁️ 多云夜晚",
            "morning": "🌅 清晨",
            "daylight": "☀️ 白天",
            "evening": "🌇 傍晚",
            "night": "🌙 夜晚"
        }
        return theme_names.get(theme, "动态主题")

    def _render_model_selection(self) -> Dict[str, Any]:
        """渲染模型选择区域"""
        st.subheader("🔬物理模型")

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
            'bbox': (-3, 3, -3, 3, -1, 1)  # 主要关注xy平面
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
        
        # 专家模式下的高级参数
        expert_params = {}
        if st.session_state['ui_config']['expert_mode']:
            st.markdown("---")
            st.subheader("🔬 边界元法专家参数")
            
            col3, col4 = st.columns(2)
            with col3:
                solver_precision = st.selectbox(
                    "求解精度",
                    options=['float32', 'float64'],
                    index=1,  # 默认选择'float64'
                    help="选择数值计算的精度"
                )
                
                max_iterations = st.number_input(
                    "最大迭代次数",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="迭代求解器的最大迭代次数"
                )
            
            with col4:
                convergence_tol = st.number_input(
                    "收敛阈值",
                    min_value=1e-12,
                    max_value=1e-6,
                    value=1e-8,
                    format="%.2e",
                    help="求解器收敛的误差阈值"
                )
                
                use_direct_solver = st.checkbox(
                    "使用直接求解器",
                    value=False,
                    help="对于小型问题，直接求解可能更快"
                )
            
            st.markdown("### 物理参数")
            epsilon_r = st.slider(
                "相对介电常数",
                min_value=1.0,
                max_value=100.0,
                value=1.0,
                help="周围介质的相对介电常数"
            )
            
            expert_params = {
                'solver_precision': solver_precision,
                'max_iterations': max_iterations,
                'convergence_tol': convergence_tol,
                'use_direct_solver': use_direct_solver,
                'epsilon_r': epsilon_r
            }

        params = {
            'radius': radius,
            'voltage': voltage,
            'divisions': res_map[resolution],
            'bbox': (-3, 3, -3, 3, -3, 3)
        }
        
        # 合并专家参数
        params.update(expert_params)
        
        return params

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

        # ML加速选项
        ml_enabled = st.checkbox(
            "启用ML加速",
            value=st.session_state['ui_config']['ml_enabled'],
            help="使用机器学习插值加速重复计算"
        )
        st.session_state['ui_config']['ml_enabled'] = ml_enabled

        if ml_enabled:
            st.success("ML加速,后续计算将显著加快")

        return {
            'grid_size': grid_size,
            'ml_enabled': ml_enabled
        }

    def _render_visualization_settings(self) -> Dict[str, Any]:
        """渲染可视化设置"""
        st.subheader("可视化设置")

        # 后端选择
        backend = st.radio(
            "渲染引擎",
            options=["plotly", "matplotlib"],
            format_func=lambda x: "🔄 Plotly (交互式)" if x == "plotly" else "📊 Matplotlib (高质量)",
            horizontal=True
        )

        col1, col2 = st.columns(2)

        with col1:
            show_vectors = st.checkbox("电场向量", value=True)
            show_contours = st.checkbox("等势面", value=True)

        with col2:
            show_charges = st.checkbox("显示电荷", value=True)
            show_field_lines = st.checkbox("电场线", value=True)

        # 颜色映射
        current_theme = self._get_current_theme()
        if "night" in current_theme:
            default_colormap = "plasma"
        elif "evening" in current_theme:
            default_colormap = "hot"
        elif "morning" in current_theme:
            default_colormap = "viridis"
        else:
            default_colormap = "cool"

        colormap = st.selectbox(
            "颜色主题",
            options=["viridis", "plasma", "hot", "cool", "rainbow"],
            index=["viridis", "plasma", "hot", "cool", "rainbow"].index(default_colormap),
            help="选择颜色映射方案"
        )

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

    def _render_advanced_settings(self) -> Dict[str, Any]:
        """渲染高级设置"""
        if st.session_state['ui_config']['expert_mode']:
            st.subheader("🔧 高级设置")

            col1, col2 = st.columns(2)

            with col1:
                validation_level = st.selectbox(
                    "验证等级",
                    options=["none", "basic", "strict"],
                    format_func=lambda x: {"none": "无", "basic": "基础", "strict": "严格"}[x]
                )

            with col2:
                cache_enabled = st.checkbox("启用缓存", value=True)

            return {
                'validation_level': validation_level,
                'cache_enabled': cache_enabled
            }

        return {}

    def _render_action_buttons(self) -> Dict[str, Any]:
        """渲染操作按钮"""
        st.markdown("---")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            calculate_btn = st.button(
                "开始计算",
                type="primary",
                use_container_width=True
            )

        with col2:
            if st.button("🔄 重置", use_container_width=True):
                self._reset_application()

        with col3:
            st.session_state['ui_config']['expert_mode'] = st.checkbox(
                "专家模式",
                value=st.session_state['ui_config']['expert_mode']
            )

        return {'calculate_requested': calculate_btn}

    def _reset_application(self):
        """重置应用状态"""
        keys_to_keep = ['user_prefs', 'ui_config']
        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]

        for key in keys_to_remove:
            del st.session_state[key]

        st.rerun()

    def render_main_content(self, params: Dict[str, Any]) -> None:
        """
        渲染主内容区域

        Args:
            params: 来自侧边栏的参数
        """
        # 应用标题
        st.title("🌌 智能静电场仿真平台")

        # 动态问候语
        current_hour = datetime.now().hour
        if current_hour < 12:
            greeting = "早安！开始今天的电场探索吧"
        elif current_hour < 18:
            greeting = "午安！享受静电场的奇妙世界"
        else:
            greeting = "晚上好！在星空下研究电磁奥秘"

        st.markdown(f"**{greeting}** • 实时交互式电磁场仿真")

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
            欢迎使用智能静电场仿真平台！
            快速开始:
            1. 在左侧面板选择物理模型
            2. 调整电荷参数和计算设置  
            3. 点击"开始计算"运行仿真
            4. 在结果面板探索可视化效果
            """)

        with col2:
            st.image("https://via.placeholder.com/300x200/4F46E5/FFFFFF?text=电场仿真",
                     caption="静电场可视化示例")

        # 系统状态卡片
        st.markdown("---")
        self._render_system_status_cards()

    def _render_system_status_cards(self):
        """渲染系统状态卡片"""
        st.subheader("系统状态")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            engine_status = "就绪" if st.session_state['engine'] else "初始化中"
            st.metric("计算引擎", engine_status)

        with col2:
            n_models = len(self._get_engine().list_models()) if st.session_state['engine'] else 0
            st.metric("可用模型", n_models)

        with col3:
            ml_status = "已启用" if self.enable_ml else "已禁用"
            st.metric("ML加速", ml_status)

        with col4:
            cache_hits = st.session_state['cache_stats']['hits']
            st.metric("缓存命中", cache_hits)

    def _render_results_dashboard(self, params: Dict[str, Any]):
        """渲染结果仪表板"""
        solution = st.session_state['current_solution']

        # 创建标签页布局
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "主可视化",
            "场分析",
            "电场线",
            "数据统计",
            "技术详情"
        ])

        with tab1:
            self._render_main_visualization(solution, params['viz_config'])

        with tab2:
            self._render_field_analysis(solution, params['viz_config'])

        with tab3:
            self._render_field_lines(solution, params['viz_config'])

        with tab4:
            self._render_data_statistics(solution)

        with tab5:
            self._render_technical_details(solution)

    def _render_main_visualization(self, solution: "FieldSolution", viz_config: "VisualizationConfig"):
        """渲染主可视化"""
        st.subheader("🎯 电场分布可视化")

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
            st.subheader("📊 场强统计")
            self._render_field_statistics(solution)

    def _render_field_lines(self, solution: "FieldSolution", viz_config: "VisualizationConfig"):
        """渲染电场线"""
        st.subheader("🔍 电场线分析")

        col1, col2 = st.columns([3, 1])

        with col1:
            try:
                backend = VisualizationBackend.create(viz_config)
                
                # 检查是否为边界元法模型（BEM只能是3D）
                is_bem_model = solution.get('metadata', {}).get('model_name') == 'bem_sphere'
                
                if is_bem_model:
                    # 边界元法模型强制使用3D显示
                    is_3d = True
                else:
                    # 其他模型保留维度选择
                    dimension_option = st.radio(
                        "维度选择",
                        options=["2D", "3D"],
                        index=0,  # 默认选择2D
                        horizontal=True
                    )
                    is_3d = dimension_option == "3D"
                
                # 电场线数量滑块（3D模式下可以适当减少默认数量以优化性能）
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
                # 安全获取观察点数和维度
                points = solution.get('points', np.array([]))
                point_count = len(points) if hasattr(points, '__len__') else 0
                st.metric("观察点数", point_count)
                
                # 安全获取空间维度
                if hasattr(points, 'shape') and len(points.shape) > 1:
                    dimensions = points.shape[1]
                else:
                    dimensions = 2  # 默认2D
                st.metric("空间维度", f"{dimensions}D")
            except Exception as e:
                logger.error(f"数据维度统计错误: {e}")
                st.metric("观察点数", "N/A")
                st.metric("空间维度", "N/A")

        with col2:
            try:
                charges = solution.get('charges', [])
                if isinstance(charges, list):
                    # 安全计算总电荷量，添加异常处理
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
                    # 安全获取最大和最小电位
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

    def _render_technical_details(self, solution: "FieldSolution"):
        """渲染技术详情"""
        st.subheader("技术详情")

        col1, col2 = st.columns(2)

        with col1:
            st.json(solution.get('metadata', {}), expanded=False)

        with col2:
            st.markdown("""
            数据结构：
            - 观察点: 场计算的位置坐标
            - 场向量: 每个点的电场向量 (Ex, Ey, Ez)  
            - 电位: 标量电位分布（如可用）
            - 电荷: 源电荷配置信息
            - 元数据: 计算参数和性能数据
            """)

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
            label="📥 下载CSV文件",
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
        # 这里需要实现截图功能
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
            with st.spinner("🔄 生成观测网格..."):
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

            # 显示详细错误信息（专家模式）
            if st.session_state['ui_config']['expert_mode']:
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())

            # 清理无效状态
            st.session_state['current_solution'] = None

    def render_footer(self):
        """渲染页脚"""
        st.markdown("---")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.caption("""
            🌟 **智能静电场仿真平台 v2.0**  
            🏗️ 基于物理优先的机器学习架构  
            """)

        with col2:
            current_theme = self._get_current_theme()
            st.caption(f"""
            **系统信息**  
            引擎: {len(self._get_engine().list_models())} 模型  
            ML加速: {'TRUE' if self.enable_ml else 'FALSE'}  
            主题: {self._get_theme_display_name(current_theme)}
            """)

        with col3:
            st.caption(f"""
            **性能统计**  
            计算次数: {len(st.session_state['performance_history'])}  
            缓存命中: {st.session_state['cache_stats']['hits']}  
            最后更新: {datetime.now().strftime('%H:%M:%S')}
            """)

    def run(self):
        """运行应用主循环"""
        try:
            # 渲染侧边栏并获取参数
            params = self.render_sidebar()
            # 处理计算请求
            self.handle_computation(params)
            # 渲染主内容
            self.render_main_content(params)
            # 渲染页脚
            self.render_footer()

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
    # 应用标题和描述
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 创建并运行应用
    app = ElectroFieldApp(
        enable_ml=True,  # 启用ML加速
        enable_cache=True  # 启用智能缓存
    )

    app.run()


if __name__ == "__main__":
    # 命令行参数处理
    import argparse

    parser = argparse.ArgumentParser(description='智能静电场仿真平台')
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