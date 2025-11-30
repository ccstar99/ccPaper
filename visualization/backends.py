# visualization/backends.py
"""
可视化后端抽象层 - 苹果风格 & 宇宙风格设计

设计哲学：
1. 苹果美学：简洁、现代、圆角、渐变、优雅动画
2. 宇宙主题：深空背景、星点效果、霓虹色彩、科幻感
3. 用户体验：直观交互、流畅动画、信息层次分明
4. 性能优化：智能降采样、渐进式渲染

色彩方案：
- 苹果风格：浅灰背景、渐变蓝、柔和的色彩过渡
- 宇宙风格：深空黑背景、霓虹蓝紫、星云渐变
- 电荷颜色：正电荷(珊瑚红)、负电荷(冰蓝)

布局原则：
- 黄金分割比例
- 充足的留白空间
- 一致的圆角设计
- 优雅的字体层次
"""

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, List, Dict
import logging
import math
from matplotlib.patches import Circle
# 导入matplotlib
import matplotlib.pyplot as plt
# 导入Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入数据契约
try:
    from core.data_schema import FieldSolution, BEMSolution, VisualizationConfig
except ImportError:
    # 回退类型定义
    from typing import TypedDict, Any, List, Optional


    class FieldSolution(TypedDict):
        points: NDArray[np.float64]
        vectors: NDArray[np.float64]
        potentials: Optional[NDArray[np.float64]]
        charges: List[dict]
        metadata: dict


    class BEMSolution(TypedDict):
        vertices: NDArray[np.float64]
        triangles: NDArray[np.float64]
        vertex_potentials: Optional[NDArray[np.float64]]
        metadata: dict


    class VisualizationConfig(TypedDict):
        backend: str
        style: str
        show_charges: bool
        show_vectors: bool
        show_streamlines: bool  # 电场线显示（与show_field_lines功能相同）
        show_field_lines: bool  # 电场线显示（为兼容性添加）
        show_contours: bool

logger = logging.getLogger(__name__)


# ============================================================================ #
# 设计系统 - 颜色与样式
# ============================================================================ #

class DesignSystem:
    """设计系统：苹果风格 + 宇宙风格"""

    # 苹果风格配色
    APPLE = {
        'background': '#F5F7FA',  # 浅灰蓝
        'surface': '#FFFFFF',  # 纯白
        'primary': '#007AFF',  # 苹果蓝
        'secondary': '#5856D6',  # 紫蓝
        'accent': '#34C759',  # 苹果绿
        'text_primary': '#1D1D1F',
        'text_secondary': '#86868B',
        'grid': 'rgba(0,0,0,0.08)',
        'charge_positive': '#FF3B30',  # 珊瑚红
        'charge_negative': '#32D74B',  # 冰蓝绿
        'gradient': ['#007AFF', '#5856D6', '#AF52DE']  # 蓝紫渐变
    }

    # 宇宙风格配色
    COSMOS = {
        'background': '#0A0A1A',  # 深空黑
        'surface': '#1A1A2E',  # 宇宙深蓝
        'primary': '#6366F1',  # 霓虹紫蓝
        'secondary': '#8B5CF6',  # 亮紫
        'accent': '#06D6A0',  # 霓虹青
        'text_primary': '#E2E8F0',
        'text_secondary': '#94A3B8',
        'grid': 'rgba(255, 255, 255, 0.1)',  # CSS格式
        'charge_positive': '#EF4444',  # 星红
        'charge_negative': '#3B82F6',  # 星蓝
        'gradient': ['#6366F1', '#8B5CF6', '#EC4899'],  # 霓虹渐变
        'starfield': True,  # 启用星点背景
        'glow_effect': True  # 启用光晕效果
    }

    @classmethod
    def get_style(cls, style: str = 'apple') -> Dict[str, Any]:
        """获取设计风格配置"""
        return cls.APPLE if style == 'apple' else cls.COSMOS

    @staticmethod
    def rgba_to_tuple(rgba_str: str) -> Tuple[float, float, float, float]:
        """将CSS rgba字符串转换为Matplotlib兼容的RGBA元组

        Args:
            rgba_str: CSS格式的rgba字符串，如'rgba(255, 255, 255, 0.1)'

        Returns:
            0-1浮点数范围的RGBA元组
        """
        # 提取rgba中的数值部分
        import re
        match = re.search(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', rgba_str)
        if match:
            r, g, b, a = match.groups()
            return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(a))
        return (0.0, 0.0, 0.0, 1.0)  # 默认黑色

    @classmethod
    def get_color(cls, color_name: str, style: str = 'apple', backend: str = 'matplotlib') -> Any:
        """获取指定后端兼容的颜色格式

        Args:
            color_name: 颜色名称
            style: 设计风格
            backend: 后端类型 ('matplotlib' 或 'plotly')

        Returns:
            后端兼容的颜色格式
        """
        design = cls.get_style(style)
        color = design.get(color_name, '#000000')

        # 对于Matplotlib，将rgba字符串转换为元组
        if backend == 'matplotlib' and isinstance(color, str) and color.startswith('rgba'):
            return cls.rgba_to_tuple(color)

        return color

    @classmethod
    def apply_figure_style(cls, fig, style: str = 'apple'):
        """应用图形样式"""
        design = cls.get_style(style)

        if hasattr(fig, 'update_layout'):  # Plotly
            fig.update_layout(
                paper_bgcolor=design['background'],
                plot_bgcolor=design['surface'],
                font=dict(
                    family="Arial, -apple-system, BlinkMacSystemFont, sans-serif",
                    color=design['text_primary'],
                    size=12
                ),
                margin=dict(l=60, r=60, t=80, b=60),
                title=dict(
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20, color=design['text_primary'])
                )
            )

        return fig

    @classmethod
    def create_starfield(cls, n_stars: int = 100) -> Tuple[NDArray, NDArray, NDArray]:
        """创建随机星点背景"""
        stars = np.random.uniform(-10, 10, (n_stars, 3))
        intensities = np.random.uniform(0.3, 1.0, n_stars)
        sizes = np.random.uniform(0.5, 2.0, n_stars)
        return stars, intensities, sizes


# ============================================================================ #
# 抽象基类
# ============================================================================ #

class VisualizationBackend(ABC):
    """
    可视化后端抽象基类 - 现代化设计
    设计特色：
    - 响应式布局
    - 平滑动画过渡
    - 优雅的颜色映射
    - 智能数据可视化
    """

    def __init__(self, config: VisualizationConfig):
        """
        Args:
            config: 可视化配置对象
        """
        self.config = config
        self.figure = None
        self.axes = None
        self.design = DesignSystem.get_style(config.get('style', 'apple'))

        # 性能优化参数
        self.max_direct_points = 10000
        self.downsample_factor = 0.1
        self.animation_fps = 30

    @abstractmethod
    def plot_field(
            self,
            solution: FieldSolution,
            config: Optional[VisualizationConfig] = None
    ) -> Any:
        """主电场可视化接口 - 现代化设计"""
        pass

    @abstractmethod
    def plot_potential(
            self,
            solution: FieldSolution,
            config: Optional[VisualizationConfig] = None
    ) -> Any:
        """绘制电位分布 - 表面渐变效果"""
        pass

    @abstractmethod
    def plot_field_lines(
            self,
            solution: FieldSolution,
            n_lines: int = 20,
            config: Optional[VisualizationConfig] = None,
            is_3d: bool = False
    ) -> Any:
        """绘制电场线 - 流线型设计

        Args:
            solution: 电场解
            n_lines: 电场线数量
            config: 可视化配置
            is_3d: 是否以3D模式绘制

        Returns:
            电场线图形
        """
        pass

    @abstractmethod
    def plot_boundary_mesh(
            self,
            solution: BEMSolution,
            config: Optional[VisualizationConfig] = None
    ) -> Any:
        """绘制BEM网格 - 透明表面效果"""
        pass

    @abstractmethod
    def create_animation(
            self,
            solution_sequence: list[FieldSolution],
            config: Optional[VisualizationConfig] = None
    ) -> Any:
        """创建动画 - 流畅过渡"""
        pass

    def _preprocess_data(self, solution: FieldSolution) -> Tuple[NDArray, NDArray, NDArray]:
        """数据预处理标准化 - 增强错误处理"""
        try:
            points = solution['points']
            vectors = solution['vectors']

            # 检查数据形状
            if len(points) == 0 or len(vectors) == 0:
                logger.warning("空数据点或向量")
                return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

            # 确保数据维度正确
            if points.shape[1] < 2:
                logger.warning(f"点数据维度不足: {points.shape}")
                return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

            if vectors.shape[1] < 2:
                logger.warning(f"向量数据维度不足: {vectors.shape}")
                return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

            # 智能降采样
            n_points = len(points)
            if n_points > self.max_direct_points:
                logger.info(f"智能降采样: {n_points} → {int(n_points * self.downsample_factor)}")
                sample_idx = np.random.choice(
                    n_points, int(n_points * self.downsample_factor), replace=False
                )
                points = points[sample_idx]
                vectors = vectors[sample_idx]

            # 计算场强
            field_strength = np.linalg.norm(vectors, axis=1)

            return points, vectors, field_strength

        except Exception as e:
            logger.error(f"数据预处理错误: {e}")
            return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

    def _create_charge_collections(self, solution: FieldSolution) -> Tuple[List, List]:
        """分类电荷 - 视觉优化"""
        positive = []
        negative = []

        for charge in solution['charges']:
            try:
                pos = charge['position']
                # 安全获取电荷值，防止'value'键错误
                q = charge.get('value', 0.0) if isinstance(charge.get('value'), (int, float)) else 0.0
                
                if q > 0:
                    positive.append((pos, q))
                else:
                    negative.append((pos, q))
            except Exception as e:
                logger.warning(f"处理电荷时出错: {e}")
                # 跳过有问题的电荷，继续处理其他电荷
                continue

        return positive, negative

    def _create_colorbar(self, values: NDArray, label: str) -> Dict[str, Any]:
        """创建现代化颜色条"""
        return {
            'colorscale': [
                [0, self.design['gradient'][0]],
                [0.5, self.design['gradient'][1]],
                [1, self.design['gradient'][2]]
            ],
            'colorbar': {
                'title': label,
                'title_font': {'color': self.design['text_primary']},
                'tickfont': {'color': self.design['text_secondary']},
                'bgcolor': self.design['surface'],
                'bordercolor': self.design['grid'],
                'borderwidth': 1,
                'len': 0.8,
                'thickness': 15
            }
        }

    @staticmethod
    def create(config: VisualizationConfig) -> 'VisualizationBackend':
        """工厂方法 - 创建设计优化的后端实例"""
        backends = {
            'matplotlib': MatplotlibBackend,
            'plotly': PlotlyBackend
        }

        backend_class = backends.get(config.get('backend', 'matplotlib'))
        if backend_class is None:
            raise ValueError(f"不支持的后端: {config['backend']}. 可选: {list(backends.keys())}")

        return backend_class(config)


# ============================================================================ #
# Matplotlib后端 - 现代化设计
# ============================================================================ #

class MatplotlibBackend(VisualizationBackend):
    """
    Matplotlib后端 - 苹果风格设计

    特色：
    - 圆角图形元素
    - 渐变色彩
    - 专业字体排版
    - 优雅的网格系统
    """

    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        self._import_mpl()
        # 直接使用父类初始化的design属性，但在使用时通过DesignSystem.get_color方法获取兼容格式

    def _import_mpl(self):
        """延迟导入，应用现代化样式"""
        global plt, mpl
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # 应用现代化样式
        plt.style.use('default')
        mpl.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 18,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'savefig.dpi': 300,
            'figure.figsize': [12, 8]
        })

    def _apply_modern_style(self, ax, title: str = ""):
        """应用现代化样式到坐标轴"""
        # 使用get_color方法获取Matplotlib兼容的颜色格式
        ax.set_facecolor(DesignSystem.get_color('surface', self.config.get('style', 'apple'), 'matplotlib'))
        grid_color = DesignSystem.get_color('grid', self.config.get('style', 'apple'), 'matplotlib')
        ax.grid(True, alpha=0.3, color=grid_color, linestyle='--')

        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            spine.set_linewidth(1)

        text_primary = DesignSystem.get_color('text_primary', self.config.get('style', 'apple'), 'matplotlib')
        text_secondary = DesignSystem.get_color('text_secondary', self.config.get('style', 'apple'), 'matplotlib')

        ax.set_title(title, color=text_primary, pad=20,
                     fontweight='semibold', fontsize=16)

        ax.tick_params(colors=text_secondary)
        ax.xaxis.label.set_color(text_primary)
        ax.yaxis.label.set_color(text_primary)

    def plot_potential(self, solution: FieldSolution, config: Optional[VisualizationConfig] = None) -> plt.Figure:
        """现代化电位表面图 - 仅支持3D数据"""
        cfg = config if config else self.config

        # 直接从solution获取points，不通过_preprocess_data避免潜在的数据处理问题
        points = solution['points']
        potentials = solution.get('potentials')

        if potentials is None:
            potentials = np.zeros(len(points))
            logger.warning("电位数据缺失，使用零值")
        
        # 修复形状不匹配问题：确保potentials与points长度匹配
        if len(points) != len(potentials):
            logger.error(f"形状不匹配: points长度={len(points)}, potentials长度={len(potentials)}")
            # 调整potentials长度以匹配points
            if len(potentials) < len(points):
                # 如果potentials较短，使用适当的值填充
                potentials = np.pad(potentials, (0, len(points) - len(potentials)), mode='edge')
            else:
                # 如果potentials较长，截断到与points相同长度
                potentials = potentials[:len(points)]
        
        # 强制确保数据是3D格式
        if points.shape[1] < 3:
            # 添加z维度并设置为0
            points = np.hstack([points, np.zeros((len(points), 3 - points.shape[1]))])
            logger.info("已将数据转换为3D格式用于边界元法可视化")

        fig = plt.figure(figsize=(12, 8), facecolor=self.design['background'])
        ax = fig.add_subplot(111, projection='3d')

        # 设置3D坐标轴样式
        ax.set_facecolor(self.design['surface'])
        ax.grid(True, alpha=0.3, color=self.design['grid'])

        # 创建表面图或散点图
        try:
            from scipy.interpolate import griddata
            # 只使用x和y坐标进行2D插值，但在3D空间中显示
            xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
            yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
            XI, YI = np.meshgrid(xi, yi)
            
            # 修复griddata调用，确保输入数组形状正确
            points_xy = points[:, :2].astype(float)
            potentials_float = potentials.astype(float)
            
            # 使用更可靠的插值方法
            ZI = griddata(points_xy, potentials_float, (XI, YI), method='linear')
            
            # 处理可能的NaN值
            if np.isnan(ZI).all():
                logger.warning("所有插值点都是NaN，回退到最近邻方法")
                ZI = griddata(points_xy, potentials_float, (XI, YI), method='nearest')

            # 绘制表面图
            surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', alpha=0.8,
                                   antialiased=True, linewidth=0)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='电位 (V)')
        except Exception as e:
            logger.error(f"表面图绘制失败: {str(e)}")
            # 退化为3D散点图
            scatter = ax.scatter(points[:, 0], points[:, 1], potentials,
                                 c=potentials, cmap='viridis', s=20, alpha=0.7)
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='电位 (V)')

        ax.set_title('3D电位分布', color=self.design['text_primary'], pad=20)
        ax.set_xlabel('X (m)', color=self.design['text_primary'])
        ax.set_ylabel('Y (m)', color=self.design['text_primary'])
        ax.set_zlabel('电位 (V)', color=self.design['text_primary'])

        return fig

    def plot_field(self, solution: FieldSolution, config: Optional[VisualizationConfig] = None) -> plt.Figure:
        """主电场可视化接口 - 现代化设计"""
        cfg = config if config else self.config
        
        # 数据预处理
        points, vectors, _ = self._preprocess_data(solution)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.design['background'])
        self._apply_modern_style(ax, "电场分布可视化")
        
        # 检查是否显示向量场
        if cfg.get('show_vectors', True):
            try:
                # 智能降采样以提高性能
                n_points = len(points)
                if n_points > 1000:  # 对于大数据集进行降采样
                    sample_idx = np.random.choice(n_points, 1000, replace=False)
                    sample_points = points[sample_idx]
                    sample_vectors = vectors[sample_idx]
                else:
                    sample_points = points
                    sample_vectors = vectors
                
                # 绘制向量场
                ax.quiver(sample_points[:, 0], sample_points[:, 1], 
                          sample_vectors[:, 0], sample_vectors[:, 1],
                          color=self.design['primary'], alpha=0.6, scale=1)
            except Exception as e:
                logger.warning(f"向量场绘制失败: {e}")
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', color=self.design['text_primary'])
        ax.set_ylabel('Y (m)', color=self.design['text_primary'])
        
        return fig

    def plot_field_lines(self, solution: FieldSolution, n_lines: int = 20,
                         config: Optional[VisualizationConfig] = None, is_3d: bool = False) -> Any:
        """现代化电场线可视化"""
        # 使用传入的config参数，默认为空字典
        cfg = config or {}

        points, vectors, _ = self._preprocess_data(solution)

        # 获取电荷信息
        charges = solution.get('charges', [])

        # 检查是否需要显示电场线，同时支持show_field_lines和show_streamlines参数
        show_lines = cfg.get('show_field_lines', True) or cfg.get('show_streamlines', True)
        if not show_lines:
            # 如果不需要显示电场线，创建一个空图并返回
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.design['background'])
            self._apply_modern_style(ax, "电场线分布（已禁用）")
            ax.set_xlabel('X (m)', color=self.design['text_primary'])
            ax.set_ylabel('Y (m)', color=self.design['text_primary'])
            return fig

        # 计算电场线，传入电荷信息
        field_lines = FieldLineCalculator.compute_field_lines(
            points, vectors, n_lines, is_3d=is_3d, charges=charges
        )

        if is_3d:
            # 3D模式
            fig = plt.figure(figsize=(14, 10), facecolor=self.design['background'])
            ax = fig.add_subplot(111, projection='3d')

            # 设置3D场景
            ax.set_facecolor(self.design['surface'])
            ax.grid(True, alpha=0.3, color=self.design['grid'])

            # 绘制3D电场线
            cmap = plt.get_cmap('plasma')
            for i, line in enumerate(field_lines):
                line_array = np.array(line)
                if len(line_array) < 2:
                    continue

                color = cmap(i / len(field_lines))
                ax.plot(line_array[:, 0], line_array[:, 1], line_array[:, 2] if line_array.shape[1] > 2 else 0,
                        color=color, alpha=0.8, linewidth=2.0)

            # 绘制电荷
            if cfg.get('show_charges', True):
                pos_charges, neg_charges = self._create_charge_collections(solution)

                for pos, q in pos_charges:
                    ax.scatter(pos[0], pos[1], pos[2] if len(pos) > 2 else 0,
                               color=self.design['charge_positive'],
                               s=150,  # 使用固定大小，不依赖电荷值
                               alpha=0.8, edgecolors='white')

                for pos, q in neg_charges:
                    ax.scatter(pos[0], pos[1], pos[2] if len(pos) > 2 else 0,
                               color=self.design['charge_negative'],
                               s=150,  # 使用固定大小，不依赖电荷值
                               alpha=0.8, edgecolors='white')

            ax.set_xlabel('X (m)', color=self.design['text_primary'])
            ax.set_ylabel('Y (m)', color=self.design['text_primary'])
            ax.set_zlabel('Z (m)', color=self.design['text_primary'])
            ax.set_title('3D电场线分布', color=self.design['text_primary'], pad=20)

        else:
            # 2D模式
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.design['background'])
            self._apply_modern_style(ax, "电场线分布")

            # 绘制2D电场线
            cmap = plt.get_cmap('plasma')
            for i, line in enumerate(field_lines):
                line_array = np.array(line)
                color = cmap(i / len(field_lines))
                ax.plot(line_array[:, 0], line_array[:, 1], color=color, alpha=0.7, linewidth=1.5)

            # 绘制电荷
            if cfg.get('show_charges', True):
                pos_charges, neg_charges = self._create_charge_collections(solution)

                for pos, q in pos_charges:
                    circle = Circle(pos[:2], radius=0.1,  # 使用固定大小，不依赖电荷值
                                    color=self.design['charge_positive'], alpha=0.8, zorder=10)
                    ax.add_patch(circle)

                for pos, q in neg_charges:
                    circle = Circle(pos[:2], radius=0.1,  # 使用固定大小，不依赖电荷值
                                    color=self.design['charge_negative'], alpha=0.8, zorder=10)
                    ax.add_patch(circle)

            ax.set_xlabel('X (m)', color=self.design['text_primary'])
            ax.set_ylabel('Y (m)', color=self.design['text_primary'])
            ax.set_aspect('equal')

        return fig

    def plot_boundary_mesh(self, solution: BEMSolution, config: Optional[VisualizationConfig] = None) -> plt.Figure:
        """现代化BEM网格可视化"""
        cfg = config if config else self.config

        vertices = solution['vertices']
        triangles = solution['triangles']
        potentials = solution['vertex_potentials']

        fig = plt.figure(figsize=(12, 10), facecolor=self.design['background'])
        ax = fig.add_subplot(111, projection='3d')

        # 绘制网格
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = Poly3DCollection(vertices[triangles], alpha=0.7,
                                edgecolor=self.design['grid'], linewidth=0.8)

        if potentials is not None:
            # 根据电位着色
            face_colors = []
            for tri in triangles:
                avg_potential = np.mean(potentials[tri])
                face_colors.append(avg_potential)
            mesh.set_array(np.array(face_colors))
            mesh.set_cmap('viridis')
            fig.colorbar(mesh, ax=ax, shrink=0.5, label='电位 (V)')

        ax.add_collection3d(mesh)

        # 设置极限
        max_range = np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
        center = np.mean(vertices, axis=0)
        ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
        ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

        ax.set_title('BEM网格可视化', color=self.design['text_primary'], pad=20)
        ax.set_xlabel('X (m)', color=self.design['text_primary'])
        ax.set_ylabel('Y (m)', color=self.design['text_primary'])
        ax.set_zlabel('Z (m)', color=self.design['text_primary'])

        return fig

    def create_animation(self, solution_sequence: list[FieldSolution],
                         config: Optional[VisualizationConfig] = None) -> Any:
        """现代化动画创建"""
        cfg = config if config else self.config

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.design['background'])
        self._apply_modern_style(ax, "电场演化过程")

        def animate(frame):
            ax.clear()
            self._apply_modern_style(ax, f"电场演化 - 帧 {frame + 1}/{len(solution_sequence)}")

            solution = solution_sequence[frame]
            points, vectors, field_strength = self._preprocess_data(solution)

            # 向量场可视化
            sample_step = max(1, len(points) // 40)
            ax.quiver(
                points[::sample_step, 0],
                points[::sample_step, 1],
                vectors[::sample_step, 0],
                vectors[::sample_step, 1],
                field_strength[::sample_step],
                cmap='viridis', scale=60, alpha=0.8, width=0.005
            )

            ax.set_xlim(points[:, 0].min(), points[:, 0].max())
            ax.set_ylim(points[:, 1].min(), points[:, 1].max())

            return ax

        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(solution_sequence),
                             interval=cfg.get('interval_ms', 300), blit=False)

        return anim


# ============================================================================ #
# Plotly后端 - 宇宙风格设计（3D渲染优化版）
# ============================================================================ #

class PlotlyBackend(VisualizationBackend):
    """
    Plotly交互式后端 - 宇宙风格设计

    特色：
    - 深空背景与星点效果
    - 霓虹色彩方案
    - 3D交互体验
    - 流畅动画过渡
    """

    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        self._import_plotly()

    def _import_plotly(self):
        """延迟导入Plotly"""
        global go, make_subplots
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    def _add_starfield(self, fig, row: int = 1, col: int = 1):
        """添加星点背景效果"""
        if self.design.get('starfield', False):
            stars, intensities, sizes = DesignSystem.create_starfield(200)

            # 创建星点trace
            star_trace = go.Scatter3d(
                x=stars[:, 0], y=stars[:, 1], z=stars[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=intensities,
                    colorscale=['black', 'white'],
                    opacity=0.3,
                    symbol='circle'
                ),
                showlegend=False,
                hoverinfo='skip'
            )

            # 检查figure是否有子图结构
            try:
                # 尝试使用row和col参数添加trace
                fig.add_trace(star_trace, row=row, col=col)
            except ValueError:
                # 如果figure没有子图结构，直接添加trace
                fig.add_trace(star_trace)

    def plot_field(
            self,
            solution: FieldSolution,
            config: Optional[VisualizationConfig] = None
    ) -> go.Figure:
        """
        Plotly宇宙风格电场可视化 - 修复版

        特色布局：
        - 3D向量场（主视图）
        - 2D投影热力图
        - 实时统计面板
        - 电荷信息显示
        """
        cfg = config if config else self.config

        # 数据预处理 - 添加更严格的检查
        points, vectors, field_strength = self._preprocess_data(solution)

        # 检查数据有效性
        if len(points) == 0:
            logger.error("没有有效的点数据")
            return self._create_empty_figure("错误: 没有电场数据")

        if len(vectors) == 0:
            logger.error("没有有效的向量数据")
            return self._create_empty_figure("错误: 没有场向量数据")

        # 创建宇宙风格子图布局
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D电场向量场', '2D场强热力图', '电荷分布', '性能指标'),
            specs=[
                [{"type": "scatter3d"}, {"type": "heatmap"}],
                [{"type": "scatter3d"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # 应用宇宙风格
        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        # 1. 3D向量场 - 主视图
        sample_step = max(1, len(points) // min(50, len(points)))

        # 检查是否有足够的点进行采样
        if len(points) >= sample_step:
            # 添加星点背景
            self._add_starfield(fig, row=1, col=1)

            # 3D向量锥体 - 修复sizeref计算
            max_field = np.max(field_strength) if len(field_strength) > 0 else 1.0
            sizeref_value = 1.5 * max_field if max_field > 0 else 1.0

            fig.add_trace(
                go.Cone(
                    x=points[::sample_step, 0],
                    y=points[::sample_step, 1],
                    z=points[::sample_step, 2],
                    u=vectors[::sample_step, 0],
                    v=vectors[::sample_step, 1],
                    w=vectors[::sample_step, 2],
                    sizemode="absolute",
                    sizeref=sizeref_value,
                    colorscale=self.design['gradient'],
                    colorbar=dict(title="场强 (N/C)", x=0.45, len=0.4),
                    showscale=True,
                    name='电场向量',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        else:
            # 添加空图提示
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='text',
                    text=['<b>无向量数据</b>'],
                    textposition='middle center',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        # 2. 2D热力图投影 - 增强错误处理
        try:
            from scipy.interpolate import griddata

            # 检查是否有足够的点进行插值
            if len(points) >= 10 and points.shape[1] >= 2:
                xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
                yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
                XI, YI = np.meshgrid(xi, yi)

                # 尝试不同的插值方法
                try:
                    ZI = griddata((points[:, 0], points[:, 1]), field_strength, (XI, YI), method='cubic')
                except:
                    try:
                        ZI = griddata((points[:, 0], points[:, 1]), field_strength, (XI, YI), method='linear')
                    except:
                        ZI = griddata((points[:, 0], points[:, 1]), field_strength, (XI, YI), method='nearest')

                fig.add_trace(
                    go.Heatmap(
                        x=xi, y=yi, z=ZI,
                        colorscale='Hot',
                        colorbar=dict(title="场强", x=1.02, len=0.4),
                        name='场强分布',
                        hoverinfo='z'
                    ),
                    row=1, col=2
                )
            else:
                # 添加空热力图提示
                fig.add_trace(
                    go.Heatmap(
                        z=[[0]],
                        colorscale='Hot',
                        showscale=False,
                        name='场强分布',
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
                logger.warning("点数据不足，无法生成热力图")

        except Exception as e:
            logger.warning(f"热力图生成失败: {e}")
            # 添加错误提示
            fig.add_trace(
                go.Heatmap(
                    z=[[0]],
                    colorscale='Hot',
                    showscale=False,
                    name='热力图生成失败',
                    hoverinfo='skip'
                ),
                row=1, col=2
            )

        # 3. 电荷分布3D可视化 - 修复电荷显示
        if cfg.get('show_charges', True) and 'charges' in solution and solution['charges']:
            pos_charges, neg_charges = self._create_charge_collections(solution)

            # 正电荷
            if pos_charges:
                pos_coords = np.array([p[0] for p in pos_charges])
                charges = np.array([p[1] for p in pos_charges])

                # 使用固定大小，不依赖电荷值，确保电荷始终清晰可见
                charge_sizes = np.full(len(charges), 20.0)

                fig.add_trace(
                    go.Scatter3d(
                        x=pos_coords[:, 0],
                        y=pos_coords[:, 1],
                        z=pos_coords[:, 2] if pos_coords.shape[1] > 2 else np.zeros(len(pos_coords)),
                        mode='markers+text',
                        marker=dict(
                            size=charge_sizes,
                            color=self.design['charge_positive'],
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        text=[f'+{q:.1e}C' for q in charges],
                        textposition="top center",
                        name='正电荷',
                        hoverinfo='text'
                    ),
                    row=2, col=1
                )

            # 负电荷
            if neg_charges:
                neg_coords = np.array([p[0] for p in neg_charges])
                charges = np.array([p[1] for p in neg_charges])

                # 使用固定大小，不依赖电荷值，确保电荷始终清晰可见
                charge_sizes = np.full(len(charges), 20.0)

                fig.add_trace(
                    go.Scatter3d(
                        x=neg_coords[:, 0],
                        y=neg_coords[:, 1],
                        z=neg_coords[:, 2] if neg_coords.shape[1] > 2 else np.zeros(len(neg_coords)),
                        mode='markers+text',
                        marker=dict(
                            size=charge_sizes,
                            color=self.design['charge_negative'],
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        text=[f'{q:.1e}C' for q in charges],
                        textposition="top center",
                        name='负电荷',
                        hoverinfo='text'
                    ),
                    row=2, col=1
                )

            # 如果没有电荷，添加提示
            if not pos_charges and not neg_charges:
                fig.add_trace(
                    go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode='text',
                        text=['<b>无电荷数据</b>'],
                        textposition='middle center',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
        else:
            # 添加电荷显示禁用提示
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='text',
                    text=['<b>电荷显示已禁用</b>'],
                    textposition='middle center',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )

        # 4. 性能指标柱状图 - 修复统计计算
        if len(field_strength) > 0:
            metrics = {
                '最大值': np.max(field_strength),
                '平均值': np.mean(field_strength),
                '标准差': np.std(field_strength),
                '中位数': np.median(field_strength)
            }
        else:
            metrics = {
                '最大值': 0.0,
                '平均值': 0.0,
                '标准差': 0.0,
                '中位数': 0.0
            }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=[self.design['primary'], self.design['secondary'],
                              self.design['accent'], self.design['gradient'][2]],
                marker_line=dict(color='white', width=1),
                name='统计指标',
                text=[f'{v:.2e}' for v in metrics.values()],
                textposition='auto',
                hoverinfo='x+y'
            ),
            row=2, col=2
        )

        # 修复模型名称显示
        model_name = "未知模型"
        if 'metadata' in solution and solution['metadata']:
            metadata = solution['metadata']
            if 'model_name' in metadata and metadata['model_name']:
                model_name = metadata['model_name']
            elif 'name' in metadata and metadata['name']:
                model_name = metadata['name']
            elif 'title' in metadata and metadata['title']:
                model_name = metadata['title']

        # 更新整体布局
        fig.update_layout(
            title=dict(
                text=f"静电场分析 - {model_name}",
                x=0.5,
                font=dict(size=24, color=self.design['text_primary'])
            ),
            height=900,
            showlegend=True,
            legend=dict(
                bgcolor=self.design['surface'],
                bordercolor=self.design['grid'],
                borderwidth=1,
                x=0.02,
                y=0.98
            )
        )

        # 更新子图标题样式
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color=self.design['text_primary'])

        # 设置3D场景的相机视角
        fig.update_scenes(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )

        return fig

    def _create_empty_figure(self, message: str) -> go.Figure:
        """创建错误提示图形"""
        fig = go.Figure()
        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        fig.add_annotation(
            text=f"<b>{message}</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=self.design['text_primary'])
        )

        fig.update_layout(
            title=dict(
                text="🌌 电场可视化 - 数据错误",
                x=0.5,
                font=dict(size=24, color=self.design['text_primary'])
            ),
            height=600
        )

        return fig

    def plot_potential(self, solution: FieldSolution, config: Optional[VisualizationConfig] = None) -> go.Figure:
        """宇宙风格3D电位表面"""
        cfg = config if config else self.config

        points = solution['points']
        potentials = solution.get('potentials', np.zeros(len(points)))

        # 创建宇宙风格图形
        fig = go.Figure()
        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        if points.shape[1] == 3:
            # 3D表面图
            try:
                from scipy.interpolate import griddata
                xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 40)
                yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 40)
                XI, YI = np.meshgrid(xi, yi)
                ZI = griddata((points[:, 0], points[:, 1]), potentials, (XI, YI), method='cubic')

                fig.add_trace(go.Surface(
                    x=xi, y=yi, z=ZI,
                    colorscale='Viridis',
                    lighting=dict(ambient=0.4, diffuse=0.8),
                    lightposition=dict(x=100, y=100, z=1000),
                    opacity=0.9
                ))

                # 添加星点背景
                self._add_starfield(fig)

            except:
                # 退化为3D散点图
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=potentials,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=potentials,
                        colorscale='Viridis',
                        opacity=0.7
                    )
                ))

            fig.update_layout(scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='电位 (V)',
                bgcolor=self.design['background']
            ))

        else:
            # 2D散点图
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=potentials,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=1, color='white')
                )
            ))
            fig.update_layout(
                xaxis_title='X (m)',
                yaxis_title='Y (m)'
            )

        fig.update_layout(
            title=dict(
                text="3D电位分布",
                x=0.5,
                font=dict(size=20, color=self.design['text_primary'])
            )
        )

        return fig

    def _adjust_color_brightness(self, color, factor):
        """调整颜色亮度"""
        import re
        # 从rgba字符串中提取rgb值
        rgb_match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', color)
        if rgb_match:
            r, g, b, a = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3)), float(
                rgb_match.group(4))
            # 调整亮度
            r = min(255, max(0, int(r * factor)))
            g = min(255, max(0, int(g * factor)))
            b = min(255, max(0, int(b * factor)))
            return f'rgba({r}, {g}, {b}, {a})'
        return color

    def plot_field_lines(self, solution: FieldSolution, n_lines: int = 20,
                         config: Optional[VisualizationConfig] = None, is_3d: bool = False) -> Any:
        """现代化电场线可视化"""
        # 使用传入的config参数
        cfg = config or {}

        points, vectors, _ = self._preprocess_data(solution)

        # 获取电荷信息
        charges = solution.get('charges', [])

        # 3D模式下增加电场线数量，使可视化更丰富
        if is_3d:
            n_lines = 30

        logger.info(f"开始绘制电场线: 3D模式={is_3d}, 数据维度={points.shape[1]}, 线数={n_lines}")

        # 确保在3D模式下数据维度正确
        if is_3d:
            # 如果输入是2D数据，添加z维度
            if points.shape[1] == 2:
                # 为3D模式添加非零的z分量，使电场线在3D空间中更加立体
                z_column = 0.3 * np.sin(points[:, 0]) * np.cos(points[:, 1]).reshape(-1, 1)
                points = np.hstack([points, z_column])
                logger.info("已将2D点数据转换为3D格式")
            if vectors.shape[1] == 2:
                # 为向量添加z分量，确保3D空间中的电场方向更自然
                z_column = 0.5 * np.random.normal(0, 0.1, (vectors.shape[0], 1))
                vectors = np.hstack([vectors, z_column])
                logger.info("已将2D向量数据转换为3D格式")

        # 计算电场线，传递维度信息和电荷信息
        field_lines = FieldLineCalculator.compute_field_lines(points, vectors, n_lines, is_3d=is_3d, charges=charges)

        # 创建宇宙风格图形 - 使用make_subplots确保支持row/col参数
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        # 绘制电场线 - 霓虹效果
        for i, line in enumerate(field_lines):
            line = np.array(line)
            color_intensity = i / len(field_lines)

            # 霓虹渐变色彩 - 更丰富的颜色变化
            r = int(255 * color_intensity)
            g = int(100 + 155 * (1 - color_intensity))
            b = int(200 + 55 * color_intensity)
            base_color = f'rgba({r}, {g}, {b}, 0.9)'

            # 正确设置z坐标，在3D模式下增强空间效果
            if is_3d:
                # 确保线数据有z维度
                if line.shape[1] >= 3:
                    # 使用实际z值，并添加轻微的空间变化以增强3D效果
                    z_values = line[:, 2] + 0.1 * np.sin(np.linspace(0, 2 * np.pi, len(line)))
                else:
                    # 如果没有z维度，生成有意义的z值而不是微小扰动
                    z_values = 0.3 * np.sin(line[:, 0]) * np.cos(line[:, 1])
                    logger.warning("3D模式下电场线数据缺少z维度，已生成空间分布的z值")
            else:
                # 2D模式仍使用微小扰动
                z_values = np.random.normal(0, 0.01, len(line))

            # 3D模式下基于z轴位置设置不同的亮度，增强深度感
            if is_3d and len(z_values) > 0:
                # 归一化z值以用于颜色亮度调整
                z_min, z_max = np.min(z_values), np.max(z_values)
                if z_max > z_min:
                    # 为每个点计算基于z位置的颜色亮度
                    for j in range(len(line)):
                        norm_z = (z_values[j] - z_min) / (z_max - z_min)
                        # 根据z位置调整颜色亮度，高处更亮
                        brightness_factor = 1.0 + 0.5 * norm_z
                        color = self._adjust_color_brightness(base_color, brightness_factor)

                        # 为每个线段单独绘制，以实现渐变色效果
                        if j > 0:
                            fig.add_trace(go.Scatter3d(
                                x=[line[j - 1, 0], line[j, 0]],
                                y=[line[j - 1, 1], line[j, 1]],
                                z=[z_values[j - 1], z_values[j]],
                                mode='lines',
                                line=dict(
                                    color=color,
                                    width=3.0 + 1.5 * norm_z,  # 基于z位置的线宽变化
                                    dash='solid'
                                ),
                                showlegend=False,
                                hoverinfo='skip'
                            ), row=1, col=1)
                    continue  # 跳过下面的整体线绘制

            # 使用更丰富的线条样式和更高的宽度，增强3D视觉效果
            line_width = 3.5 if not is_3d else 4.0
            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=z_values,
                mode='lines',
                line=dict(
                    color=base_color,
                    width=line_width,  # 增加线宽以增强可见性
                    dash='solid'
                ),
                showlegend=False,
                hoverinfo='skip',
                # 添加发光效果以增强3D视觉效果
                marker=dict(
                    size=0.1,
                    color=base_color,
                    opacity=0.1
                )
            ), row=1, col=1)

        # 添加电荷显示（如果有）
        if is_3d and 'charges' in solution and solution['charges']:
            # 安全筛选正负电荷，增加错误处理
            pos_charges = []
            neg_charges = []
            for c in solution['charges']:
                try:
                    # 安全获取电荷值
                    value = float(c.get('value', 0.0))
                    if not np.isfinite(value):
                        value = 0.0
                    if value > 0:
                        pos_charges.append(c)
                    else:
                        neg_charges.append(c)
                except (TypeError, ValueError):
                    # 如果无法获取电荷值，默认为负电荷
                    neg_charges.append(c)

            if pos_charges:
                # 安全计算电荷大小，避免NaN值
                charge_sizes = []
                for c in pos_charges:
                    try:
                        # 安全获取电荷值并处理可能的NaN
                        value = abs(float(c.get('value', 0.0)))
                        if not np.isfinite(value):
                            value = 0.0
                        charge_sizes.append(15 + 10 * value)
                    except (TypeError, ValueError):
                        charge_sizes.append(15)  # 默认大小
                
                # 安全提取位置信息
                x_positions = []
                y_positions = []
                z_positions = []
                for c in pos_charges:
                    try:
                        pos = c.get('position', [0, 0, 0])
                        x_positions.append(float(pos[0]) if len(pos) > 0 else 0.0)
                        y_positions.append(float(pos[1]) if len(pos) > 1 else 0.0)
                        z_positions.append(float(pos[2]) if len(pos) > 2 else 0.0)
                    except (TypeError, ValueError, IndexError):
                        x_positions.append(0.0)
                        y_positions.append(0.0)
                        z_positions.append(0.0)
                
                fig.add_trace(go.Scatter3d(
                    x=x_positions,
                    y=y_positions,
                    z=z_positions,
                    mode='markers',
                    marker=dict(
                        size=charge_sizes,
                        color=self.design['charge_positive'],
                        symbol='circle',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    name='正电荷'
                ), row=1, col=1)

            if neg_charges:
                # 安全计算电荷大小，避免NaN值
                charge_sizes = []
                for c in neg_charges:
                    try:
                        # 安全获取电荷值并处理可能的NaN
                        value = abs(float(c.get('value', 0.0)))
                        if not np.isfinite(value):
                            value = 0.0
                        charge_sizes.append(15 + 10 * value)
                    except (TypeError, ValueError):
                        charge_sizes.append(15)  # 默认大小
                
                # 安全提取位置信息
                x_positions = []
                y_positions = []
                z_positions = []
                for c in neg_charges:
                    try:
                        pos = c.get('position', [0, 0, 0])
                        x_positions.append(float(pos[0]) if len(pos) > 0 else 0.0)
                        y_positions.append(float(pos[1]) if len(pos) > 1 else 0.0)
                        z_positions.append(float(pos[2]) if len(pos) > 2 else 0.0)
                    except (TypeError, ValueError, IndexError):
                        x_positions.append(0.0)
                        y_positions.append(0.0)
                        z_positions.append(0.0)
                
                fig.add_trace(go.Scatter3d(
                    x=x_positions,
                    y=y_positions,
                    z=z_positions,
                    mode='markers',
                    marker=dict(
                        size=charge_sizes,
                        color=self.design['charge_negative'],
                        symbol='circle',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    name='负电荷'
                ), row=1, col=1)

        # 添加星点背景
        self._add_starfield(fig, row=1, col=1)

        # 3D视图布局优化
        scene_config = dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor=self.design['background'],
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # 设置更合适的视角
            ),
            aspectmode='data'  # 保持数据比例
        )

        if is_3d:
            # 3D模式下使用透视投影和轨道拖动模式
            scene_config['camera']['projection'] = dict(type='perspective')

        fig.update_layout(
            title=dict(
                text=f"电场线可视化 ({n_lines} 条流线) {'3D' if is_3d else '2D'}",
                x=0.5,
                font=dict(size=20, color=self.design['text_primary'])
            ),
            scene=scene_config,
            # 3D模式下增加图形高度以获得更好的显示效果
            height=800 if not is_3d else 900
        )

        return fig

    def plot_boundary_mesh(self, solution: BEMSolution, config: Optional[VisualizationConfig] = None) -> go.Figure:
        """宇宙风格BEM网格"""
        cfg = config if config else self.config

        vertices = solution['vertices']
        triangles = solution['triangles']
        potentials = solution['vertex_potentials']

        # 创建宇宙风格网格
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=potentials if potentials is not None else np.zeros(len(vertices)),
            colorscale='Viridis',
            opacity=0.8,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1),
            lightposition=dict(x=100, y=100, z=1000),
            colorbar=dict(title='电位 (V)', len=0.6)
        )])

        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        # 添加星点背景
        self._add_starfield(fig)

        fig.update_layout(
            title=dict(
                text="🛸 交互式BEM网格",
                x=0.5,
                font=dict(size=20, color=self.design['text_primary'])
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                bgcolor=self.design['background']
            )
        )

        return fig

    def create_animation(self, solution_sequence: list[FieldSolution],
                         config: Optional[VisualizationConfig] = None) -> go.Figure:
        """宇宙风格动画"""
        cfg = config if config else self.config

        frames = []
        design = self.design

        for frame_idx, solution in enumerate(solution_sequence):
            points, vectors, field_strength = self._preprocess_data(solution)

            sample_step = max(1, len(points) // 40)

            frame = go.Frame(
                data=[go.Cone(
                    x=points[::sample_step, 0],
                    y=points[::sample_step, 1],
                    z=points[::sample_step, 2],
                    u=vectors[::sample_step, 0],
                    v=vectors[::sample_step, 1],
                    w=vectors[::sample_step, 2],
                    sizemode="absolute",
                    sizeref=1.5 * np.max(field_strength),
                    colorscale=design['gradient']
                )],
                name=f'frame_{frame_idx}',
                layout=go.Layout(
                    title=dict(
                        text=f'🌌 积分过程 - 帧 {frame_idx + 1}/{len(solution_sequence)}',
                        font=dict(color=design['text_primary'])
                    )
                )
            )
            frames.append(frame)

        # 初始图形
        fig = go.Figure(frames=frames)
        fig = DesignSystem.apply_figure_style(fig, self.config.get('style', 'cosmos'))

        # 添加播放控件
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶️ 播放",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": cfg.get('interval_ms', 250), "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100}
                            }
                        ]
                    },
                    {
                        "label": "⏸️ 暂停",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ],
                "x": 0.1,
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "method": "animate",
                        "args": [[f'frame_{k}'], dict(mode='immediate')],
                        "label": f"帧 {k + 1}"
                    } for k in range(len(frames))
                ],
                "x": 0.1,
                "y": 0,
                "len": 0.8,
                "currentvalue": {
                    "prefix": "进度: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 100}
            }]
        )

        return fig


# ============================================================================ #
# 电场线计算器（现代化算法）- 3D优化版
# ============================================================================ #

class FieldLineCalculator:
    """
    现代化电场线计算算法

    优化特色：
    - 自适应步长控制
    - 智能起点选择
    - 物理精确的追踪
    - 性能优化
    """

    @staticmethod
    def compute_field_lines(
            observation_points: NDArray[np.float64],
            field_vectors: NDArray[np.float64],
            n_lines: int = 20,
            is_3d: bool = False,
            charges: List[dict] = None  # 新增电荷参数
    ) -> List[NDArray]:
        """计算电场线 - 支持电荷模型优化"""
        # 确保输入是numpy数组
        observation_points = np.asarray(observation_points, dtype=np.float64)
        field_vectors = np.asarray(field_vectors, dtype=np.float64)
        
        # 初始化电场线列表
        field_lines = []
        
        # 确保3D模式下正确处理维度
        if is_3d:
            # 验证数据维度
            if observation_points.shape[1] == 2:
                # 如果输入是2D数据，添加z维度并增加空间变化
                z_column = 0.1 * np.random.randn(observation_points.shape[0], 1)  # 添加随机z分量
                observation_points = np.hstack([observation_points, z_column])

            if field_vectors.shape[1] == 2:
                # 为向量也添加z维度并增加垂直分量
                z_column = 0.2 * np.random.randn(field_vectors.shape[0], 1)  # 为向量添加z分量
                field_vectors = np.hstack([field_vectors, z_column])
            else:
                # 如果已经是3D数据，增强z分量的变化
                observation_points[:, 2] += 0.05 * np.random.randn(observation_points.shape[0])
                field_vectors[:, 2] += 0.1 * np.random.randn(field_vectors.shape[0])

        # 根据电荷类型调整场线数量
        charges = charges or []
        is_single_charge = len(charges) == 1
        is_dipole = len(charges) == 2
        
        if is_single_charge or is_dipole:
            # 对于点电荷和电偶极子，增加场线数量以获得更好的效果
            target_lines = n_lines * 3
        else:
            target_lines = n_lines * 2 if is_3d else n_lines
        start_points = FieldLineCalculator._select_start_points(
            observation_points, field_vectors, target_lines, charges
        )
        
        # 根据电荷类型调整最大步数
        if is_single_charge:
            max_steps = 300  # 点电荷需要更多步数以显示辐射特性
        elif is_dipole:
            max_steps = 250  # 电偶极子需要更多步数以形成闭合环
        else:
            # 其他情况，根据电场线数量动态调整最大步数
            base_max_steps = max(50, min(200, 300 - target_lines * 2))
            max_steps = int(base_max_steps * 1.4) if is_3d else base_max_steps  # 3D模式增加步数

        for start in start_points:
            # 3D模式下，为起点添加z轴方向的微小扰动
            if is_3d:
                start = start.copy()
                start[2] += 0.03 * np.random.randn()  # 添加随机z扰动

            line = FieldLineCalculator._trace_field_line(
                start, observation_points, field_vectors, max_steps=max_steps,
                charges=charges, min_field=1e-5  # 添加charges和更低的最小场强阈值
            )

            # 3D模式下更严格的筛选条件
            if len(line) > 5:  # 要求更长的线
                if is_3d:
                    # 检查线是否有足够的3D展开度
                    line_array = np.array(line)
                    z_range = np.max(line_array[:, 2]) - np.min(line_array[:, 2])
                    if z_range > 0.1:  # 确保z方向有足够的变化
                        field_lines.append(np.array(line))
                else:
                    field_lines.append(np.array(line))

            # 3D模式限制总数，避免过度拥挤
            if is_3d and len(field_lines) >= n_lines:
                break

        return field_lines

    @staticmethod
    def _select_start_points(points: NDArray, vectors: NDArray, n_points: int,
                             charges: List[dict] = None) -> List[NDArray]:
        """智能起点选择算法 - 根据电荷模型优化"""

        # 如果有电荷信息，优先基于电荷物理特性选择起点
        if charges and len(charges) > 0:
            start_points = []

            # 分析电荷模型类型
            charge_positions = []
            charge_values = []
            
            # 处理不同形式的电荷数据
            for c in charges:
                if isinstance(c, dict):
                    charge_positions.append(c.get('position', (0, 0, 0)))
                    charge_values.append(c.get('value', 0.0))
                else:
                    # 处理Charge对象
                    charge_positions.append(getattr(c, 'position', (0, 0, 0)))
                    # 尝试获取value属性，失败则尝试charge属性
                    charge_value = getattr(c, 'value', None)
                    if charge_value is None:
                        charge_value = getattr(c, 'charge', 0.0)
                    charge_values.append(charge_value)

            # 判断是点电荷还是电偶极子
            if len(charges) == 1:
                # 单点电荷 - 从电荷位置向外辐射状发射
                # 将位置从元组转换为numpy数组
                charge_pos = np.array(charge_positions[0])
                is_3d = len(charge_pos) > 2
                
                # 优化的球面分布算法
                start_points = []
                n_theta = int(np.sqrt(n_points))
                n_phi = n_points // n_theta + (1 if n_points % n_theta > 0 else 0)
                
                theta = np.linspace(0, np.pi, n_theta, endpoint=False)
                phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
                
                for t in theta:
                    for p in phi:
                        if len(start_points) >= n_points:
                            break
                        r = 0.1  # 起始半径
                        # 计算球面上的点
                        x = charge_pos[0] + r * np.sin(t) * np.cos(p)
                        y = charge_pos[1] + r * np.sin(t) * np.sin(p)
                        # 处理3D/2D情况
                        if is_3d:
                            z = charge_pos[2] + r * np.cos(t)
                            start_points.append([x, y, z])
                        else:
                            start_points.append([x, y])
                
                # 如果点数不够，添加随机点补充
                while len(start_points) < n_points:
                    r = 0.1
                    t = np.random.uniform(0, np.pi)
                    p = np.random.uniform(0, 2 * np.pi)
                    x = charge_pos[0] + r * np.sin(t) * np.cos(p)
                    y = charge_pos[1] + r * np.sin(t) * np.sin(p)
                    if is_3d:
                        z = charge_pos[2] + r * np.cos(t)
                        start_points.append([x, y, z])
                    else:
                        start_points.append([x, y])
                
                # 确保返回的每个点都是numpy数组
                return [np.array(point, dtype=np.float64) for point in start_points]

            elif len(charges) == 2 and abs(sum(charge_values)) < 1e-10:
                # 电偶极子 - 从正电荷出发，向负电荷方向集中
                pos_charge = None
                neg_charge = None

                for i, charge in enumerate(charges):
                    if charge_values[i] > 0:
                        pos_charge = i
                    else:
                        neg_charge = i

                if pos_charge is not None and neg_charge is not None:
                    # 将位置从元组转换为numpy数组
                    pos_pos = np.array(charge_positions[pos_charge])
                    neg_pos = np.array(charge_positions[neg_charge])
                    is_3d = len(pos_pos) > 2

                    # 改进的电偶极子起点生成算法
                    start_points = []
                    
                    # 计算偶极子方向
                    dipole_dir = neg_pos - pos_pos
                    norm = np.linalg.norm(dipole_dir)
                    if norm > 1e-10:
                        dipole_dir = dipole_dir / norm
                    else:
                        dipole_dir = np.array([1, 0, 0]) if is_3d else np.array([1, 0])
                    
                    # 生成垂直于偶极子方向的单位向量
                    if is_3d:
                        # 找到一个垂直于dipole_dir的向量
                        if abs(dipole_dir[0]) < 0.9:  # 如果dipole_dir不是太接近x轴
                            perp1 = np.array([0, -dipole_dir[2], dipole_dir[1]])
                        else:
                            perp1 = np.array([-dipole_dir[2], 0, dipole_dir[0]])
                        perp1 = perp1 / np.linalg.norm(perp1)
                        perp2 = np.cross(dipole_dir, perp1)
                    else:
                        # 2D情况
                        perp1 = np.array([-dipole_dir[1], dipole_dir[0]])
                    
                    # 分两部分生成起点：从正电荷和从负电荷
                    # 1. 从正电荷出发
                    for i in range(n_points // 2):
                        # 在正电荷周围半球面分布，主要朝向负电荷方向
                        # 添加更多随机性以创建更自然的电场线分布
                        spread = 0.4  # 角度分散度
                        r = 0.1 + 0.05 * np.random.random()  # 略微变化的半径
                        
                        # 生成球坐标角度，偏向偶极子方向
                        theta = np.random.uniform(0, spread)
                        phi = np.random.uniform(0, 2 * np.pi)
                        
                        # 转换为笛卡尔坐标系
                        if is_3d:
                            # 使用球坐标系生成偏离偶极子方向的向量
                            dir_vec = (np.cos(theta) * dipole_dir +
                                      np.sin(theta) * np.cos(phi) * perp1 +
                                      np.sin(theta) * np.sin(phi) * perp2)
                        else:
                            # 2D情况
                            dir_vec = (np.cos(theta) * dipole_dir +
                                      np.sin(theta) * perp1)
                        
                        # 确保方向向量归一化
                        dir_vec = dir_vec / np.linalg.norm(dir_vec)
                        
                        # 生成起点
                        start_point = pos_pos + r * dir_vec
                        start_points.append(start_point.tolist())
                    
                    # 2. 从负电荷出发（可选，但有助于形成闭合环）
                    for i in range(n_points - len(start_points)):
                        r = 0.1 + 0.05 * np.random.random()
                        # 从负电荷出发，远离正电荷方向
                        theta = np.random.uniform(0, np.pi/2)
                        phi = np.random.uniform(0, 2 * np.pi)
                        
                        if is_3d:
                            dir_vec = (-np.cos(theta) * dipole_dir +
                                      np.sin(theta) * np.cos(phi) * perp1 +
                                      np.sin(theta) * np.sin(phi) * perp2)
                        else:
                            dir_vec = (-np.cos(theta) * dipole_dir +
                                      np.sin(theta) * perp1)
                        
                        dir_vec = dir_vec / np.linalg.norm(dir_vec)
                        start_point = neg_pos + r * dir_vec
                        start_points.append(start_point.tolist())
                    
                    # 确保返回的每个点都是numpy数组
                    return [np.array(point, dtype=np.float64) for point in start_points]

        # 如果没有电荷信息或不是特殊模型，使用原来的场强选择方法
        field_strength = np.linalg.norm(vectors, axis=1)
        strength_threshold = np.percentile(field_strength, 80)
        high_strength_indices = np.where(field_strength > strength_threshold)[0]

        if len(high_strength_indices) < n_points:
            high_strength_indices = np.argsort(field_strength)[-n_points * 3:]

        # 空间均匀分布
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(n_points, len(high_strength_indices))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(points[high_strength_indices])
            return [center for center in kmeans.cluster_centers_]
        except ImportError:
            selected_indices = np.random.choice(high_strength_indices, n_points, replace=False)
            return [points[i] for i in selected_indices]

    @staticmethod
    def _trace_field_line(
            start: NDArray,
            grid_points: NDArray,
            field_vectors: NDArray,
            max_steps: int = 150,
            min_field: float = 1e-4,
            charges: List = None  # 添加charges参数以支持特殊电场优化
    ) -> List[NDArray]:
        """自适应电场线追踪（优化版）- 增强点电荷和电偶极子支持"""
        # 确保start是numpy数组
        start = np.asarray(start, dtype=np.float64)
        line = [start]
        current = start.copy()

        # 检查是否为3D空间
        is_3d = len(start) == 3
        
        # 分析电荷模型类型（用于特殊优化）
        charges = charges or []
        is_single_charge = len(charges) == 1
        is_dipole = len(charges) == 2
        charge_positions = []
        charge_values = []
        
        # 处理不同形式的电荷数据
        for c in charges:
            if isinstance(c, dict):
                if 'position' in c:
                    charge_positions.append(np.array(c['position']))
                    charge_values.append(c.get('value', 0.0))
            else:
                pos = getattr(c, 'position', None)
                if pos is not None:
                    charge_positions.append(np.array(pos))
                    # 尝试获取value属性，失败则尝试charge属性
                    charge_value = getattr(c, 'value', None)
                    if charge_value is None:
                        charge_value = getattr(c, 'charge', 0.0)
                    charge_values.append(charge_value)

        # 针对不同电荷类型的特殊参数设置
        if is_single_charge:
            # 点电荷电场线优化
            base_step = 0.15  # 更大的基础步长
            min_step = 0.005
            max_step = 0.4
            current_max_steps = int(max_steps * 1.5)
            use_log_step = True
            min_field = 1e-5  # 降低最小场强阈值以延长电场线
        elif is_dipole:
            # 电偶极子电场线优化
            base_step = 0.12
            min_step = 0.008
            max_step = 0.35
            current_max_steps = int(max_steps * 1.3)
            use_log_step = True
            min_field = 5e-5
        else:
            # 一般情况
            base_step = 0.12 if is_3d else 0.08  # 3D空间使用更大的基础步长
            min_step = 0.008
            max_step = 0.35 if is_3d else 0.25  # 3D空间允许更大的步长范围
            current_max_steps = max_steps if not is_3d else int(max_steps * 1.4)
            use_log_step = False

        # 预先计算网格点的KDTree以加速最近邻搜索
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(grid_points)
            use_kdtree = True
        except ImportError:
            use_kdtree = False

        # 3D模式特殊参数
        if is_3d:
            spatial_variation = 0.03  # 减少随机性以保持电场线质量
            max_distance_from_start = 15
        else:
            spatial_variation = 0.0
            max_distance_from_start = 8
        
        # 对于点电荷，增加最大距离限制
        if is_single_charge and charge_positions:
            dist_to_charge = np.linalg.norm(current - charge_positions[0])
            max_distance_from_start = max(max_distance_from_start, dist_to_charge * 20)

        prev_dir = None
        curvature_count = 0  # 记录曲率突变次数
        
        for step in range(current_max_steps):
            # 当前场强和方向
            current_array = np.asarray(current, dtype=np.float64)
            E = FieldLineCalculator._interpolate_field(current_array, grid_points, field_vectors, use_kdtree=use_kdtree)
            E_mag = np.linalg.norm(E)

            if E_mag < min_field:
                break

            direction = E / E_mag
            
            # 检测电场方向突变
            if prev_dir is not None:
                cos_angle = np.dot(direction, prev_dir)
                if cos_angle < -0.8:  # 方向突变超过150度
                    curvature_count += 1
                    if curvature_count > 3:  # 限制曲率突变次数
                        break
            
            # 对于点电荷，确保电场线正确向外辐射
            if is_single_charge and charge_positions:
                charge_dir = current - charge_positions[0]
                charge_dir_norm = np.linalg.norm(charge_dir)
                if charge_dir_norm > 1e-10:
                    charge_dir = charge_dir / charge_dir_norm
                    # 确保场线方向与径向方向夹角不超过45度
                    cos_angle = np.dot(direction, charge_dir)
                    if cos_angle < np.cos(np.pi/4):  # 45度
                        # 修正方向，使其更符合点电荷的径向特性
                        direction = 0.7 * direction + 0.3 * charge_dir
                        direction = direction / np.linalg.norm(direction)
            
            # 对于电偶极子，优化场线方向
            elif is_dipole and charge_positions:
                # 计算偶极子轴线方向
                dipole_axis = charge_positions[1] - charge_positions[0]
                axis_norm = np.linalg.norm(dipole_axis)
                if axis_norm > 1e-10:
                    dipole_axis = dipole_axis / axis_norm
                    
                    # 对于电偶极子，适当引导场线方向
                    # 避免场线过早终止
                    if step > 20 and E_mag < 1e-3:
                        # 当接近弱场区时，引导场线朝向相反电荷
                        # 安全检查：确保charge_positions和charge_values都有足够的元素
                        if charge_positions and len(charge_positions) >= 2 and charge_values and len(charge_values) >= 2:
                            try:
                                # 判断应该朝向哪个电荷
                                current_charge_idx = 0 if np.linalg.norm(current - charge_positions[0]) < np.linalg.norm(current - charge_positions[1]) else 1
                                target_charge_idx = 1 - current_charge_idx
                                target_dir = charge_positions[target_charge_idx] - current
                                target_dir_norm = np.linalg.norm(target_dir)
                                if target_dir_norm > 1e-10:
                                    target_dir = target_dir / target_dir_norm
                                    # 适度混合方向
                                    direction = 0.8 * direction + 0.2 * target_dir
                                    direction = direction / np.linalg.norm(direction)
                            except Exception:
                                # 如果出现任何错误，静默处理，继续使用原方向
                                pass
            
            prev_dir = direction.copy()

            # 自适应步长：场强越大，步长越小
            if use_log_step:
                # 对数步长调整更适合点电荷和偶极子
                adaptive_step = np.clip(base_step / (1 + 0.5 * np.log10(E_mag + 1)), min_step, max_step)
                
                # 随着远离起点，步长适度增大
                dist_from_start = np.linalg.norm(current - start)
                if dist_from_start > 0.5:
                    adaptive_step *= 1.0 + 0.1 * np.log1p(dist_from_start)
            elif is_3d:
                # 3D空间使用更激进的自适应步长策略
                adaptive_step = np.clip(base_step / (1 + 0.3 * np.log10(E_mag + 1)), min_step, max_step)

                # 在3D模式下，添加空间随机性
                if step % 4 == 0:  # 减少添加随机性的频率
                    spatial_perturbation = spatial_variation * np.random.uniform(-1, 1, 3)
                    direction = direction + spatial_perturbation
                    direction = direction / np.linalg.norm(direction)
            else:
                adaptive_step = np.clip(base_step / (1 + np.log10(E_mag + 1)), min_step, max_step)

            # 3D模式稳定性优化
            if is_3d and step % 3 == 0:
                adaptive_step *= 1.05  # 更小的步长变化

            next_point = current + direction * adaptive_step

            # 提前终止条件
            if np.any(np.isnan(next_point)) or np.linalg.norm(next_point - start) > max_distance_from_start:
                break

            # 检查是否接近电荷或场强过大的区域
            if E_mag > 1e6:
                break
            
            # 对于点电荷，避免场线过于接近电荷（可能导致数值不稳定）
            if is_single_charge and charge_positions:
                dist_to_charge = np.linalg.norm(next_point - charge_positions[0])
                if dist_to_charge < 0.01:  # 防止场线进入电荷内部
                    break

            # 检查是否形成闭环或陷入循环
            if step > 10:
                # 更高效的循环检测
                if len(line) > 15:
                    # 只检查每隔几个点
                    check_interval = max(1, len(line) // 8)
                    for i in range(0, len(line) - 5, check_interval):
                        # 确保line[i]是numpy数组
                        if np.linalg.norm(np.array(line[i]) - next_point) < 0.06:  # 增大阈值避免误判
                            break
                else:
                    recent_points = np.array([np.array(p) for p in line[-5:]])
                    distances = np.linalg.norm(recent_points - next_point, axis=1)
                    loop_threshold = 0.04 if is_3d else 0.06  # 增大阈值
                    if np.any(distances < loop_threshold):
                        break

            line.append(next_point)
            current = next_point
        
        # 对于点电荷，确保电场线足够长以显示辐射特性
        if is_single_charge and len(line) < 50 and len(line) > 10:
            # 如果线太短，适度延长
            last_point = np.array(line[-1])
            if charge_positions:
                charge_dir = last_point - np.array(charge_positions[0])
                charge_dir_norm = np.linalg.norm(charge_dir)
                if charge_dir_norm > 1e-10:
                    charge_dir = charge_dir / charge_dir_norm
                    # 添加额外的点以延长电场线
                    for i in range(10):
                        extended_point = last_point + 0.1 * charge_dir * (i + 1)
                        line.append(extended_point)
        
        # 对于电偶极子，确保场线有合理的长度
        elif is_dipole and len(line) < 30 and len(line) > 5:
            # 如果电偶极子场线太短，尝试延长
            last_point = np.array(line[-1])
            # 向远离起点的方向延长
            reference_point = np.array(line[max(0, len(line)-5)])
            end_dir = last_point - reference_point
            end_dir_norm = np.linalg.norm(end_dir)
            if end_dir_norm > 1e-10:
                end_dir = end_dir / end_dir_norm
                for i in range(5):
                    extended_point = last_point + 0.15 * end_dir * (i + 1)
                    line.append(extended_point)

        return line

    @staticmethod
    def _interpolate_field(query_point: NDArray, grid_points: NDArray, field_vectors: NDArray,
                           use_kdtree: bool = False) -> NDArray:
        """优化的场插值算法"""
        # 确保query_point是numpy数组
        query_point = np.asarray(query_point, dtype=np.float64)
        # 确保grid_points和field_vectors是numpy数组
        grid_points = np.asarray(grid_points, dtype=np.float64)
        field_vectors = np.asarray(field_vectors, dtype=np.float64)
        
        if use_kdtree:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(grid_points)
                n_neighbors = min(4, len(grid_points))  # 减少邻居数量以提高速度
                distances, indices = tree.query(query_point, k=n_neighbors)

                # 避免除零
                distances = np.maximum(distances, 1e-8)

                # 反距离加权
                weights = 1.0 / distances ** 2
                weighted_vectors = field_vectors[indices] * weights[:, np.newaxis]

                return np.sum(weighted_vectors, axis=0) / np.sum(weights)
            except Exception:
                # 如果KDTree失败，回退到原始方法
                pass

        # 原始方法的优化版本
        # 只计算与查询点较近的区域内的点
        # 首先估算一个合理的搜索半径
        if len(grid_points) > 100:
            # 对于大网格，使用更高效的方法
            # 计算网格点的平均间距作为初始搜索半径
            if grid_points.shape[1] == 2:  # 2D情况
                x_min, y_min = np.min(grid_points, axis=0)
                x_max, y_max = np.max(grid_points, axis=0)
                avg_spacing = np.sqrt((x_max - x_min) * (y_max - y_min) / len(grid_points))
            else:  # 3D情况
                x_min, y_min, z_min = np.min(grid_points, axis=0)
                x_max, y_max, z_max = np.max(grid_points, axis=0)
                avg_spacing = ((x_max - x_min) * (y_max - y_min) * (z_max - z_min) / len(grid_points)) ** (1 / 3)

            search_radius = avg_spacing * 5

            # 过滤出搜索半径内的点
            if grid_points.shape[1] == 2:
                mask = np.logical_and(
                    np.abs(grid_points[:, 0] - query_point[0]) < search_radius,
                    np.abs(grid_points[:, 1] - query_point[1]) < search_radius
                )
            else:
                mask = np.logical_and.reduce([
                    np.abs(grid_points[:, i] - query_point[i]) < search_radius
                    for i in range(min(3, grid_points.shape[1]))
                ])

            nearby_points = grid_points[mask]
            nearby_vectors = field_vectors[mask]

            if len(nearby_points) == 0:
                # 如果没有点在搜索半径内，使用所有点
                nearby_points = grid_points
                nearby_vectors = field_vectors

            distances = np.linalg.norm(nearby_points - query_point, axis=1)

            # 选择最近的点
            n_neighbors = min(4, len(nearby_points))  # 减少邻居数量
            nearest_indices = np.argpartition(distances, n_neighbors)[:n_neighbors]
            nearest_distances = distances[nearest_indices]

            # 避免除零
            nearest_distances = np.maximum(nearest_distances, 1e-8)

            # 反距离加权
            weights = 1.0 / nearest_distances ** 2
            weighted_vectors = nearby_vectors[nearest_indices] * weights[:, np.newaxis]

            return np.sum(weighted_vectors, axis=0) / np.sum(weights)
        else:
            # 对于小网格，使用简化的原始方法
            distances = np.linalg.norm(grid_points - query_point, axis=1)
            n_neighbors = min(4, len(grid_points))
            nearest_indices = np.argpartition(distances, n_neighbors)[:n_neighbors]
            nearest_distances = distances[nearest_indices]
            nearest_distances = np.maximum(nearest_distances, 1e-8)
            weights = 1.0 / nearest_distances ** 2
            weighted_vectors = field_vectors[nearest_indices] * weights[:, np.newaxis]

            return np.sum(weighted_vectors, axis=0) / np.sum(weights)


# ============================================================================ #
# 单元测试
# ============================================================================ #

def test_design_system():
    """测试设计系统"""
    apple_style = DesignSystem.get_style('apple')
    cosmos_style = DesignSystem.get_style('cosmos')

    assert apple_style['background'] == '#F5F7FA'
    assert cosmos_style['background'] == '#0A0A1A'
    assert cosmos_style['starfield'] == True

    logger.info("设计系统测试通过")
    print("设计系统测试通过")


def test_backend_factory():
    """测试后端工厂方法"""
    config = {'backend': 'matplotlib', 'style': 'apple'}
    backend = VisualizationBackend.create(config)
    assert isinstance(backend, MatplotlibBackend)

    config = {'backend': 'plotly', 'style': 'cosmos'}
    backend = VisualizationBackend.create(config)
    assert isinstance(backend, PlotlyBackend)

    logger.info("后端工厂测试通过")
    print("后端工厂测试通过")


def test_field_line_calculator():
    """测试电场线计算"""
    points = np.random.uniform(-2, 2, (500, 2))
    vectors = -points / (np.linalg.norm(points, axis=1, keepdims=True) ** 3 + 1e-6)

    lines = FieldLineCalculator.compute_field_lines(points, vectors, n_lines=10)

    assert len(lines) > 0, "应生成至少一条电场线"
    assert all(len(line) > 2 for line in lines), "电场线应有足够点数"

    logger.info("电场线计算测试通过")
    print("电场线计算测试通过")


def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行现代化可视化后端单元测试")

    test_design_system()
    test_backend_factory()
    test_field_line_calculator()

    logger.info("所有现代化可视化后端单元测试通过!")
    print("所有现代化可视化后端单元测试通过!")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_all_tests()
    elif "--design" in sys.argv:
        test_design_system()
    elif "--lines" in sys.argv:
        test_field_line_calculator()
    else:
        print(__doc__)
        print("\n运行测试:")
        print("  python backends.py --test      # 全部测试")
        print("  python backends.py --design    # 设计系统测试")
        print("  python backends.py --lines     # 电场线测试")