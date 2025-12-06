# visualization/main.py
"""
可视化主接口模块
提供四个核心功能的统一调用接口
"""
import numpy as np
from typing import List, Optional, Tuple
from core.field_calculator import FieldCalculator
from core.potential_calculator import PotentialCalculator
from core.field_line_tracer import AdaptiveFieldLineTracer

from visualization.charge_distribution_3d import ChargeDistributionPlot3D
from visualization.potential_distribution_3d import PotentialDistributionPlot3D
from visualization.field_lines_2d import AppleStyleFieldLines2D
from visualization.field_lines_3d import CosmosFieldLines3D


class ElectricFieldVisualizer:
    """
    静电场可视化统一接口
    
    提供四个核心可视化功能：
    1. 电荷分布图3D (宇宙风格)
    2. 电势分布图3D (宇宙风格) 
    3. 电场线分布图2D (苹果风格)
    4. 电场线分布图3D (宇宙风格)
    """
    
    def __init__(self, charges: List):
        """
        初始化可视化器
        
        Args:
            charges: 电荷对象列表
        """
        self.charges = charges
        
        # 初始化计算器
        self.field_calculator = FieldCalculator(charges)
        self.potential_calculator = PotentialCalculator(charges)
        
        # 初始化追踪器
        self.field_line_tracer = AdaptiveFieldLineTracer(
            charges, 
            min_step=1e-4,
            max_step=0.1,
            max_iter=5000,
            term_field=1e-6,
            boundary_radius=10.0,
            dim=3
        )
        
        # 缓存电场线
        self._field_lines_3d = None
        self._field_lines_2d = None
    
    def generate_field_lines(self, force_rerun: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        生成电场线（3D和2D）
        
        Returns:
            (field_lines_3d, field_lines_2d)
        """
        if self._field_lines_3d is None or force_rerun:
            print("正在生成3D电场线...")
            self._field_lines_3d = self.field_line_tracer.trace_all_field_lines()
            
            print("正在生成2D电场线...")
            tracer_2d = AdaptiveFieldLineTracer(
                self.charges,
                dim=2,
                max_iter=3000
            )
            self._field_lines_2d = tracer_2d.trace_all_field_lines()
        
        return self._field_lines_3d, self._field_lines_2d
    
    def plot_charge_distribution_3d(self, 
                                   title: str = "宇宙电荷分布",
                                   show_electric_field: bool = False,
                                   save_html: Optional[str] = None):
        """
        绘制电荷分布图3D
        
        Args:
            title: 图表标题
            show_electric_field: 是否显示电场矢量
            save_html: HTML保存路径（可选）
        """
        plotter = ChargeDistributionPlot3D(
            self.charges,
            title=title,
            show_grid=True,
            size_scale=1.0
        )
        
        fig = plotter.plot()
        
        if show_electric_field:
            # 创建电场箭头网格
            x_range = np.linspace(-2, 2, 8)
            y_range = np.linspace(-2, 2, 8)
            z_range = np.linspace(-1, 1, 5)
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            
            fig = plotter.add_electric_field_arrows(
                fig, grid_points, self.field_calculator
            )
        
        if save_html:
            fig.write_html(save_html)
        
        fig.show()
        
        return fig
    
    def plot_potential_distribution_3d(self,
                                      grid_bounds: Tuple = (-2, 2, -2, 2, -1, 1),
                                      resolution: int = 30,
                                      title: str = "宇宙电势分布",
                                      save_html: Optional[str] = None):
        """
        绘制电势分布图3D
        
        Args:
            grid_bounds: 网格边界 (x_min, x_max, y_min, y_max, z_min, z_max)
            resolution: 网格分辨率
            title: 图表标题
            save_html: HTML保存路径
        """
        plotter = PotentialDistributionPlot3D(
            self.potential_calculator,
            grid_bounds=grid_bounds,
            grid_resolution=resolution,
            show_isosurfaces=True,
            show_slices=False
        )
        
        fig = plotter.plot(title=title)
        
        if save_html:
            fig.write_html(save_html)
        
        fig.show()
        
        return fig
    
    def plot_field_lines_2d(self,
                           bounds: Tuple = (-3, 3, -3, 3),
                           title: str = "电场线分布 (2D)",
                           save_path: Optional[str] = None):
        """
        绘制电场线分布图2D
        
        Args:
            bounds: 绘图边界 (x_min, x_max, y_min, y_max)
            title: 图表标题
            save_path: 图片保存路径
        """
        # 生成2D电场线
        _, field_lines_2d = self.generate_field_lines()
        
        if not field_lines_2d:
            print("警告：未生成电场线")
            return None
        
        plotter = AppleStyleFieldLines2D(
            field_lines_2d,
            self.charges,
            bounds=bounds,
            line_width=1.5,
            arrow_scale=1.0,
            color_by_field_strength=True
        )
        
        fig = plotter.plot(
            self.field_calculator,
            title=title,
            save_path=save_path
        )
        
        return fig
    
    def plot_field_lines_3d(self,
                           title: str = "宇宙电场线",
                           show_arrows: bool = True,
                           show_animation: bool = True,
                           save_html: Optional[str] = None):
        """
        绘制电场线分布图3D
        
        Args:
            title: 图表标题
            show_arrows: 是否显示箭头
            show_animation: 是否显示动画
            save_html: HTML保存路径
        """
        # 生成3D电场线
        field_lines_3d, _ = self.generate_field_lines()
        
        if not field_lines_3d:
            print("警告：未生成电场线")
            return None
        
        plotter = CosmosFieldLines3D(
            field_lines_3d,
            self.charges,
            show_arrows=show_arrows,
            arrow_density=0.2,
            tube_radius=0.02,
            opacity=0.7,
            color_by_potential=True
        )
        
        fig = plotter.plot(
            potential_calculator=self.potential_calculator,
            title=title,
            show_animation=show_animation
        )
        
        if save_html:
            fig.write_html(save_html)
        
        fig.show()
        
        return fig
    
    def plot_all(self, save_dir: str = "./visualizations"):
        """
        一键生成所有可视化
        
        Args:
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*60)
        print("开始生成静电场可视化...")
        print("="*60)
        
        # 1. 电荷分布图
        print("\n1. 生成电荷分布图3D...")
        fig1 = self.plot_charge_distribution_3d(
            title="静电场 - 电荷分布",
            show_electric_field=False,
            save_html=f"{save_dir}/charge_distribution.html"
        )
        
        # 2. 电势分布图
        print("\n2. 生成电势分布图3D...")
        fig2 = self.plot_potential_distribution_3d(
            grid_bounds=(-2, 2, -2, 2, -1, 1),
            resolution=25,
            title="静电场 - 电势分布",
            save_html=f"{save_dir}/potential_distribution.html"
        )
        
        # 3. 2D电场线图
        print("\n3. 生成电场线分布图2D...")
        fig3 = self.plot_field_lines_2d(
            bounds=(-3, 3, -3, 3),
            title="静电场 - 电场线分布 (2D)",
            save_path=f"{save_dir}/field_lines_2d.png"
        )
        
        # 4. 3D电场线图
        print("\n4. 生成电场线分布图3D...")
        fig4 = self.plot_field_lines_3d(
            title="静电场 - 电场线分布 (3D)",
            show_arrows=True,
            show_animation=True,
            save_html=f"{save_dir}/field_lines_3d.html"
        )
        
        print("\n" + "="*60)
        print(f"可视化生成完成！文件已保存至: {save_dir}")
        print("="*60)
        
        return fig1, fig2, fig3, fig4