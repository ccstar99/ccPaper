# visualization/potential_distribution_3d.py
import plotly.graph_objects as go
import numpy as np
from typing import Tuple
from core.potential_calculator import PotentialCalculator

class PotentialDistributionPlot3D:
    """
    宇宙风格3D电势分布可视化
    
    设计特性：
        - 电势等值面 + 颜色映射
        - 对数刻度增强对比
        - 交互式电势探测
        - 梯度箭头显示
    """
    
    COSMOS_COLORS = {
        'high_potential': '#ff3366',  # 高电势（红色）
        'low_potential': '#3366ff',   # 低电势（蓝色）
        'zero_potential': '#9d4edd',  # 零电势（紫色）
        'background': '#0b0b1f',
        'surface': '#1e2a47'
    }
    
    def __init__(self, potential_calculator: PotentialCalculator,
                 grid_bounds: Tuple[float, float, float, float, float, float] = (-2, 2, -2, 2, -1, 1),
                 grid_resolution: int = 30,
                 show_isosurfaces: bool = True,
                 show_slices: bool = False):
        """
        Args:
            grid_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
            grid_resolution: 每个维度的网格点数
        """
        self.calc = potential_calculator
        self.grid_bounds = grid_bounds
        self.grid_resolution = grid_resolution
        self.show_isosurfaces = show_isosurfaces
        self.show_slices = show_slices
        
    def _create_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """创建计算网格"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.grid_bounds
        
        x = np.linspace(x_min, x_max, self.grid_resolution)
        y = np.linspace(y_min, y_max, self.grid_resolution)
        z = np.linspace(z_min, z_max, self.grid_resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return X, Y, Z, np.stack((X, Y, Z), axis=-1)
    
    def calculate_potential_grid(self) -> np.ndarray:
        """计算网格上的电势值"""
        X, Y, Z, grid_points = self._create_grid()
        
        # 重塑为二维数组用于批量计算
        points_flat = grid_points.reshape(-1, 3)
        V_flat = self.calc.potential(points_flat, absolute=True)
        
        # 重塑回网格形状
        V_grid = V_flat.reshape(X.shape)
        return V_grid
    
    def create_isosurfaces(self, V_grid: np.ndarray, 
                          num_isosurfaces: int = 8) -> List[go.Isosurface]:
        """创建等势面"""
        V_min = np.min(V_grid)
        V_max = np.max(V_grid)
        
        # 使用对数刻度选择等值面（处理数量级差异）
        if V_min > 0 and V_max > 0:
            # 全正电势
            levels = np.logspace(np.log10(V_min+1e-12), np.log10(V_max), num_isosurfaces)
        elif V_min < 0 and V_max < 0:
            # 全负电势
            levels = -np.logspace(np.log10(-V_min+1e-12), np.log10(-V_max), num_isosurfaces)
        else:
            # 混合正负电势
            pos_levels = np.linspace(0, V_max, num_isosurfaces//2) if V_max > 0 else []
            neg_levels = np.linspace(V_min, 0, num_isosurfaces//2) if V_min < 0 else []
            levels = np.concatenate([neg_levels, pos_levels])
        
        surfaces = []
        for i, value in enumerate(levels):
            # 根据电势值选择颜色
            if value > 0:
                color = self._value_to_color(value, V_max, is_positive=True)
            else:
                color = self._value_to_color(abs(value), abs(V_min), is_positive=False)
            
            surface = go.Isosurface(
                x=self.X.flatten(),
                y=self.Y.flatten(),
                z=self.Z.flatten(),
                value=V_grid.flatten(),
                isomin=value,
                isomax=value,
                surface_count=1,
                colorscale=[[0, color], [1, color]],
                opacity=0.4,
                showscale=False,
                name=f'电势 = {value:.2e} V',
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
            surfaces.append(surface)
        
        return surfaces
    
    def create_potential_slice(self, V_grid: np.ndarray, 
                              slice_axis: str = 'z',
                              slice_value: float = 0.0) -> go.Surface:
        """创建电势切片"""
        x_min, x_max, y_min, y_max, z_min, z_max = self.grid_bounds
        
        if slice_axis == 'z':
            # 找到最接近slice_value的z索引
            z_values = np.linspace(z_min, z_max, self.grid_resolution)
            slice_idx = np.argmin(np.abs(z_values - slice_value))
            
            slice_grid = V_grid[:, :, slice_idx]
            X_slice, Y_slice = np.meshgrid(
                np.linspace(x_min, x_max, self.grid_resolution),
                np.linspace(y_min, y_max, self.grid_resolution)
            )
            Z_slice = np.full_like(X_slice, slice_value)
            
            surface = go.Surface(
                x=X_slice,
                y=Y_slice,
                z=Z_slice,
                surfacecolor=slice_grid.T,
                colorscale='RdBu_r',  # 红蓝渐变（红为正，蓝为负）
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title="电势 (V)",
                    titleside='right',
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ),
                name=f'电势切片 (z={slice_value:.2f}m)',
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="white",
                        project=dict(z=True)
                    )
                )
            )
            
        return surface
    
    def _value_to_color(self, value: float, max_value: float, 
                       is_positive: bool = True) -> str:
        """将电势值映射到颜色"""
        if max_value < 1e-12:
            return self.COSMOS_COLORS['zero_potential']
        
        normalized = value / max_value
        
        if is_positive:
            # 红色系渐变
            r = int(255 * normalized)
            g = int(51 * (1 - normalized))
            b = int(102 * (1 - normalized))
        else:
            # 蓝色系渐变
            r = int(51 * (1 - normalized))
            g = int(102 * (1 - normalized))
            b = int(255 * normalized)
        
        return f'rgb({r},{g},{b})'
    
    def create_gradient_arrows(self, V_grid: np.ndarray, 
                              arrow_density: float = 0.2) -> go.Cone:
        """创建电势梯度箭头（电场方向）"""
        # 计算梯度（电场方向）
        dV_dx, dV_dy, dV_dz = np.gradient(V_grid)
        
        # 下采样以减少箭头数量
        step = int(1 / arrow_density)
        indices = np.arange(0, self.grid_resolution, step)
        
        # 创建箭头位置和方向
        positions = []
        directions = []
        
        for i in indices:
            for j in indices:
                for k in indices:
                    pos = np.array([
                        self.X[i, j, k],
                        self.Y[i, j, k],
                        self.Z[i, j, k]
                    ])
                    
                    grad = np.array([
                        dV_dx[i, j, k],
                        dV_dy[i, j, k],
                        dV_dz[i, j, k]
                    ])
                    
                    grad_mag = np.linalg.norm(grad)
                    if grad_mag > 1e-6:
                        # 电场方向 = -梯度方向
                        E_dir = -grad / grad_mag
                        
                        positions.append(pos)
                        directions.append(E_dir)
        
        positions = np.array(positions)
        directions = np.array(directions)
        
        if len(positions) == 0:
            return None
        
        gradient_trace = go.Cone(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            u=directions[:, 0],
            v=directions[:, 1],
            w=directions[:, 2],
            colorscale='Viridis',
            sizemode='scaled',
            sizeref=0.3,
            showscale=False,
            name='电场方向 (-∇V)',
            anchor='tail'
        )
        
        return gradient_trace
    
    def plot(self, title: str = "宇宙电势分布") -> go.Figure:
        """生成完整的电势分布图"""
        # 计算电势网格
        self.X, self.Y, self.Z, grid_points = self._create_grid()
        V_grid = self.calculate_potential_grid()
        
        # 创建布局
        layout = go.Layout(
            title=dict(
                text=title,
                font=dict(size=24, color='white', family='Arial Black'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title='X (m)',
                    gridcolor='#1e2a47',
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                yaxis=dict(
                    title='Y (m)',
                    gridcolor='#1e2a47',
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                zaxis=dict(
                    title='Z (m)',
                    gridcolor='#1e2a47',
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                bgcolor=self.COSMOS_COLORS['background'],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            paper_bgcolor=self.COSMOS_COLORS['background'],
            font=dict(color='white'),
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig = go.Figure(layout=layout)
        
        # 添加等势面
        if self.show_isosurfaces:
            isosurfaces = self.create_isosurfaces(V_grid)
            for surface in isosurfaces:
                fig.add_trace(surface)
        
        # 添加电势切片
        if self.show_slices:
            slice_surface = self.create_potential_slice(V_grid, 'z', 0)
            fig.add_trace(slice_surface)
        
        # 添加梯度箭头
        gradient_arrows = self.create_gradient_arrows(V_grid)
        if gradient_arrows:
            fig.add_trace(gradient_arrows)
        
        # 添加电势值标注
        self._add_potential_annotations(fig, V_grid)
        
        return fig
    
    def _add_potential_annotations(self, fig: go.Figure, V_grid: np.ndarray):
        """添加关键电势值标注"""
        V_min = np.min(V_grid)
        V_max = np.max(V_grid)
        V_abs_max = max(abs(V_min), abs(V_max))
        
        if V_abs_max > 1e-6:
            annotation_text = f"""
            <b>电势范围</b>: {V_min:.2e} V 到 {V_max:.2e} V<br>
            <b>电势差</b>: {V_max - V_min:.2e} V
            """
            
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(color='white', size=12),
                bgcolor='rgba(11, 11, 31, 0.8)',
                bordercolor='#1e2a47',
                borderwidth=1,
                borderpad=10
            )