import numpy as np
import pyvista as pv
import plotly.graph_objects as go
import plotly.io as pio
from scipy.integrate import solve_ivp
from typing import Optional, List
from Bemmodel import spherical_centroid

# 设置Plotly宇宙主题
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0a0a1a",
        "plot_bgcolor": "#0a0a1a",
        "font": {"color": "#e0e0ff"},  # 移除字体族设置，使用系统默认字体
        "xaxis": {
            "gridcolor": "#2a2a4a",
            "linecolor": "#4a4a8a",
            "tickcolor": "#e0e0ff"
        },
        "yaxis": {
            "gridcolor": "#2a2a4a",
            "linecolor": "#4a4a8a",
            "tickcolor": "#e0e0ff"
        },
        "scene": {
            "xaxis": {"backgroundcolor": "#0a0a1a", "gridcolor": "#2a2a4a"},
            "yaxis": {"backgroundcolor": "#0a0a1a", "gridcolor": "#2a2a4a"},
            "zaxis": {"backgroundcolor": "#0a0a1a", "gridcolor": "#2a2a4a"}
        }
    }
}
pio.templates["cosmic"] = PLOTLY_TEMPLATE
pio.templates.default = "cosmic"


# ==================== 1. 宇宙风格PyVista渲染器 ====================

class CosmicFieldVisualizer:
    """
    电场三维可视化（PyVista）
    风格：深空背景 + 发光材质 + 流体美学
    """

    def __init__(self, solver, background_color: str = "#0a0a1a",
                 starfield_density: int = 100):  # 默认大幅减少宇宙粒子数量
        """
        参数:
            solver: SphericalBEMSolver实例（含计算结果）
            background_color: 深空背景色
            starfield_density: 背景星点数量
        """
        # 设置允许空网格绘制，解决PyVista误判问题
        pv.global_theme.allow_empty_mesh = True
        
        self.solver = solver
        self.mesh = solver.mesh

        # 创建PyVista绘图器
        self.plotter = pv.Plotter(
            notebook=False,
            off_screen=False,
            window_size=(1920, 1080)
        )

        # 配置宇宙风格
        self.plotter.set_background(background_color)
        self._add_starfield(starfield_density)

        # 预设相机参数
        self.camera_position = None
        self.camera_focus = self.mesh.center
        self.camera_up = [0, 0, 1]

    def _add_starfield(self, density: int):
        """添加随机分布的星点背景"""
        np.random.seed(42)  # 可复现性
        stars = np.random.randn(density, 3) * 10  # 在球壳上分布
        stars = stars / np.linalg.norm(stars, axis=1)[:, None] * np.random.uniform(5, 15, size=(density, 1))

        star_cloud = pv.PolyData(stars)
        self.plotter.add_mesh(
            star_cloud,
            color="white",
            point_size=2,
            render_points_as_spheres=True,
            opacity=0.8,
            name="starfield"
        )

    def render_sphere_surface(self, colormap: str = "plasma",
                              show_edges: bool = True,
                              edge_opacity: float = 0.3,
                              scalar_bar_title: str = "面电荷密度 (C/m²)"):
        """
        渲染球面三角形网格（颜色映射电荷密度）

        参数:
            colormap: 颜色映射（plasma, viridis, coolwarm）
            show_edges: 是否显示三角形边线
            edge_opacity: 边线透明度
            scalar_bar_title: 色标标题
        """
        # 导出几何数据
        vertices, faces = self.mesh.export_mesh_data()

        # 创建PyVista多面体网格
        print(f"  调试：在visualization.py中，vertices.shape = {vertices.shape}")
        print(f"  调试：在visualization.py中，faces.shape = {faces.shape}")
        print(f"  调试：在visualization.py中，faces[:10] = {faces[:10]}")
        sphere_mesh = pv.PolyData(vertices, faces)
        print(f"  调试：在visualization.py中，sphere_mesh.n_points = {sphere_mesh.n_points}")
        print(f"  调试：在visualization.py中，sphere_mesh.n_cells = {sphere_mesh.n_cells}")

        # 将电荷密度映射到单元中心
        cell_centers = np.array([
            spherical_centroid(tri.vertices, self.mesh.center, self.mesh.radius)
            for tri in self.mesh.spherical_triangles
        ])

        # 为每个单元附加标量值
        sphere_mesh.cell_data["charge_density"] = self.solver.charge_density

        # 主网格渲染（发光效果）
        print(f"  调试：在visualization.py中，调用add_mesh前，sphere_mesh.n_points = {sphere_mesh.n_points}")
        print(f"  调试：在visualization.py中，调用add_mesh前，sphere_mesh.n_cells = {sphere_mesh.n_cells}")
        print(f"  调试：在visualization.py中，调用add_mesh前，sphere_mesh.cell_data.keys() = {list(sphere_mesh.cell_data.keys())}")
        if 'charge_density' in sphere_mesh.cell_data:
            print(f"  调试：在visualization.py中，调用add_mesh前，charge_density.shape = {sphere_mesh.cell_data['charge_density'].shape}")
        self.plotter.add_mesh(
            sphere_mesh,
            scalars="charge_density",
            cmap=colormap,
            show_edges=show_edges,
            edge_color="cyan",
            edge_opacity=edge_opacity,
            opacity=0.8,
            smooth_shading=True,
            specular=0.5,  # 镜面反射
            specular_power=30,
            scalar_bar_args={
                "title": scalar_bar_title,
                "title_font_size": 14,
                "label_font_size": 12,
                "color": "white",
                "position_x": 0.8,
                "position_y": 0.05
            },
            name="sphere_surface"
        )

        # 添加轮廓线（增强立体感）
        self.plotter.add_mesh(
            sphere_mesh.extract_feature_edges(),
            color="cyan",
            line_width=1,
            opacity=0.4,
            name="silhouette"
        )

        return self

    def trace_field_lines(self, num_lines: int = 30,
                          integration_length: float = 3.0,
                          max_step: float = 0.05,
                          tube_radius: float = 0.005,
                          tube_opacity: float = 0.7):
        """
        追踪并渲染电场线（流线）

        参数:
            num_lines: 电场线数量
            integration_length: 积分长度（单位：球半径倍数）
            max_step: 最大积分步长
            tube_radius: 流线管半径（视觉粗细）
            tube_opacity: 流线透明度
        """
        # 获取电场线和起点（使用analytic方法确保球对称性）
        field_lines, start_points = self.solver.compute_electric_field_lines(
            num_lines=num_lines,
            start_radius_factor=1.01,
            method='analytic'
        )

        print(f"[CosmicViz] 追踪{len(field_lines)}条电场线...")

        # 积分参数
        s_max = integration_length * self.mesh.radius

        # 流线颜色映射（按起始电荷密度）
        start_tri_indices = self._locate_start_triangles(start_points)
        line_colors = self._get_line_colors(start_tri_indices)

        # 渲染电场线
        streamlines = []
        print(f"[调试] field_lines数量: {len(field_lines)}")
        for i, line_points in enumerate(field_lines):
            print(f"[调试] 电场线{i}点数: {len(line_points)}")
            if len(line_points) > 5:  # 至少有5个点
                try:
                    # 创建平滑的线
                    spline = pv.Spline(line_points, n_points=200)
                    
                    # 创建管状物（使用固定半径，避免数组参数问题）
                    tube = spline.tube(radius=tube_radius)

                    # 按电荷密度着色
                    self.plotter.add_mesh(
                        tube,
                        color=line_colors[i],
                        opacity=tube_opacity,
                        smooth_shading=True,
                        specular=0.3,
                        name=f"streamline_{i}"
                    )
                    streamlines.append(line_points)
                    print(f"[调试] 成功渲染电场线{i}")
                except Exception as e:
                    print(f"[调试] 渲染电场线{i}失败: {e}")
                    
                    # 尝试使用简单的线代替管状物
                    try:
                        self.plotter.add_lines(
                            line_points,
                            color=line_colors[i],
                            width=2,
                            name=f"streamline_{i}_simple"
                        )
                        streamlines.append(line_points)
                        print(f"[调试] 使用简单线渲染电场线{i}成功")
                    except Exception as e2:
                        print(f"[调试] 简单线渲染电场线{i}也失败: {e2}")

        print(f"[CosmicViz] 成功渲染{len(streamlines)}条电场线")
        return self

    def _locate_start_triangles(self, start_points: np.ndarray) -> List[int]:
        """定位每个起点所在的三角形单元"""
        indices = []
        for point in start_points:
            found = False
            for idx, tri in enumerate(self.mesh.spherical_triangles):
                if tri.contains_point(point, tol=1e-3):
                    indices.append(idx)
                    found = True
                    break
            if not found:
                indices.append(0)  # 默认
        return indices

    def _get_line_colors(self, tri_indices: List[int]) -> List[tuple]:
        """根据电荷密度获取电场线颜色"""
        colors = []
        for idx in tri_indices:
            sigma = self.solver.charge_density[idx]
            # 映射到颜色（蓝色=负，红色=正）
            norm_sigma = (sigma - self.solver.charge_density.min())
            norm_sigma /= (self.solver.charge_density.max() - self.solver.charge_density.min())

            # RGB插值（蓝→紫→红），使用RGB元组格式(0-1之间的浮点数)
            r = float(norm_sigma)
            b = float(1 - norm_sigma)
            colors.append((r, 0.0, b))  # pyvista接受的RGB元组格式

        return colors

    def add_probe_points(self, points: np.ndarray, radius: float = 0.02):
        """
        添加探测点（显示电场矢量）

        参数:
            points: (n_points, 3) 探测点坐标
            radius: 探测点球半径
        """
        # Calculate E field for each point
        E_field = np.array([self.solver.calculate_electric_field_at_point(point)[0] for point in points])

        # 绘制探测点
        probe_cloud = pv.PolyData(points)
        probe_cloud["E_magnitude"] = np.linalg.norm(E_field, axis=1)

        self.plotter.add_mesh(
            probe_cloud,
            render_points_as_spheres=True,
            point_size=10,
            scalars="E_magnitude",
            cmap="viridis",
            name="probe_points"
        )

        # 添加电场矢量箭头
        arrows = pv.PolyData(points)
        arrows["vectors"] = E_field

        self.plotter.add_arrows(
            points,
            E_field,
            mag=0.5,
            color="cyan",
            opacity=0.8,
            name="field_vectors"
        )

        return self

    def set_camera(self, position: Optional[List[float]] = None,
                   focal_point: Optional[List[float]] = None,
                   view_up: Optional[List[float]] = None,
                   zoom: float = 1.0):
        """
        配置相机参数

        参数:
            position: 相机位置
            focal_point: 焦点
            view_up: 上方向
            zoom: 缩放倍数
        """
        if position is None:
            # 默认位置：球坐标系
            theta = np.pi / 3
            phi = np.pi / 4
            r = 3.0 * self.mesh.radius / zoom
            position = [
                self.mesh.center[0] + r * np.sin(theta) * np.cos(phi),
                self.mesh.center[1] + r * np.sin(theta) * np.sin(phi),
                self.mesh.center[2] + r * np.cos(theta)
            ]

        if focal_point is None:
            focal_point = self.mesh.center

        if view_up is None:
            view_up = self.camera_up

        self.camera_position = position
        self.plotter.camera.position = position
        self.plotter.camera.focal_point = focal_point
        self.plotter.camera.up = view_up

        return self

    def show(self, title: str = "球形电极电场 - 球面三角形边界元法",
             save_path: Optional[str] = None,
             show_window: bool = True):
        """
        显示或保存渲染结果

        参数:
            title: 窗口标题
            save_path: 保存图片路径（如"render.png"）
            show_window: 是否显示窗口
        """
        self.plotter.add_title(title, font="arial", font_size=18, color="white")
        self.plotter.add_axes(line_width=3, color="white")

        if save_path:
            try:
                # 先检查plotter是否可用
                if self.plotter is None:
                    print("[调试] plotter不可用")
                    return self
                
                # 确保窗口已创建
                if not hasattr(self.plotter, 'window') or self.plotter.window is None:
                    print("[调试] 创建临时窗口用于截图")
                    temp_plotter = pv.Plotter(window_size=(1920, 1080), off_screen=True)
                    temp_plotter.background_color = self.plotter.background_color
                    
                    # 重新添加所有对象
                    try:
                        # 尝试访问_actors属性（旧版本PyVista）
                        for actor in self.plotter.renderers[0]._actors.values():
                            temp_plotter.add_actor(actor)
                    except AttributeError:
                        # 尝试访问actors属性（新版本PyVista）
                        for actor in self.plotter.renderers[0].actors.values():
                            temp_plotter.add_actor(actor)
                    
                    temp_plotter.add_title(title, font="arial", font_size=18, color="white")
                    temp_plotter.add_axes(line_width=3, color="white")
                    
                    # 保存截图
                    temp_plotter.screenshot(save_path, transparent_background=True)
                    temp_plotter.close()
                else:
                    # 正常截图
                    self.plotter.screenshot(save_path, transparent_background=True)
                
                print(f"[CosmicViz] 渲染保存至 {save_path}")
            except Exception as e:
                print(f"[调试] 截图保存失败: {e}")

        if show_window:
            try:
                self.plotter.show()
            except Exception as e:
                print(f"[调试] 显示窗口失败: {e}")
        return self


# ==================== 2. Plotly 2D分析图表 ====================

class PlotlyAnalyzer:
    """
    电场分析图表（Plotly）
    生成论文图3/4样式的交互式曲线
    """

    def __init__(self, solver):
        self.solver = solver

    def plot_elevation_distribution(self, output_path: Optional[str] = None) -> go.Figure:
        """
        绘制电场强度随极角分布（论文图3样式）

        返回:
            Plotly Figure对象
        """
        # 生成极角采样点
        theta = np.linspace(0, np.pi, 180)
        phi = np.zeros_like(theta)

        # 球坐标转笛卡尔
        points = np.vstack([
            self.solver.mesh.radius * 1.01 * np.sin(theta) * np.cos(phi),
            self.solver.mesh.radius * 1.01 * np.sin(theta) * np.sin(phi),
            self.solver.mesh.radius * 1.01 * np.cos(theta)
        ]).T

        # 计算电场强度
        # Calculate E field for each point
        E_field = np.array([self.solver.calculate_electric_field_at_point(point)[0] for point in points])
        E_magnitude = np.linalg.norm(E_field, axis=1)

        # 创建交互式图表
        fig = go.Figure()

        # 数值解曲线
        fig.add_trace(go.Scatter(
            x=np.degrees(theta),
            y=E_magnitude,
            mode='lines+markers',
            name='球面三角形BEM',
            line=dict(color="#00f2ff", width=3),
            marker=dict(size=4, color="#00f2ff"),
            hovertemplate="θ: %{x:.1f}°<br>|E|: %{y:.2f} V/m<extra></extra>"
        ))

        # 解析解（参考线）
        analytical_E = self.solver.voltage / self.solver.mesh.radius
        fig.add_hline(
            y=analytical_E,
            line_dash="dash",
            line_color="#ff4d6d",
            annotation_text=f"解析解: {analytical_E:.2f} V/m",
            annotation_position="top right"
        )

        # 填充区域（误差带）
        error = np.abs(E_magnitude - analytical_E)
        fig.add_trace(go.Scatter(
            x=np.degrees(theta),
            y=analytical_E + error,
            fill=None,
            mode='lines',
            line_color="rgba(255,77,109,0.3)",
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=np.degrees(theta),
            y=analytical_E - error,
            fill='tonexty',
            mode='lines',
            line_color="rgba(255,77,109,0.3)",
            name='误差带',
            hoverinfo='skip'
        ))

        # 样式配置
        fig.update_layout(
            title={
                "text": "电场强度随极角分布<br><sub>球面三角形边界元 vs 解析解</sub>",
                "x": 0.5,
                "font": {"size": 20, "family": "arial"}
            },
            xaxis_title="极角 θ (°)",
            yaxis_title="电场强度 |E| (V/m)",
            xaxis=dict(range=[0, 180]),
            yaxis=dict(range=[0, max(E_magnitude) * 1.1]),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#4a4a8a",
                borderwidth=1
            ),
            hovermode='x unified'
        )

        if output_path:
            fig.write_html(output_path)
            print(f"[Plotly] 图表保存至 {output_path}")

        return fig

    def plot_field_line_3d_interactive(self, num_lines: int = 10,
                                       output_path: Optional[str] = None) -> go.Figure:
        """
        交互式3D电场线（Plotly）
        可在浏览器中旋转查看
        """
        # 获取电场线和起点（使用analytic方法确保球对称性）
        field_lines, start_points = self.solver.compute_electric_field_lines(
            num_lines=num_lines,
            start_radius_factor=1.01,
            method='analytic'
        )

        fig = go.Figure()

        # 添加球面（半透明）
        vertices, faces = self.solver.mesh.export_mesh_data()
        # Plotly需要faces格式: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        plotly_faces = []
        for face in self.solver.mesh.triangles:
            plotly_faces.extend([3, face[0], face[1], face[2]])

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[faces[i + 1] for i in range(0, len(faces), 4)],
            j=[faces[i + 2] for i in range(0, len(faces), 4)],
            k=[faces[i + 3] for i in range(0, len(faces), 4)],
            opacity=0.3,
            color="#4a4a8a",
            name="电极表面"
        ))

        # 绘制电场线
        for i, line in enumerate(field_lines[:num_lines]):
            # 颜色映射（按长度衰减）
            colors = np.linspace(1, 0, len(line))

            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(
                    color=colors,
                    colorscale="Plasma",
                    width=4
                ),
                name=f"电场线_{i}",
                showlegend=False,
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            ))

        # 相机视角
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X (m)", gridcolor="#2a2a4a"),
                yaxis=dict(title="Y (m)", gridcolor="#2a2a4a"),
                zaxis=dict(title="Z (m)", gridcolor="#2a2a4a"),
                bgcolor="#0a0a1a",
                camera=dict(
                    eye=dict(x=2, y=2, z=1),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            title={
                "text": "三维电场线分布",
                "x": 0.5,
                "font": {"size": 18}
            }
        )

        if output_path:
            fig.write_html(output_path)
            print(f"[Plotly] 3D视图保存至 {output_path}")

        return fig

    def plot_charge_density_map(self, output_path: Optional[str] = None) -> go.Figure:
        """
        球面电荷密度云图（2D投影）
        """
        import matplotlib.tri as tri

        # 获取顶点数据
        vertices = self.solver.mesh.vertices
        charge_at_vertices = np.zeros(self.solver.mesh.num_vertices)

        # 将单元电荷密度平均到节点
        for i in range(self.solver.mesh.num_vertices):
            # 遍历所有三角形，找到包含该节点的三角形
            related_tris = []
            for tri_idx, tri in enumerate(self.solver.mesh.triangles):
                if i in tri:
                    related_tris.append(tri_idx)
            
            sigma_vals = [self.solver.charge_density[tri_idx] for tri_idx in related_tris]
            charge_at_vertices[i] = np.mean(sigma_vals)

        # 球坐标转换
        r = np.linalg.norm(vertices - self.solver.mesh.center, axis=1)
        theta = np.arccos((vertices[:, 2] - self.solver.mesh.center[2]) / r)
        phi = np.arctan2(vertices[:, 1] - self.solver.mesh.center[1],
                         vertices[:, 0] - self.solver.mesh.center[0])

        # 创建三角剖分
        fig = go.Figure(data=go.Contour(
            x=np.degrees(phi),
            y=np.degrees(theta),
            z=charge_at_vertices,
            colorscale="Plasma",
            opacity=0.8,
            hovertemplate="φ: %{x:.1f}°<br>θ: %{y:.1f}°<br>σ: %{z:.3e} C/m²<extra></extra>"
        ))

        fig.update_layout(
            title="球面电荷密度分布",
            xaxis_title="方位角 φ (°)",
            yaxis_title="极角 θ (°)"
        )

        if output_path:
            fig.write_html(output_path)

        return fig


# ==================== 3. 统一可视化接口 ====================

class UnifiedVisualizer:
    """
    统一可视化接口
    一键生成所有图表
    """

    def __init__(self, solver):
        self.solver = solver
        self.pyvista_viz = CosmicFieldVisualizer(solver)
        self.plotly_analyzer = PlotlyAnalyzer(solver)

    def render_all(self, num_lines: int = 30, save_dir: str = "./output"):
        """
        渲染所有可视化结果

        参数:
            num_lines: 电场线数量
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 1. PyVista 3D渲染
        print("[Unified] 渲染3D场景...")
        self.pyvista_viz.render_sphere_surface()
        self.pyvista_viz.trace_field_lines(num_lines=num_lines)
        self.pyvista_viz.set_camera()
        self.pyvista_viz.show(save_path=os.path.join(save_dir, "sphere_render.png"))

        # 2. Plotly 极角分布
        print("[Unified] 生成极角分布图...")
        fig1 = self.plotly_analyzer.plot_elevation_distribution(
            output_path=os.path.join(save_dir, "elevation_plot.html")
        )
        fig1.show()

        # 3. Plotly 3D交互电场线
        print("[Unified] 生成交互式3D电场线...")
        fig2 = self.plotly_analyzer.plot_field_line_3d_interactive(
            num_lines=15,
            output_path=os.path.join(save_dir, "field_lines_3d.html")
        )
        fig2.show()

        print(f"[Unified] 所有渲染完成，保存至 {save_dir}")

    def interactive_explorer(self):
        """
        启动交互式探索模式
        可在Jupyter中动态调整参数
        """
        from ipywidgets import interact, IntSlider, FloatSlider

        def explore(num_lines=20, tube_radius=0.005, opacity=0.7):
            self.pyvista_viz.plotter.clear()
            self.pyvista_viz.render_sphere_surface()
            self.pyvista_viz.trace_field_lines(
                num_lines=num_lines,
                tube_radius=tube_radius,
                tube_opacity=opacity
            )
            self.pyvista_viz.show()

        return interact(
            explore,
            num_lines=IntSlider(min=5, max=50, step=5, value=20),
            tube_radius=FloatSlider(min=0.001, max=0.01, step=0.001, value=0.005),
            opacity=FloatSlider(min=0.3, max=1.0, step=0.1, value=0.7)
        )


# ==================== 4. 测试 ====================

if __name__ == "__main__":
    from Bemmodel import generate_icosphere
    from compute import SphericalBEMSolver

    print("=== 宇宙可视化测试 ===")

    # 1. 创建模型与求解
    mesh = generate_icosphere(radius=1.0, subdivisions=1)
    solver = SphericalBEMSolver(mesh, voltage=100.0)
    solver.assemble_matrix()
    solver.solve()

    # 2. 创建可视化器
    viz = CosmicFieldVisualizer(solver)

    # 3. 渲染
    viz.render_sphere_surface(colormap="plasma")
    viz.trace_field_lines(num_lines=20)
    viz.set_camera()
    viz.show(save_path="cosmic_sphere.png")

    # 4. Plotly图表
    analyzer = PlotlyAnalyzer(solver)
    fig = analyzer.plot_elevation_distribution()
    fig.show()