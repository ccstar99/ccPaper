'''    项目结构：
    ├── main.py                 # 主程序：参数设置与流程控制
    ├── 模型.py                 # 核心：球面三角形几何与离散化
    ├── 模型计算器.py           # 核心：边界元物理量计算
    ├── 渲染工具.py             # 核心：PyVista/Plotly可视化
'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序：球形电极三维静电场可视化系统
基于论文《球形电极三维静电场的球面三角形边界元算法》实现
"""
import time
import warnings
from pathlib import Path

# 导入三个核心模块
from Bemmodel import generate_icosphere, validate_mesh
from compute import SphericalBEMSolver
from visualization import CosmicFieldVisualizer, PlotlyAnalyzer, UnifiedVisualizer

# ==================== 1. 参数配置区 ====================

# 几何参数（与论文一致）
RADIUS = 1.0  # 球半径 (m)
CENTER = (0, 0, 0)  # 球心坐标
SUBDIVISIONS = 1  # 网格细分次数（0=20单元, 1=80单元, 2=320单元）

# 物理参数
VOLTAGE = 100.0  # 导体球电势 (V)

# 可视化参数
NUM_FIELD_LINES = None  # 电场线数量（None表示使用单元数量，细分1级为80条）
INTEGRATION_LENGTH = 3.0  # 电场线积分长度（球半径倍数）
CAMERA_ZOOM = 1.2  # 相机缩放

# 输出设置
OUTPUT_DIR = Path("./render_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== 2. 主执行流程 ====================

def run_simulation():
    """主模拟流程：模型生成 → 求解 → 可视化 → 验证"""

    print("=" * 70)
    print("球形电极三维静电场可视化系统")
    print("基于《电工技术学报》2009年 球面三角形边界元算法")
    print("=" * 70)

    # 步骤1：生成球面三角形网格
    print("\n【步骤1】生成球面三角形网格...")
    print(f"  半径: {RADIUS} m, 细分: {SUBDIVISIONS}次")

    start_time = time.time()
    mesh = generate_icosphere(radius=RADIUS, center=CENTER, subdivisions=SUBDIVISIONS)

    # 验证网格几何精度
    errors = validate_mesh(mesh)
    print(f"  ✓ 网格生成完成: {mesh.num_vertices}节点, {mesh.num_triangles}单元")
    print(f"  ✓ 几何误差: 半径偏差={errors['vertex_radius_error']:.2e}, 面积偏差={errors['area_error_rel']:.2e}")
    print(f"  ✓ 顶点数组形状: {mesh.vertices.shape}, 三角形数组长度: {len(mesh.triangles)}")

    # 步骤2：边界元求解
    print("\n【步骤2】边界元方程求解...")
    solver = SphericalBEMSolver(mesh, voltage=VOLTAGE)

    print("  组装系数矩阵...")
    G, H = solver.assemble_system_matrices(gauss_order=4)  # 使用4阶高斯积分

    print("  求解线性方程组...")
    sigma_elements, sigma_nodes, E_elements = solver.solve_electric_field(G, H)
    solve_time = time.time() - start_time
    print(f"  ✓ 求解完成，耗时: {solve_time:.2f} 秒")

    # 步骤3：解析解验证（仅单个导体球）
    print("\n【步骤3】解析解验证...")
    analytical_E = VOLTAGE / RADIUS  # V/m

    results = solver.validate_solution(sigma_elements, E_elements)
    print(f"  解析解: |E| = {analytical_E:.3f} V/m")
    print(f"  最大相对误差: {results['max_E_error']:.3f}%")
    print(f"  平均相对误差: {results['mean_E_error']:.3f}%")
    print(f"  总电荷相对误差: {results['charge_error']:.3f}%")

    # 步骤4：生成分析图表
    print("\n【步骤4】生成Plotly交互式图表...")
    
    analyzer = PlotlyAnalyzer(solver)
    
    # 图表1：电场强度-极角分布（论文图3样式）
    print("  生成极角分布图...")
    fig1 = analyzer.plot_elevation_distribution(
        output_path=str(OUTPUT_DIR / "elevation_distribution.html")
    )
    fig1.show()  # 显示极角分布图
    
    # 图表2：交互式3D电场线
    print("  生成交互式3D视图...")
    fig2 = analyzer.plot_field_line_3d_interactive(
        num_lines=NUM_FIELD_LINES or mesh.num_triangles,  # 使用NUM_FIELD_LINES或单元数量(80)
        output_path=str(OUTPUT_DIR / "interactive_3d.html")
    )
    fig2.show()  # 显示交互式3D电场线
    
    # 图表3：电荷密度云图
    print("  生成电荷密度云图...")
    fig3 = analyzer.plot_charge_density_map(
        output_path=str(OUTPUT_DIR / "charge_density_map.html")
    )
    fig3.show()  # 显示电荷密度云图

    # 步骤5：宇宙风格3D渲染
    print("\n【步骤5】生成宇宙风格3D渲染...")
    
    # 调试：检查mesh对象
    print(f"  调试：mesh.vertices.shape = {solver.mesh.vertices.shape}")
    print(f"  调试：len(mesh.triangles) = {len(solver.mesh.triangles)}")
    
    # 检查export_mesh_data的输出
    vertices, faces = solver.mesh.export_mesh_data()
    print(f"  调试：export_mesh_data返回的vertices.shape = {vertices.shape}")
    print(f"  调试：export_mesh_data返回的faces.shape = {faces.shape}")

    # 创建PyVista可视化器
    viz = CosmicFieldVisualizer(solver, starfield_density=100)  # 大幅减少宇宙粒子数量

    # 渲染球面（电荷密度映射）
    viz.render_sphere_surface(
        colormap="plasma",
        show_edges=True,
        edge_opacity=0.3,
        scalar_bar_title="面电荷密度 (C/m²)"
    )

    # 追踪并渲染电场线
    viz.trace_field_lines(
        num_lines=NUM_FIELD_LINES,
        integration_length=INTEGRATION_LENGTH,
        max_step=0.03,
        tube_radius=0.008,
        tube_opacity=0.75
    )

    # 添加探测点（可选）
    # probe_points = np.array([[1.2, 0, 0], [1.5, 0, 0], [2.0, 0, 0]])
    # viz.add_probe_points(probe_points)

    # 设置相机并保存多角度渲染
    # 视角1：正面
    print("  渲染视角1：正面视图")
    viz.set_camera(position=[3, 2, 1], zoom=CAMERA_ZOOM)
    viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_front.png"))

    # 视角2：侧面
    print("  渲染视角2：俯视图")
    viz.set_camera(position=[0, 0, 4], zoom=CAMERA_ZOOM)
    viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_top.png"))

    # 视角3：等距视角
    print("  渲染视角3：等距视图")
    viz.set_camera(position=[3, 3, 3], zoom=CAMERA_ZOOM)
    viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_iso.png"))

    # 步骤6：统一可视化接口（一键全部生成）- 暂时注释
    # print("\n【步骤6】一键生成所有可视化...")
    # unified = UnifiedVisualizer(solver)
    # unified.render_all(
    #     num_lines=30,
    #     save_dir=str(OUTPUT_DIR / "unified")
    # )

    return solver, viz, analyzer


# ==================== 3. 交互探索模式 ====================

def interactive_explore(solver: SphericalBEMSolver):
    """
    交互式探索模式（需在Jupyter Notebook中运行）
    动态调整参数并实时查看效果
    """
    try:
        from ipywidgets import interact, IntSlider, FloatSlider

        print("\n启动交互式探索模式...")
        print("  可调整参数：电场线数量、粗细、透明度")

        unified = UnifiedVisualizer(solver)
        return unified.interactive_explorer()

    except ImportError:
        print("警告：未安装ipywidgets，跳过交互模式")
        return None


# ==================== 4. 性能测试 ====================

def benchmark_performance():
    """性能测试：不同网格密度下的计算时间"""
    print("\n【性能测试】不同网格密度对比...")

    results = []

    for subdiv in [0, 1, 2]:
        print(f"\n  测试细分级别: {subdiv}")

        # 生成网格
        start = time.time()
        mesh = generate_icosphere(radius=1.0, subdivisions=subdiv)
        mesh_time = time.time() - start

        # 求解
        start = time.time()
        solver = SphericalBEMSolver(mesh, voltage=100.0)
        solver.assemble_matrix(gauss_order=3)
        solver.solve()
        solve_time = time.time() - start

        # 验证精度
        errors = solver.validate_analytical_solution()

        results.append({
            "subdivisions": subdiv,
            "num_nodes": mesh.num_vertices,
            "num_elements": mesh.num_triangles,
            "mesh_time": mesh_time,
            "solve_time": solve_time,
            "max_error": errors["max_relative_error"] * 100
        })

        print(f"    节点数: {mesh.num_vertices}, 单元数: {mesh.num_triangles}")
        print(f"    组装+求解: {solve_time:.2f} 秒")
        print(f"    最大误差: {errors['max_relative_error'] * 100:.3f}%")

    # 打印对比表格
    print("\n性能对比结果:")
    print("-" * 70)
    print(f"{'细分':<8} {'节点数':<10} {'单元数':<10} {'时间(s)':<10} {'误差(%)':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['subdivisions']:<8} {r['num_nodes']:<10} {r['num_elements']:<10} {r['solve_time']:<10.2f} {r['max_error']:<10.3f}")

    return results


# ==================== 5. 主入口 ====================

def main():
    """主入口函数"""
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        # 运行完整模拟
        solver, viz, analyzer = run_simulation()

        # 性能测试（可选）
        # benchmark_performance()

        # 交互探索（Jupyter环境）
        # interactive_explore(solver)

        # 输出完成信息
        print("\n" + "=" * 70)
        print("✓ 所有渲染完成！")
        print(f"✓ 输出目录: {OUTPUT_DIR.absolute()}")
        print("✓ 包含文件：")
        for file in sorted(OUTPUT_DIR.rglob("*.png")):
            print(f"  - {file.relative_to(OUTPUT_DIR)}")
        for file in sorted(OUTPUT_DIR.rglob("*.html")):
            print(f"  - {file.relative_to(OUTPUT_DIR)}")
        print("=" * 70)
    except Exception as e:
        import traceback
        print(f"\n发生错误：{type(e).__name__}: {e}")
        print("完整错误信息：")
        traceback.print_exc()


# ==================== 6. 直接执行 ====================

if __name__ == "__main__":
    main()