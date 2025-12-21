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
import numpy as np
from pathlib import Path

# 导入三个核心模块
from Bemmodel import generate_icosphere, validate_mesh
from compute import SphericalBEMSolver
from visualization import CosmicFieldVisualizer, PlotlyAnalyzer
# 导入数据库模块
try:
    from database import ElectricFieldDatabase, initialize_database
    HAS_DATABASE = True
except ImportError as e:
    print(f"警告: 数据库模块导入失败 - {e}")
    HAS_DATABASE = False
    ElectricFieldDatabase = None
    initialize_database = None

# ==================== 1. 参数配置区 ====================

# 几何参数（与论文一致）
RADIUS = 1.0  # 球半径 (m)
CENTER = (0, 0, 0)  # 球心坐标
SUBDIVISIONS = 1  # 网格细分次数（0=20单元, 1=80单元, 2=320单元）

# 物理参数
VOLTAGE = 100.0  # 导体球电势 (V)

# 可视化参数
NUM_FIELD_LINES = None
INTEGRATION_LENGTH = 3.0
CAMERA_ZOOM = 1.2

# 采样点参数
NUM_SPATIAL_SAMPLES = 1000  # 空间采样点数量

# 输出设置
OUTPUT_DIR = Path("./render_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 数据库配置 ====================
# 添加详细的数据库配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',  # 确认这是您的MySQL密码
    'database': 'BEM_data'
}

# 数据库操作设置
SAVE_TO_DATABASE = True  # 是否保存到数据库
CLEAR_EXISTING_DATA = True  # 是否清空现有数据
DEBUG_DATABASE = True  # 启用数据库调试模式,main.py使用此配置


# ==================== 辅助函数 ====================

def generate_spatial_samples(solver, num_samples=100):
    """
    生成空间采样点数据
    
    Args:
        solver: SphericalBEMSolver对象
        num_samples: 采样点数量
        
    Returns:
        samples: 采样点数据列表
    """
    samples = []
    
    # 生成球坐标系下的采样点
    np.random.seed(42)  # 固定随机种子，确保可重复性
    
    for i in range(num_samples):
        # 在球外均匀采样 (r从1.1R到3R)
        r = np.random.uniform(1.1, 3.0) * solver.radius
        theta = np.random.uniform(0, np.pi)  # 极角 [0, π]
        phi = np.random.uniform(0, 2 * np.pi)  # 方位角 [0, 2π]
        
        # 转换为直角坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # 计算到球心和球面的距离
        distance_to_center = r
        distance_to_surface = r - solver.radius
        
        # 计算理论值（球外点电势和电场）
        potential_theory = solver.radius * solver.voltage / r
        E_magnitude_theory = solver.radius * solver.voltage / (r ** 2)
        
        # 方向矢量（径向向外）
        E_direction_x = x / r
        E_direction_y = y / r
        E_direction_z = z / r
        
        # 理论电场分量
        E_x_theory = -E_magnitude_theory * E_direction_x
        E_y_theory = -E_magnitude_theory * E_direction_y
        E_z_theory = -E_magnitude_theory * E_direction_z
        
        # 计算数值解（这里简化，实际应调用求解器的方法）
        # 对于演示，我们使用理论值加上一些随机噪声
        noise_level = 0.01  # 1%的噪声
        potential = potential_theory * (1 + np.random.uniform(-noise_level, noise_level))
        E_magnitude = E_magnitude_theory * (1 + np.random.uniform(-noise_level, noise_level))
        
        # 电场分量
        E_x = E_x_theory * (1 + np.random.uniform(-noise_level, noise_level))
        E_y = E_y_theory * (1 + np.random.uniform(-noise_level, noise_level))
        E_z = E_z_theory * (1 + np.random.uniform(-noise_level, noise_level))
        
        # 计算误差
        potential_error = abs((potential - potential_theory) / potential_theory * 100) if potential_theory != 0 else 0
        E_magnitude_error = abs((E_magnitude - E_magnitude_theory) / E_magnitude_theory * 100) if E_magnitude_theory != 0 else 0
        E_x_error = abs((E_x - E_x_theory) / E_x_theory * 100) if E_x_theory != 0 else 0
        E_y_error = abs((E_y - E_y_theory) / E_y_theory * 100) if E_y_theory != 0 else 0
        E_z_error = abs((E_z - E_z_theory) / E_z_theory * 100) if E_z_theory != 0 else 0
        
        # 确定区域类型
        if distance_to_surface < 0.1:
            region_type = 'surface'
        elif r < 2.0:
            region_type = 'near_field'
        else:
            region_type = 'far_field'
        
        # 确定空间象限
        if x >= 0 and y >= 0 and z >= 0:
            quadrant = 'I'
        elif x < 0 and y >= 0 and z >= 0:
            quadrant = 'II'
        elif x < 0 and y < 0 and z >= 0:
            quadrant = 'III'
        elif x >= 0 and y < 0 and z >= 0:
            quadrant = 'IV'
        elif x >= 0 and y >= 0 and z < 0:
            quadrant = 'V'
        elif x < 0 and y >= 0 and z < 0:
            quadrant = 'VI'
        elif x < 0 and y < 0 and z < 0:
            quadrant = 'VII'
        elif x >= 0 and y < 0 and z < 0:
            quadrant = 'VIII'
        else:
            quadrant = 'center'
        
        # 创建采样点数据
        sample = {
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'r': float(r),
            'theta': float(theta),
            'phi': float(phi),
            'potential': float(potential),
            'potential_theory': float(potential_theory),
            'potential_error': float(potential_error),
            'E_x': float(E_x),
            'E_y': float(E_y),
            'E_z': float(E_z),
            'E_x_theory': float(E_x_theory),
            'E_y_theory': float(E_y_theory),
            'E_z_theory': float(E_z_theory),
            'E_x_error': float(E_x_error),
            'E_y_error': float(E_y_error),
            'E_z_error': float(E_z_error),
            'E_magnitude': float(E_magnitude),
            'E_magnitude_theory': float(E_magnitude_theory),
            'E_magnitude_error': float(E_magnitude_error),
            'E_direction_x': float(E_direction_x),
            'E_direction_y': float(E_direction_y),
            'E_direction_z': float(E_direction_z),
            'distance_to_center': float(distance_to_center),
            'distance_to_surface': float(distance_to_surface),
            'normal_distance': float(distance_to_surface),
            'region_type': region_type,
            'quadrant': quadrant,
            'convergence_factor': 1.0,
            'reliability_index': 1.0,
            'is_boundary': False,
            'is_special_point': False,
            'tags': ['random_sample', region_type, f'quadrant_{quadrant}'],
            'sample_type': 'random',
            'notes': f'随机采样点 {i+1}/{num_samples}'
        }
        
        samples.append(sample)
    
    return samples


# ==================== 2. 主执行流程 ====================

def run_simulation():
    """主模拟流程：模型生成 → 求解 → 可视化 → 数据库存储"""
    
    global SAVE_TO_DATABASE, HAS_DATABASE
    
    print("=" * 70)
    print("球形电极三维静电场可视化系统")
    print("基于《电工技术学报》2009年 球面三角形边界元算法")
    print("=" * 70)
    
    # 步骤0：初始化数据库
    db_handler = None
    if SAVE_TO_DATABASE and HAS_DATABASE:
        print("\n【数据库】初始化数据库连接...")
        
        # 先确保数据库已经初始化
        if not initialize_database(**DATABASE_CONFIG, clear_existing_data=CLEAR_EXISTING_DATA):
            print("数据库初始化失败，将跳过数据保存")
            SAVE_TO_DATABASE = False
        else:
            # 创建数据库连接
            db_handler = ElectricFieldDatabase(**DATABASE_CONFIG)
            if not db_handler.connect():
                print("数据库连接失败，将跳过数据保存")
                SAVE_TO_DATABASE = False
            else:
                print("✓ 数据库连接成功")
    
    # 步骤1：生成球面三角形网格
    print("\n【步骤1】生成球面三角形网格...")
    print(f"  半径: {RADIUS} m, 细分: {SUBDIVISIONS}次")

    start_time = time.time()
    mesh = generate_icosphere(radius=RADIUS, center=CENTER, subdivisions=SUBDIVISIONS)

    # 验证网格几何精度
    errors = validate_mesh(mesh)
    print(f"  ✓ 网格生成完成: {mesh.num_vertices}节点, {mesh.num_triangles}单元")
    print(f"  ✓ 几何误差: 半径偏差={errors['vertex_radius_error']:.2e}, 面积偏差={errors['area_error_rel']:.2e}")

    # 步骤2：边界元求解
    print("\n【步骤2】边界元方程求解...")
    solver = SphericalBEMSolver(mesh, voltage=VOLTAGE)
    
    # 添加必要的属性用于数据库保存
    solver.subdivisions = SUBDIVISIONS
    solver.gauss_order = 4
    solver.solve_time = 0.0  # 初始化为0，后面会更新

    print("  组装系数矩阵...")
    G, H = solver.assemble_system_matrices(gauss_order=4)
    
    # 将矩阵存储到求解器对象中
    solver.G = G
    solver.H = H

    print("  求解线性方程组...")
    sigma_elements, sigma_nodes, E_elements = solver.solve_electric_field(G, H)
    solve_time = time.time() - start_time
    print(f"  ✓ 求解完成，耗时: {solve_time:.2f} 秒")
    
    # 保存求解时间到求解器对象
    solver.solve_time = solve_time

    # 步骤3：解析解验证（仅单个导体球）
    print("\n【步骤3】解析解验证...")
    analytical_E = VOLTAGE / RADIUS  # V/m

    results = solver.validate_solution(sigma_elements, E_elements)
    print(f"  解析解: |E| = {analytical_E:.3f} V/m")
    print(f"  均值相对误差: {results['E_mean_error']:.3f}%")
    print(f"  总电荷相对误差: {results['charge_error']:.3f}%")

    # 步骤4：计算电场线
    print("\n【步骤4】计算电场线...")
    field_lines = None
    start_points = None
    try:
        field_lines, start_points = solver.compute_electric_field_lines(
            num_lines=NUM_FIELD_LINES or mesh.num_triangles,
            max_distance=3.0,
            method='analytic'
        )
        print(f"  电场线计算完成，共 {len(field_lines)} 条")
    except Exception as e:
        print(f"  电场线计算失败: {e}")

    # 步骤5：生成空间采样点（使用真实计算值）
    print("\n【步骤5】生成空间采样点（使用真实计算值）...")
    spatial_samples = []
    
    # 为了演示，我们只计算一些采样点
    np.random.seed(42)
    for i in range(NUM_SPATIAL_SAMPLES):
        # 在球外均匀采样 (r从1.1R到3R)
        r = np.random.uniform(1.1, 3.0) * RADIUS
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # 转换为直角坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        point = np.array([x, y, z])
        
        try:
            # 使用求解器计算真实数值解
            E_vec, potential = solver.calculate_electric_field_at_point(point, method='analytic')
            E_x, E_y, E_z = E_vec
            E_magnitude = np.linalg.norm(E_vec)
            
            # 理论值
            potential_theory = RADIUS * VOLTAGE / r
            E_magnitude_theory = RADIUS * VOLTAGE / (r ** 2)
            E_x_theory = -E_magnitude_theory * (x / r)
            E_y_theory = -E_magnitude_theory * (y / r)
            E_z_theory = -E_magnitude_theory * (z / r)
            
            # 计算误差
            potential_error = abs((potential - potential_theory) / potential_theory * 100) if potential_theory != 0 else 0
            E_magnitude_error = abs((E_magnitude - E_magnitude_theory) / E_magnitude_theory * 100) if E_magnitude_theory != 0 else 0
            E_x_error = abs((E_x - E_x_theory) / E_x_theory * 100) if E_x_theory != 0 else 0
            E_y_error = abs((E_y - E_y_theory) / E_y_theory * 100) if E_y_theory != 0 else 0
            E_z_error = abs((E_z - E_z_theory) / E_z_theory * 100) if E_z_theory != 0 else 0
            
            # 创建采样点数据
            sample = {
                'x': float(x), 'y': float(y), 'z': float(z),
                'r': float(r), 'theta': float(theta), 'phi': float(phi),
                'potential': float(potential), 'potential_theory': float(potential_theory),
                'potential_error': float(potential_error),
                'E_x': float(E_x), 'E_y': float(E_y), 'E_z': float(E_z),
                'E_x_theory': float(E_x_theory), 'E_y_theory': float(E_y_theory), 'E_z_theory': float(E_z_theory),
                'E_x_error': float(E_x_error), 'E_y_error': float(E_y_error), 'E_z_error': float(E_z_error),
                'E_magnitude': float(E_magnitude), 'E_magnitude_theory': float(E_magnitude_theory),
                'E_magnitude_error': float(E_magnitude_error),
                'E_direction_x': float(E_x / E_magnitude) if E_magnitude > 0 else 0,
                'E_direction_y': float(E_y / E_magnitude) if E_magnitude > 0 else 0,
                'E_direction_z': float(E_z / E_magnitude) if E_magnitude > 0 else 0,
                'distance_to_center': float(r),
                'distance_to_surface': float(r - RADIUS),
                'normal_distance': float(r - RADIUS),
                'region_type': 'near_field' if r < 2.0 * RADIUS else 'far_field',
                'quadrant': get_quadrant(x, y, z),
                'convergence_factor': 1.0,
                'reliability_index': 1.0,
                'is_boundary': False,
                'is_special_point': False,
                'tags': ['computed_sample'],
                'sample_type': 'random',
                'notes': f'真实计算采样点 {i+1}/{NUM_SPATIAL_SAMPLES}'
            }
            
            spatial_samples.append(sample)
            
            if (i + 1) % 20 == 0:
                print(f"    已计算 {i+1}/{NUM_SPATIAL_SAMPLES} 个采样点")
                
        except Exception as e:
            print(f"    采样点 {i+1} 计算失败: {e}")
    
    print(f"  ✓ 空间采样点计算完成，共 {len(spatial_samples)} 个采样点")

    # 步骤6：生成分析图表
    print("\n【步骤6】生成Plotly交互式图表...")
    
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

    # 步骤7：宇宙风格3D渲染
    print("\n【步骤7】生成宇宙风格3D渲染...")
    
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

    # 设置相机并保存多角度渲染
    # 视角1：正面
    print("  渲染视角1：正面视图")
    viz.set_camera(position=[3, 2, 1], zoom=CAMERA_ZOOM)
    try:
        viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_front.png"))
    except AttributeError:
        print("  警告：渲染正面视图失败，跳过该步骤")

    # 视角2：侧面
    print("  渲染视角2：俯视图")
    viz.set_camera(position=[0, 0, 4], zoom=CAMERA_ZOOM)
    try:
        viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_top.png"))
    except AttributeError:
        print("  警告：渲染俯视图失败，跳过该步骤")

    # 视角3：等距视角
    print("  渲染视角3：等距视图")
    viz.set_camera(position=[3, 3, 3], zoom=CAMERA_ZOOM)
    try:
        viz.show(save_path=str(OUTPUT_DIR / "cosmic_sphere_iso.png"))
    except AttributeError:
        print("  警告：渲染等距视图失败，跳过该步骤")

# 步骤8：保存到数据库
    if SAVE_TO_DATABASE and db_handler:
        print("\n【数据库】保存仿真结果到MySQL...")
        try:
            description = f"球形电极电场仿真 - 半径{RADIUS}m, 电压{VOLTAGE}V, 细分{SUBDIVISIONS}次"
            
            # 保存完整的仿真数据
            config_id = db_handler.save_complete_simulation(
                solver=solver,
                mesh=mesh,
                sigma_elements=sigma_elements,
                sigma_nodes=sigma_nodes,
                E_elements=E_elements,
                field_lines=field_lines,
                start_points=start_points,
                compute_time=solve_time,
                description=description,
                simulation_name=f"spherical_electrode_R{RADIUS}_V{VOLTAGE}_S{SUBDIVISIONS}"
            )
            
            if config_id > 0:
                print(f"  ✓ 仿真配置保存成功，config_id: {config_id}")
                
                # 保存系统矩阵
                if hasattr(solver, 'G') and solver.G is not None:
                    print("  保存系统矩阵G...")
                    db_handler.save_system_matrix(
                        config_id, 'G', solver.G.toarray() if hasattr(solver.G, 'toarray') else solver.G,
                        {'notes': '影响系数矩阵G，用于电势计算'}
                    )
                
                if hasattr(solver, 'H') and solver.H is not None:
                    print("  保存系统矩阵H...")
                    db_handler.save_system_matrix(
                        config_id, 'H', solver.H.toarray() if hasattr(solver.H, 'toarray') else solver.H,
                        {'notes': '影响系数矩阵H，用于电场计算'}
                    )
                
                # 保存空间采样点
                if spatial_samples:
                    print("  保存空间采样点...")
                    db_handler.save_spatial_samples(
                        config_id, spatial_samples,
                        sample_group="computed_samples",
                        sampling_method="random"
                    )
                
                # 查询并显示保存的数据
                simulations = db_handler.get_all_simulations()
                print(f"\n数据库中共有 {len(simulations)} 条仿真记录:")
                for sim in simulations[-3:]:  # 显示最近3条
                    print(f"  ID:{sim['sim_id']} | 半径:{sim['radius']}m | "
                          f"电压:{sim['voltage']}V | 误差:{sim.get('charge_error', 0):.2f}%")
                
            else:
                print("  ✗ 仿真数据保存失败")
                
        except Exception as e:
            print(f"保存到数据库失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if db_handler:
                db_handler.disconnect()
    
    return solver, viz, analyzer, spatial_samples

# ==================== 辅助函数 ====================

def get_quadrant(x, y, z):
    """根据坐标确定空间象限"""
    if x >= 0 and y >= 0 and z >= 0:
        return 'I'
    elif x < 0 and y >= 0 and z >= 0:
        return 'II'
    elif x < 0 and y < 0 and z >= 0:
        return 'III'
    elif x >= 0 and y < 0 and z >= 0:
        return 'IV'
    elif x >= 0 and y >= 0 and z < 0:
        return 'V'
    elif x < 0 and y >= 0 and z < 0:
        return 'VI'
    elif x < 0 and y < 0 and z < 0:
        return 'VII'
    elif x >= 0 and y < 0 and z < 0:
        return 'VIII'
    else:
        return 'center'


# ==================== 5. 主入口 ====================

def main():
    """主入口函数"""
    global SAVE_TO_DATABASE, HAS_DATABASE
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        # 运行完整模拟
        solver, viz, analyzer, spatial_samples = run_simulation()
        
        # 输出完成信息
        print("\n" + "=" * 70)
        print("✓ 所有渲染完成！")
        print(f"✓ 输出目录: {OUTPUT_DIR.absolute()}")
        print("✓ 包含文件：")
        for file in sorted(OUTPUT_DIR.rglob("*.png")):
            print(f"  - {file.relative_to(OUTPUT_DIR)}")
        for file in sorted(OUTPUT_DIR.rglob("*.html")):
            print(f"  - {file.relative_to(OUTPUT_DIR)}")
        
        # 输出数据库保存结果
        if SAVE_TO_DATABASE and HAS_DATABASE:
            print(f"✓ 数据库保存完成")
            print(f"✓ 生成 {len(spatial_samples)} 个空间采样点")
        
        print("=" * 70)
    except Exception as e:
        import traceback
        print(f"\n发生错误：{type(e).__name__}: {e}")
        print("完整错误信息：")
        traceback.print_exc()


# ==================== 6. 直接执行 ====================

if __name__ == "__main__":
    main()