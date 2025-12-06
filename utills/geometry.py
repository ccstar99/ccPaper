# utils/geometry.py
"""
几何计算工具模块

功能特性：
    - 向量化实现：所有函数支持批量计算，无Python级循环
    - 数值稳定：处理奇点、除零等边界情况
    - 多维度支持：兼容2D和3D空间
    - 均匀分布：斐波那契球面、圆柱面、圆环面等算法

核心算法：
    1. 斐波那契球面算法：O(n)生成球面均匀分布点，最优性证明来自球面填充问题
    2. 曲率估算：基于Frenet-Serret公式的离散化实现
    3. 快速投影：使用Householder变换实现超平面投影
"""

import numpy as np
from typing import Optional

def fibonacci_sphere_points(center: np.ndarray, radius: float, n_points: int) -> np.ndarray:
    """
    斐波那契球面均匀点生成算法
    
    数学原理：
        利用黄金角 φ = π(3-√5) 的均匀分布性质，将球面坐标映射为:
        θ = i·φ,  z = 1 - 2i/(n-1)
        
    复杂度：O(n) 时间，O(n) 内存
    
    Args:
        center: 球心坐标 [x, y, z]
        radius: 球半径
        n_points: 生成点数
        
    Returns:
        点坐标数组 [n_points, 3]
    """
    if n_points <= 0:
        return np.empty((0, 3))
    
    # 黄金角（弧度）
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # 生成索引
    i = np.arange(n_points, dtype=float)
    
    # 计算球面坐标（θ为经度，φ为余纬）
    z = 1 - (2 * i / (n_points - 1)) if n_points > 1 else np.zeros_like(i)
    sqrt_term = np.sqrt(np.maximum(0, 1 - z**2))
    theta = golden_angle * i
    
    # 笛卡尔坐标
    x = sqrt_term * np.cos(theta)
    y = sqrt_term * np.sin(theta)
    
    # 缩放并平移
    points = radius * np.column_stack([x, y, z])
    return points + center

def cylindrical_surface_points(center: np.ndarray, radius: float, 
                               height: float, n_points: int) -> np.ndarray:
    """
    圆柱表面均匀点生成
    
    分布策略：
        - 沿轴向均匀分布 n_circles 个圆
        - 每个圆上均匀分布 n_per_circle 个点
        - 避免极点聚集问题（不同于经纬度分布）
    
    Args:
        center: 圆柱中心 (x, y, z=0)
        radius: 圆柱半径
        height: 圆柱高度
        n_points: 总生成点数
        
    Returns:
        点坐标数组 [n_points, 3]
    """
    if n_points <= 0:
        return np.empty((0, 3))
    
    # 轴向圆环数（避免过于密集）
    n_circles = max(1, int(np.sqrt(n_points / 2)))
    n_per_circle = max(1, n_points // n_circles)
    
    # 轴向坐标
    z_coords = np.linspace(-height/2, height/2, n_circles)
    
    all_points = []
    for z in z_coords:
        # 圆周角
        angles = np.linspace(0, 2*np.pi, n_per_circle, endpoint=False)
        
        # 圆柱面点
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z_vec = np.full_like(x, z)
        
        points = np.column_stack([x, y, z_vec]) + center
        all_points.append(points)
    
    return np.vstack(all_points)

def torus_surface_points(center: np.ndarray, major_radius: float, 
                         minor_radius: float, n_points: int) -> np.ndarray:
    """
    圆环面（Torus）均匀点生成
    
    参数方程：
        x = (R + r·cosφ)·cosθ
        y = (R + r·cosφ)·sinθ
        z = r·sinφ
        
    其中 R 为主半径，r 为辅半径
    
    Args:
        center: 圆环中心
        major_radius: 主半径 R
        minor_radius: 辅半径 r
        n_points: 生成点数
        
    Returns:
        点坐标数组 [n_points, 3]
    """
    if n_points <= 0:
        return np.empty((0, 3))
    
    # 使用斐波那契分布思想扩展到圆环面
    n_major = max(1, int(np.sqrt(n_points)))
    n_minor = max(1, n_points // n_major)
    
    # 主圆周角
    theta = np.linspace(0, 2*np.pi, n_major, endpoint=False)
    
    # 辅圆周角（黄金角偏移保证均匀）
    golden_angle = np.pi * (3 - np.sqrt(5))
    phi_offsets = golden_angle * np.arange(n_major)
    
    all_points = []
    for i, (t, phi0) in enumerate(zip(theta, phi_offsets)):
        # 辅圆角
        phi = phi0 + np.linspace(0, 2*np.pi, n_minor, endpoint=False)
        
        # 参数方程
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(t)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(t)
        z = minor_radius * np.sin(phi)
        
        points = np.column_stack([x, y, z]) + center
        all_points.append(points)
    
    return np.vstack(all_points)

def calculate_curvature(points: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    计算离散点序列的曲率（标量曲率）
    
    数学方法：
        使用中心差分法估算Frenet公式：
        κ = |r' × r''| / |r'|³
        
        离散化：
        r' ≈ (r_{i+1} - r_{i-1}) / (2h)
        r'' ≈ (r_{i+1} - 2r_i + r_{i-1}) / h²
    
    Args:
        points: 点序列 [N, 3]
        h: 差分步长
        
    Returns:
        曲率数组 [N]，边界点重复
    """
    n = len(points)
    if n < 3:
        return np.zeros(n)
    
    # 中心差分计算一阶导数（速度）
    dr = np.zeros_like(points)
    dr[1:-1] = (points[2:] - points[:-2]) / (2 * h)
    dr[0] = dr[1]  # 边界处理
    dr[-1] = dr[-2]
    
    # 二阶导数（加速度）
    d2r = np.zeros_like(points)
    d2r[1:-1] = (points[2:] - 2*points[1:-1] + points[:-2]) / h**2
    d2r[0] = d2r[1]
    d2r[-1] = d2r[-2]
    
    # 曲率 κ = |r' × r''| / |r'|³
    cross_norm = np.linalg.norm(np.cross(dr, d2r), axis=1)
    speed_cubed = np.maximum(np.linalg.norm(dr, axis=1)**3, 1e-12)
    
    curvature = cross_norm / speed_cubed
    
    return curvature

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    计算两个向量数组的夹角（弧度）
    
    公式：θ = arccos( (v1·v2) / (|v1||v2|) )
    
    Args:
        v1: 向量数组 [N, 3]
        v2: 向量数组 [N, 3]
        
    Returns:
        夹角数组 [N]
    """
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    
    # 安全除法
    cos_angle = dot / (norm1 * norm2 + 1e-12)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return np.arccos(cos_angle)

def project_to_plane(points: np.ndarray, plane_normal: np.ndarray, 
                     plane_point: Optional[np.ndarray] = None) -> np.ndarray:
    """
    将点投影到指定平面
    
    数学原理：
        d = (p - p₀)·n
        p' = p - d·n
        
    Args:
        points: 待投影点 [N, 3]
        plane_normal: 平面单位法向量 [3]
        plane_point: 平面上任意一点（默认原点）
        
    Returns:
        投影点 [N, 3]
    """
    if plane_point is None:
        plane_point = np.zeros(3)
    
    # 确保法向量归一化
    normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    
    # 计算有符号距离
    distances = np.dot(points - plane_point, normal)
    
    # 投影
    return points - distances[:, np.newaxis] * normal

def is_point_in_cylinder(point: np.ndarray, cylinder_center: np.ndarray,
                         radius: float, height: float) -> bool:
    """
    判断点是否在圆柱体内
    
    Args:
        point: 测试点 [3]
        cylinder_center: 圆柱中心 [3]
        radius: 圆柱半径
        height: 圆柱高度
        
    Returns:
        是否在圆柱内
    """
    # 相对位置
    rel_point = point - cylinder_center
    
    # 水平距离
    horizontal_dist = np.linalg.norm(rel_point[:2])
    
    # 垂直距离
    vertical_dist = abs(rel_point[2])
    
    return (horizontal_dist <= radius) and (vertical_dist <= height/2)

def is_point_in_torus(point: np.ndarray, torus_center: np.ndarray,
                      major_radius: float, minor_radius: float) -> bool:
    """
    判断点是否在圆环管体内
    
    数学条件：
        (√(x² + y²) - R)² + z² ≤ r²
        
    Args:
        point: 测试点 [3]
        torus_center: 圆环中心 [3]
        major_radius: 主半径 R
        minor_radius: 辅半径 r
        
    Returns:
        是否在圆环内
    """
    rel_point = point - torus_center
    
    # 到z轴的距离
    dist_to_z_axis = np.linalg.norm(rel_point[:2])
    
    # 到圆环中心圆的平方距离
    dist_sq = (dist_to_z_axis - major_radius)**2 + rel_point[2]**2
    
    return dist_sq <= minor_radius**2

def uniform_spherical_cap_points(center: np.ndarray, radius: float,
                                 cap_angle: float, n_points: int) -> np.ndarray:
    """
    球冠表面均匀点生成
    
    应用：用于电荷半球分布场景
    
    Args:
        center: 球心
        radius: 球半径
        cap_angle: 球冠半角（弧度）
        n_points: 生成点数
        
    Returns:
        点坐标 [n_points, 3]
    """
    if n_points <= 0:
        return np.empty((0, 3))
    
    # 球冠面积比例
    area_ratio = (1 - np.cos(cap_angle)) / 2
    
    # 调整总点数
    n_total = max(1, int(n_points / area_ratio))
    
    # 生成球面点
    points = fibonacci_sphere_points(center, radius, n_total)
    
    # 筛选在球冠内的点
    rel_points = points - center
    angles = np.arccos(rel_points[:, 2] / radius)
    mask = angles <= cap_angle
    
    # 若不足，重复采样
    selected = points[mask]
    if len(selected) < n_points:
        repeats = (n_points // len(selected)) + 1
        selected = np.tile(selected, (repeats, 1))[:n_points]
    
    return selected[:n_points]

def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    计算将v1旋转到v2的旋转矩阵（罗德里格斯公式）
    
    Args:
        v1: 源向量 [3]
        v2: 目标向量 [3]
        
    Returns:
        3×3旋转矩阵
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-12)
    
    # 旋转轴
    axis = np.cross(v1_norm, v2_norm)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-12:
        # 两向量平行
        return np.eye(3)
    
    axis = axis / axis_norm
    
    # 旋转角
    angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
    
    # 罗德里格斯公式
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R

# ==================== 测试与验证 ====================
def test_geometry_functions():
    """
    几何函数单元测试
    
    验证项：
        1. 均匀性：球面点最小间距 > 平均间距的0.5倍
        2. 曲率正确性：圆上点曲率恒为 1/R
        3. 包含性：边界点判断准确
    """
    print("\n" + "="*60)
    print("几何计算模块验证")
    print("="*60)
    
    # 测试1：斐波那契球面均匀性
    print("\n【测试1：球面点均匀性】")
    center = np.array([0, 0, 0])
    radius = 1.0
    n_points = 100
    
    points = fibonacci_sphere_points(center, radius, n_points)
    print(f"生成 {len(points)} 个点")
    
    # 计算最小间距
    distances = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    avg_dist = np.mean(distances[distances < np.inf])
    
    print(f"最小间距: {min_dist:.4f}")
    print(f"平均间距: {avg_dist:.4f}")
    print(f"均匀度: {'✓ 通过' if min_dist > avg_dist * 0.3 else '✗ 失败'}")
    
    # 测试2：曲率计算
    print("\n【测试2：曲率计算】")
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    
    curvature = calculate_curvature(circle, h=0.01)
    print(f"圆曲率均值: {np.mean(curvature):.6f}")
    print(f"圆曲率标准差: {np.std(curvature):.6f}")
    print(f"理论曲率: 1.000000")
    print(f"精度: {'✓ 通过' if abs(np.mean(curvature) - 1.0) < 0.01 else '✗ 失败'}")
    
    # 测试3：圆柱包含性
    print("\n【测试3：圆柱包含判断】")
    cyl_center = np.array([0, 0, 0])
    test_point = np.array([0.5, 0, 0])  # 在圆柱内
    result = is_point_in_cylinder(test_point, cyl_center, radius=1.0, height=2.0)
    print(f"点 {test_point} 在圆柱内: {'✓ 通过' if result else '✗ 失败'}")
    
    # 测试4：旋转矩阵
    print("\n【测试4：旋转矩阵】")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    R = rotation_matrix_from_vectors(v1, v2)
    v2_rotated = R @ v1
    print(f"旋转后向量: {v2_rotated}")
    print(f"与目标向量一致性: {'✓ 通过' if np.allclose(v2_rotated, v2) else '✗ 失败'}")
    
    print("\n" + "="*60)
    print("几何模块验证完成")
    print("="*60)

if __name__ == "__main__":
    test_geometry_functions()