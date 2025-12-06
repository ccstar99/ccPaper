# physics/line.py
import numpy as np
from typing import Union, List
from utils.constants import VACUUM_PERMITTIVITY

class LineCharge:
    """
    无限长均匀带电直线物理模型
    
    电场强度（高斯定理推导）：
        E = λ / (2πε₀r) · r̂
        其中r为到导线的垂直距离，方向沿径向
    
    电势：
        V = -λ / (2πε₀) · ln(r/r₀)
        其中r₀为参考距离（默认取导线半径）
        注意：无限长导线电势在无穷远处不收敛，必须指定参考点
    
    假设：
        - 导线沿z轴方向无限延伸
        - 线电荷密度λ均匀分布
        - 仅考虑径向电场，无轴向分量
    """
    
    def __init__(self, lambda_val: float, position: Union[List[float], np.ndarray], 
                 radius: float = 0.1, reference_radius: float = None):
        """
        初始化线电荷
        
        Args:
            lambda_val: 线电荷密度λ（C/m），正为负电荷
            position: 导线在xy平面的位置 [x0, y0]
            radius: 导线有效半径（m），用于奇点保护
            reference_radius: 电势参考距离r₀（默认=radius）
        """
        self.lambda_val = float(lambda_val)
        self.position = np.array(position, dtype=float).flatten()[:2]  # 仅xy坐标
        self.radius = float(radius)
        self.reference_radius = float(reference_radius) if reference_radius else self.radius
        
        # 预计算常数 λ/(2πε₀) 提升性能
        self.lambda_over_2pi_eps = self.lambda_val / (2 * np.pi * VACUUM_PERMITTIVITY)
        
        # 电势缩放因子（包含参考点）
        self.potential_prefactor = -self.lambda_over_2pi_eps
    
    def electric_field(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电场强度矢量 E = λ/(2πε₀r) · r̂
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电场矢量数组，形状与输入相同 [N, 3]
            注意：z分量始终为0（无限长导线假设）
        """
        points = np.atleast_2d(points)  # 统一转换为N×3数组
        r_xy = points[:, :2] - self.position  # xy平面投影矢量
        
        # 计算垂直距离
        r_mag = np.linalg.norm(r_xy, axis=1, keepdims=True)
        
        # 安全距离处理奇点
        safe_r_mag = np.maximum(r_mag, self.radius)
        
        # 计算电场大小 λ/(2πε₀r)
        E_magnitude = self.lambda_over_2pi_eps / safe_r_mag
        
        # 单位化方向矢量（仅xy平面）
        r_hat_xy = r_xy / safe_r_mag
        
        # 组装三维电场矢量（z分量为0）
        E_xy = E_magnitude * r_hat_xy
        E_z = np.zeros((len(points), 1))
        E = np.hstack((E_xy, E_z))
        
        # 导线内部电场为0
        E = np.where(r_mag < self.radius, 0.0, E)
        
        return E.squeeze()
    
    def potential(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电势 V = -λ/(2πε₀) · ln(r/r₀)
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电势值数组，形状为 (N,)
            注意：与点电荷不同，电势有负号且为对数关系
        """
        points = np.atleast_2d(points)
        r_xy = points[:, :2] - self.position
        r_mag = np.linalg.norm(r_xy, axis=1)
        
        # 安全距离
        safe_r_mag = np.maximum(r_mag, self.radius)
        
        # 计算电势 V = -λ/(2πε₀) * ln(r/r₀)
        # 使用log(r/r₀) = log(r) - log(r₀)
        V = self.potential_prefactor * np.log(safe_r_mag / self.reference_radius)
        
        return V.squeeze()
    
    def is_inside(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        判断点是否在导线内部（用于电场线终止条件）
        
        Args:
            points: 单个点或点数组
            
        Returns:
            布尔数组，True表示在导线内部（r < radius）
        """
        points = np.atleast_2d(points)
        r_xy = points[:, :2] - self.position
        r_mag = np.linalg.norm(r_xy, axis=1)
        return r_mag < self.radius
    
    def get_line_axis_points(self, z_range: List[float] = [-5, 5], num: int = 100) -> np.ndarray:
        """
        获取用于可视化导线的三维线段点集
        
        Args:
            z_range: 导线在z轴方向的延伸范围
            num: 采样点数
            
        Returns:
            三维坐标数组 [num, 3]
        """
        z_vals = np.linspace(z_range[0], z_range[1], num)
        line_points = np.column_stack((
            np.full(num, self.position[0]),
            np.full(num, self.position[1]),
            z_vals
        ))
        return line_points


# 使用示例与测试
if __name__ == "__main__":
    # 创建λ=+1e-9 C/m的线电荷，位于x=1, y=0
    line = LineCharge(lambda_val=1e-9, position=[1.0, 0.0], radius=0.05)
    
    # 测试单个点
    point = np.array([2, 0, 3])  # r=1.0m
    E = line.electric_field(point)
    V = line.potential(point)
    print(f"点{point}处：E={E}, |E|={np.linalg.norm(E):.2e} N/C")
    print(f"      V={V:.2e} V")
    print(f"正确性验证：E_magnitude ≈ λ/(2πε₀r) = {line.lambda_over_2pi_eps:.2e}/1.0 = {line.lambda_over_2pi_eps:.2e} N/C")
    
    # 测试电场方向
    print(f"\n电场方向：E_xy={E[:2]}，z分量={E[2]}（应为0）")
    
    # 测试点数组
    points = np.array([[2, 0, 0], [1.5, 0.5, 2], [3, 4, 0]])  # r=1.0, √(0.5²+0.5²), 5.0
    E_array = line.electric_field(points)
    V_array = line.potential(points)
    print(f"\n多点电场：\n{E_array}")
    print(f"\n多点电势：{V_array}")
    
    # 验证1/r关系
    r_vals = np.array([0.5, 1.0, 2.0, 5.0])
    test_points = np.column_stack((
        line.position[0] + r_vals,
        np.full_like(r_vals, line.position[1]),
        np.zeros_like(r_vals)
    ))
    E_mags = np.linalg.norm(line.electric_field(test_points), axis=1)
    
    print("\n距离反比验证（E ∝ 1/r）：")
    print(f"r\tdE∝1/r")
    for r, E_mag in zip(r_vals, E_mags):
        print(f"{r:.1f}\t{E_mag:.2e}")
    
    ratio = E_mags[0] / E_mags[2]
    expected_ratio = r_vals[2] / r_vals[0]
    print(f"\nr=0.5与r=2.0场强比：{ratio:.2f} (理论:{expected_ratio:.2f})")
    
    # 电势对数关系验证
    V_vals = line.potential(test_points)
    print("\n电势对数关系验证（V ∝ ln(r)）：")
    for r, V_val in zip(r_vals, V_vals):
        print(f"r={r:.1f}m, V={V_val:.2e} V")
    
    # 测试奇点保护
    point_near = np.array([1.01, 0, 0])  # r=0.01m < radius
    E_near = line.electric_field(point_near)
    V_near = line.potential(point_near)
    print(f"\n奇点保护测试（r=0.01m < radius={line.radius}m）：")
    print(f"E={E_near}, V={V_near:.2e}（应被saturate）")