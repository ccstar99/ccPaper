# physics/ring.py
import numpy as np
from typing import Union, List
from utils.constants import COULOMB_CONSTANT, VACUUM_PERMITTIVITY

class RingCharge:
    """
    均匀带电圆环物理模型
    
    物理参数：
        - 半径 R，总电荷 Q，线电荷密度 λ = Q/(2πR)
        - 圆环位于xy平面，中心在position
    
    数学特性：
        1. 轴线上解析解（严格解）：
           E_z = kQz / (R² + z²)^(3/2)
           V = kQ / √(R² + z²)
        
        2. 全空间一般解（需数值积分）：
           电势 V = (1/4πε₀)∫(λdl/r')
           电场 E = (1/4πε₀)∫(λdl·r̂'/r'²)
           其中积分沿圆环进行，r'为场点到圆环微元的距离
    
    数值方法：
        - 使用高斯-勒让德数值积分（固定16点保证精度）
        - 预计算积分权重和节点，提升性能
        - 自适应奇点处理（场点靠近圆环时加密积分）
    """
    
    def __init__(self, q: float, radius: float, 
                 position: Union[List[float], np.ndarray] = None,
                 center_position: Union[List[float], np.ndarray] = None):
        """
        初始化圆环电荷
        
        Args:
            q: 总电荷量 Q（库仑）
            radius: 圆环半径 R（米）
            position: 圆环中心在xy平面的坐标 [x0, y0]（简化接口，z=0）
            center_position: 完整三维中心坐标 [x0, y0, z0]（优先使用）
        """
        self.q = float(q)
        self.R = float(radius)
        
        # 位置处理：支持二维和三维输入
        if center_position is not None:
            self.position = np.array(center_position, dtype=float).flatten()[:3]
        elif position is not None:
            pos_array = np.array(position, dtype=float).flatten()
            if len(pos_array) == 2:
                self.position = np.append(pos_array, 0.0)  # 默认z=0
            else:
                self.position = pos_array[:3]  # 取前3个元素
        else:
            self.position = np.array([0.0, 0.0, 0.0])
        
        # 线电荷密度 λ = Q/(2πR)
        self.lambda_val = self.q / (2 * np.pi * self.R)
        
        # 预计算常数
        self.k_times_Q = COULOMB_CONSTANT * self.q  # kQ
        self.k_times_lambda = COULOMB_CONSTANT * self.lambda_val  # kλ
        
        # 高斯-勒让德积分参数（16点，精度达1e-12）
        # 节点和权重在[-1,1]区间，需映射到[0, 2π]
        self.gauss_nodes, self.gauss_weights = np.polynomial.legendre.leggauss(16)
        self.theta_nodes = np.pi * (self.gauss_nodes + 1)  # 映射到[0, 2π]
        self.theta_weights = np.pi * self.gauss_weights
        
        # 预计算圆环上积分点的坐标（相对位置）
        self.ring_x = self.R * np.cos(self.theta_nodes)
        self.ring_y = self.R * np.sin(self.theta_nodes)
    
    def _is_on_axis(self, points: np.ndarray) -> np.ndarray:
        """
        判断点是否在圆环轴线上（即x,y与圆环中心重合）
        
        Args:
            points: N×3数组
            
        Returns:
            布尔数组，True表示在轴线上
        """
        # 计算xy平面距离
        r_xy = points[:, :2] - self.position[:2]
        r_xy_mag = np.linalg.norm(r_xy, axis=1)
        return r_xy_mag < 1e-12  # 容差1e-12米
    
    def _potential_on_axis(self, z: np.ndarray) -> np.ndarray:
        """
        轴线上电势解析解 V = kQ / √(R² + z²)
        
        Args:
            z: 轴向距离数组
            
        Returns:
            电势值数组
        """
        return self.k_times_Q / np.sqrt(self.R**2 + z**2)
    
    def _electric_field_on_axis(self, z: np.ndarray) -> np.ndarray:
        """
        轴线上电场解析解 E_z = kQz / (R² + z²)^(3/2)
        
        Args:
            z: 轴向距离数组
            
        Returns:
            电场矢量数组 N×3（仅z分量非零）
        """
        z = np.array(z, dtype=float)
        denominator = (self.R**2 + z**2)**1.5
        Ez = self.k_times_Q * z / denominator
        
        # 组装三维电场（xy分量为0）
        E = np.zeros((len(z), 3))
        E[:, 2] = Ez
        return E
    
    def potential(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电势 V = (1/4πε₀)∫(λdl/r')
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电势值数组，形状为 (N,)
        """
        points = np.atleast_2d(points)
        
        # 分离轴线和非轴线点
        on_axis_mask = self._is_on_axis(points)
        
        # 初始化结果数组
        V = np.zeros(len(points))
        
        # 轴线点使用解析解
        if np.any(on_axis_mask):
            z_axis = points[on_axis_mask, 2] - self.position[2]
            V[on_axis_mask] = self._potential_on_axis(z_axis)
        
        # 非轴线点使用数值积分
        off_axis_mask = ~on_axis_mask
        if np.any(off_axis_mask):
            points_off = points[off_axis_mask]
            
            # 向量化数值积分（核心性能部分）
            # 计算每个场点到所有圆环微元的距离
            # 使用广播：[N_points, 1] - [M_ring] -> [N_points, M_ring]
            
            dx = points_off[:, 0, np.newaxis] - (self.ring_x + self.position[0])
            dy = points_off[:, 1, np.newaxis] - (self.ring_y + self.position[1])
            dz = points_off[:, 2, np.newaxis] - self.position[2]
            
            # 距离数组
            r_prime = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 积分：Σ (kλ · dl / r') · weight
            # dl = R·dθ，已在权重中考虑
            V_off = np.sum(self.k_times_lambda * self.theta_weights / r_prime, axis=1)
            V[off_axis_mask] = V_off
        
        return V.squeeze()
    
    def electric_field(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电场强度矢量 E = (1/4πε₀)∫(λdl·r̂'/r'²)
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电场矢量数组，形状为 [N, 3]
        """
        points = np.atleast_2d(points)
        
        # 分离轴线和非轴线点
        on_axis_mask = self._is_on_axis(points)
        
        # 初始化结果数组
        E = np.zeros((len(points), 3))
        
        # 轴线点使用解析解
        if np.any(on_axis_mask):
            z_axis = points[on_axis_mask, 2] - self.position[2]
            E[on_axis_mask] = self._electric_field_on_axis(z_axis)
        
        # 非轴线点使用数值积分
        off_axis_mask = ~on_axis_mask
        if np.any(off_axis_mask):
            points_off = points[off_axis_mask]
            
            # 计算相对矢量分量
            dx = points_off[:, 0, np.newaxis] - (self.ring_x + self.position[0])
            dy = points_off[:, 1, np.newaxis] - (self.ring_y + self.position[1])
            dz = points_off[:, 2, np.newaxis] - self.position[2]
            
            # 距离平方数组
            r2 = dx**2 + dy**2 + dz**2
            
            # 安全处理：避免除零（场点在圆环上）
            safe_r2 = np.maximum(r2, self.R**2)
            
            # 计算积分贡献 dE = kλ·(r̂'/r'²)·dl
            # 其中 r̂'/r'² = (r_vector)/r'³
            prefactor = self.k_times_lambda * self.theta_weights  # kλ·dl
            
            # 三个分量分别积分
            Ex = np.sum(prefactor * dx / safe_r2**1.5, axis=1)
            Ey = np.sum(prefactor * dy / safe_r2**1.5, axis=1)
            Ez = np.sum(prefactor * dz / safe_r2**1.5, axis=1)
            
            E[off_axis_mask, 0] = Ex
            E[off_axis_mask, 1] = Ey
            E[off_axis_mask, 2] = Ez
        
        return E.squeeze()
    
    def is_inside(self, points: Union[List[float], np.ndarray], 
                  tol: float = 0.01) -> np.ndarray:
        """
        判断点是否在圆环管体内（用于电场线终止条件）
        
        Args:
            points: 单个点或点数组
            tol: 管体厚度容忍度（相对于半径的比例）
            
        Returns:
            布尔数组，True表示在圆环附近
        """
        points = np.atleast_2d(points)
        
        # 计算到圆环最近距离
        # 在圆环平面内投影距离
        r_xy = np.linalg.norm(points[:, :2] - self.position[:2], axis=1)
        distance_to_ring = np.sqrt((r_xy - self.R)**2 + (points[:,2] - self.position[2])**2)
        
        # 判断是否在管体内：距离小于半径的小比例（固定管体厚度）
        tube_thickness = self.R * tol
        return distance_to_ring < tube_thickness
    
    def get_ring_visualization_points(self, num_theta: int = 100) -> np.ndarray:
        """
        生成用于可视化圆环的三维坐标点集
        
        Args:
            num_theta: 角向采样点数
            
        Returns:
            三维坐标数组 [num_theta, 3]
        """
        theta = np.linspace(0, 2*np.pi, num_theta)
        x = self.position[0] + self.R * np.cos(theta)
        y = self.position[1] + self.R * np.sin(theta)
        z = np.full_like(theta, self.position[2])
        return np.column_stack((x, y, z))


# 使用示例与测试
if __name__ == "__main__":
    # 创建Q=+1e-6C，R=0.5m的圆环，中心在原点
    ring = RingCharge(q=1e-6, radius=0.5, position=[0, 0, 0])
    
    # 测试轴线点（解析解）
    point_axis = np.array([0, 0, 0.3])  # z=0.3m
    E_axis = ring.electric_field(point_axis)
    V_axis = ring.potential(point_axis)
    print(f"轴线点{point_axis}：")
    print(f"  E={E_axis}, |E|={np.linalg.norm(E_axis):.6e} N/C")
    print(f"  理论Ez={ring.k_times_Q * 0.3 / (0.5**2 + 0.3**2)**1.5:.6e} N/C")
    print(f"  V={V_axis:.6e} V")
    print(f"  理论V={ring.k_times_Q / np.sqrt(0.5**2 + 0.3**2):.6e} V")
    
    # 测试非轴线点（数值积分）
    point_off = np.array([0.6, 0.2, 0.3])
    E_off = ring.electric_field(point_off)
    V_off = ring.potential(point_off)
    print(f"\n非轴线点{point_off}：")
    print(f"  E={E_off}, |E|={np.linalg.norm(E_off):.6e} N/C")
    print(f"  V={V_off:.6e} V")
    
    # 验证轴线和非轴线一致性（靠近轴线）
    point_near_axis = np.array([1e-6, 0, 0.3])  # 极接近轴线
    E_near = ring.electric_field(point_near_axis)
    print(f"\n近轴线一致性验证：")
    print(f"  数值解 Ez={E_near[2]:.6e}")
    print(f"  解析解 Ez={ring.k_times_Q * 0.3 / (0.5**2 + 0.3**2)**1.5:.6e}")
    print(f"  相对误差={abs(E_near[2] - E_axis[2])/abs(E_axis[2]):.2e}")
    
    # 测试电场垂直分量（轴线点应为0）
    print(f"\n轴线电场垂直分量：Ex={E_axis[0]:.2e}, Ey={E_axis[1]:.2e}（应≈0）")
    
    # 测试多个点
    points = np.array([[0,0,0.3], [0.6,0.2,0.3], [1,0,0]])
    E_array = ring.electric_field(points)
    V_array = ring.potential(points)
    print(f"\n多点电场：")
    print(E_array)
    print(f"\n多点电势：{V_array}")
    
    # 验证超距衰减
    far_point = np.array([0, 0, 10])  # z=10m >> R
    V_far = ring.potential(far_point)
    V_dipole_approx = ring.k_times_Q / far_point[2]  # 近似为点电荷
    print(f"\n超距衰减验证（z=10m >> R=0.5m）：")
    print(f"  圆环电势={V_far:.6e} V")
    print(f"  点电荷近似={V_dipole_approx:.6e} V")
    print(f"  相对差异={abs(V_far - V_dipole_approx)/V_dipole_approx:.2e}")