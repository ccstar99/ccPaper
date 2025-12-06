# physics/point.py
import numpy as np
from typing import Union, List
from utils.constants import COULOMB_CONSTANT

class PointCharge:
    """
    点电荷物理模型
    电场强度：E = kQ/r² · r̂
    电势：V = kQ/r
    严格遵循距离反平方定律
    """
    
    def __init__(self, q: float, position: Union[List[float], np.ndarray], radius: float = 0.1):
        """
        初始化点电荷
        
        Args:
            q: 电荷量（库仑）
            position: 位置坐标 [x, y, z]
            radius: 电荷可视化半径（米），用于电场线起始点和奇点保护
        """
        self.q = float(q)
        self.position = np.array(position, dtype=float).flatten()
        self.radius = float(radius)
        
        # 预计算常数提升性能
        self.kq = COULOMB_CONSTANT * self.q
    
    def electric_field(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电场强度矢量 E = kQ/r² · r̂
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电场矢量数组，形状与输入points相同
        """
        points = np.atleast_2d(points)  # 统一转换为N×3数组
        r = points - self.position  # 相对位置矢量
        
        # 计算距离（列向量范数）
        r_mag = np.linalg.norm(r, axis=1, keepdims=True)
        
        # 安全处理奇点：距离小于半径时返回0
        # 这里使用np.where避免条件分支，保持向量化性能
        safe_r_mag = np.maximum(r_mag, self.radius)
        
        # 计算电场大小 kQ/r²
        E_magnitude = self.kq / (safe_r_mag  ** 2)
        
        # 单位化方向矢量（r̂），并处理零距离情况
        r_hat = r / safe_r_mag
        
        # 电场矢量
        E = E_magnitude * r_hat
        
        # 强制电荷内部电场为0（物理上未定义）
        E = np.where(r_mag < self.radius, 0.0, E)
        
        return E.squeeze()
    
    def potential(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电势 V = kQ/r
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            电势值数组，形状为 (N,)
        """
        points = np.atleast_2d(points)
        r = points - self.position
        r_mag = np.linalg.norm(r, axis=1)
        
        # 安全距离处理奇点
        safe_r_mag = np.maximum(r_mag, self.radius)
        
        # 计算电势
        V = self.kq / safe_r_mag
        
        # 电荷内部电势按表面值计算
        V = np.where(r_mag < self.radius, self.kq / self.radius, V)
        
        return V.squeeze()
    
    def is_inside(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        判断点是否在电荷内部（用于电场线终止条件）
        
        Args:
            points: 单个点或点数组
            
        Returns:
            布尔数组，True表示在内部
        """
        points = np.atleast_2d(points)
        r = points - self.position
        r_mag = np.linalg.norm(r, axis=1)
        return r_mag < self.radius


# 使用示例与测试
if __name__ == "__main__":
    # 创建+1C点电荷位于原点
    charge = PointCharge(q=1.0, position=[0, 0, 0], radius=0.1)
    
    # 测试单个点
    point = np.array([1, 0, 0])
    E = charge.electric_field(point)
    V = charge.potential(point)
    print(f"点{point}处：E={E}, V={V:.2e}")
    
    # 测试点数组（批量计算）
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    E_array = charge.electric_field(points)
    V_array = charge.potential(points)
    print(f"\n多点电场：\n{E_array}")
    print(f"\n多点电势：{V_array}")
    
    # 验证距离反平方律
    r_values = np.array([0.5, 1.0, 2.0, 5.0])
    test_points = np.column_stack((r_values, np.zeros_like(r_values), np.zeros_like(r_values)))
    E_magnitudes = np.linalg.norm(charge.electric_field(test_points), axis=1)
    
    print("\n距离反平方验证：")
    for r, E in zip(r_values, E_magnitudes):
        print(f"r={r:.1f}m, |E|={E:.2e} N/C")
    
    # 理论值：E ∝ 1/r²
    ratio = E_magnitudes[0] / E_magnitudes[1]
    expected_ratio = (r_values[1]/r_values[0])**2
    print(f"\nr=0.5与r=1.0场强比：{ratio:.2f} (理论:{expected_ratio:.2f})")