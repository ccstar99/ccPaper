# core/field_calculator.py
import numpy as np
from typing import List, Union, Protocol, runtime_checkable
from utils.constants import COULOMB_CONSTANT
from physics.point import PointCharge
from physics.line import LineCharge
from physics.ring import RingCharge

@runtime_checkable
class ChargeProtocol(Protocol):
    """协议类：所有电荷模型必须实现此接口"""
    def electric_field(self, points: np.ndarray) -> np.ndarray: ...
    q: float  # 允许实例变量或属性

class FieldCalculator:
    """
    电场强度矢量计算器（支持多电荷叠加原理）
    
    数学原理：
        电场叠加原理：E_total = Σ E_i
        库仑定律（矢量形式）：E_i = k·q_i·r̂ / r²
    
    支持电荷类型：
        - PointCharge：点电荷
        - LineCharge：无限长线电荷
        - RingCharge：圆环电荷
    
    核心特性：
        - 完全向量化运算，无Python循环
        - 支持批量点计算
        - 严格遵循国际单位制
    """
    
    def __init__(self, charges: List[ChargeProtocol] = None):
        """
        初始化场计算器
        
        Args:
            charges: 电荷对象列表，每个对象需实现electric_field方法
        """
        self.charges: List[ChargeProtocol] = charges if charges is not None else []
    
    def add_charge(self, charge: ChargeProtocol):
        """添加单个电荷"""
        self.charges.append(charge)
    
    def add_charges(self, charges: List[ChargeProtocol]):
        """批量添加电荷"""
        self.charges.extend(charges)
    
    def electric_field(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算多电荷在空间点产生的总电场（矢量叠加）
        
        Args:
            points: 单个点 [x,y,z] 或点数组 N×3
            
        Returns:
            总电场矢量，形状与输入相同
            - 单点: [3,]
            - 多点: [N, 3]
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        
        # 零场初始化
        total_E = np.zeros_like(points)
        
        # 叠加所有电荷的贡献（核心算法）
        for charge in self.charges:
            # 每个charge.electric_field已经是向量化实现
            total_E += charge.electric_field(points)
        
        return total_E.squeeze()
    
    def field_magnitude(self, points: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        计算电场强度大小 |E|
        
        Args:
            points: 空间点坐标
            
        Returns:
            标量场强数组
        """
        E = self.electric_field(points)
        return np.linalg.norm(E, axis=-1)
    
    def field_line_equation(self, start_point: np.ndarray, 
                           direction: float = 1.0,
                           step_size: float = 0.01,
                           max_steps: int = 1000) -> np.ndarray:
        """
        计算单条电场线（简化版欧拉积分，用于验证）
        
        Args:
            start_point: 起始点
            direction: +1（正向）或 -1（反向）
            step_size: 积分步长
            max_steps: 最大步数
            
        Returns:
            电场线路径点数组
        """
        path = [start_point]
        point = np.array(start_point, dtype=float)
        
        for _ in range(max_steps):
            E = self.electric_field(point)
            E_mag = np.linalg.norm(E)
            
            if E_mag < 1e-10:  # 场强太弱则终止
                break
                
            # 归一化方向并移动
            direction_vec = direction * E / E_mag
            point = point + step_size * direction_vec
            path.append(point.copy())
        
        return np.array(path)


# 参考算法：库仑定律标量力（力的大小，非场）
# 此函数仅为演示，实际应使用矢量化的electric_field方法
def coulombs_law_force(q1: float, q2: float, radius: float) -> float:
    """
    计算两点电荷间的库仑力大小（标量）
    
    这是传统标量版，用于教学对比。实际计算请使用FieldCalculator.
    
    Args:
        q1: 源电荷量（C）
        q2: 测试电荷量（C）
        radius: 电荷间距离（m），必须>0
        
    Returns:
        力的大小（N），正为排斥，负为吸引
        
    Raises:
        ValueError: radius <= 0
        
    Examples:
        >>> coulombs_law_force(15.5, 20, 15)
        12382849136.06
        >>> coulombs_law_force(1, 1, 1)  # 测试电荷为1C时即为场强大小
        8987551792.3
    """
    if radius <= 0:
        raise ValueError("距离必须是正数")
    
    # F = k·q₁·q₂ / r²
    # 当q₂=1C时，F即为电场强度E的大小
    force = (COULOMB_CONSTANT * q1 * q2) / (radius  ** 2)
    return round(force, 2)


# 性能对比测试：向量化 vs 标量循环
def performance_comparison():
    """
    演示向量化计算相对于标量循环的性能优势
    """
    from time import time
    
    # 创建几个电荷
    charges = [
        PointCharge(1e-6, [0, 0, 0]),
        PointCharge(-0.5e-6, [1, 0, 0]),
        LineCharge(1e-9, [0.5, 0.5]),
        RingCharge(0.8e-6, 0.3, position=[0, 0.5])
    ]
    
    calculator = FieldCalculator(charges)
    
    # 生成测试点（1000个点）
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    z = np.linspace(-1, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # 向量化计算
    t0 = time()
    E_vectorized = calculator.electric_field(points)
    t1 = time()
    
    # 标量循环计算（模拟传统方法）
    t2 = time()
    E_scalar = np.zeros_like(points)
    for i, pt in enumerate(points):
        E_scalar[i] = calculator.electric_field(pt)
    t3 = time()
    
    print(f"向量化计算 {len(points)} 个点: {(t1-t0)*1000:.2f} ms")
    print(f"标量循环计算: {(t3-t2)*1000:.2f} ms")
    print(f"加速比: {(t3-t2)/(t1-t0):.1f}x")
    print(f"结果一致性（最大误差）: {np.max(np.abs(E_vectorized - E_scalar)):.2e}")


if __name__ == "__main__":
    # 单元测试
    import doctest
    doctest.testmod()
    
    # 功能性验证
    print("=== 电场计算器测试 ===\n")
    
    # 创建电荷系统
    charges = [
        PointCharge(q=1e-6, position=[0, 0, 0], radius=0.1),
        PointCharge(q=-0.5e-6, position=[1, 0, 0], radius=0.1),
        RingCharge(q=0.8e-6, radius=0.3, position=[0, 0.5])
    ]
    
    calc = FieldCalculator(charges)
    
    # 测试点
    test_point = np.array([0.5, 0.5, 0])
    E_total = calc.electric_field(test_point)
    
    # 手动叠加验证
    E1 = charges[0].electric_field(test_point)
    E2 = charges[1].electric_field(test_point)
    E3 = charges[2].electric_field(test_point)
    E_manual = E1 + E2 + E3
    
    print(f"测试点: {test_point}")
    print(f"计算器结果: {E_total}")
    print(f"手动叠加结果: {E_manual}")
    print(f"一致性验证: {np.allclose(E_total, E_manual)}")
    print(f"场强大小: {np.linalg.norm(E_total):.6e} N/C")
    
    # 对比标量库仑定律
    r1 = np.linalg.norm(test_point - charges[0].position)
    r2 = np.linalg.norm(test_point - charges[1].position)
    
    # 注意：标量力是q1*q2，场是q1*1
    F1 = coulombs_law_force(charges[0].q, 1.0, r1)
    F2 = coulombs_law_force(charges[1].q, 1.0, r2)
    print(f"\n标量力验证（q2=1C）：")
    print(f"  电荷1产生的力大小: {F1:.2f} N（即场强大小）")
    print(f"  电荷2产生的力大小: {F2:.2f} N（即场强大小）")
    print(f"  与矢量场大小误差: {abs(F1 - np.linalg.norm(E1)):.2e}")