# core/potential_calculator.py
import numpy as np
from typing import List, Union, Protocol, runtime_checkable


@runtime_checkable
class ChargeProtocol(Protocol):
    """
    电荷协议：所有电荷模型必须实现此接口
    通过runtime_checkable支持运行时类型检查
    """
    def potential(self, points: np.ndarray) -> np.ndarray: ...
    def electric_field(self, points: np.ndarray) -> np.ndarray: ...

class PotentialCalculator:
    """
    电势标量计算器（严格遵循叠加原理与保守场理论）
    
    物理原理：
        1. 电势叠加原理：V_total(P) = Σ V_i(P)
        2. 电场-电势关系：E = -∇V （微分形式）
        3. 库仑势：V = k·q/r （点电荷）
        4. 电势相对性：电势值依赖于参考点选择，仅差值具有物理意义
    """
    
    def __init__(self, charges: List[ChargeProtocol] = None, 
                 reference_point: Union[List[float], np.ndarray] = None):
        """
        初始化电势计算器
        
        Args:
            charges: 电荷对象列表，每个必须实现potential方法
            reference_point: 电势参考点，默认为无穷远（V=0）
                           格式：[x_ref, y_ref, z_ref]
                           若指定，则所有电势值相对于该点
        """
        self.charges: List[ChargeProtocol] = charges if charges is not None else []
        self.reference_point = (
            np.array(reference_point, dtype=float) if reference_point is not None else None
        )
        
        # 参考点电势值（用于相对化）
        self._reference_potential = None
    
    def potential(self, points: Union[List[float], np.ndarray], 
                  absolute: bool = False) -> np.ndarray:
        """
        计算总电势 V_total = Σ V_i
        
        数学实现：
            V = Σ k·q_i/r_i  （对每种电荷类型使用其特定公式）
        
        Args:
            points: 空间点，支持单点[x,y,z]或多点N×3数组
            absolute: 是否返回绝对电势（相对于参考点）
                     False时：V = V_abs - V_ref
                     True时：V = V_abs
        
        Returns:
            标量电势数组，形状 (N,)
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        
        # 初始化零电势标量场
        V_total = np.zeros(len(points), dtype=np.float64)
        
        # 核心叠加：遍历所有电荷贡献
        for charge in self.charges:
            V_total += charge.potential(points)
        
        # 相对化处理（若指定参考点）
        if not absolute and self.reference_point is not None:
            # 懒计算参考点电势
            if self._reference_potential is None:
                self._reference_potential = self.potential(
                    self.reference_point.reshape(1, -1), absolute=True
                )[0]
            
            # 减去参考点电势
            V_total -= self._reference_potential
        
        return V_total.squeeze()
