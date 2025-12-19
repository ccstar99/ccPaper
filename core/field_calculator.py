# core/field_calculator.py
import numpy as np
from typing import List, Union, Protocol, runtime_checkable

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
