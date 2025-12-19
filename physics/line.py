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
        - 导线沿指定方向无限延伸（x轴、y轴或z轴）
        - 线电荷密度λ均匀分布
        - 仅考虑径向电场，无轴向分量
    """
    
    def __init__(self, lambda_val: float, position: Union[List[float], np.ndarray], 
                 radius: float = 0.1, reference_radius: float = None, direction: str = 'x'):
        """
        初始化线电荷
        
        Args:
            lambda_val: 线电荷密度λ（C/m），正为负电荷
            position: 导线在垂直于方向平面的位置 [x0, y0] 或 [x0, z0] 或 [y0, z0]
            radius: 导线有效半径（m），用于奇点保护
            reference_radius: 电势参考距离r₀（默认=radius）
            direction: 导线方向，可选值：'x', 'y', 'z'，默认沿x轴
        """
        self.lambda_val = float(lambda_val)
        # 确保position始终是3D坐标，不足3个元素用0填充
        pos_array = np.array(position, dtype=float).flatten()
        # 扩展为3D坐标
        if len(pos_array) < 3:
            pos_array = np.pad(pos_array, (0, 3 - len(pos_array)), 'constant')
        self.position = pos_array[:3]  # 确保只取前3个元素
        self.radius = float(radius)
        self.reference_radius = float(reference_radius) if reference_radius else self.radius
        self.direction = direction.lower()  # 标准化方向
        
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
            注意：轴向分量始终为0（无限长导线假设）
        """
        points = np.atleast_2d(points)  # 统一转换为N×3数组
        
        # 根据线电荷方向计算垂直距离和径向分量
        if self.direction == 'x':
            # 沿x轴方向：电场在y-z平面
            r_perp = points[:, 1:] - self.position[1:]  # y-z平面投影矢量
            r_mag = np.linalg.norm(r_perp, axis=1, keepdims=True)
        elif self.direction == 'y':
            # 沿y轴方向：电场在x-z平面
            r_perp = np.hstack((points[:, 0:1], points[:, 2:3])) - np.hstack((self.position[0:1], self.position[2:3]))
            r_mag = np.linalg.norm(r_perp, axis=1, keepdims=True)
        else:  # 'z'轴方向
            # 沿z轴方向：电场在x-y平面
            r_perp = points[:, :2] - self.position[:2]  # x-y平面投影矢量
            r_mag = np.linalg.norm(r_perp, axis=1, keepdims=True)
        
        # 安全距离处理奇点
        safe_r_mag = np.maximum(r_mag, self.radius)
        
        # 计算电场大小 λ/(2πε₀r)
        E_magnitude = self.lambda_over_2pi_eps / safe_r_mag
        
        # 单位化方向矢量（垂直于线电荷方向）
        r_hat_perp = r_perp / safe_r_mag
        
        # 组装三维电场矢量
        E = np.zeros_like(points)
        
        if self.direction == 'x':
            # 沿x轴：电场分量为 (0, Ey, Ez)
            E[:, 1:] = E_magnitude * r_hat_perp
        elif self.direction == 'y':
            # 沿y轴：电场分量为 (Ex, 0, Ez)
            E[:, 0] = r_hat_perp[:, 0] * E_magnitude.squeeze()
            E[:, 2] = r_hat_perp[:, 1] * E_magnitude.squeeze()
        else:  # 'z'轴
            # 沿z轴：电场分量为 (Ex, Ey, 0)
            E[:, :2] = E_magnitude * r_hat_perp
        
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
        
        # 根据线电荷方向计算垂直距离
        if self.direction == 'x':
            # 沿x轴方向：距离在y-z平面
            r_perp = points[:, 1:] - self.position[1:]
        elif self.direction == 'y':
            # 沿y轴方向：距离在x-z平面
            r_perp = np.hstack((points[:, 0:1], points[:, 2:3])) - np.hstack((self.position[0:1], self.position[2:3]))
        else:  # 'z'轴方向
            # 沿z轴方向：距离在x-y平面
            r_perp = points[:, :2] - self.position[:2]
        
        r_mag = np.linalg.norm(r_perp, axis=1)
        
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
        
        # 根据线电荷方向计算垂直距离
        if self.direction == 'x':
            # 沿x轴方向：距离在y-z平面
            r_perp = points[:, 1:] - self.position[1:]
        elif self.direction == 'y':
            # 沿y轴方向：距离在x-z平面
            r_perp = np.hstack((points[:, 0:1], points[:, 2:3])) - np.hstack((self.position[0:1], self.position[2:3]))
        else:  # 'z'轴方向
            # 沿z轴方向：距离在x-y平面
            r_perp = points[:, :2] - self.position[:2]
        
        r_mag = np.linalg.norm(r_perp, axis=1)
        return r_mag < self.radius
    
    def get_line_axis_points(self, length: float = 10.0, num_points: int = 200) -> np.ndarray:
        """
        获取线电荷的可视化线段点集
        
        由于无限长导线无法完全绘制，我们使用一条足够长的线段来表示它
        
        Args:
            length: 线段长度
            num_points: 线段上的点数量
            
        Returns:
            线段点集数组，形状为 (num_points, 3)
        """
        # 根据线电荷方向生成线段
        if self.direction == 'x':
            # 沿x轴方向：生成水平线段
            x = np.linspace(self.position[0] - length/2, self.position[0] + length/2, num_points)
            y = np.full(num_points, self.position[1])
            z = np.full(num_points, self.position[2])
        elif self.direction == 'y':
            # 沿y轴方向：生成垂直线段（在y方向）
            x = np.full(num_points, self.position[0])
            y = np.linspace(self.position[1] - length/2, self.position[1] + length/2, num_points)
            z = np.full(num_points, self.position[2])
        else:  # 'z'轴方向
            # 沿z轴方向：生成垂直线段（在z方向）
            x = np.full(num_points, self.position[0])
            y = np.full(num_points, self.position[1])
            z = np.linspace(self.position[2] - length/2, self.position[2] + length/2, num_points)
        
        # 组合成三维点集
        points = np.vstack((x, y, z)).T
        return points