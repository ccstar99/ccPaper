"""
静电学物理模型实现
基于经典电磁学理论和边界元法
参考：赵凯华《电磁学》、李亚莎《三维静电场线性插值边界元中的解析积分方法》
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import integrate
from scipy import special
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.engine import BaseFieldModel, FieldSolution, ValidationResult


class PointChargeModel(BaseFieldModel):
    """点电荷模型类"""

    def __init__(self, model_name="point_charge", dimension="3D"):
        super().__init__(model_name, dimension)
        self.charge = 1.0  # 默认电荷
        self.position = (0, 0, 0)  # 默认位置
        self.name = "点电荷模型"
        self._charges = []  # 存储电荷列表

    def compute_field(self, observation_points: np.ndarray) -> FieldSolution:
        """计算电场
        
        Args:
            observation_points: 观察点坐标数组 (n, 3)
            
        Returns:
            FieldSolution: 包含电场和电势的解
        """
        # 如果没有设置电荷，使用默认电荷
        if not self._charges:
            from core.engine import Charge
            self._charges = [Charge(charge=self.charge, position=self.position)]
        
        # 确保输入是3D数组
        if observation_points.ndim == 1:
            observation_points = observation_points.reshape(1, -1)
        
        # 计算电场
        field = np.zeros((len(observation_points), 3))
        potential = np.zeros(len(observation_points))
        
        k = 8.99e9  # 库仑常数
        
        # 预先处理电荷数据，避免重复判断类型
        processed_charges = []
        for charge in self._charges:
            if isinstance(charge, dict):
                processed_charges.append({
                    'value': charge.get('value', 0.0),
                    'position': charge.get('position', (0, 0, 0))
                })
            else:
                processed_charges.append({
                    'value': getattr(charge, 'value', 0.0),
                    'position': getattr(charge, 'position', (0, 0, 0))
                })
        
        # 批量计算电场和电势
        for i, point in enumerate(observation_points):
            Ex, Ey, Ez = 0.0, 0.0, 0.0
            pot = 0.0
            
            for charge in processed_charges:
                dx = point[0] - charge['position'][0]
                dy = point[1] - charge['position'][1]
                dz = point[2] - charge['position'][2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if r > 1e-12:  # 避免除以零
                    # 计算电场
                    E_mag = k * charge['value'] / (r ** 2)
                    Ex += E_mag * (dx / r)
                    Ey += E_mag * (dy / r)
                    Ez += E_mag * (dz / r)
                    
                    # 计算电势
                    pot += k * charge['value'] / r
            
            field[i] = [float(Ex), float(Ey), float(Ez)]
            potential[i] = float(pot)
        
        # 为电场线生成添加额外的元数据
        # 计算电荷范围用于电场线密度调整
        charge_values = [charge['value'] for charge in processed_charges]
        max_charge_magnitude = max(abs(q) for q in charge_values) if charge_values else 0
        
        return FieldSolution(
            points=observation_points,
            vectors=field,
            potentials=potential,
            charges=self._charges,
            metadata={
                'model_type': 'point_charge',
                'charge_count': len(self._charges),
                'max_charge_magnitude': max_charge_magnitude,
                'status': 'computed',
                'dimension': '3D',
                'field_accuracy': 'high',  # 标记为高精度电场计算
                'line_integration_hint': 'use_adaptive_step'  # 提示使用自适应步长积分
            }
        )
    
    def validate_parameters(self) -> ValidationResult:
        """验证模型参数"""
        # 简单验证电荷范围
        is_valid = all(-1e-6 <= c.charge <= 1e-6 for c in self._charges) if self._charges else True
        
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                message="电荷值超出有效范围 (-1e-6 到 1e-6 C)"
            )
        
        return ValidationResult(is_valid=True, message="参数验证通过")
    
    def _calculate_total_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """计算所有电荷在给定点产生的总电场"""
        total_Ex, total_Ey, total_Ez = 0.0, 0.0, 0.0
        
        for charge in self._charges:
            # 处理字典格式的电荷数据
            if isinstance(charge, dict):
                charge_value = charge.get('value', 0.0)
                position = charge.get('position', (0, 0, 0))
            else:
                # 假设是Charge对象
                charge_value = getattr(charge, 'value', 0.0)
                position = getattr(charge, 'position', (0, 0, 0))
            
            Ex, Ey, Ez = self.point_charge_field(x, y, z, charge_value, position)
            total_Ex += Ex
            total_Ey += Ey
            total_Ez += Ez
        
        return total_Ex, total_Ey, total_Ez

    def get_parameters(self) -> Dict:
        """获取模型参数"""
        return {
            'charge': self.charge,
            'position': self.position,
            'name': self.name
        }

    @staticmethod
    def point_charge_field(x: float, y: float, z: float,
                           charge: float,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
        """
        点电荷电场计算
        参数:
            x, y, z: 观察点坐标
            charge: 电荷量 (C)
            position: 电荷位置
        返回:
            Ex, Ey, Ez: 电场分量 (N/C)
        """
        k = 8.99e9  # 库仑常数

        # 相对位置向量
        dx = x - position[0]
        dy = y - position[1]
        dz = z - position[2]

        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if r < 1e-12:  # 避免除以零
            return 0, 0, 0

        # 电场大小
        E_mag = k * charge / (r ** 2)

        # 电场方向
        Ex = E_mag * (dx / r)
        Ey = E_mag * (dy / r)
        Ez = E_mag * (dz / r)

        return float(Ex), float(Ey), float(Ez)


class DipoleModel(BaseFieldModel):
    """电偶极子模型类"""

    def __init__(self, charge: float = 1.0, distance: float = 0.1,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 direction: Tuple[float, float, float] = (1, 0, 0),
                 model_name="dipole", dimension="3D"):
        super().__init__(model_name, dimension)
        self.charge = charge
        self.distance = distance
        self.position = position
        self.direction = direction
        self.name = "电偶极子模型"
        self._charges = []  # 存储电荷列表
        self._initialize_charges()
    
    def _initialize_charges(self):
        """初始化电偶极子的正负电荷"""
        from core.engine import Charge
        half_distance = self.distance / 2
        direction_norm = np.array(self.direction) / np.linalg.norm(self.direction)

        pos_positive = np.array(self.position) + half_distance * direction_norm
        pos_negative = np.array(self.position) - half_distance * direction_norm
        
        # 添加正负电荷到电荷列表
        self._charges = [
            Charge(charge=self.charge, position=tuple(pos_positive)),
            Charge(charge=-self.charge, position=tuple(pos_negative))
        ]

    def calculate_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """计算电偶极子电场"""
        # 计算正负电荷位置
        half_distance = self.distance / 2
        direction_norm = np.array(self.direction) / np.linalg.norm(self.direction)

        pos_positive = np.array(self.position) + half_distance * direction_norm
        pos_negative = np.array(self.position) - half_distance * direction_norm

        # 计算正负电荷产生的电场
        Ex1, Ey1, Ez1 = PointChargeModel.point_charge_field(
            x, y, z, self.charge, tuple(pos_positive)
        )
        Ex2, Ey2, Ez2 = PointChargeModel.point_charge_field(
            x, y, z, -self.charge, tuple(pos_negative)
        )

        # 叠加电场
        Ex = Ex1 + Ex2
        Ey = Ey1 + Ey2
        Ez = Ez1 + Ez2

        return float(Ex), float(Ey), float(Ez)
    
    def compute_field(self, observation_points: np.ndarray) -> FieldSolution:
        """计算电场（与BaseFieldModel兼容的接口）
        
        Args:
            observation_points: 观察点坐标数组 (n, 3)
            
        Returns:
            FieldSolution: 包含电场和电势的解
        """
        # 确保电荷已初始化
        if not self._charges:
            self._initialize_charges()
        
        # 计算电场
        field = np.zeros((len(observation_points), 3))
        potential = np.zeros(len(observation_points))
        
        k = 8.99e9  # 库仑常数
        
        for i, point in enumerate(observation_points):
            # 计算电场
            Ex, Ey, Ez = self.calculate_field(point[0], point[1], point[2])
            field[i] = [Ex, Ey, Ez]
            
            # 计算电势
            for charge in self._charges:
                dx = point[0] - charge.position[0]
                dy = point[1] - charge.position[1]
                dz = point[2] - charge.position[2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if r > 1e-12:
                    potential[i] += k * charge.charge / r
        
        return FieldSolution(
            points=observation_points,
            vectors=field,
            potentials=potential,
            charges=self._charges,
            metadata={
                'model_type': 'dipole',
                'charge': self.charge,
                'distance': self.distance,
                'status': 'computed',
                'dimension': '3D'
            }
        )

    def get_parameters(self) -> Dict:
        """获取模型参数"""
        return {
            'charge': self.charge,
            'distance': self.distance,
            'position': self.position,
            'direction': self.direction,
            'name': self.name
        }
    
    def validate_parameters(self) -> ValidationResult:
        """验证模型参数"""
        # 验证电荷范围
        is_valid = -1e-6 <= self.charge <= 1e-6
        
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                message="电荷值超出有效范围 (-1e-6 到 1e-6 C)"
            )
        
        # 验证距离
        if self.distance <= 0 or self.distance > 10.0:
            return ValidationResult(
                is_valid=False,
                message="电偶极子距离必须在 (0, 10] 米范围内"
            )
        
        return ValidationResult(is_valid=True, message="参数验证通过")


class ElectrostaticModels:
    """静电学物理模型集合"""

    @staticmethod
    def finite_line_charge_field(x: float, y: float, z: float,
                                 charge_density: float,
                                 length: float,
                                 position: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
        """
        有限长直导线电场计算
        参数:
            x, y, z: 观察点坐标
            charge_density: 线电荷密度 (C/m)
            length: 导线长度 (m)
            position: 导线中心位置
        返回:
            Ex, Ey, Ez: 电场分量 (N/C)
        """
        k = 8.99e9  # 库仑常数

        # 相对坐标
        x_rel = x - position[0]
        y_rel = y - position[1]
        z_rel = z - position[2]

        # 导线沿着z轴方向
        L = length
        z1 = -L / 2
        z2 = L / 2

        # 计算电场分量
        rho = np.sqrt(x_rel ** 2 + y_rel ** 2)

        if rho < 1e-12:  # 在导线上的点
            return 0, 0, 0

        # 解析解公式
        term1 = 1 / np.sqrt(rho ** 2 + (z_rel - z1) ** 2)
        term2 = 1 / np.sqrt(rho ** 2 + (z_rel - z2) ** 2)

        E_rho = k * charge_density / rho * (term1 - term2)

        # 转换为直角坐标
        if rho > 1e-12:
            Ex = E_rho * (x_rel / rho)
            Ey = E_rho * (y_rel / rho)
        else:
            Ex, Ey = 0, 0

        # z方向分量
        Ez = k * charge_density * ((z_rel - z1) / np.sqrt(rho ** 2 + (z_rel - z1) ** 2) -
                                   (z_rel - z2) / np.sqrt(rho ** 2 + (z_rel - z2) ** 2))

        return float(Ex), float(Ey), float(Ez)

    @staticmethod
    def charged_ring_field(x: float, y: float, z: float,
                           total_charge: float,
                           radius: float,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
        """
        带电圆环电场计算
        参数:
            x, y, z: 观察点坐标
            total_charge: 总电荷量 (C)
            radius: 圆环半径 (m)
            position: 圆环中心位置
        返回:
            Ex, Ey, Ez: 电场分量 (N/C)
        """
        k = 8.99e9  # 库仑常数

        # 相对坐标
        x_rel = x - position[0]
        y_rel = y - position[1]
        z_rel = z - position[2]

        R = radius
        Q = total_charge

        # 计算径向距离
        rho = np.sqrt(x_rel ** 2 + y_rel ** 2)
        r = np.sqrt(rho ** 2 + z_rel ** 2)

        if r < 1e-12:  # 在中心的点
            return 0, 0, 0

        # 完全椭圆积分计算
        k_sq = 4 * R * rho / ((R + rho) ** 2 + z_rel ** 2)

        try:
            # 使用scipy的完全椭圆积分
            K = special.ellipk(k_sq)
            E = special.ellipe(k_sq)

            # 电场分量
            common_factor = k * Q / (2 * np.pi * np.sqrt((R + rho) ** 2 + z_rel ** 2))

            if rho > 1e-12:
                E_rho = common_factor * ((K - E) * z_rel / (rho * np.sqrt((R - rho) ** 2 + z_rel ** 2))
                                         - E * (R ** 2 - rho ** 2 - z_rel ** 2) / ((R - rho) ** 2 + z_rel ** 2))
                E_phi = 0  # 轴对称，方位角分量为0
            else:
                E_rho = 0

            E_z = common_factor * (K + E * (R ** 2 - rho ** 2 - z_rel ** 2) / ((R - rho) ** 2 + z_rel ** 2))

            # 转换为直角坐标
            if rho > 1e-12:
                Ex = E_rho * (x_rel / rho)
                Ey = E_rho * (y_rel / rho)
            else:
                Ex, Ey = 0, 0

            return float(Ex), float(Ey), float(E_z)

        except (ValueError, ZeroDivisionError):
            # 数值不稳定时的近似计算
            return 0, 0, 0

    @staticmethod
    def charged_disk_field(x: float, y: float, z: float,
                           surface_charge_density: float,
                           radius: float,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
        """
        带电圆盘电场计算
        参数:
            x, y, z: 观察点坐标
            surface_charge_density: 面电荷密度 (C/m²)
            radius: 圆盘半径 (m)
            position: 圆盘中心位置
        返回:
            Ex, Ey, Ez: 电场分量 (N/C)
        """
        k = 8.99e9  # 库仑常数
        sigma = surface_charge_density

        # 相对坐标
        x_rel = x - position[0]
        y_rel = y - position[1]
        z_rel = z - position[2]

        R = radius
        rho = np.sqrt(x_rel ** 2 + y_rel ** 2)

        # 沿着对称轴的解析解
        if rho < 1e-12:
            if z_rel > 0:
                Ez = 2 * np.pi * k * sigma * (1 - z_rel / np.sqrt(z_rel ** 2 + R ** 2))
            elif z_rel < 0:
                Ez = -2 * np.pi * k * sigma * (1 - abs(z_rel) / np.sqrt(z_rel ** 2 + R ** 2))
            else:
                Ez = 0
            return 0, 0, float(Ez)

        # 一般位置的数值积分
        def integrand(phi, r_prime):
            distance = np.sqrt(rho ** 2 + r_prime ** 2 - 2 * rho * r_prime * np.cos(phi) + z_rel ** 2)
            cos_theta = z_rel / distance
            return r_prime / (distance ** 2) * cos_theta

        try:
            # 双重积分计算Ez
            Ez_result, _ = integrate.dblquad(
                integrand, 0, R,
                lambda r_prime: 0,
                lambda r_prime: 2 * np.pi
            )
            Ez = k * sigma * Ez_result

            # 近似计算径向分量（简化）
            if z_rel != 0:
                E_rho_approx = (k * sigma * z_rel * rho /
                                (2 * (rho ** 2 + z_rel ** 2) ** (3 / 2)) *
                                (1 - 1 / np.sqrt(1 + (R / rho) ** 2)))
            else:
                E_rho_approx = 0

            # 转换为直角坐标
            if rho > 1e-12:
                Ex = E_rho_approx * (x_rel / rho)
                Ey = E_rho_approx * (y_rel / rho)
            else:
                Ex, Ey = 0, 0

            return float(Ex), float(Ey), float(Ez)

        except (ValueError, integrate.IntegrationWarning):
            return 0, 0, 0


class IntegrationProcess:
    """积分过程动态演示"""

    def __init__(self):
        self.integration_steps = []
        self.current_step = 0

    def setup_line_charge_integration(self, charge_density: float, length: float,
                                      observation_points: np.ndarray) -> List[Dict]:
        """
        设置有限长直导线积分过程
        返回积分步骤数据
        """
        steps = []
        L = length
        n_segments = 50  # 积分微元数量

        for i in range(n_segments + 1):
            # 当前积分长度
            current_length = i * L / n_segments

            # 计算当前积分状态下的电场
            fields = []
            for point in observation_points:
                if i == 0:
                    Ex, Ey, Ez = 0, 0, 0
                else:
                    # 简化计算：使用从 -current_length/2 到 current_length/2 的积分
                    Ex, Ey, Ez = ElectrostaticModels.finite_line_charge_field(
                        point[0], point[1], point[2], charge_density, current_length
                    )

                fields.append([Ex, Ey, Ez])

            steps.append({
                'step': i,
                'integrated_length': current_length,
                'fields': np.array(fields),
                'total_segments': n_segments
            })

        self.integration_steps = steps
        return steps

    def setup_ring_charge_integration(self, total_charge: float, radius: float,
                                      observation_points: np.ndarray) -> List[Dict]:
        """
        设置带电圆环积分过程
        返回积分步骤数据
        """
        steps = []
        n_segments = 36  # 将圆环分为36段

        for i in range(n_segments + 1):
            # 当前积分角度
            current_angle = i * 2 * np.pi / n_segments

            # 计算当前积分状态下的电场
            fields = []
            for point in observation_points:
                if i == 0:
                    Ex, Ey, Ez = 0, 0, 0
                else:
                    # 使用离散点电荷近似积分
                    Ex, Ey, Ez = 0, 0, 0
                    segment_charge = total_charge * current_angle / (2 * np.pi)

                    for j in range(i):
                        angle = j * 2 * np.pi / n_segments
                        charge_x = radius * np.cos(angle)
                        charge_y = radius * np.sin(angle)
                        charge_z = 0

                        dEx, dEy, dEz = PointChargeModel.point_charge_field(
                            point[0], point[1], point[2],
                            segment_charge / i,  # 平均分配电荷
                            (charge_x, charge_y, charge_z)
                        )

                        Ex += dEx
                        Ey += dEy
                        Ez += dEz

                fields.append([Ex, Ey, Ez])

            steps.append({
                'step': i,
                'integrated_angle': current_angle,
                'fields': np.array(fields),
                'total_segments': n_segments
            })

        self.integration_steps = steps
        return steps


class BoundaryElementSolver:
    """
    边界元法求解器
    基于李亚莎论文《三维静电场线性插值边界元中的解析积分方法》
    """

    def __init__(self):
        self.triangles = None
        self.vertices = None
        self.potentials = None

    def create_sphere_mesh(self, radius: float = 1.0, divisions: int = 2):
        """
        创建球面网格用于边界元计算
        简化实现，实际应用应使用专业网格生成库
        """
        # 这里应该实现球面三角形网格生成
        # 简化版本：返回一个立方体近似
        vertices = np.array([
            [radius, radius, radius],
            [radius, radius, -radius],
            [radius, -radius, radius],
            [radius, -radius, -radius],
            [-radius, radius, radius],
            [-radius, radius, -radius],
            [-radius, -radius, radius],
            [-radius, -radius, -radius]
        ])

        triangles = np.array([
            [0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7],
            [0, 1, 4], [1, 4, 5], [2, 3, 6], [3, 6, 7],
            [0, 2, 4], [2, 4, 6], [1, 3, 5], [3, 5, 7]
        ])

        self.vertices = vertices
        self.triangles = triangles
        return vertices, triangles

    def calculate_influence_matrix(self):
        """
        计算影响矩阵（系数矩阵）
        基于边界积分方程中的积分项
        """
        if self.vertices is None or self.triangles is None:
            raise ValueError("请先创建网格")

        n_vertices = len(self.vertices)
        n_triangles = len(self.triangles)

        # 简化实现，实际应按照论文中的解析积分方法
        G = np.zeros((n_vertices, n_vertices))
        H = np.zeros((n_vertices, n_vertices))

        for i in range(n_vertices):
            for j in range(n_vertices):
                if i == j:
                    # 对角元素处理（奇异积分）
                    G[i, j] = 0.5  # 光滑边界假设
                    H[i, j] = 0.0
                else:
                    r = np.linalg.norm(self.vertices[i] - self.vertices[j])
                    G[i, j] = 1.0 / (4 * np.pi * r)
                    # H矩阵计算需要法向量信息，这里简化处理
                    H[i, j] = 0.0

        return G, H

    def solve_potential(self, boundary_conditions: Dict):
        """
        求解边界电位分布
        参数:
            boundary_conditions: 边界条件字典
        """
        G, H = self.calculate_influence_matrix()

        # 简化解法，实际应处理混合边界条件
        n = len(self.vertices)
        A = np.eye(n)  # 简化系统矩阵
        b = np.zeros(n)

        # 设置边界条件（简化）
        for vertex_idx, potential in boundary_conditions.items():
            if vertex_idx < n:
                A[vertex_idx, :] = 0
                A[vertex_idx, vertex_idx] = 1
                b[vertex_idx] = potential

        # 求解线性系统
        self.potentials = np.linalg.solve(A, b)
        return self.potentials

    def calculate_field(self, observation_points: np.ndarray) -> np.ndarray:
        """
        计算观察点的电场
        参数:
            observation_points: 观察点坐标数组
        返回:
            field_vectors: 电场向量数组
        """
        if self.potentials is None:
            raise ValueError("请先求解电位分布")

        field_vectors = []
        k = 8.99e9

        for point in observation_points:
            E = np.zeros(3)

            # 简化计算：基于电位梯度
            for i, vertex in enumerate(self.vertices):
                r_vec = point - vertex
                r_mag = np.linalg.norm(r_vec)

                if r_mag > 1e-10:
                    # 点电荷近似
                    E += k * self.potentials[i] * r_vec / (r_mag ** 3)

            field_vectors.append(E)

        return np.array(field_vectors)


# 确保这些类可以被导入
__all__ = [
    'PointChargeModel',
    'DipoleModel',
    'ElectrostaticModels',
    'IntegrationProcess',
    'BoundaryElementSolver'
]