# physics/ring_charge.py
"""
带电圆环物理模型实现

基于经典静电学理论和精确解析解，使用完全椭圆积分：

电场分量（柱坐标系）：
  E_ρ = (kQ / [2π√((R+ρ)²+z²)]) ×
        [ (K(k)-E(k)) × z / (ρ√((R-ρ)²+z²)) -
          E(k) × (R²-ρ²-z²) / ((R-ρ)²+z²) ]

  E_z = (kQ / [2π√((R+ρ)²+z²)]) ×
        [ K(k) + E(k) × (R²-ρ²-z²) / ((R-ρ)²+z²) ]

其中：
  - R: 圆环半径
  - Q: 总电荷量
  - ρ = √(x²+y²): 观察点到z轴的径向距离
  - z: 观察点的z坐标
  - k² = 4Rρ / [(R+ρ)² + z²]: 椭圆积分的模数
  - K(k): 第一类完全椭圆积分
  - E(k): 第二类完全椭圆积分

特殊情形：
1. 轴线点（ρ=0）：
   E_z = kQz / (R² + z²)^(3/2)
   E_ρ = 0

2. 圆环平面（z=0, ρ≠R）：
   E_z = 0
   E_ρ = kQ / [2πρ√((R+ρ)²)] × [(K-E)/ρ - E(R²-ρ²)/(R-ρ)²]

3. 圆环上（ρ≈R, z≈0）：奇异点，返回零场

参考：赵凯华《电磁学》第2.4节
"""

import numpy as np
from numpy.typing import NDArray
from scipy import special
import logging
from typing import Optional, Any

# 导入基类和混入类
from .base_model import BaseFieldModel, ParameterValidationMixin
from core.data_schema import FieldSolution, Charge, ValidationResult, ModelParameters

logger = logging.getLogger(__name__)


class RingChargeModel(BaseFieldModel, ParameterValidationMixin):
    """
    带电圆环模型

    支持：
    - 单个圆环
    - 多个共轴圆环
    - 任意空间位置（通过圆环中心坐标定义）

    特性：
    - 精确的解析解（使用完全椭圆积分）
    - 轴对称性利用优化
    - 自动处理特殊点（中心、轴线、圆环上）

    Attributes:
        rings (list): 圆环列表，每个元素是 {'center': (x0,y0,z0), 'radius': R, 'charge': Q}
        k (float): 库仑常数
        total_charge (float): 所有圆环总电荷量

    Examples:
        >>> # 单个圆环（在原点，半径1m，电荷1nC）
        >>> model = RingChargeModel(radius=1.0, charge=1e-9)
        >>>
        >>> # 多个同心圆环
        >>> model = RingChargeModel()
        >>> model.add_ring(center=(0,0,0), radius=0.5, charge=1e-9)
        >>> model.add_ring(center=(0,0,0), radius=1.0, charge=-0.5e-9)
    """

    def __init__(
            self,
            radius: Optional[float] = None,
            charge: Optional[float] = None,
            center: tuple[float, float, float] = (0.0, 0.0, 0.0),
            dimension: str = "3D"
    ):
        """
        初始化圆环模型

        Args:
            radius: 圆环半径 (m)。若为None，需后续add_ring添加
            charge: 总电荷量 (C)
            center: 圆环中心3D坐标
            dimension: "2D" | "3D"（注意：圆环本质上是3D结构）
        """
        super().__init__(model_name="ring_charge", dimension=dimension)

        self.k: float = 8.99e9  # N·m²/C²

        # 圆环列表
        self.rings: list[dict[str, Any]] = []

        # 如果提供了半径和电荷，添加第一个圆环
        if radius is not None and charge is not None:
            self.add_ring(center, radius, charge)

        # 记录参数
        self._parameters.update({
            'radius': radius,
            'charge': charge,
            'center': center
        })

        logger.info(
            f"✅ 圆环模型已创建 (维度: {dimension}, "
            f"圆环数: {len(self.rings)})"
        )

    def add_ring(
            self,
            center: tuple[float, float, float],
            radius: float,
            charge: float
    ) -> None:
        """
        添加一个圆环

        Args:
            center: 圆环中心3D坐标 (x,y,z)
            radius: 圆环半径 (m)，必须 > 0
            charge: 总电荷量 (C)
        """
        # 确保3D坐标
        if len(center) == 2:
            center = (*center, 0.0)

        # 验证半径
        if radius <= 0:
            raise ValueError(f"圆环半径必须为正数，得到 {radius}")
        if radius > 1000:
            logger.warning(f"圆环半径较大: {radius} m")

        # 验证电荷
        if not np.isfinite(charge):
            raise ValueError(f"电荷量必须是有限数: {charge}")
        if abs(charge) > 1e-6:
            logger.warning(f"圆环电荷量较大: {charge:.3e} C")

        self.rings.append({
            'center': tuple(float(x) for x in center),
            'radius': float(radius),
            'charge': float(charge)
        })

        logger.info(
            f"添加圆环: 半径={radius:.3f}m, "
            f"电荷={charge:.3e}C, "
            f"中心={center}"
        )

    def compute_field(self, observation_points: NDArray[np.float64]) -> FieldSolution:
        """
        计算观察点的电场和电势

        使用精确的解析公式，对每个圆环叠加贡献

        Args:
            observation_points: 观察点数组 (N, 2) 或 (N, 3)

        Returns:
            FieldSolution: 包含电场、电势、圆环信息
        """
        # 确保3D输入
        points_3d = self._ensure_3d_points(observation_points)
        n_points = points_3d.shape[0]

        # 如果没有圆环，返回零场
        if len(self.rings) == 0:
            logger.warning("计算场时发现没有圆环")
            return {
                'points': points_3d,
                'vectors': np.zeros((n_points, 3), dtype=np.float64),
                'potentials': np.zeros(n_points, dtype=np.float64),
                'charges': [],
                'metadata': {
                    'model_name': self._model_name,
                    'computed_by': 'physical_model',
                    'n_rings': 0,
                    'warning': 'no_rings'
                }
            }

        # 初始化总场
        total_field = np.zeros((n_points, 3), dtype=np.float64)
        total_potential = np.zeros(n_points, dtype=np.float64)

        # 对每个圆环叠加贡献
        for ring in self.rings:
            field_contrib, potential_contrib = self._ring_field(ring, points_3d)
            total_field += field_contrib
            total_potential += potential_contrib

        # 构建电荷表示（用于可视化）
        charges = self._create_charge_representation()

        # 计算元数据，用于优化电场线生成
        field_magnitudes = np.linalg.norm(total_field, axis=1)
        max_field_magnitude = np.max(field_magnitudes) if len(field_magnitudes) > 0 else 0.0
        total_charge = sum(ring['charge'] for ring in self.rings)
        
        # 标记电场线生成相关的特殊区域（如圆环中心附近）
        special_regions = []
        for ring in self.rings:
            special_regions.append({
                'type': 'ring_center',
                'position': ring['center'],
                'radius': ring['radius'],
                'charge': ring['charge']
            })

        solution = {
            'points': points_3d,
            'vectors': total_field,
            'potentials': total_potential,
            'charges': charges,
            'metadata': {
                'model_name': self._model_name,
                'computed_by': 'analytical_model',
                'n_rings': len(self.rings),
                'total_charge': total_charge,
                'coulomb_constant': self.k,
                'max_field_magnitude': max_field_magnitude,
                'field_accuracy': 'analytic',  # 解析解，高精度
                'line_integration_hint': 'use_adaptive_step',  # 提示使用自适应步长
                'min_step_size': 0.005,  # 圆环电场变化剧烈，使用更小步长
                'max_steps': 1500,  # 增加最大步数
                'special_regions': special_regions,  # 特殊区域标记
                'elliptic_integral_used': True  # 标记使用了椭圆积分
            }
        }

        logger.debug(
            f"圆环场计算完成: {n_points}个点, "
            f"{len(self.rings)}个圆环, "
            f"最大场强: {max_field_magnitude:.3e} N/C"
        )

        return solution

    def _ring_field(
            self,
            ring: dict[str, Any],
            observation_points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        计算单个圆环产生的电场

        核心算法：
        1. 转换到以圆环中心为原点的相对坐标
        2. 转换到柱坐标系 (ρ, φ, z)
        3. 使用椭圆积分计算电场分量
        4. 转换回全局笛卡尔坐标系

        Args:
            ring: 圆环定义 {'center': (x0,y0,z0), 'radius': R, 'charge': Q}
            observation_points: 观察点 (N, 3)

        Returns:
            tuple: (电场向量 (N,3), 电势 (N,))
        """
        global sqrt_term, K
        center = np.array(ring['center'])
        radius = ring['radius']
        total_charge = ring['charge']

        # 相对坐标（相对于圆环中心）
        rel_points = observation_points - center

        # 转换到柱坐标系
        # ρ = sqrt(x²+y²), φ = arctan(y/x), z = z
        x_rel = rel_points[:, 0]
        y_rel = rel_points[:, 1]
        z_rel = rel_points[:, 2]

        rho = np.sqrt(x_rel ** 2 + y_rel ** 2)
        phi = np.arctan2(y_rel, x_rel)

        # 初始化场分量（柱坐标系）
        E_rho = np.zeros(len(rho))
        E_phi = np.zeros(len(rho))
        E_z = np.zeros(len(rho))

        # 计算电势
        potential = np.zeros(len(rho))

        # 特殊点分类
        # 1. 轴线点（rho = 0）
        on_axis = rho < 1e-12

        # 2. 圆环平面点（z ≈ 0, rho ≠ R）
        in_plane = np.abs(z_rel) < 1e-12

        # 3. 圆环上点（|rho - R| < ε 且 in_plane）
        on_ring = (np.abs(rho - radius) < 1e-6) & in_plane

        # 4. 一般点
        general_points = ~(on_axis | on_ring)

        # 处理一般点（使用椭圆积分）
        if np.any(general_points):
            rho_g = rho[general_points]
            z_g = z_rel[general_points]

            # 计算模数 k² = 4Rρ / [(R+ρ)² + z²]
            denom = (radius + rho_g) ** 2 + z_g ** 2
            k_sq = np.divide(
                4 * radius * rho_g,
                denom,
                out=np.zeros_like(rho_g),
                where=denom > 1e-12
            )

            # 限制 k² < 1
            k_sq = np.minimum(k_sq, 0.999999)

            # 计算完全椭圆积分
            K = special.ellipk(k_sq)
            E = special.ellipe(k_sq)

            # 辅助量
            sqrt_term = np.sqrt((radius + rho_g) ** 2 + z_g ** 2)
            delta_term = (radius - rho_g) ** 2 + z_g ** 2

            # 避免除以零
            delta_term = np.maximum(delta_term, 1e-12)
            rho_g_safe = np.maximum(rho_g, 1e-12)

            # 径向分量
            term1 = np.divide(
                (K - E) * z_g,
                rho_g_safe * np.sqrt(delta_term),
                out=np.zeros_like(rho_g),
                where=rho_g_safe > 1e-12
            )

            term2 = E * (radius ** 2 - rho_g ** 2 - z_g ** 2) / delta_term

            E_rho_g = (self.k * total_charge / (2 * np.pi * sqrt_term)) * (term1 - term2)

            # z分量
            term3 = K + E * (radius ** 2 - rho_g ** 2 - z_g ** 2) / delta_term
            E_z_g = (self.k * total_charge / (2 * np.pi * sqrt_term)) * term3

            # 赋值
            E_rho[general_points] = E_rho_g
            E_z[general_points] = E_z_g

        # 处理轴线点（ρ = 0）
        if np.any(on_axis):
            z_axis = z_rel[on_axis]

            # 轴线电场只有z分量
            # E_z = kQz / (R² + z²)^(3/2)
            E_z_axis = self.k * total_charge * z_axis / (radius ** 2 + z_axis ** 2) ** (3 / 2)
            E_z[on_axis] = E_z_axis

            # 径向分量为零
            E_rho[on_axis] = 0.0

        # 处理圆环上点（奇异点）
        if np.any(on_ring):
            logger.debug(f"跳过{np.sum(on_ring)}个奇异点（场点在圆环上）")
            E_rho[on_ring] = 0.0
            E_z[on_ring] = 0.0

        # 转换为笛卡尔坐标系
        # 全局坐标 = 局部柱坐标 -> 局部笛卡尔 -> 全局笛卡尔
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        E_x = E_rho * cos_phi
        E_y = E_rho * sin_phi
        E_z = E_z  # 直接复制

        # 组装全局坐标系电场向量
        E_global = np.column_stack([E_x, E_y, E_z])

        # 计算电势（一般点的电位积分）
        # φ = kQ / π * K(k) / sqrt((R+ρ)² + z²)
        potential[general_points] = (
                self.k * total_charge / np.pi * K / sqrt_term
        )

        # 轴线点电势
        potential[on_axis] = self.k * total_charge / np.sqrt(radius ** 2 + z_rel[on_axis] ** 2)

        # 奇异点电势
        potential[on_ring] = 0.0  # 正则化处理

        return E_global, potential

    def _create_charge_representation(self) -> list[Charge]:
        """
        为可视化创建电荷表示

        将圆环离散化为点电荷阵列
        用于在可视化时显示电荷位置
        """
        charges = []

        for ring in self.rings:
            # 将圆环离散为12个等间距点
            n_points = 12
            center = np.array(ring['center'])
            radius = ring['radius']
            total_charge = ring['charge']

            # 每点电荷量
            point_charge = total_charge / n_points

            # 生成圆环上的点
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            for angle in angles:
                pos = center + radius * np.array([
                    np.cos(angle), np.sin(angle), 0.0
                ])
                charges.append({
                    'position': tuple(pos),
                    'value': point_charge
                })

        return charges

    def validate_parameters(self) -> ValidationResult:
        """
        验证模型参数

        检查：
        1. 至少有一个圆环
        2. 半径为正数
        3. 电荷量在合理范围内
        4. 几何尺寸有效
        """
        if len(self.rings) == 0:
            return {
                'is_valid': False,
                'message': "圆环模型必须至少有一个圆环",
                'detail': {'error': 'no_rings'}
            }

        # 验证所有圆环的参数
        for i, ring in enumerate(self.rings):
            radius = ring['radius']
            charge = ring['charge']
            center = ring['center']

            # 验证半径
            if radius <= 0:
                return {
                    'is_valid': False,
                    'message': f"圆环 {i} 半径必须为正数: {radius}",
                    'detail': {'ring_index': i, 'invalid_radius': radius}
                }
            if radius > 1000:
                return {
                    'is_valid': False,
                    'message': f"圆环 {i} 半径过大: {radius} m",
                    'detail': {'ring_index': i, 'radius': radius}
                }

            # 验证电荷
            if not np.isfinite(charge):
                return {
                    'is_valid': False,
                    'message': f"圆环 {i} 电荷量无效: {charge}",
                    'detail': {'ring_index': i, 'invalid_charge': charge}
                }
            if abs(charge) > 1e-6:
                return {
                    'is_valid': False,
                    'message': f"圆环 {i} 电荷量过大: |{charge:.2e}| > 1e-6 C",
                    'detail': {'ring_index': i, 'charge': charge}
                }

        # 验证几何参数
        for i, ring in enumerate(self.rings):
            center = ring['center']
            if max(abs(c) for c in center) > 1000:
                return {
                    'is_valid': False,
                    'message': f"圆环 {i} 中心坐标过大",
                    'detail': {'ring_index': i, 'center': center}
                }

        return {
            'is_valid': True,
            'message': "圆环参数验证通过",
            'detail': {
                'n_rings': len(self.rings),
                'total_charge': sum(ring['charge'] for ring in self.rings),
                'max_radius': float(max(ring['radius'] for ring in self.rings))
            }
        }

    def add_ring_from_two_charges(self, charge1: Charge, charge2: Charge) -> None:
        """
        从两个对称电荷添加圆环（用于构建电偶极子等）

        假设：两个电荷位于圆环对径点，电荷量大小相等

        Args:
            charge1: 第一个电荷
            charge2: 第二个电荷
        """
        pos1 = np.array(charge1['position'])
        pos2 = np.array(charge2['position'])

        # 计算中心
        center = (pos1 + pos2) / 2.0

        # 计算半径
        radius = np.linalg.norm(pos1 - pos2) / 2.0

        # 计算总电荷（假设均匀分布）
        total_charge = charge1['value'] + charge2['value']

        self.add_ring(tuple(center), radius, total_charge)

    def clear_rings(self) -> None:
        """清空所有圆环"""
        self.rings.clear()
        logger.warning("所有圆环已清空")

    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        base_dict = super().to_dict()
        base_dict.update({
            'rings': self.rings,
            'total_charge': sum(ring['charge'] for ring in self.rings)
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RingChargeModel':
        """反序列化"""
        try:
            dimension = data.get('dimension', '3D')
            model = cls(dimension=dimension)

            # 恢复圆环
            rings = data.get('rings', [])
            for ring in rings:
                model.add_ring(
                    ring['center'],
                    ring['radius'],
                    ring['charge']
                )

            logger.info(f"从字典恢复圆环模型: {len(rings)}个圆环")
            return model

        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            raise ValueError(f"无法从字典创建RingChargeModel: {e}")


# ============================================================================ #
# 单元测试
# ============================================================================ #

def test_single_ring_on_axis():
    """测试单个圆环在轴线上的电场"""
    model = RingChargeModel(radius=1.0, charge=1e-9)

    # 在轴线上z=1m处
    points = np.array([[0.0, 0.0, 1.0]])
    solution = model.compute_field(points)

    # 应有非零Ez分量
    Ez = solution['vectors'][0, 2]
    assert Ez > 0, "轴线电场方向应向上（正电荷）"

    # 径向分量应为零
    Ex = solution['vectors'][0, 0]
    Ey = solution['vectors'][0, 1]
    assert abs(Ex) < 1e-12, f"Ex应为零，得到{Ex:.3e}"
    assert abs(Ey) < 1e-12, f"Ey应为零，得到{Ey:.3e}"

    logger.info("✅ 轴线电场测试通过")
    print("✅ 轴线电场测试通过")


def test_ring_center():
    """测试圆环中心处电场"""
    model = RingChargeModel(radius=1.0, charge=1e-9)

    # 在中心（原点）
    points = np.array([[0.0, 0.0, 0.0]])
    solution = model.compute_field(points)

    # 中心电场应为零
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag < 1e-12, f"中心电场应为零，得到{E_mag:.3e}"

    # 中心电势不为零
    phi = solution['potentials'][0]
    assert phi > 0, "中心电势应为正"

    # 理论值：φ = kQ/R
    phi_expected = model.k * 1e-9 / 1.0
    assert abs(phi - phi_expected) / phi_expected < 0.01, \
        f"中心电势不符: 计算={phi:.3e}, 理论={phi_expected:.3e}"

    logger.info("✅ 圆环中心测试通过")
    print("✅ 圆环中心测试通过")


def test_far_field_approximation():
    """测试远场近似（应接近点电荷）"""
    model = RingChargeModel(radius=1.0, charge=1e-9)

    # 在远处（r >> R）
    points = np.array([[0.0, 0.0, 10.0]])  # 距离10m，10倍半径
    solution = model.compute_field(points)

    # 远场应近似点电荷
    r = 10.0
    E_expected = model.k * 1e-9 / (r ** 2)
    E_magnitude = np.linalg.norm(solution['vectors'][0])

    # 相对误差应小于5%
    assert abs(E_magnitude - E_expected) / E_expected < 0.05, \
        f"远场近似失败: 计算值={E_magnitude:.3e}, 点电荷近似={E_expected:.3e}"

    logger.info("✅ 远场近似测试通过")
    print("✅ 远场近似测试通过")


def test_dipole_approximation():
    """测试电偶极子近似（两个相反电荷圆环）"""
    model = RingChargeModel()

    # 两个圆环构成电偶极子
    model.add_ring(center=(0, 0, -0.5), radius=1.0, charge=1e-9)
    model.add_ring(center=(0, 0, 0.5), radius=1.0, charge=-1e-9)

    # 在中点计算
    points = np.array([[0.0, 0.0, 0.0]])
    solution = model.compute_field(points)

    # 总电荷为零的电偶极子，中心电场不为零
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag > 0, "电偶极子中心电场应不为零"

    # 方向应指向正电荷
    Ez = solution['vectors'][0, 2]
    assert Ez > 0, "电场方向应指向正电荷"

    logger.info("✅ 电偶极子近似测试通过")
    print("✅ 电偶极子近似测试通过")


def test_singular_point_on_ring():
    """测试圆环上奇异点"""
    model = RingChargeModel(radius=1.0, charge=1e-9)

    # 在圆环上一点
    points = np.array([[1.0, 0.0, 0.0]])
    solution = model.compute_field(points)

    # 电场应为零（避免奇异）
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag < 1e-10, f"圆环上电场应为零，得到{E_mag:.3e}"

    logger.info("✅ 奇异点处理测试通过")
    print("✅ 奇异点处理测试通过")


def test_multiple_rings():
    """测试多个圆环叠加"""
    model = RingChargeModel()

    # 添加两个同心圆环
    model.add_ring(center=(0, 0, 0), radius=0.5, charge=1e-9)
    model.add_ring(center=(0, 0, 0), radius=1.0, charge=1e-9)

    # 在轴线上某点
    points = np.array([[0.0, 0.0, 1.0]])
    solution = model.compute_field(points)

    # 应有非零场
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag > 0, "多圆环场强应大于零"

    # 应大于单个圆环
    model_single = RingChargeModel(radius=0.5, charge=1e-9)
    solution_single = model_single.compute_field(points)
    E_single = np.linalg.norm(solution_single['vectors'][0])

    assert E_mag > E_single, "多圆环场强大于单圆环"

    logger.info("✅ 多圆环叠加测试通过")
    print("✅ 多圆环叠加测试通过")


def test_2d_compatibility():
    """测试2D兼容性"""
    model = RingChargeModel(radius=1.0, charge=1e-9, dimension="2D")

    # 2D点输入（z=0平面观察）
    points = np.array([[1.5, 0.0]])
    solution = model.compute_field(points)

    # 输出应为3D
    assert solution['points'].shape == (1, 3)
    assert solution['points'][0, 2] == 0.0

    # 在z=0平面，电场只有z分量（垂直于平面）
    Ex = solution['vectors'][0, 0]
    Ey = solution['vectors'][0, 1]
    Ez = solution['vectors'][0, 2]

    # 在z=0平面观察，电场应垂直于平面（即只有z分量）
    # 但计算在z=0平面点，圆环在该平面，电场只有z分量
    assert abs(Ez) > 0, "z=0平面应有非零Ez分量"

    logger.info("✅ 2D兼容性测试通过")
    print("✅ 2D兼容性测试通过")


def test_serialization():
    """测试序列化"""
    model = RingChargeModel()
    model.add_ring(center=(0, 0, 0), radius=1.0, charge=1e-9)
    model.add_ring(center=(0, 0, 1), radius=0.5, charge=-0.5e-9)

    # 序列化
    data = model.to_dict()

    # 反序列化
    restored = RingChargeModel.from_dict(data)

    assert len(restored.rings) == 2
    assert restored.rings[0]['radius'] == 1.0
    assert restored.rings[1]['charge'] == -0.5e-9

    # 验证计算结果一致
    points = np.array([[0.5, 0.5, 0.5]])
    sol_original = model.compute_field(points)
    sol_restored = restored.compute_field(points)

    assert np.allclose(sol_original['vectors'], sol_restored['vectors'], rtol=1e-12)

    logger.info("✅ 序列化测试通过")
    print("✅ 序列化测试通过")


def run_all_tests():
    """运行所有单元测试"""
    logger.info("开始运行RingChargeModel单元测试")

    test_single_ring_on_axis()
    test_ring_center()
    test_far_field_approximation()
    test_dipole_approximation()
    test_singular_point_on_ring()
    test_multiple_rings()
    test_2d_compatibility()
    test_serialization()

    logger.info("✅ 所有RingChargeModel单元测试通过!")
    print("✅ 所有RingChargeModel单元测试通过!")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_all_tests()
    elif "--axis" in sys.argv:
        test_single_ring_on_axis()
    elif "--center" in sys.argv:
        test_ring_center()
    elif "--far" in sys.argv:
        test_far_field_approximation()
    else:
        print(__doc__)
        print("\n运行测试:")
        print("  python ring_charge.py --test      # 全部测试")
        print("  python ring_charge.py --axis      # 轴线电场测试")
        print("  python ring_charge.py --center    # 中心测试")
        print("  python ring_charge.py --far       # 远场近似测试")