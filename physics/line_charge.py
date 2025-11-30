# physics/line_charge.py
"""
有限长直导线线电荷物理模型
基于经典电磁学的解析解公式：
电场分量（导线沿z轴方向，中心在原点）：
  E_ρ = k*λ/rho * [ (z+L/2)/sqrt(rho²+(z+L/2)²) - (z-L/2)/sqrt(rho²+(z-L/2)²) ]
  E_z = k*λ * [ 1/sqrt(rho²+(z+L/2)²) - 1/sqrt(rho²+(z-L/2)²) ]
其中：
  - ρ = sqrt(x²+y²) 是径向距离
  - L 是导线长度
  - λ 是线电荷密度 (C/m)
  - k = 8.99e9 N·m²/C²
模型特点：
- 精确解析解（无数值积分误差）
- 支持任意位置、任意方向的导线
- 支持多段导线叠加
- 自动处理奇异点（ρ=0）
- 3D优先，2D自动兼容
参考：赵凯华《电磁学》第二版，第2.3节
"""

import numpy as np
from numpy.typing import NDArray
import logging
from typing import Optional, Any

# 导入基类和混入类
from .base_model import BaseFieldModel, ParameterValidationMixin
from core.data_schema import FieldSolution, Charge, ValidationResult, ModelParameters

logger = logging.getLogger(__name__)


class LineChargeModel(BaseFieldModel, ParameterValidationMixin):
    """
    有限长直导线线电荷模型

    支持：
    - 单根导线
    - 多段导线（分段线性近似任意形状）
    - 任意空间取向（通过起点/终点定义）

    Attributes:
        segments (list): 线段列表，每个元素是 {'start': (x1,y1,z1), 'end': (x2,y2,z2), 'density': λ}
        k (float): 库仑常数
        total_length (float): 所有线段总长度

    Examples:
        >>> # 单根导线
        >>> model = LineChargeModel(
        ...     start=(-1, 0, 0), end=(1, 0, 0), density=1e-9
        ... )
        >>> # 多段导线（L形状）
        >>> model = LineChargeModel()
        >>> model.add_segment(start=(0,0,0), end=(1,0,0), density=1e-9)
        >>> model.add_segment(start=(1,0,0), end=(1,1,0), density=1e-9)
    """

    def __init__(
            self,
            start: Optional[tuple[float, float, float]] = None,
            end: Optional[tuple[float, float, float]] = None,
            density: float = 1e-9,
            dimension: str = "3D"
    ):
        """
        初始化线电荷模型

        Args:
            start: 线段起点 (x,y,z)。若为None，需后续add_segment添加
            end: 线段终点 (x,y,z)
            density: 线电荷密度 (C/m)
            dimension: "2D" | "3D"
        """
        super().__init__(model_name="line_charge", dimension=dimension)

        self.k: float = 8.99e9  # N·m²/C²

        # 线段列表
        self.segments: list[dict[str, Any]] = []

        # 如果提供了起点终点，添加第一个线段
        if start is not None and end is not None:
            self.add_segment(start, end, density)

        # 记录参数
        self._parameters.update({
            'start': start,
            'end': end,
            'density': density
        })

        logger.info(
            f"✅ 线电荷模型已创建 (维度: {dimension}, "
            f"线段数: {len(self.segments)})"
        )

    def add_segment(
            self,
            start: tuple[float, float, float],
            end: tuple[float, float, float],
            density: float
    ) -> None:
        """
        添加一个线段

        Args:
            start: 起点3D坐标
            end: 终点3D坐标
            density: 线电荷密度 (C/m)
        """
        # 确保3D坐标
        if len(start) == 2:
            start = (*start, 0.0)
        if len(end) == 2:
            end = (*end, 0.0)

        # 验证密度
        if not np.isfinite(density):
            raise ValueError(f"线电荷密度必须是有限数: {density}")
        if abs(density) > 1e-6:
            logger.warning(f"线电荷密度较大: {density:.3e} C/m")

        # 计算长度
        start_arr = np.array(start)
        end_arr = np.array(end)
        length = np.linalg.norm(end_arr - start_arr)

        if length < 1e-12:
            logger.warning("添加长度接近零的线段，已忽略")
            return

        self.segments.append({
            'start': tuple(float(x) for x in start),
            'end': tuple(float(x) for x in end),
            'density': float(density),
            'length': length
        })

        logger.info(
            f"添加线段: 长度={length:.3f}m, "
            f"密度={density:.3e}C/m, "
            f"起点={start}, 终点={end}"
        )

    def compute_field(self, observation_points: NDArray[np.float64]) -> FieldSolution:
        """
        计算观察点的电场和电势

        使用精确的解析公式，对每个线段叠加贡献

        Args:
            observation_points: 观察点数组 (N, 2) 或 (N, 3)

        Returns:
            FieldSolution: 包含电场、电势、线密度信息
        """
        # 确保3D输入
        points_3d = self._ensure_3d_points(observation_points)
        n_points = points_3d.shape[0]

        # 如果没有线段，返回零场
        if len(self.segments) == 0:
            logger.warning("计算场时发现没有线段")
            return {
                'points': points_3d,
                'vectors': np.zeros((n_points, 3), dtype=np.float64),
                'potentials': np.zeros(n_points, dtype=np.float64),
                'charges': [],
                'metadata': {
                    'model_name': self._model_name,
                    'computed_by': 'physical_model',
                    'n_segments': 0,
                    'warning': 'no_segments'
                }
            }

        # 预计算线段属性，提高效率
        segment_props = []
        for segment in self.segments:
            start = np.array(segment['start'])
            end = np.array(segment['end'])
            direction = end - start
            length = np.linalg.norm(direction)
            
            # 避免零长度线段
            if length < 1e-12:
                continue
            
            segment_props.append(segment)

        # 初始化总场
        total_field = np.zeros((n_points, 3), dtype=np.float64)
        total_potential = np.zeros(n_points, dtype=np.float64)

        # 对每个线段叠加贡献
        for i, segment in enumerate(segment_props):
            field_contrib, potential_contrib = self._segment_field(
                segment, points_3d
            )
            total_field += field_contrib
            total_potential += potential_contrib

        # 构建电荷表示（用于可视化）
        charges = self._create_charge_representation()

        # 计算电场线可视化所需的元数据
        field_magnitudes = np.linalg.norm(total_field, axis=1)
        max_field_magnitude = np.max(field_magnitudes) if n_points > 0 else 0.0
        total_charge = sum(seg['density'] * seg['length'] for seg in segment_props)
        
        # 找出高梯度区域，用于指导电场线采样
        high_gradient_regions = []
        if n_points > 10:  # 只有当点足够多时才计算
            for i in range(1, n_points - 1):
                if field_magnitudes[i] > 0.5 * max_field_magnitude:
                    high_gradient_regions.append(i)

        solution = {
            'points': points_3d,
            'vectors': total_field,
            'potentials': total_potential,
            'charges': charges,
            'metadata': {
                'model_name': self._model_name,
                'computed_by': 'analytical_model',
                'n_segments': len(self.segments),
                'total_length': sum(seg['length'] for seg in segment_props),
                'coulomb_constant': self.k,
                'max_field_magnitude': max_field_magnitude,
                'total_charge': total_charge,
                'field_accuracy': 'analytic',  # 解析解，高精度
                'line_integration_hint': 'use_adaptive_step',  # 提示使用自适应步长
                'min_step_size': 0.01,  # 建议的最小步长
                'max_steps': 1000,  # 建议的最大步数
                'high_gradient_regions': high_gradient_regions
            }
        }

        logger.debug(
            f"线电荷场计算完成: {n_points}个点, "
            f"{len(segment_props)}个有效线段, "
            f"最大场强: {max_field_magnitude:.3e} N/C"
        )

        return solution

    def _segment_field(
            self,
            segment: dict[str, Any],
            observation_points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        计算单个线段产生的电场

        核心算法：
        1. 旋转坐标系使线段与z'轴对齐
        2. 在局部坐标系中应用解析公式
        3. 旋转回全局坐标系

        Args:
            segment: 线段定义 {'start': (x1,y1,z1), 'end': (x2,y2,z2), 'density': λ}
            observation_points: 观察点 (N, 3)

        Returns:
            tuple: (电场向量 (N,3), 电势 (N,))
        """
        start = np.array(segment['start'])
        end = np.array(segment['end'])
        density = segment['density']
        length = segment['length']

        # 计算线段方向向量（从起点到终点）
        direction = end - start
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-12:
            return np.zeros_like(observation_points), np.zeros(len(observation_points))

        # 单位方向向量
        u = direction / direction_norm

        # 线段中点
        midpoint = (start + end) / 2.0

        # 线段半长
        half_length = length / 2.0

        # 将观察点转换到线段局部坐标系
        # 原点：线段中点
        # z'轴：沿线段方向
        # x'、y'轴：任意正交基

        # 计算局部坐标系的基向量
        # 使用Gram-Schmidt正交化
        if abs(u[2]) < 0.9:  # 避免与z轴平行
            temp = np.array([0.0, 0.0, 1.0])
        else:
            temp = np.array([1.0, 0.0, 0.0])

        # 第一个正交基（垂直于u）
        v1 = temp - np.dot(temp, u) * u
        v1 = v1 / np.linalg.norm(v1)

        # 第二个正交基
        v2 = np.cross(u, v1)

        # 构建旋转矩阵（全局 -> 局部）
        R = np.array([v1, v2, u])  # 每行是局部坐标轴在全局中的表示

        # 观察点相对于中点的坐标
        rel_points = observation_points - midpoint

        # 转换到局部坐标系 (x', y', z')
        local_points = rel_points @ R.T  # (N, 3) @ (3, 3) = (N, 3)

        # 在局部坐标系中计算场
        # 此时线段位于z'轴，从 -half_length 到 +half_length
        x_p = local_points[:, 0]
        y_p = local_points[:, 1]
        z_p = local_points[:, 2]

        rho_p = np.sqrt(x_p ** 2 + y_p ** 2)  # 到z'轴的径向距离

        # 计算电场分量（局部坐标系）
        # 避免除以零
        rho_p = np.maximum(rho_p, 1e-12)

        # 预计算距离项
        zp_plus_L = z_p + half_length
        zp_minus_L = z_p - half_length

        dist_plus = np.sqrt(rho_p ** 2 + zp_plus_L ** 2)
        dist_minus = np.sqrt(rho_p ** 2 + zp_minus_L ** 2)

        # 径向分量 E_rho
        E_rho = self.k * density / rho_p * (zp_plus_L / dist_plus - zp_minus_L / dist_minus)

        # z'分量 E_z
        E_z = self.k * density * (1.0 / dist_plus - 1.0 / dist_minus)

        # 局部坐标系中的电场向量
        E_local = np.column_stack([
            E_rho * (x_p / rho_p),  # E_x'
            E_rho * (y_p / rho_p),  # E_y'
            E_z  # E_z'
        ])

        # 旋转回全局坐标系
        # 注意：R的列向量是局部坐标轴在全局中的表示
        # 所以 E_global = E_local @ R
        E_global = E_local @ R

        # 计算电势
        # φ = k*λ * ln( (z+L/2 + sqrt(rho^2+(z+L/2)^2)) / (z-L/2 + sqrt(rho^2+(z-L/2)^2)) )
        phi = self.k * density * np.log(
            (zp_plus_L + dist_plus) / (zp_minus_L + dist_minus)
        )

        # 处理无穷大（场点在导线上）
        on_wire = rho_p < 1e-9  # 在导线轴线上
        E_global[on_wire] = 0.0
        phi[on_wire] = 0.0

        return E_global, phi

    def _create_charge_representation(self) -> list[Charge]:
        """
        为可视化创建电荷表示

        将线段离散化为点电荷阵列
        """
        charges = []

        for segment in self.segments:
            # 将线段离散为10个点
            n_points = 10
            start = np.array(segment['start'])
            end = np.array(segment['end'])
            positions = np.linspace(start, end, n_points)

            # 每点电荷量 = λ * ΔL
            total_charge = segment['density'] * segment['length']
            point_charge = total_charge / n_points

            for pos in positions:
                charges.append({
                    'position': tuple(pos),
                    'value': point_charge
                })

        return charges

    def validate_parameters(self) -> ValidationResult:
        """
        验证模型参数

        检查：
        1. 至少有一个线段
        2. 线密度在合理范围内
        3. 几何尺寸有效
        """
        if len(self.segments) == 0:
            return {
                'is_valid': False,
                'message': "线电荷模型必须至少有一个线段",
                'detail': {'error': 'no_segments'}
            }

        # 验证所有线段的密度
        for i, seg in enumerate(self.segments):
            density = seg['density']
            if not np.isfinite(density):
                return {
                    'is_valid': False,
                    'message': f"线段 {i} 的线密度无效: {density}",
                    'detail': {'segment_index': i, 'invalid_density': density}
                }
            if abs(density) > 1e-6:
                return {
                    'is_valid': False,
                    'message': f"线段 {i} 线密度过大: |{density:.2e}| > 1e-6 C/m",
                    'detail': {'segment_index': i, 'density': density}
                }

        # 验证几何参数
        for i, seg in enumerate(self.segments):
            length = seg['length']
            if length < 1e-12:
                return {
                    'is_valid': False,
                    'message': f"线段 {i} 长度过小: {length:.2e} m",
                    'detail': {'segment_index': i, 'length': length}
                }
            if length > 1000:
                return {
                    'is_valid': False,
                    'message': f"线段 {i} 长度过大: {length:.2e} m",
                    'detail': {'segment_index': i, 'length': length}
                }

        return {
            'is_valid': True,
            'message': "线电荷参数验证通过",
            'detail': {
                'n_segments': len(self.segments),
                'total_length': sum(seg['length'] for seg in self.segments),
                'max_density': float(max(abs(seg['density']) for seg in self.segments))
            }
        }

    def add_segment_from_charges(self, charge1: Charge, charge2: Charge) -> None:
        """
        从两个端点电荷添加线段

        用于从离散电荷构建连续线段

        Args:
            charge1: 起点电荷
            charge2: 终点电荷
        """
        # 假设电荷密度均匀，取平均值
        avg_density = (charge1['value'] + charge2['value']) / 2.0
        self.add_segment(charge1['position'], charge2['position'], avg_density)

    def clear_segments(self) -> None:
        """清空所有线段"""
        self.segments.clear()
        logger.warning("所有线段已清空")

    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        base_dict = super().to_dict()
        base_dict.update({
            'segments': self.segments,
            'total_length': sum(seg['length'] for seg in self.segments)
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LineChargeModel':
        """反序列化"""
        try:
            dimension = data.get('dimension', '3D')
            model = cls(dimension=dimension)

            # 恢复线段
            segments = data.get('segments', [])
            for seg in segments:
                model.add_segment(
                    seg['start'],
                    seg['end'],
                    seg['density']
                )

            logger.info(f"从字典恢复线电荷模型: {len(segments)}个线段")
            return model

        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            raise ValueError(f"无法从字典创建LineChargeModel: {e}")


# ============================================================================ #
# 单元测试
# ============================================================================ #

def test_single_segment():
    """测试单一线段"""
    model = LineChargeModel(
        start=(-1, 0, 0),
        end=(1, 0, 0),
        density=1e-9
    )

    # 验证参数
    validation = model.validate_parameters()
    assert validation['is_valid'], validation['message']
    assert len(model.segments) == 1

    # 在中垂线计算（x=0, y=1, z=0）
    points = np.array([[0.0, 1.0, 0.0]])
    solution = model.compute_field(points)

    # 在中垂线，电场应指向-y方向
    Ey = solution['vectors'][0, 1]
    assert Ey < 0, "中垂线电场方向错误"

    # x和z分量应接近零
    assert abs(solution['vectors'][0, 0]) < 1e-12, "Ex分量应为0"
    assert abs(solution['vectors'][0, 2]) < 1e-12, "Ez分量应为0"

    logger.info("✅ 单线段测试通过")
    print("✅ 单线段测试通过")


def test_infinite_wire_approximation():
    """测试无限长导线近似"""
    # 很长的线段，在中心附近观察，近似无限长导线
    model = LineChargeModel(
        start=(-100, 0, 0),
        end=(100, 0, 0),
        density=1e-9
    )

    # 在中心附近（x=0, y=0.5, z=0）
    points = np.array([[0.0, 0.5, 0.0]])
    solution = model.compute_field(points)

    # 对于无限长导线，E = 2kλ/r，方向径向
    r = 0.5
    E_expected = 2 * model.k * 1e-9 / r

    E_magnitude = np.linalg.norm(solution['vectors'][0])

    # 相对误差应小于5%
    assert abs(E_magnitude - E_expected) / E_expected < 0.05, \
        f"无限长导线近似失败: 计算值={E_magnitude:.3e}, 理论值={E_expected:.3e}"

    logger.info("✅ 无限长导线近似测试通过")
    print("✅ 无限长导线近似测试通过")


def test_multiple_segments():
    """测试多段导线"""
    model = LineChargeModel()

    # L形导线
    model.add_segment(start=(0, 0, 0), end=(1, 0, 0), density=1e-9)
    model.add_segment(start=(1, 0, 0), end=(1, 1, 0), density=1e-9)

    # 在拐角处计算
    points = np.array([[1.0, 0.5, 0.0]])
    solution = model.compute_field(points)

    # 应有非零场
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag > 0, "多段导线场强应为正"

    logger.info("✅ 多段导线测试通过")
    print("✅ 多段导线测试通过")


def test_singular_handling():
    """测试奇异点处理（场点在导线上）"""
    model = LineChargeModel(
        start=(0, 0, 0),
        end=(1, 0, 0),
        density=1e-9
    )

    # 在导线中点
    points = np.array([[0.5, 0, 0]])
    solution = model.compute_field(points)

    # 电场应为零（避免奇异）
    E_mag = np.linalg.norm(solution['vectors'][0])
    assert E_mag < 1e-12, f"导线中点电场应为零，得到{E_mag:.3e}"

    logger.info("✅ 奇异点处理测试通过")
    print("✅ 奇异点处理测试通过")


def test_2d_compatibility():
    """测试2D兼容性"""
    model = LineChargeModel(
        start=(0, -1),
        end=(0, 1),
        density=1e-9,
        dimension="2D"
    )

    # 2D点输入
    points = np.array([[1.0, 0.0]])
    solution = model.compute_field(points)

    # 输出应为3D，z=0
    assert solution['points'].shape == (1, 3)
    assert solution['points'][0, 2] == 0.0
    assert solution['vectors'][0, 2] == 0.0

    logger.info("✅ 2D兼容性测试通过")
    print("✅ 2D兼容性测试通过")


def test_serialization():
    """测试序列化"""
    model = LineChargeModel()
    model.add_segment((0, 0, 0), (1, 0, 0), 1e-9)
    model.add_segment((1, 0, 0), (1, 1, 0), 1e-9)

    # 序列化
    data = model.to_dict()

    # 反序列化
    restored = LineChargeModel.from_dict(data)

    assert len(restored.segments) == 2
    assert restored.segments[0]['density'] == 1e-9

    logger.info("✅ 序列化测试通过")
    print("✅ 序列化测试通过")


def run_all_tests():
    """运行所有单元测试"""
    logger.info("开始运行LineChargeModel单元测试")

    test_single_segment()
    test_infinite_wire_approximation()
    test_multiple_segments()
    test_singular_handling()
    test_2d_compatibility()
    test_serialization()

    logger.info("所有LineChargeModel单元测试通过!")
    print("所有LineChargeModel单元测试通过!")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_all_tests()
    elif "--single" in sys.argv:
        test_single_segment()
    elif "--infinite" in sys.argv:
        test_infinite_wire_approximation()
    else:
        print(__doc__)
        print("\n运行测试:")
        print("  python line_charge.py --test      # 全部测试")
        print("  python line_charge.py --single    # 单线段测试")
        print("  python line_charge.py --infinite  # 无限长近似测试")