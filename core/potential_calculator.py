# core/potential_calculator.py
import numpy as np
from typing import List, Union, Protocol, runtime_checkable
from utils.constants import COULOMB_CONSTANT, VACUUM_PERMITTIVITY
from physics.point import PointCharge
from physics.line import LineCharge
from physics.ring import RingCharge

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
    
    数学特性：
        - 标量场叠加：线性代数加法
        - 梯度运算：中心差分法数值计算∇V
        - 奇点处理：电荷内部电势定义为表面电势
        - 参考点归一化：支持指定零势点（默认无穷远）
    
    验证方法：
        通过E = -∇V恒等式反推验证电场计算精度
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
        
    def add_charge(self, charge: ChargeProtocol) -> None:
        """动态添加电荷，运行时构建场景"""
        if not isinstance(charge, ChargeProtocol):
            raise TypeError(f"电荷必须实现potential方法，获得{type(charge)}")
        self.charges.append(charge)
        
        # 重置参考电势缓存
        self._reference_potential = None
    
    def add_charges(self, charges: List[ChargeProtocol]) -> None:
        """批量添加电荷"""
        for charge in charges:
            self.add_charge(charge)
    
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
    
    def electric_field_from_gradient(self, points: Union[List[float], np.ndarray],
                                   h: float = 1e-6) -> np.ndarray:
        """
        **数学验证函数**：通过电势梯度计算电场 E = -∇V
        
        数值方法：
            中心差分法（二阶精度）：
            ∂V/∂x ≈ [V(x+h) - V(x-h)] / (2h)
            
            因此 E = -∇V = -(∂V/∂x, ∂V/∂y, ∂V/∂z)
        
        Args:
            points: 计算点
            h: 差分步长，默认1e-6米
            
        Returns:
            电场矢量数组，形状与points一致
        """
        points = np.atleast_2d(points)
        
        # 基坐标电势
        V0 = self.potential(points, absolute=True)
        
        # 初始化电场数组
        E = np.zeros_like(points, dtype=float)
        
        # 逐方向数值微分（向量化实现）
        for i, axis in enumerate([0, 1, 2]):  # x, y, z
            # 正向偏移点
            points_plus = points.copy()
            points_plus[:, axis] += h
            
            # 负向偏移点
            points_minus = points.copy()
            points_minus[:, axis] -= h
            
            # 计算差分
            V_plus = self.potential(points_plus, absolute=True)
            V_minus = self.potential(points_minus, absolute=True)
            
            # 中心差分：∂V/∂axis = (V+ - V-)/(2h)
            # 电场分量：E_axis = -∂V/∂axis
            E[:, axis] = -(V_plus - V_minus) / (2 * h)
        
        return E.squeeze()
    
    def validate_conservation_law(self, points: np.ndarray, 
                                 field_calculator, tol: float = 1e-6) -> dict:
        """
        **物理定律验证**：检验 E = -∇V 是否严格成立
        
        验证逻辑：
            Δ = max(|E_direct + ∇V| / |E_direct|) < tol
        
        Args:
            points: 测试点集
            field_calculator: FieldCalculator实例（直接计算电场）
            tol: 相对误差容忍度
            
        Returns:
            验证报告字典
        """
        points = np.atleast_2d(points)
        
        # 直接电场计算
        E_direct = field_calculator.electric_field(points)
        
        # 梯度电势计算电场
        E_from_V = self.electric_field_from_gradient(points, h=1e-6)
        
        # 计算误差
        error_vector = E_direct + E_from_V  # 应为零矢量（因E = -∇V）
        error_magnitude = np.linalg.norm(error_vector, axis=1)
        
        # 相对误差（相对于直接电场大小）
        E_mag = np.linalg.norm(E_direct, axis=1) + 1e-12  # 避免除零
        relative_error = error_magnitude / E_mag
        
        max_error = np.max(relative_error)
        
        return {
            "passed": max_error < tol,
            "max_relative_error": max_error,
            "mean_relative_error": np.mean(relative_error),
            "points_tested": len(points),
            "error_per_point": error_magnitude
        }
    
    def equipotential_surface(self, seed_point: np.ndarray, 
                             target_potential: float = None,
                             npoints: int = 200,
                             method: str = "implicit") -> np.ndarray:
        """
        **高级功能**：生成等势面采样点
        
        算法选项：
            implicit: 求解 V(r) = V₀ 的隐式曲面（暴力网格采样）
            gradient: 从种子点沿等势梯度行走（需特殊微分几何）
            
        Args:
            seed_point: 种子点
            target_potential: 目标电势值（默认使用种子点电势）
            npoints: 采样点数
            method: 算法选择
            
        Returns:
            等势面三维点集 [npoints, 3]
        """
        seed_point = np.asarray(seed_point, dtype=float)
        
        if target_potential is None:
            target_potential = self.potential(seed_point)
        
        if method == "implicit":
            # 暴力法：在种子点附近网格均匀采样，筛选电势相近点
            # 计算包围盒
            radius = 0.5  # 搜索半径
            grid_resolution = int(npoints ** (1/3)) * 2
            
            # 生成网格
            x = np.linspace(seed_point[0] - radius, seed_point[0] + radius, grid_resolution)
            y = np.linspace(seed_point[1] - radius, seed_point[1] + radius, grid_resolution)
            z = np.linspace(seed_point[2] - radius, seed_point[2] + radius, grid_resolution)
            
            # 暴力计算所有网格点电势（计算量大）
            # 实际应用中可改用八叉树或KD树优化
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            
            V_grid = self.potential(grid_points, absolute=True)
            
            # 筛选电势相近点（容差自适应）
            v_diff = np.abs(V_grid - target_potential)
            tolerance = np.std(v_diff) * 0.1
            
            valid_mask = v_diff < tolerance
            surface_points = grid_points[valid_mask]
            
            # 若点过多，随机采样
            if len(surface_points) > npoints:
                indices = np.random.choice(len(surface_points), npoints, replace=False)
                surface_points = surface_points[indices]
            
            return surface_points
        
        elif method == "gradient":
            # 高级方法：沿等势面切向行走
            # 需要求解 V(seed) = V₀ 且 ∇V·dr = 0
            raise NotImplementedError("梯度法需微分几何支持，当前版本略")
        
        else:
            raise ValueError(f"未知方法: {method}")

def test_potential_accuracy():
    """
    **精度测试**：与解析解对比
    """
    import math
    print("\n" + "="*60)
    print("电势计算器精度验证")
    print("="*60)
    
    # 验证库仑常数与真空介电常数的关系
    calculated_k = 1.0 / (4 * math.pi * VACUUM_PERMITTIVITY)
    print(f"\n【物理常数验证】")
    print(f"从VACUUM_PERMITTIVITY计算的库仑常数: {calculated_k:.6e} N·m²/C²")
    print(f"直接导入的库仑常数: {COULOMB_CONSTANT:.6e} N·m²/C²")
    print(f"常数一致性: {'✓' if abs(calculated_k - COULOMB_CONSTANT) < 1e-12 else '✗'}")
    
    # 场景1：单点电荷解析解对比
    point_charge = PointCharge(q=1e-6, position=[0,0,0])
    calc = PotentialCalculator([point_charge])
    
    test_point = np.array([1, 0, 0])
    V_numeric = calc.potential(test_point)
    V_analytic = COULOMB_CONSTANT * 1e-6 / 1.0
    
    print(f"\n【点电荷解析验证】")
    print(f"数值解: {V_numeric:.6e} V")
    print(f"解析解: {V_analytic:.6e} V")
    print(f"绝对误差: {abs(V_numeric - V_analytic):.2e}")
    print(f"相对误差: {abs(V_numeric - V_analytic)/abs(V_analytic):.2e}")
    
    # 场景2：叠加原理
    charges = [
        PointCharge(q=1e-6, position=[0,0,0]),
        PointCharge(q=-0.5e-6, position=[1,0,0])
    ]
    calc_multi = PotentialCalculator(charges)
    
    point_mid = np.array([0.5, 0, 0])
    V_total = calc_multi.potential(point_mid)
    V1 = charges[0].potential(point_mid)
    V2 = charges[1].potential(point_mid)
    
    print(f"\n【叠加原理验证】")
    print(f"V₁: {V1:.6e} V")
    print(f"V₂: {V2:.6e} V")
    print(f"V₁+V₂: {V1+V2:.6e} V")
    print(f"V_total: {V_total:.6e} V")
    print(f"叠加误差: {abs(V_total - (V1+V2)):.2e}")
    
    # 场景3：参考点相对化
    calc_ref = PotentialCalculator(charges, reference_point=[0.5, 0, 0])
    V_relative = calc_ref.potential([0,0,0])
    V_abs = calc_ref.potential([0,0,0], absolute=True)
    V_ref = calc_ref.potential([0.5, 0, 0], absolute=True)
    
    print(f"\n【参考点相对化验证】")
    print(f"绝对电势V(0,0,0): {V_abs:.6e} V")
    print(f"参考点电势V_ref: {V_ref:.6e} V")
    print(f"相对电势V_rel = V_abs - V_ref: {V_relative:.6e} V")
    print(f"手动计算差值: {V_abs - V_ref:.6e} V")
    print(f"一致性: {'✓' if abs(V_relative - (V_abs - V_ref)) < 1e-12 else '✗'}")
    
    # 场景4：线电荷电势计算
    line_charge = LineCharge(lambda_val=1e-9, position=[1.0, 0.0], radius=0.05)
    calc_line = PotentialCalculator([line_charge])
    
    # 测试线电荷电势的对数关系
    test_points_line = np.array([
        [2.0, 0.0, 0.0],  # r=1.0m
        [3.0, 0.0, 0.0],  # r=2.0m
        [6.0, 0.0, 0.0]   # r=5.0m
    ])
    
    V_line = calc_line.potential(test_points_line)
    print(f"\n【线电荷电势验证】")
    print(f"线电荷λ=1e-9 C/m，位于(1.0, 0.0, 0.0)")
    for i, (point, V) in enumerate(zip(test_points_line, V_line)):
        r = np.linalg.norm(point[:2] - np.array([1.0, 0.0]))
        print(f"  点{point} (r={r:.1f}m): V={V:.2e} V")
    
    # 验证电势比值与距离对数的关系
    V_ratio = V_line[1] / V_line[0]
    expected_ratio = np.log(2.0/line_charge.reference_radius) / np.log(1.0/line_charge.reference_radius)
    print(f"  电势比值(V2/V1): {V_ratio:.2f}")
    print(f"  对数比值(ln(r2/r0)/ln(r1/r0)): {expected_ratio:.2f}")
    print(f"  电势对数关系一致性: {'✓' if abs(V_ratio - expected_ratio) < 1e-6 else '✗'}")
    
    # 场景5：圆环电荷电势计算
    ring_charge = RingCharge(q=1e-6, radius=0.5, position=[0.0, 0.0], center_position=[0.0, 0.0, 0.0])
    calc_ring = PotentialCalculator([ring_charge])
    
    # 测试圆环轴线上的电势（解析解对比）
    z_values = np.array([0.0, 0.3, 0.5, 1.0])
    test_points_ring = np.array([[0.0, 0.0, z] for z in z_values])
    
    V_ring = calc_ring.potential(test_points_ring)
    print(f"\n【圆环电荷电势验证】")
    print(f"圆环电荷Q=1e-6 C，半径R=0.5m，中心在原点")
    for i, (point, V) in enumerate(zip(test_points_ring, V_ring)):
        z = point[2]
        # 轴线上电势解析解：V = kQ / sqrt(R² + z²)
        R = ring_charge.R
        kQ = ring_charge.k_times_Q
        V_analytic = kQ / np.sqrt(R**2 + z**2)
        print(f"  轴线点(0,0,{z:.1f})m: V_num={V:.2e} V, V_ana={V_analytic:.2e} V")
        print(f"    相对误差: {abs(V - V_analytic)/V_analytic:.2e}")
    
    # 测试非轴线上的电势
    test_points_off_axis = np.array([
        [0.6, 0.2, 0.3],  # 非轴线上的点
        [0.8, 0.8, 0.5]   # 另一个非轴线上的点
    ])
    V_off_axis = calc_ring.potential(test_points_off_axis)
    print(f"\n  非轴线点电势：")
    for point, V in zip(test_points_off_axis, V_off_axis):
        print(f"    点{point}: V={V:.2e} V")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    
    # 运行精度测试
    test_potential_accuracy()
    
    print("\n" + "="*60)
    print("电势计算器验证完成")
    print("="*60)