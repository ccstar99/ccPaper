# core/field_line_tracer.py
import numpy as np
from typing import List
from physics.point import PointCharge
from physics.line import LineCharge
from physics.ring import RingCharge
from core.field_calculator import FieldCalculator
from utils.constants import COULOMB_CONSTANT

class AdaptiveFieldLineTracer:
    """
    自适应电场线追踪器（核心算法）
    
    物理原理：
        电场线是矢量场E(r)的积分曲线，满足 dr/ds = E(r)/|E(r)|
        其中s为弧长参数，方向表示电场方向
        
    数学方法：
        - 四阶龙格-库塔积分（RK4）：高精度常微分方程求解
        - 自适应步长：h ∝ 1/|E|，在电场强或高曲率区域自动加密
        - 均匀初始分布：在正电荷表面生成均匀分布的起始点
        
    特性：
        - 从正电荷表面出发，终止于负电荷或无穷远
        - 起始点数量与电荷量成正比，最小100条
        - 支持2D和3D场景
    """
    
    def _get_charge_value(self, charge):
        """
        获取电荷的电荷量值
        
        Args:
            charge: 电荷对象
            
        Returns:
            电荷量值
        """
        if hasattr(charge, 'q'):
            return charge.q
        elif hasattr(charge, 'lambda_val'):
            return charge.lambda_val
        return 0
    
    def __init__(self, charges: List, 
                 min_step: float = 1e-4,
                 max_step: float = 0.1,
                 max_iter: int = 10000,
                 term_field: float = 1e-6,
                 boundary_radius: float = 10.0,
                 dim: int = 3,
                 n_field_lines: int = 150):
        """
        初始化追踪器
        
        Args:
            charges: 电荷对象列表
            min_step: 最小步长（电场强区域）
            max_step: 最大步长（电场弱区域）
            max_iter: 单条线最大迭代次数
            term_field: 终止场强阈值
            boundary_radius: 计算边界半径
            dim: 维度（2或3）
            n_field_lines: 电场线数量
        """
        self.charges = charges
        self.min_step = min_step
        self.max_step = max_step
        self.max_iter = max_iter
        self.term_field = term_field
        self.boundary_radius = boundary_radius
        self.dim = dim
        self.n_field_lines = n_field_lines
        
        self.field_calc = FieldCalculator(charges)
        
        # 分离正负电荷
        self.positive_charges = [c for c in charges if self._get_charge_value(c) > 0]
        self.negative_charges = [c for c in charges if self._get_charge_value(c) < 0]
        
        # 总正电荷量
        self.total_positive_charge = sum(self._get_charge_value(c) for c in self.positive_charges)
        
        # 参考场强（用于自适应步长）
        if self.positive_charges:
            min_charge = min(self._get_charge_value(c) for c in self.positive_charges)
            self.E_ref = COULOMB_CONSTANT * abs(min_charge) / (0.1 ** 2)
        else:
            self.E_ref = 1e6
        
    def _generate_seed_points(self) -> List[np.ndarray]:
        """在所有正电荷表面生成均匀起始点"""
        if self.total_positive_charge == 0:
            print("警告：未找到正电荷，无法生成电场线")
            return []
            
        all_seed_points = []
        
        # 单位电荷线密度（使用用户指定的电场线数量）
        base_line_density = self.n_field_lines / self.total_positive_charge
        
        for charge in self.positive_charges:
            # 按电荷量比例分配线数（物理要求）
            num_points = max(10, int(base_line_density * self._get_charge_value(charge)))
            
            if num_points == 0:
                continue
                
            # 根据电荷类型生成表面点
            if isinstance(charge, PointCharge):
                points = self._generate_points_on_sphere(charge, num_points)
            elif isinstance(charge, LineCharge):
                points = self._generate_points_on_line_cylinder(charge, num_points)
            elif isinstance(charge, RingCharge):
                points = self._generate_points_on_ring_torus(charge, num_points)
            else:
                raise TypeError(f"不支持的电荷类型: {type(charge)}")
                
            all_seed_points.extend(points)
            
        return all_seed_points
    
    def _generate_points_on_sphere(self, charge: PointCharge, num_points: int) -> List[np.ndarray]:
        """在球面生成均匀点（斐波那契球算法）"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # 黄金角
        
        # 确保电荷位置是3D向量
        center = np.array(charge.position, dtype=float)
        if len(center) == 2:
            center = np.append(center, 0.0)
        radius = charge.radius
        
        for i in range(num_points):
            y = 1 - (i / (num_points - 1)) * 2 if num_points > 1 else 0
            radius_at_y = np.sqrt(max(0, 1 - y * y))
            theta = phi * i
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            point = center + radius * np.array([x, y, z])
            
            if self.dim == 2:
                point[2] = 0
            
            points.append(point)
            
        return points
    
    def _generate_points_on_line_cylinder(self, charge: LineCharge, num_points: int) -> List[np.ndarray]:
        """在线电荷周围圆柱面生成起始点"""
        points = []
        
        # 确保电荷位置是3D向量
        charge_pos = np.array(charge.position, dtype=float)
        if len(charge_pos) == 2:
            charge_pos = np.append(charge_pos, 0.0)
        
        if self.dim == 2:
            # 2D模式：在线电荷周围生成均匀分布的起始点
            # 生成多个同心圆，每个圆上均匀分布点
            n_circles = 3  # 2D模式下生成3个同心圆
            points_per_circle = num_points // n_circles
            
            for circle_idx in range(n_circles):
                # 不同半径的圆
                radius = charge.radius * (1 + circle_idx * 0.5)
                
                for i in range(points_per_circle):
                    angle = 2 * np.pi * i / points_per_circle
                    # 2D模式下只在xy平面生成
                    offset = radius * np.array([np.cos(angle), np.sin(angle), 0])
                    point = charge_pos.copy()
                    point += offset
                    point[2] = 0  # 2D模式下强制z=0
                    points.append(point)
        else:
            # 3D模式：沿轴线分布多个圆
            num_circles = max(1, num_points // 10)
            
            for circle_idx in range(num_circles):
                z_offset = (circle_idx - num_circles/2) * charge.radius * 2
                
                n_per_circle = num_points // num_circles
                for i in range(n_per_circle):
                    angle = 2 * np.pi * i / n_per_circle
                    offset = charge.radius * np.array([np.cos(angle), np.sin(angle), 0])
                    point = charge_pos.copy()
                    point[2] = z_offset  # 使用z_offset替换原z坐标
                    point += offset
                    points.append(point)
        
        # 确保生成足够数量的点
        while len(points) < num_points:
            # 随机选择一个现有起始点并添加轻微扰动
            base_point = points[np.random.randint(0, len(points))].copy()
            perturbation = np.random.normal(0, charge.radius * 0.1, 3)
            new_point = base_point + perturbation
            if self.dim == 2:
                new_point[2] = 0  # 2D模式下保持z=0
            points.append(new_point)
                
        return points
    
    def _generate_points_on_ring_torus(self, charge: RingCharge, num_points: int) -> List[np.ndarray]:
        """在圆环电荷周围环面生成起始点"""
        points = []
        
        # 环面分布：增加跨截面点数，生成真正的3D起始点
        n_along_ring = max(1, num_points // 5)  # 沿圆环方向的点数
        n_around_cross = 5  # 跨截面方向的点数（增加到5个，生成3D分布）
        total_per_ring = n_along_ring * n_around_cross
        
        # 确保总点数足够
        if total_per_ring < num_points:
            n_along_ring += 1
            total_per_ring = n_along_ring * n_around_cross
        
        # 圆环半径作为可视化半径
        visual_radius = charge.R * 0.1
        
        # 确保电荷位置是3D向量
        charge_pos = np.array(charge.position, dtype=float)
        if len(charge_pos) == 2:
            charge_pos = np.append(charge_pos, 0.0)
        
        for i in range(n_along_ring):
            ring_angle = 2 * np.pi * i / n_along_ring
            
            # 圆环上的点
            ring_center = charge_pos + np.array([
                charge.R * np.cos(ring_angle),
                charge.R * np.sin(ring_angle),
                0
            ])
            
            # 生成环面的三个正交方向
            # 1. 径向（指向圆环中心）
            radial = np.array([-np.cos(ring_angle), -np.sin(ring_angle), 0])
            # 2. 轴向（垂直于圆环平面）
            axial = np.array([0, 0, 1])
            # 3. 切线方向（沿圆环切线）
            tangent = np.array([-np.sin(ring_angle), np.cos(ring_angle), 0])
            
            # 在环面上生成均匀分布的点
            for j in range(n_around_cross):
                # 跨截面角度：从0到2π，生成环面上的均匀点
                cross_angle = 2 * np.pi * j / n_around_cross
                
                # 使用球坐标生成环面上的点
                # 径向偏移（在径向和轴向平面内）
                radial_offset = visual_radius * np.cos(cross_angle)
                axial_offset = visual_radius * np.sin(cross_angle)
                
                # 计算最终点：圆环中心 + 径向偏移 + 轴向偏移
                point = ring_center + radial_offset * radial + axial_offset * axial
                
                # 对于3D模式，保留z分量；2D模式下强制z=0
                if self.dim == 2:
                    point[2] = 0
                    
                points.append(point)
                
        # 确保生成足够数量的点
        while len(points) < num_points:
            # 随机选择一个现有起始点并添加轻微扰动
            base_point = points[np.random.randint(0, len(points))].copy()
            perturbation = np.random.normal(0, visual_radius * 0.1, 3)
            points.append(base_point + perturbation)
                
        return points
    
    def _trace_single_line(self, start_point: np.ndarray) -> np.ndarray:
        """用RK4追踪单条电场线"""
        path = [start_point.copy()]
        point = start_point.copy()
        
        for iteration in range(self.max_iter):
            E = self.field_calc.electric_field(point)
            E_mag = np.linalg.norm(E)
            
            # 终止条件：场强过弱
            if E_mag < self.term_field:
                break
                
            # 终止条件：到达负电荷
            for neg_charge in self.negative_charges:
                # 确保电荷位置是3D向量
                charge_pos = np.array(neg_charge.position, dtype=float)
                if len(charge_pos) == 2:
                    charge_pos = np.append(charge_pos, 0.0)
                
                # 根据电荷类型处理不同的终止条件
                if hasattr(neg_charge, 'radius'):
                    # PointCharge和LineCharge有radius属性
                    dist_to_neg = np.linalg.norm(point - charge_pos)
                    if dist_to_neg < neg_charge.radius:
                        path.append(point)
                        return np.array(path)
                elif isinstance(neg_charge, RingCharge):
                    # RingCharge使用is_inside方法
                    if neg_charge.is_inside(point):
                        path.append(point)
                        return np.array(path)
            
            # 终止条件：意外返回正电荷内部（数值误差）
            for pos_charge in self.positive_charges:
                # 确保电荷位置是3D向量
                charge_pos = np.array(pos_charge.position, dtype=float)
                if len(charge_pos) == 2:
                    charge_pos = np.append(charge_pos, 0.0)
                
                # 根据电荷类型处理不同的终止条件
                if hasattr(pos_charge, 'radius'):
                    # PointCharge和LineCharge有radius属性
                    dist_to_pos = np.linalg.norm(point - charge_pos)
                    if dist_to_pos < pos_charge.radius * 0.8:
                        return np.array(path)
                elif isinstance(pos_charge, RingCharge):
                    # RingCharge使用is_inside方法
                    if pos_charge.is_inside(point):
                        return np.array(path)
            
            # 终止条件：到达边界且向外
            if np.linalg.norm(point) > self.boundary_radius:
                if np.dot(point, E) > 0:
                    break
            
            # 自适应步长
            step_size = self._adaptive_step_size(E_mag)
            
            # RK4积分
            k1 = self._field_direction(point)
            k2 = self._field_direction(point + 0.5 * step_size * k1)
            k3 = self._field_direction(point + 0.5 * step_size * k2)
            k4 = self._field_direction(point + step_size * k3)
            
            direction = (k1 + 2*k2 + 2*k3 + k4) / 6.0
            point += step_size * direction
            
            if self.dim == 2:
                point[2] = 0
                
            path.append(point.copy())
            
        return np.array(path)
    
    def _field_direction(self, point: np.ndarray) -> np.ndarray:
        """获取电场单位方向矢量"""
        E = self.field_calc.electric_field(point)
        E_mag = np.linalg.norm(E)
        
        if E_mag < 1e-12:
            return np.zeros(3, dtype=float)
            
        return E / E_mag
    
    def _adaptive_step_size(self, E_mag: float) -> float:
        """基于场强的自适应步长"""
        if E_mag < 1e-12:
            return self.max_step
            
        # 指数衰减：场强越大步长越小
        step = self.min_step + (self.max_step - self.min_step) * np.exp(-E_mag / self.E_ref)
        return np.clip(step, self.min_step, self.max_step)
    
    def trace_all_field_lines(self) -> List[np.ndarray]:
        """追踪所有电场线（主入口）"""
        seed_points = self._generate_seed_points()
        
        if not seed_points:
            return []
            
        print(f"信息：生成 {len(seed_points)} 个起始点，开始追踪...")
        
        # 顺序追踪（可扩展为并行）
        lines = []
        for i, point in enumerate(seed_points):
            line = self._trace_single_line(point)
            if len(line) >= 2:
                lines.append(line)
            
            if (i + 1) % 20 == 0:
                print(f"  已完成 {i + 1}/{len(seed_points)} 条...")
        
        print(f"成功追踪 {len(lines)} 条电场线")
        return lines


# ==================== 测试套件 ====================
def create_test_scene(dim: int = 3) -> List:
    """
    创建标准测试场景（电偶极子）
    """
    if dim == 2:
        return [
            PointCharge(q=1e-6, position=[0, 0, 0], radius=0.1),
            PointCharge(q=-1e-6, position=[1, 0, 0], radius=0.1)
        ]
    else:
        return [
            PointCharge(q=1e-6, position=[0, 0, 0], radius=0.1),
            PointCharge(q=-1e-6, position=[1, 0, 0], radius=0.1),
            LineCharge(lambda_val=0.5e-9, position=[0.5, 0.5], radius=0.05)
        ]

def test_tracer():
    """
    综合验证测试
    """
    print("\n" + "="*60)
    print("电场线追踪器验证测试")
    print("="*60)
    
    # 测试1：3D场景
    print("\n【测试1：3D场景追踪】")
    charges_3d = create_test_scene(dim=3)
    tracer_3d = AdaptiveFieldLineTracer(charges_3d, dim=3, max_iter=500)
    lines_3d = tracer_3d.trace_all_field_lines()
    
    print(f"生成 {len(lines_3d)} 条3D电场线")
    if lines_3d:
        avg_length = np.mean([len(line) for line in lines_3d])
        print(f"平均长度: {avg_length:.1f} 个点")
        
        # 验证终止于负电荷
        neg_pos = charges_3d[1].position
        neg_radius = charges_3d[1].radius
        terminated_correctly = sum(
            1 for line in lines_3d 
            if len(line) > 1 and np.linalg.norm(line[-1] - neg_pos) <= neg_radius * 1.5
        )
        print(f"正确终止比例: {terminated_correctly}/{len(lines_3d)} = {terminated_correctly/len(lines_3d):.1%}")
    
    # 测试2：2D场景
    print("\n【测试2：2D场景追踪】")
    charges_2d = create_test_scene(dim=2)
    tracer_2d = AdaptiveFieldLineTracer(charges_2d, dim=2, max_iter=500)
    lines_2d = tracer_2d.trace_all_field_lines()
    
    print(f"生成 {len(lines_2d)} 条2D电场线")
    if lines_2d:
        avg_length = np.mean([len(line) for line in lines_2d])
        print(f"平均长度: {avg_length:.1f} 个点")
        
        # 验证Z轴约束
        max_z_error = max(np.abs(line[:, 2]).max() for line in lines_2d)
        print(f"Z轴约束误差: {max_z_error:.2e} (应≈0)")
    
    # 测试3：最少线数验证
    print("\n【测试3：最少线数验证】")
    single_charge = [PointCharge(q=1e-6, position=[0, 0, 0], radius=0.1)]
    single_charge.append(PointCharge(q=-1e-9, position=[10, 0, 0], radius=0.1))
    
    tracer_min = AdaptiveFieldLineTracer(single_charge, dim=3)
    lines_min = tracer_min.trace_all_field_lines()
    
    print(f"场景生成 {len(lines_min)} 条线")
    status = "✓" if len(lines_min) >= 100 else "✗"
    print(f"{status} 最小100条线要求")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_tracer()