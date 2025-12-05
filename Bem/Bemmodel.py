import numpy as np
from typing import List, Tuple
# ==================== 1. 数学函数库 ====================
# 严格遵循球面三角学公式，所有角度单位为弧度

def great_circle_arc_length(p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float) -> float:
    """
    计算两点间大圆弧长（论文式6）
    p1, p2: 球面上两点坐标
    center: 球心O
    radius: 球半径r
    公式: s = r·cos⁻¹[(p1-O)·(p2-O)/r²]
    """
    v1 = p1 - center
    v2 = p2 - center
    cos_angle = np.clip(np.dot(v1, v2) / (radius ** 2), -1.0, 1.0)
    return radius * np.arccos(cos_angle)


def spherical_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, center: np.ndarray) -> float:
    """
    计算球面三角形在顶点b处的角度（即平面Oab与Obc的二面角）
    用于球面面积计算
    """
    # 确保所有输入都是3D坐标点
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    c = np.asarray(c).reshape(-1)
    center = np.asarray(center).reshape(-1)
    
    # 处理形状为(12,)的情况，这可能是4个3D坐标拼接在一起
    if a.shape[0] == 12:
        # 提取第一个3D坐标
        a = a[:3]
    if b.shape[0] == 12:
        # 提取第二个3D坐标
        b = b[3:6]
    if c.shape[0] == 12:
        # 提取第三个3D坐标
        c = c[6:9]
    if center.shape[0] == 12:
        # 提取第四个3D坐标作为球心
        center = center[9:12]
    
    # 检查维度
    if a.shape[0] != 3 or b.shape[0] != 3 or c.shape[0] != 3 or center.shape[0] != 3:
        raise ValueError(f"所有输入点必须是3D坐标。当前形状：a={a.shape}, b={b.shape}, c={c.shape}, center={center.shape}")
    
    # 向量归一化
    ba = a - center
    bc = c - center
    
    # 计算模长
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    
    # 归一化
    if ba_norm > 1e-10:
        ba_norm_vec = ba / ba_norm
    else:
        ba_norm_vec = np.zeros_like(ba)
    
    if bc_norm > 1e-10:
        bc_norm_vec = bc / bc_norm
    else:
        bc_norm_vec = np.zeros_like(bc)

    # 计算两平面法向量
    # 平面Oba的法向（指向球面外）
    cross1 = np.cross(ba_norm_vec, bc_norm_vec)
    n1 = np.cross(ba_norm_vec, cross1)
    
    # 归一化n1
    n1_norm = np.linalg.norm(n1)
    if n1_norm > 1e-10:
        n1 = n1 / n1_norm
    else:
        n1 = np.zeros_like(n1)
    
    # 平面Obc的法向（指向球面外）
    cross2 = np.cross(bc_norm_vec, ba_norm_vec)
    n2 = np.cross(bc_norm_vec, cross2)
    
    # 归一化n2
    n2_norm = np.linalg.norm(n2)
    if n2_norm > 1e-10:
        n2 = n2 / n2_norm
    else:
        n2 = np.zeros_like(n2)

    # 计算二面角的余弦值
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    
    return np.arccos(cos_angle)


def spherical_triangle_area(vertices: List[np.ndarray], center: np.ndarray, radius: float) -> float:
    """
    计算球面三角形面积（球面角超公式，论文隐含使用）
    公式: S = r²(A + B + C - π)
    A,B,C为三个球面内角
    """
    A = spherical_angle(vertices[2], vertices[0], vertices[1], center)  # 顶点0处的角
    B = spherical_angle(vertices[0], vertices[1], vertices[2], center)  # 顶点1处的角
    C = spherical_angle(vertices[1], vertices[2], vertices[0], center)  # 顶点2处的角

    area = radius ** 2 * (A + B + C - np.pi)

    # 数值稳定性处理：极小三角形面积可能为负
    return max(area, 1e-12)


def project_to_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    将点投影到球面上（保持径向距离严格等于r）
    论文式3的逆变换
    """
    vectors = points - center
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return center + radius * vectors / norms


def spherical_centroid(tri_coords: List[np.ndarray], center: np.ndarray, radius: float) -> np.ndarray:
    """
    计算球面三角形的"质心"（三个顶点向量和的单位化）
    用于高斯点生成
    """
    centroid_vec = sum(tri_coords) / 3.0 - center
    return center + radius * centroid_vec / np.linalg.norm(centroid_vec)


# ==================== 2. 球面三角形单元类 ====================

class SphericalTriangle:
    """
    球面三角形单元（论文核心数据结构）
    三条边均为大圆弧，严格位于球面上
    """

    def __init__(self, vertex_indices: Tuple[int, int, int],
                 vertex_coords: List[np.ndarray],
                 center: np.ndarray,
                 radius: float):
        """
        vertex_indices: 三个顶点在全局列表中的索引
        vertex_coords: 三个顶点的笛卡尔坐标（必须严格在球面上）
        center: 球心O
        radius: 球半径r
        """
        assert len(vertex_indices) == 3
        assert len(vertex_coords) == 3

        self.vertex_indices = tuple(vertex_indices)
        self.vertices = [np.array(v, dtype=np.float64) for v in vertex_coords]
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)

        # 预计算几何属性
        self.normal = self._compute_normal()  # 严格球面法线（论文式4）
        self.area = spherical_triangle_area(self.vertices, self.center, self.radius)

        # 缓存形状函数计算中间结果
        self._edge_lengths = self._compute_edge_lengths()
        self._spherical_angles = self._compute_spherical_angles()

    def _compute_normal(self) -> np.ndarray:
        """计算严格球面外法线（论文式4）"""
        vec = self.vertices[0] - self.center
        norm = np.linalg.norm(vec)
        return vec / norm

    def _compute_edge_lengths(self) -> np.ndarray:
        """计算三条大圆弧边长"""
        lengths = np.array([
            great_circle_arc_length(self.vertices[0], self.vertices[1],
                                    self.center, self.radius),
            great_circle_arc_length(self.vertices[1], self.vertices[2],
                                    self.center, self.radius),
            great_circle_arc_length(self.vertices[2], self.vertices[0],
                                    self.center, self.radius)
        ])
        return lengths

    def _compute_spherical_angles(self) -> np.ndarray:
        """计算三个球面内角（弧度）"""
        angles = np.array([
            spherical_angle(self.vertices[2], self.vertices[0], self.vertices[1], self.center),
            spherical_angle(self.vertices[0], self.vertices[1], self.vertices[2], self.center),
            spherical_angle(self.vertices[1], self.vertices[2], self.vertices[0], self.center)
        ])
        return angles

    def contains_point(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """
        改进的判断点是否在球面三角形内部的方法
        使用球面重心坐标符号判断，更鲁棒
        """
        p = np.array(point, dtype=np.float64)
        
        # 确保点在球面上（归一化）
        if not np.isclose(np.linalg.norm(p - self.center), self.radius, rtol=1e-8):
            p = p - self.center
            p = self.center + self.radius * p / np.linalg.norm(p)
        
        # 计算球面重心坐标（面积坐标）
        areas = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            # 计算子三角形面积
            sub_area = spherical_triangle_area(
                [self.vertices[i], self.vertices[j], p],
                self.center, self.radius
            )
            areas.append(sub_area)
        
        # 计算重心坐标
        total_area = sum(areas)
        if total_area < 1e-12:  # 退化情况
            return False
        
        barycentric = [area / total_area for area in areas]
        
        # 判断：所有重心坐标都在[0,1]范围内（允许小误差）
        for coord in barycentric:
            if coord < -tol or coord > 1 + tol:
                return False
        
        return True

    def shape_functions(self, point: np.ndarray) -> np.ndarray:
        """
        改进的形状函数计算方法
        直接使用球面重心坐标，无需先判断点是否在内部
        """
        # 确保点在球面上
        point_vec = point - self.center
        point_norm = np.linalg.norm(point_vec)
        if not np.isclose(point_norm, self.radius, rtol=1e-8):
            point = self.center + self.radius * point_vec / point_norm
        
        # 计算三个子三角形的面积
        areas = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            
            # 确保计算顺序正确
            area_i = spherical_triangle_area(
                [self.vertices[j], self.vertices[k], point],
                self.center, self.radius
            )
            areas[i] = max(area_i, 0)  # 避免负面积
        
        # 计算重心坐标（即使点在外部也有效）
        total_area = np.sum(areas)
        if total_area < 1e-12:
            # 退化情况，返回平均值
            return np.array([1/3, 1/3, 1/3], dtype=np.float64)
        
        # 形状函数就是重心坐标
        N = areas / total_area
        
        # 归一化确保和为1（处理浮点误差）
        N = N / np.sum(N)
        
        return N

    def gauss_quadrature_points(self, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        改进的高斯积分点生成
        直接在球面三角形上生成积分点，无需平面投影
        """
        # 使用球面三角形上的高斯点公式
        # 基于球面三角形的重心坐标
        if order == 1:
            # 1点公式（重心）
            bary_coords = np.array([[1/3, 1/3, 1/3]])
            weights = np.array([self.area])
        elif order == 3:
            # 3点公式
            # 使用对称点
            alpha = 1/3
            beta = 2/3
            gamma = 0
            bary_coords = np.array([
                [alpha, beta, gamma],
                [gamma, alpha, beta],
                [beta, gamma, alpha]
            ])
            weights = np.array([self.area/3, self.area/3, self.area/3])
        elif order == 4:
            # 4点公式（二阶精度）
            # 使用文献中的权重和坐标
            a = 0.5
            b = 0.5
            c = 0.0
            bary_coords = np.array([
                [1/3, 1/3, 1/3],  # 重心
                [a, b, c],
                [c, a, b],
                [b, c, a]
            ])
            weights = np.array([
                -9/16 * self.area,
                25/48 * self.area,
                25/48 * self.area,
                25/48 * self.area
            ])
        else:
            raise ValueError(f"不支持的积分阶数: {order}")
        
        # 将重心坐标转换为球面上的点
        n_points = len(bary_coords)
        points = np.zeros((n_points, 3))
        
        for i, (w1, w2, w3) in enumerate(bary_coords):
            # 球面线性插值（Slerp）
            # 使用单位向量插值
            v1 = self.vertices[0] - self.center
            v2 = self.vertices[1] - self.center
            v3 = self.vertices[2] - self.center
            
            # 归一化
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)
            
            # 球面线性插值
            if w1 + w2 + w3 < 1e-12:
                # 零向量，使用中心
                point_vec = np.zeros(3)
            else:
                # 权重和插值
                point_vec = w1 * v1 + w2 * v2 + w3 * v3
                point_norm = np.linalg.norm(point_vec)
                if point_norm > 1e-12:
                    point_vec = point_vec / point_norm
            
            # 缩放到球面
            points[i] = self.center + self.radius * point_vec
        
        return points, weights

    def gauss_points_with_jacobian(self, order: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """ 
        生成球面三角形上的高斯积分点，包含雅可比行列式 
        按照论文式(1)-(4)实现 
        
        返回: 
            points_sphere: (n_points, 3) 球面上积分点坐标 
            points_plane: (n_points, 3) 对应平面三角形上的点坐标 
            jacobians: (n_points,) 雅可比行列式值 |J| 
        """ 
        # 平面三角形的高斯点（面积坐标） 
        if order == 1: 
            bary_coords = np.array([[1/3, 1/3, 1/3]]) 
            weights = np.array([1.0]) 
        elif order == 3: 
            bary_coords = np.array([ 
                [2/3, 1/6, 1/6], 
                [1/6, 2/3, 1/6], 
                [1/6, 1/6, 2/3] 
            ]) 
            weights = np.array([1/3, 1/3, 1/3]) 
        elif order == 4: 
            # 二阶精度4点公式 
            a = (5 + 3*np.sqrt(5))/20 
            b = (5 - np.sqrt(5))/20 
            bary_coords = np.array([ 
                [a, b, b], 
                [b, a, b], 
                [b, b, a], 
                [1/3, 1/3, 1/3] 
            ]) 
            weights = np.array([(5-np.sqrt(5))/20, (5-np.sqrt(5))/20, 
                               (5-np.sqrt(5))/20, (5+3*np.sqrt(5))/20]) 
        else: 
            raise ValueError(f"不支持的积分阶数: {order}") 
        
        n_points = len(bary_coords) 
        points_plane = np.zeros((n_points, 3)) 
        points_sphere = np.zeros((n_points, 3)) 
        jacobians = np.zeros(n_points) 
        
        # 计算平面三角形法向量 
        v0, v1, v2 = self.vertices 
        edge1 = v1 - v0 
        edge2 = v2 - v0 
        normal_plane = np.cross(edge1, edge2) 
        normal_plane = normal_plane / np.linalg.norm(normal_plane) 
        
        for i, (w1, w2, w3) in enumerate(bary_coords): 
            # 平面三角形上的点（论文中的a点） 
            point_plane = w1 * v0 + w2 * v1 + w3 * v2 
            points_plane[i] = point_plane 
            
            # 投影到球面（论文中的b点，式3） 
            vec_plane = point_plane - self.center 
            r_prime = np.linalg.norm(vec_plane) 
            
            if r_prime < 1e-12: 
                point_sphere = self.center + self.radius * normal_plane 
            else: 
                point_sphere = self.center + self.radius * vec_plane / r_prime 
            
            points_sphere[i] = point_sphere 
            
            # 计算雅可比行列式 |J| = (r/r')² cosα（论文式2） 
            if r_prime < 1e-12: 
                jacobians[i] = 0.0 
            else: 
                # 球面法向量（径向，论文式4） 
                normal_sphere = (point_sphere - self.center) / self.radius 
                
                # cosα = n·n' 
                cos_alpha = np.dot(normal_sphere, normal_plane) 
                cos_alpha = np.clip(cos_alpha, 0.0, 1.0)  # 避免数值误差 
                
                jacobians[i] = (self.radius / r_prime)**2 * cos_alpha 
        
        return points_sphere, points_plane, jacobians * weights.reshape(-1, 1)

    def subdivide(self) -> List['SphericalTriangle']:
        """
        将球面三角形细分为4个子三角形（用于自适应细分）
        细分后所有新顶点严格在球面上

        返回: 4个SphericalTriangle实例
        """
        # 计算各边中点并投影到球面
        mid12 = project_to_sphere(
            ((self.vertices[0] + self.vertices[1]) / 2).reshape(1, -1),
            self.center, self.radius
        )[0]
        mid23 = project_to_sphere(
            ((self.vertices[1] + self.vertices[2]) / 2).reshape(1, -1),
            self.center, self.radius
        )[0]
        mid31 = project_to_sphere(
            ((self.vertices[2] + self.vertices[0]) / 2).reshape(1, -1),
            self.center, self.radius
        )[0]

        # 当前单元索引
        i, j, k = self.vertex_indices

        # 新顶点坐标
        new_vertices = [mid12, mid23, mid31]

        # 生成4个子三角形（顶点顺序保持外法向一致）
        triangles = [
            SphericalTriangle((i, j, len(new_vertices) + 0),
                              [self.vertices[0], self.vertices[1], mid12],
                              self.center, self.radius),
            SphericalTriangle((j, k, len(new_vertices) + 1),
                              [self.vertices[1], self.vertices[2], mid23],
                              self.center, self.radius),
            SphericalTriangle((k, i, len(new_vertices) + 2),
                              [self.vertices[2], self.vertices[0], mid31],
                              self.center, self.radius),
            SphericalTriangle((len(new_vertices) + 0, len(new_vertices) + 1, len(new_vertices) + 2),
                              [mid12, mid23, mid31],
                              self.center, self.radius)
        ]

        return triangles

    def __repr__(self):
        return (f"SphericalTriangle(vertices={self.vertex_indices}, "
                f"area={self.area:.6f}, center={self.center})")


# ==================== 3. 球面网格生成器 ====================

class SphericalMesh:
    """
    球面三角形网格生成器（基于正二十面体细分）
    确保所有单元均为严格球面三角形
    """

    def __init__(self, radius: float = 1.0, center: Tuple[float, float, float] = (0, 0, 0),
                 subdivisions: int = 0):
        """
        参数:
            radius: 球半径 r
            center: 球心坐标 (x0, y0, z0)
            subdivisions: 细分次数，0=正二十面体（20个单元），
                         1=80个单元，2=320个单元
        """
        self.radius = float(radius)
        self.center = np.array(center, dtype=np.float64)

        # 初始化正二十面体顶点（黄金比例）
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]

        # 归一化到单位球面并缩放
        self.vertices = np.array([
            self.center + radius * (v / np.linalg.norm(v)) for v in vertices
        ], dtype=np.float64)

        # 正二十面体面定义（顶点索引）
        self.triangles = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ]

        # 执行细分
        for _ in range(subdivisions):
            self._subdivide()

        # 创建SphericalTriangle对象列表
        self.spherical_triangles = self._create_triangle_objects()

    def _subdivide(self):
        """执行一次球面三角形细分（每条边一分为二）"""
        new_triangles = []
        edge_midpoint_cache = {}  # 避免重复计算中点

        def get_midpoint(i, j):
            """获取边(i,j)的中点（缓存并投影到球面）"""
            key = tuple(sorted((i, j)))
            if key not in edge_midpoint_cache:
                mid = project_to_sphere(
                    ((self.vertices[i] + self.vertices[j]) / 2).reshape(1, -1),
                    self.center, self.radius
                )[0]
                edge_midpoint_cache[key] = len(self.vertices)
                self.vertices = np.vstack([self.vertices, mid])
            return edge_midpoint_cache[key]

        for tri in self.triangles:
            i, j, k = tri

            # 获取各边中点
            mid_ij = get_midpoint(i, j)
            mid_jk = get_midpoint(j, k)
            mid_ki = get_midpoint(k, i)

            # 生成4个子三角形（保持外法向一致）
            new_triangles.extend([
                (i, mid_ij, mid_ki),
                (j, mid_jk, mid_ij),
                (k, mid_ki, mid_jk),
                (mid_ij, mid_jk, mid_ki)
            ])

        self.triangles = new_triangles

    def _create_triangle_objects(self) -> List[SphericalTriangle]:
        """将拓扑数据转换为SphericalTriangle对象列表"""
        return [
            SphericalTriangle(
                tri,
                [self.vertices[i] for i in tri],
                self.center,
                self.radius
            )
            for tri in self.triangles
        ]

    def get_vertex(self, idx: int) -> np.ndarray:
        """获取顶点坐标"""
        return self.vertices[idx].copy()

    def get_triangle(self, idx: int) -> SphericalTriangle:
        """获取球面三角形单元"""
        return self.spherical_triangles[idx]

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    @property
    def total_area(self) -> float:
        """计算球面总表面积（验证用，应接近4πr²）"""
        return sum(tri.area for tri in self.spherical_triangles)

    def export_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        导出网格数据供可视化使用
        返回: (vertices, faces)
        faces格式: [n_verts, v0, v1, v2, n_verts, v3, v4, v5, ...]
        """
        faces = []
        for tri in self.triangles:
            faces.extend([3, tri[0], tri[1], tri[2]])

        return self.vertices.copy(), np.array(faces, dtype=np.int32)

    def __repr__(self):
        return (f"SphericalMesh(radius={self.radius}, center={self.center}, "
                f"vertices={self.num_vertices}, triangles={self.num_triangles})")


# ==================== 4. 工具函数 ====================

def generate_icosphere(radius: float = 1.0, center: Tuple[float, float, float] = (0, 0, 0),
                       subdivisions: int = 1) -> SphericalMesh:
    """
    生成球面三角形网格的工厂函数

    参数:
        radius: 球半径
        center: 球心坐标
        subdivisions: 细分次数

    返回:
        SphericalMesh实例

    示例:
        # 生成分辨率为80个三角形的球面
        mesh = generate_icosphere(radius=1.0, subdivisions=1)
    """
    return SphericalMesh(radius=radius, center=center, subdivisions=subdivisions)


def validate_mesh(mesh: SphericalMesh) -> dict:
    """
    验证球面网格的几何精度

    返回:
        包含误差统计的字典
    """
    errors = {
        "vertex_radius_error": 0.0,
        "area_error_rel": 0.0
    }

    # 验证所有顶点严格在球面上
    radius_deviations = np.abs(np.linalg.norm(mesh.vertices - mesh.center, axis=1) - mesh.radius)
    errors["vertex_radius_error"] = np.max(radius_deviations)

    # 验证总表面积
    exact_area = 4 * np.pi * mesh.radius ** 2
    computed_area = mesh.total_area
    errors["area_error_rel"] = abs(computed_area - exact_area) / exact_area

    return errors


# ==================== 5. 测试与验证 ====================

if __name__ == "__main__":
    # 测试正二十面体生成
    print("=== 测试1: 正二十面体 ===")
    mesh0 = generate_icosphere(radius=1.0, subdivisions=0)
    errors0 = validate_mesh(mesh0)
    print(f"顶点数: {mesh0.num_vertices}, 单元数: {mesh0.num_triangles}")
    print(f"半径最大误差: {errors0['vertex_radius_error']:.2e}")
    print(f"面积相对误差: {errors0['area_error_rel']:.2e}")
    print(f"总面积: {mesh0.total_area:.6f} (理论: {4 * np.pi:.6f})")

    # 测试一次细分
    print("\n=== 测试2: 一次细分（80单元） ===")
    mesh1 = generate_icosphere(radius=1.0, subdivisions=1)
    errors1 = validate_mesh(mesh1)
    print(f"顶点数: {mesh1.num_vertices}, 单元数: {mesh1.num_triangles}")
    print(f"半径最大误差: {errors1['vertex_radius_error']:.2e}")
    print(f"面积相对误差: {errors1['area_error_rel']:.2e}")
    print(f"总面积: {mesh1.total_area:.6f} (理论: {4 * np.pi:.6f})")

    # 测试形状函数
    print("\n=== 测试3: 形状函数 ===")
    tri = mesh1.get_triangle(0)
    centroid = spherical_centroid(tri.vertices, mesh1.center, mesh1.radius)
    N = tri.shape_functions(centroid)
    print(f"质心处形状函数: N1={N[0]:.6f}, N2={N[1]:.6f}, N3={N[2]:.6f}")
    print(f"形状函数和: {np.sum(N):.15f}")

    # 测试顶点处形状函数
    N_v0 = tri.shape_functions(tri.vertices[0])
    print(f"顶点0处形状函数: N1={N_v0[0]:.6f}, N2={N_v0[1]:.6f}, N3={N_v0[2]:.6f}")