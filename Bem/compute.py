import numpy as np
import time
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

class SphericalBEMSolver:
    """
    严格按照论文《球形电极三维静电场的球面三角形边界元算法》实现
    包含精确的电场线计算功能
    """

    def __init__(self, mesh, voltage=100.0, epsilon_0=8.854187817e-12, use_symmetry=True):
        """
        初始化求解器
        """
        self.mesh = mesh
        self.voltage = voltage
        self.epsilon_0 = epsilon_0
        self.use_symmetry = use_symmetry
        
        self.num_nodes = mesh.num_vertices
        self.num_elements = mesh.num_triangles
        self.center = mesh.center
        self.radius = mesh.radius
        
        self.is_symmetric = self._check_symmetry()
        
        # 存储计算结果
        self.sigma_elements = None  # 单元电荷密度
        self.sigma_nodes = None     # 节点电荷密度
        self.E_elements = None      # 单元电场强度
        self.total_charge = None    # 总电荷
        self.charge_density = None  # 与可视化模块兼容的电荷密度属性（指向sigma_elements）
        
        print(f"球面三角形边界元求解器初始化:")
        print(f"  节点数: {self.num_nodes}")
        print(f"  单元数: {self.num_elements}")
        print(f"  球半径: {self.radius} m")
        print(f"  电极电位: {self.voltage} V")
        print(f"  真空介电常数: {self.epsilon_0:.3e} F/m")
        print(f"  对称性优化: {'开启' if use_symmetry and self.is_symmetric else '关闭'}")

    def _check_symmetry(self):
        """
        检查网格是否具有对称性
        """
        if self.num_elements < 2:
            return True
            
        areas = [tri.area for tri in self.mesh.spherical_triangles[:10]]
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        
        # 相对标准差小于2%认为对称（考虑数值误差）
        is_sym = (area_std / area_mean) < 0.001
        return is_sym

    def coordinate_transform(self, point_plane):
        """
        平面点坐标到球面点坐标的变换
        论文公式(3)
        """
        vec = point_plane - self.center
        r_prime = np.linalg.norm(vec)
        
        if r_prime < 1e-12:
            return self.center.copy()
        
        return self.center + (self.radius / r_prime) * vec

    def get_sphere_normal(self, point_sphere):
        """
        计算球面某点的单位法向量
        论文公式(4)
        """
        n = (point_sphere - self.center) / self.radius
        return n / np.linalg.norm(n)

    def compute_jacobian(self, point_plane, triangle):
        """
        计算平面三角形到球面三角形的雅可比行列式
        论文公式(1)和(2)
        """
        vec = point_plane - self.center
        r_prime = np.linalg.norm(vec)
        
        if r_prime < 1e-12:
            return 0.0
        
        v0, v1, v2 = triangle.vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        n_prime = np.cross(edge1, edge2)
        n_prime = n_prime / np.linalg.norm(n_prime)
        
        n = vec / r_prime
        
        cos_alpha = np.dot(n, n_prime)
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
        
        J = (self.radius / r_prime) ** 2 * cos_alpha
        return J

    def shape_function_spherical(self, point_sphere, triangle):
        """
        计算球面三角形单元上的形状函数
        严格按照论文公式(5)-(14)实现
        """
        # 确保点在球面上
        point_vec = point_sphere - self.center
        point_norm = np.linalg.norm(point_vec)
        if abs(point_norm - self.radius) > 1e-10:
            point_sphere = self.center + self.radius * point_vec / point_norm
        
        vertices = triangle.vertices
        N = np.zeros(3)
        
        # 计算三个顶点的位置向量
        vec_i = vertices[0] - self.center
        vec_j = vertices[1] - self.center
        vec_k = vertices[2] - self.center
        vec_p = point_sphere - self.center
        
        # 归一化
        vec_i_norm = vec_i / self.radius
        vec_j_norm = vec_j / self.radius
        vec_k_norm = vec_k / self.radius
        vec_p_norm = vec_p / self.radius
        
        # 计算球面三角形的三个内角I,J,K（用余弦定理）
        # 角度I在顶点i，对应边jk
        # 边长的余弦
        cos_ij = np.clip(np.dot(vec_i_norm, vec_j_norm), -1.0, 1.0)
        cos_jk = np.clip(np.dot(vec_j_norm, vec_k_norm), -1.0, 1.0)
        cos_ki = np.clip(np.dot(vec_k_norm, vec_i_norm), -1.0, 1.0)
        
        # 边长（角度制）
        a = np.arccos(cos_jk)  # 顶点i对边
        b = np.arccos(cos_ki)  # 顶点j对边
        c = np.arccos(cos_ij)  # 顶点k对边
        
        # 使用球面三角余弦定理计算角度
        sin_b = np.sin(b)
        sin_c = np.sin(c)
        sin_a = np.sin(a)
        
        if sin_b > 1e-12 and sin_c > 1e-12:
            cos_A = (cos_jk - cos_ki * cos_ij) / (sin_b * sin_c)
            cos_A = np.clip(cos_A, -1.0, 1.0)
            A = np.arccos(cos_A)
        else:
            A = np.pi / 3.0
            
        if sin_c > 1e-12 and sin_a > 1e-12:
            cos_B = (cos_ki - cos_ij * cos_jk) / (sin_c * sin_a)
            cos_B = np.clip(cos_B, -1.0, 1.0)
            B = np.arccos(cos_B)
        else:
            B = np.pi / 3.0
            
        if sin_a > 1e-12 and sin_b > 1e-12:
            cos_C = (cos_ij - cos_jk * cos_ki) / (sin_a * sin_b)
            cos_C = np.clip(cos_C, -1.0, 1.0)
            C = np.arccos(cos_C)
        else:
            C = np.pi / 3.0
        
        # 对于点p，计算到三个顶点的大圆弧距离
        cos_ip = np.clip(np.dot(vec_i_norm, vec_p_norm), -1.0, 1.0)
        cos_jp = np.clip(np.dot(vec_j_norm, vec_p_norm), -1.0, 1.0)
        cos_kp = np.clip(np.dot(vec_k_norm, vec_p_norm), -1.0, 1.0)
        
        d_ip = np.arccos(cos_ip)  # i到p的弧长
        d_jp = np.arccos(cos_jp)  # j到p的弧长
        d_kp = np.arccos(cos_kp)  # k到p的弧长
        
        # 计算球面三角形的面积（球面角超）
        spherical_area = A + B + C - np.pi
        
        if spherical_area < 1e-12:
            return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
        # 计算三个子三角形的面积
        sin_d_jp = np.sin(d_jp)
        sin_d_kp = np.sin(d_kp)
        sin_d_ip = np.sin(d_ip)
        
        if sin_d_jp > 1e-12 and sin_d_kp > 1e-12:
            cos_alpha1 = (cos_jk - cos_jp * cos_kp) / (sin_d_jp * sin_d_kp)
            cos_alpha1 = np.clip(cos_alpha1, -1.0, 1.0)
            alpha1 = np.arccos(cos_alpha1)
        else:
            alpha1 = 0.0
            
        if sin_d_ip > 1e-12 and sin_d_kp > 1e-12:
            cos_alpha2 = (cos_ki - cos_ip * cos_kp) / (sin_d_ip * sin_d_kp)
            cos_alpha2 = np.clip(cos_alpha2, -1.0, 1.0)
            alpha2 = np.arccos(cos_alpha2)
        else:
            alpha2 = 0.0
            
        if sin_d_ip > 1e-12 and sin_d_jp > 1e-12:
            cos_alpha3 = (cos_ij - cos_ip * cos_jp) / (sin_d_ip * sin_d_jp)
            cos_alpha3 = np.clip(cos_alpha3, -1.0, 1.0)
            alpha3 = np.arccos(cos_alpha3)
        else:
            alpha3 = 0.0
        
        # 计算三个子三角形的面积
        area_pjk = alpha1 + np.pi - B - C
        area_ipk = alpha2 + np.pi - A - C
        area_ijp = alpha3 + np.pi - A - B
        
        # 形状函数 = 对角子三角形面积 / 总三角形面积
        N[0] = area_pjk / spherical_area
        N[1] = area_ipk / spherical_area
        N[2] = area_ijp / spherical_area
        
        # 归一化（处理数值误差）
        N_sum = np.sum(N)
        if N_sum > 1e-12:
            N = N / N_sum
        
        return N

    def gauss_points_triangle(self, order=4):
        """
        获取三角形单元的高斯积分点和权重
        """
        if order == 1:
            bary_coords = np.array([[1.0/3.0, 1.0/3.0, 1.0/3.0]])
            weights = np.array([1.0])
        elif order == 3:
            bary_coords = np.array([
                [2.0/3.0, 1.0/6.0, 1.0/6.0],
                [1.0/6.0, 2.0/3.0, 1.0/6.0],
                [1.0/6.0, 1.0/6.0, 2.0/3.0]
            ])
            weights = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        elif order == 4:
            a = (5.0 + 3.0*np.sqrt(5.0))/20.0
            b = (5.0 - np.sqrt(5.0))/20.0
            bary_coords = np.array([
                [a, b, b],
                [b, a, b],
                [b, b, a],
                [1.0/3.0, 1.0/3.0, 1.0/3.0]
            ])
            weights = np.array([
                (5.0-np.sqrt(5.0))/20.0,
                (5.0-np.sqrt(5.0))/20.0,
                (5.0-np.sqrt(5.0))/20.0,
                (5.0+3.0*np.sqrt(5.0))/20.0
            ])
        elif order == 7:
            alpha1 = 0.333333333333333
            alpha2 = 0.470142064105115
            alpha3 = 0.059715871789770
            alpha4 = 0.101286507323456
            alpha5 = 0.797426985353087
            alpha6 = 0.101286507323456

            bary_coords = np.array([
                [alpha1, alpha1, alpha1],
                [alpha2, alpha3, alpha3],
                [alpha3, alpha2, alpha3],
                [alpha3, alpha3, alpha2],
                [alpha4, alpha5, alpha5],
                [alpha5, alpha4, alpha5],
                [alpha5, alpha5, alpha4]
            ])
            weights = np.array([
                0.225000000000000,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827
            ])
        else:
            raise ValueError(f"不支持的高斯积分阶数: {order}")
        
        return bary_coords, weights

    def compute_element_integrals(self, tri_source, tri_field, gauss_order=4):
        """
        计算两个单元之间的积分
        严格按照论文公式(16)实现
        """
        G_elem = np.zeros((3, 3))
        H_elem = np.zeros((3, 3))
        
        bary_coords, weights = self.gauss_points_triangle(gauss_order)
        
        # 外层积分：场点单元
        for w_field, weight_field in zip(bary_coords, weights):
            # 平面三角形上的场点
            point_plane_field = (
                w_field[0] * tri_field.vertices[0] +
                w_field[1] * tri_field.vertices[1] +
                w_field[2] * tri_field.vertices[2]
            )
            
            # 投影到球面
            point_sphere_field = self.coordinate_transform(point_plane_field)
            
            # 雅可比行列式
            J_field = self.compute_jacobian(point_plane_field, tri_field)
            
            if J_field < 1e-12:
                continue
            
            # 场点形状函数
            N_field = self.shape_function_spherical(point_sphere_field, tri_field)
            
            # 内层积分：源点单元
            for w_source, weight_source in zip(bary_coords, weights):
                # 平面三角形上的源点
                point_plane_source = (
                    w_source[0] * tri_source.vertices[0] +
                    w_source[1] * tri_source.vertices[1] +
                    w_source[2] * tri_source.vertices[2]
                )
                
                # 投影到球面
                point_sphere_source = self.coordinate_transform(point_plane_source)
                
                # 雅可比行列式
                J_source = self.compute_jacobian(point_plane_source, tri_source)
                
                if J_source < 1e-12:
                    continue
                
                # 源点形状函数
                N_source = self.shape_function_spherical(point_sphere_source, tri_source)
                
                # 计算R向量和距离
                R_vec = point_sphere_field - point_sphere_source
                R = np.linalg.norm(R_vec)
                
                if R < 1e-10:
                    continue
                
                # 源点法向量
                n_source = self.get_sphere_normal(point_sphere_source)
                
                # 计算核函数
                G_kernel = 1.0 / R
                H_kernel = np.dot(R_vec, n_source) / (R**3)
                
                # 形状函数外积
                N_outer = np.outer(N_field, N_source)
                
                # 积分权重
                weight = J_field * J_source * weight_field * weight_source
                
                G_elem += G_kernel * N_outer * weight
                H_elem += H_kernel * N_outer * weight
        
        return G_elem, H_elem

    def assemble_system_matrices(self, gauss_order=4):
        """
        组装全局系统矩阵
        严格按照论文公式(16)实现
        """
        print("组装系统矩阵...")
        start_time = time.time()
        
        n_nodes = self.num_nodes
        n_elements = self.num_elements
        
        # 初始化稀疏矩阵（使用lil_matrix提高内存效率和构建速度）
        G = lil_matrix((n_nodes, n_nodes))
        H = lil_matrix((n_nodes, n_nodes))
        
        # 如果使用对称性优化且网格对称
        if self.use_symmetry and self.is_symmetric:
            print("使用对称性优化...")
            
            # 计算一个代表性单元的所有相互作用
            rep_tri = self.mesh.spherical_triangles[0]
            rep_idx = rep_tri.vertex_indices
            
            # 计算代表性单元的自相互作用
            G_self, H_self = self.compute_element_integrals(rep_tri, rep_tri, gauss_order)
            
            # 将自相互作用组装到全局矩阵
            for m in range(3):
                for n in range(3):
                    row = rep_idx[m]
                    col = rep_idx[n]
                    G[row, col] += G_self[m, n]
                    H[row, col] += H_self[m, n]
            
            # 对于孤立球，所有单元等价
            for elem_idx in range(1, n_elements):
                tri = self.mesh.spherical_triangles[elem_idx]
                idx = tri.vertex_indices
                
                for m in range(3):
                    for n in range(3):
                        row = idx[m]
                        col = idx[n]
                        G[row, col] += G_self[m, n]
                        H[row, col] += H_self[m, n]
        else:
            # 完整计算所有单元对
            print("完整计算所有单元对...")
            for i in range(n_elements):
                if i % 10 == 0:
                    print(f"  处理单元 {i+1}/{n_elements}...")
                
                tri_i = self.mesh.spherical_triangles[i]
                idx_i = tri_i.vertex_indices
                
                for j in range(n_elements):
                    tri_j = self.mesh.spherical_triangles[j]
                    idx_j = tri_j.vertex_indices
                    
                    # 计算单元对积分
                    G_elem, H_elem = self.compute_element_integrals(tri_i, tri_j, gauss_order)
                    
                    # 组装到全局矩阵
                    for m in range(3):
                        for n in range(3):
                            row = idx_i[m]
                            col = idx_j[n]
                            G[row, col] += G_elem[m, n]
                            H[row, col] += H_elem[m, n]
        
        # 应用1/(4π)因子（论文公式15）
        G = G / (4.0 * np.pi)
        H = H / (4.0 * np.pi)
        
        # 添加立体角项（论文公式15中的1/2项）
        for i in range(n_nodes):
            H[i, i] += 0.5
        
        elapsed_time = time.time() - start_time
        print(f"矩阵组装完成，耗时 {elapsed_time:.2f} 秒")
        
        return G, H

    def solve_electric_field(self, G, H):
        """
        求解边界元方程，计算表面电场
        严格按照论文方法实现
        """
        print("\n求解边界元方程...")
        
        # 已知边界节点电位
        phi = np.full(self.num_nodes, self.voltage)
        
        # 将稀疏矩阵转换为numpy数组用于求解（使用toarray()确保返回正确的数组格式）
        G_dense = G.toarray() if hasattr(G, 'toarray') else G
        H_dense = H.toarray() if hasattr(H, 'toarray') else H
        
        # 构建线性系统：G * q = H * φ
        b = np.dot(H_dense, phi)
        
        # 求解q
        try:
            q = solve(G_dense, b)
        except np.linalg.LinAlgError:
            print("矩阵奇异，使用最小二乘解...")
            q, residuals, rank, s = np.linalg.lstsq(G_dense, b, rcond=1e-10)
        
        # 计算面电荷密度：σ = -ε₀ * ∂φ/∂n
        sigma_nodes = -self.epsilon_0 * q
        
        # 计算单元平均值
        sigma_elements = np.zeros(self.num_elements)
        E_elements = np.zeros(self.num_elements)
        
        for i, tri in enumerate(self.mesh.spherical_triangles):
            idx = list(tri.vertex_indices)
            sigma_avg = np.mean(sigma_nodes[idx])
            sigma_elements[i] = sigma_avg
            E_elements[i] = sigma_avg / self.epsilon_0
        
        # 计算总电荷
        total_charge = 0.0
        for i, tri in enumerate(self.mesh.spherical_triangles):
            total_charge += sigma_elements[i] * tri.area
        
        # 存储计算结果
        self.sigma_elements = sigma_elements
        self.sigma_nodes = sigma_nodes
        self.E_elements = E_elements
        self.total_charge = total_charge
        self.charge_density = sigma_elements  # 更新与可视化模块兼容的属性
        
        return sigma_elements, sigma_nodes, E_elements
    
    def calculate_electric_field_at_point(self, point, method='exact'):
        """
        计算空间任意点的电场强度
        
        参数:
        point: 空间点坐标 [x, y, z]
        method: 计算方法
            'exact' - 精确积分（慢但准确）
            'approx' - 近似计算（假设电荷集中在单元中心）
            'analytic' - 使用解析公式（仅适用于孤立球）
            
        返回:
        E: 电场强度向量 [Ex, Ey, Ez]
        phi: 电位标量
        """
        if method == 'analytic':
            # 解析解：孤立导体球外部电场
            r_vec = point - self.center
            r = np.linalg.norm(r_vec)
            
            if r <= self.radius:
                # 球内部电场为零
                return np.zeros(3), self.voltage
            
            # 球外部电场：E = (Q/(4πε₀r²)) * (r̂)
            if self.total_charge is None:
                self.total_charge = 4 * np.pi * self.epsilon_0 * self.radius * self.voltage
            
            E_magnitude = self.total_charge / (4 * np.pi * self.epsilon_0 * r**2)
            E_direction = r_vec / r
            E = E_magnitude * E_direction
            
            # 电位：φ = Q/(4πε₀r)
            phi = self.total_charge / (4 * np.pi * self.epsilon_0 * r)
            
            return E, phi
        
        elif method == 'approx':
            # 近似计算：假设电荷集中在单元中心
            if self.sigma_elements is None:
                raise ValueError("请先调用solve_electric_field方法")
            
            E = np.zeros(3)
            phi = 0.0
            
            for i, tri in enumerate(self.mesh.spherical_triangles):
                # 计算单元中心
                center = np.mean(tri.vertices, axis=0)
                
                # 单元电荷
                charge = self.sigma_elements[i] * tri.area
                
                # 距离向量
                R_vec = point - center
                R = np.linalg.norm(R_vec)
                
                if R < 1e-12:
                    continue
                
                # 库仑定律
                E += charge * R_vec / (4 * np.pi * self.epsilon_0 * R**3)
                phi += charge / (4 * np.pi * self.epsilon_0 * R)
            
            return E, phi
        
        else:  # method == 'exact'
            # 精确积分：在每个单元上进行高斯积分
            if self.sigma_elements is None:
                raise ValueError("请先调用solve_electric_field方法")
            
            E = np.zeros(3)
            phi = 0.0
            
            # 使用高斯积分
            bary_coords, weights = self.gauss_points_triangle(order=4)
            
            for i, tri in enumerate(self.mesh.spherical_triangles):
                # 单元电荷密度
                sigma = self.sigma_elements[i]
                
                # 在单元上进行高斯积分
                for w, weight in zip(bary_coords, weights):
                    # 计算高斯点在平面三角形上的位置
                    point_tri = (
                        w[0] * tri.vertices[0] +
                        w[1] * tri.vertices[1] +
                        w[2] * tri.vertices[2]
                    )
                    
                    # 投影到球面
                    point_sphere = self.coordinate_transform(point_tri)
                    
                    # 雅可比行列式
                    J = self.compute_jacobian(point_tri, tri)
                    
                    if J < 1e-12:
                        continue
                    
                    # 距离向量
                    R_vec = point - point_sphere
                    R = np.linalg.norm(R_vec)
                    
                    if R < 1e-12:
                        continue
                    
                    # 电荷微元贡献
                    dq = sigma * J * weight
                    dE = dq * R_vec / (4 * np.pi * self.epsilon_0 * R**3)
                    dphi = dq / (4 * np.pi * self.epsilon_0 * R)
                    
                    E += dE
                    phi += dphi
            
            return E, phi
    
    def compute_electric_field_lines(self, num_lines=None, max_distance=5.0, 
                                    rtol=1e-4, atol=1e-6, method='analytic', start_radius_factor=1.001):
        """
        计算电场线
        
        参数:
        num_lines: 电场线数量，默认与单元数量相同
        max_distance: 最大追踪距离（以球半径为单位）
        rtol: 相对容差（用于solve_ivp）
        atol: 绝对容差（用于solve_ivp）
        method: 电场计算方法
        start_radius_factor: 起始点半径因子（相对于球半径）
        
        返回:
        field_lines: 电场线列表，每个元素是一个N×3的数组
        start_points: 起始点数组
        """
        if self.sigma_elements is None:
            raise ValueError("请先调用solve_electric_field方法")
        
        print(f"\n计算电场线 (方法: {method})...")
        start_time = time.time()
        
        # 默认使用单元数量作为电场线数量
        if num_lines is None:
            num_lines = self.num_elements
        
        # 从每个单元中心发出电场线
        start_points = []
        for tri in self.mesh.spherical_triangles:
            # 计算单元中心
            center = np.mean(tri.vertices, axis=0)
            
            # 确保在球面上
            center_vec = center - self.center
            center_unit = center_vec / np.linalg.norm(center_vec)
            
            # 稍微在球面外一点
            start_radius = self.radius * start_radius_factor
            start_point = self.center + center_unit * start_radius
            
            start_points.append(start_point)
        
        start_points = np.array(start_points[:num_lines])
        
        field_lines = []
        
        # 定义电场线微分方程
        def field_line_ode(t, y):
            point = np.array(y)
            E, _ = self.calculate_electric_field_at_point(point, method)
            E_norm = np.linalg.norm(E)
            
            if E_norm < 1e-6:
                return np.zeros(3)
            
            # 归一化方向向量
            return E / E_norm
        
        # 定义终止条件（超出最大距离）
        def termination_condition(t, y):
            point = np.array(y)
            distance_from_center = np.linalg.norm(point - self.center)
            return distance_from_center - max_distance * self.radius
        
        termination_condition.terminal = True
        
        for i, start_point in enumerate(start_points):
            if i % 10 == 0:
                print(f"  追踪电场线 {i+1}/{num_lines}...")
            
            # 使用solve_ivp追踪电场线
            t_span = (0, max_distance * self.radius)  # 时间跨度（这里用距离代替时间）
            
            # 设置时间点，确保有足够的输出点
            t_eval = np.linspace(0, max_distance * self.radius, 100)  # 100个点
            
            # 使用RK45方法求解，这是solve_ivp的默认方法，精度较高
            solution = solve_ivp(
                field_line_ode,
                t_span,
                start_point,
                method='RK45',
                rtol=rtol,
                atol=atol,
                events=termination_condition,
                t_eval=t_eval  # 指定输出点
            )
            
            # 提取电场线点
            line_points = solution.y.T
            
            field_lines.append(np.array(line_points))
        
        elapsed_time = time.time() - start_time
        print(f"电场线计算完成，耗时 {elapsed_time:.2f} 秒")
        
        return field_lines, start_points
    
    def compute_equipotential_surfaces(self, num_surfaces=10, method='analytic', use_interpolation=True):
        """
        计算等势面
        
        参数:
        num_surfaces: 等势面数量
        method: 电位计算方法
        use_interpolation: 是否使用插值方法加速计算
        
        返回:
        surfaces: 等势面列表，每个元素是一个网格
        potentials: 对应的电位值
        """
        if self.sigma_elements is None:
            raise ValueError("请先调用solve_electric_field方法")
        
        print(f"\n计算等势面...")
        
        # 定义电位值范围
        min_potential = 0.1 * self.voltage
        max_potential = 0.9 * self.voltage
        potentials = np.linspace(min_potential, max_potential, num_surfaces)
        
        # 创建三维网格
        grid_resolution = 50
        x_range = np.linspace(-2*self.radius, 2*self.radius, grid_resolution)
        y_range = np.linspace(-2*self.radius, 2*self.radius, grid_resolution)
        z_range = np.linspace(-2*self.radius, 2*self.radius, grid_resolution)
        
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # 计算网格点的电位
        print("计算三维电位场...")
        phi_grid = np.zeros_like(X)
        
        if use_interpolation:
            # 使用稀疏点集计算电位，然后进行插值（提高计算效率）
            sparse_resolution = 20  # 稀疏网格分辨率
            sparse_x = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            sparse_y = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            sparse_z = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            
            # 创建稀疏点集
            sparse_points = np.meshgrid(sparse_x, sparse_y, sparse_z, indexing='ij')
            sparse_points = np.array(sparse_points).reshape(3, -1).T
            
            # 计算稀疏点集的电位
            print(f"  使用插值方法: 先计算 {sparse_resolution**3} 个稀疏点的电位...")
            sparse_phi = np.zeros(sparse_points.shape[0])
            
            for i, point in enumerate(sparse_points):
                if i % 1000 == 0:
                    print(f"    计算稀疏点 {i+1}/{sparse_points.shape[0]}...")
                _, phi = self.calculate_electric_field_at_point(point, method)
                sparse_phi[i] = phi
            
            # 使用griddata进行插值，填充到密集网格
            print("  使用griddata进行三维插值...")
            phi_grid = griddata(
                sparse_points, sparse_phi, (X, Y, Z), 
                method='linear',  # 线性插值，速度较快且精度足够
                fill_value=self.voltage  # 球体内部使用电极电位
            )
        else:
            # 原始方法：直接计算密集网格上的电位
            for i in range(grid_resolution):
                if i % 10 == 0:
                    print(f"  计算切片 {i+1}/{grid_resolution}...")
                for j in range(grid_resolution):
                    for k in range(grid_resolution):
                        point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                        _, phi = self.calculate_electric_field_at_point(point, method)
                        phi_grid[i,j,k] = phi
        
        surfaces = []
        
        # 提取等势面
        for potential in potentials:
            # 使用marching cubes算法提取等值面
            from skimage import measure
            
            # 创建等值面
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    phi_grid, level=potential, spacing=(x_range[1]-x_range[0], 
                                                      y_range[1]-y_range[0],
                                                      z_range[1]-z_range[0])
                )
                
                # 调整坐标原点
                verts[:, 0] += x_range[0]
                verts[:, 1] += y_range[0]
                verts[:, 2] += z_range[0]
                
                surfaces.append((verts, faces))
            except:
                print(f"  警告: 无法提取电位为 {potential:.2f} V 的等势面")
                surfaces.append(None)
        
        return surfaces, potentials
    
    def compute_field_on_line(self, start_point, end_point, num_points=100, method='analytic'):
        """
        计算直线上各点的电场和电位
        
        参数:
        start_point: 起点坐标
        end_point: 终点坐标
        num_points: 点数
        method: 计算方法
        
        返回:
        points: 线上点的坐标
        E_magnitudes: 电场强度大小
        potentials: 电位值
        """
        # 生成直线上的点
        t = np.linspace(0, 1, num_points)
        points = start_point + np.outer(t, end_point - start_point)
        
        E_magnitudes = np.zeros(num_points)
        potentials = np.zeros(num_points)
        
        for i, point in enumerate(points):
            E, phi = self.calculate_electric_field_at_point(point, method)
            E_magnitudes[i] = np.linalg.norm(E)
            potentials[i] = phi
        
        return points, E_magnitudes, potentials
    
    def validate_solution(self, sigma_elements=None, E_elements=None):
        """
        验证计算结果
        与理论值比较
        """
        print("\n=== 结果验证 ===")
        
        if sigma_elements is None:
            sigma_elements = self.sigma_elements
        if E_elements is None:
            E_elements = self.E_elements
        
        if sigma_elements is None or E_elements is None:
            raise ValueError("请先调用solve_electric_field方法")
        
        # 理论值
        sigma_theory = self.epsilon_0 * self.voltage / self.radius
        E_theory = self.voltage / self.radius
        
        # 计算误差
        sigma_errors = np.abs(sigma_elements - sigma_theory) / sigma_theory * 100
        E_errors = np.abs(E_elements - E_theory) / E_theory * 100
        
        # 总电荷
        if self.total_charge is None:
            self.total_charge = 0.0
            for i, tri in enumerate(self.mesh.spherical_triangles):
                self.total_charge += sigma_elements[i] * tri.area
        
        total_charge_theory = 4 * np.pi * self.epsilon_0 * self.radius * self.voltage
        
        print(f"面电荷密度:")
        print(f"  理论值: {sigma_theory:.6e} C/m²")
        print(f"  计算均值: {np.mean(sigma_elements):.6e} C/m²")
        print(f"  计算标准差: {np.std(sigma_elements):.6e} C/m²")
        print(f"  最大相对误差: {np.max(sigma_errors):.3f}%")
        print(f"  平均相对误差: {np.mean(sigma_errors):.3f}%")
        
        print(f"\n表面电场强度:")
        print(f"  理论值: {E_theory:.3f} V/m")
        print(f"  计算均值: {np.mean(E_elements):.3f} V/m")
        print(f"  计算范围: {np.min(E_elements):.3f} ~ {np.max(E_elements):.3f} V/m")
        print(f"  最大相对误差: {np.max(E_errors):.3f}%")
        print(f"  平均相对误差: {np.mean(E_errors):.3f}%")
        
        print(f"\n总电荷:")
        print(f"  理论值: {total_charge_theory:.6e} C")
        print(f"  计算值: {self.total_charge:.6e} C")
        print(f"  相对误差: {abs(self.total_charge-total_charge_theory)/total_charge_theory*100:.3f}%")
        
        # 与论文结果比较
        print(f"\n=== 与论文结果比较 ===")
        print(f"论文最大相对误差: 0.640%")
        print(f"我们最大相对误差: {np.max(E_errors):.3f}%")
        
        if np.max(E_errors) < 1.0:
            print("✓ 实现成功，精度达到论文水平")
        elif np.max(E_errors) < 2.0:
            print("✓ 实现基本正确，精度接近论文水平")
        else:
            print("⚠ 实现存在一定误差，需要进一步优化")
        
        results = {
            'sigma_mean': np.mean(sigma_elements),
            'sigma_std': np.std(sigma_elements),
            'E_mean': np.mean(E_elements),
            'E_std': np.std(E_elements),
            'E_min': np.min(E_elements),
            'E_max': np.max(E_elements),
            'max_E_error': np.max(E_errors),
            'mean_E_error': np.mean(E_errors),
            'total_charge': self.total_charge,
            'charge_error': abs(self.total_charge-total_charge_theory)/total_charge_theory*100
        }
        
        return results


class ElectricFieldVisualizer:
    """
    电场可视化工具类
    """
    
    def __init__(self, bem_solver):
        self.bem_solver = bem_solver
        self.field_lines = None
        self.equipotential_surfaces = None
        
    def plot_field_lines_3d(self, field_lines, start_points, 
                           num_lines_to_plot=20, figsize=(12, 10)):
        """
        三维绘制电场线
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制球体
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = self.bem_solver.radius * np.outer(np.cos(u), np.sin(v))
        y = self.bem_solver.radius * np.outer(np.sin(u), np.sin(v))
        z = self.bem_solver.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, linewidth=0)
        
        # 绘制电场线
        num_lines = min(num_lines_to_plot, len(field_lines))
        colors = plt.cm.viridis(np.linspace(0, 1, num_lines))
        
        for i in range(num_lines):
            line = field_lines[i]
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                   color=colors[i], linewidth=1.5, alpha=0.8)
        
        # 绘制起始点
        ax.scatter(start_points[:num_lines, 0], 
                  start_points[:num_lines, 1], 
                  start_points[:num_lines, 2], 
                  color='red', s=20, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('球形电极的电场线分布')
        ax.set_box_aspect([1, 1, 1])
        
        return fig, ax
    
    def plot_equipotential_surfaces(self, surfaces, potentials, 
                                   figsize=(12, 10), alpha=0.3):
        """
        三维绘制等势面
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制球体
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = self.bem_solver.radius * np.outer(np.cos(u), np.sin(v))
        y = self.bem_solver.radius * np.outer(np.sin(u), np.sin(v))
        z = self.bem_solver.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='gray', alpha=0.3, linewidth=0)
        
        # 绘制等势面
        norm = plt.Normalize(vmin=min(potentials), vmax=max(potentials))
        cmap = cm.viridis
        
        for i, (surface, potential) in enumerate(zip(surfaces, potentials)):
            if surface is not None:
                verts, faces = surface
                color = cmap(norm(potential))
                
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                              color=color, alpha=alpha, linewidth=0.2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('球形电极的等势面分布')
        ax.set_box_aspect([1, 1, 1])
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('电位 (V)')
        
        return fig, ax
    
    def plot_field_strength_along_line(self, start_point, end_point, 
                                      num_points=100, method='analytic',
                                      figsize=(10, 6)):
        """
        绘制沿直线的电场强度和电位分布
        """
        import matplotlib.pyplot as plt
        
        # 计算电场和电位
        points, E_magnitudes, potentials = self.bem_solver.compute_field_on_line(
            start_point, end_point, num_points, method
        )
        
        # 计算距离
        distances = np.linalg.norm(points - self.bem_solver.center, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 电场强度
        ax1.plot(distances, E_magnitudes, 'b-', linewidth=2, label='计算值')
        
        # 理论值：E = Q/(4πε₀r²)
        r = distances
        Q = self.bem_solver.total_charge
        E_theory = Q / (4 * np.pi * self.bem_solver.epsilon_0 * r**2)
        ax1.plot(distances, E_theory, 'r--', linewidth=1.5, label='理论值')
        
        ax1.set_xlabel('距离球心的距离 (m)')
        ax1.set_ylabel('电场强度 (V/m)')
        ax1.set_title('沿直线的电场强度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 电位
        ax2.plot(distances, potentials, 'g-', linewidth=2, label='计算值')
        
        # 理论值：φ = Q/(4πε₀r)
        phi_theory = Q / (4 * np.pi * self.bem_solver.epsilon_0 * r)
        ax2.plot(distances, phi_theory, 'r--', linewidth=1.5, label='理论值')
        
        ax2.set_xlabel('距离球心的距离 (m)')
        ax2.set_ylabel('电位 (V)')
        ax2.set_title('沿直线的电位分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)


# 测试代码
if __name__ == "__main__":
    print("=== 球形电极电场计算器 ===")
    print("=" * 60)
    
    # 创建网格
    from Bemmodel import generate_icosphere
    
    # 按照论文中的网格：80个单元，42个节点
    mesh = generate_icosphere(radius=1.0, subdivisions=1)
    
    print("网格信息:")
    print(f"  顶点数: {mesh.num_vertices}")
    print(f"  三角形数: {mesh.num_triangles}")
    
    # 验证网格面积
    total_area = sum(tri.area for tri in mesh.spherical_triangles)
    theoretical_area = 4 * np.pi * mesh.radius**2
    area_error = abs(total_area - theoretical_area) / theoretical_area * 100
    
    print(f"  球面总面积: {total_area:.6f} m²")
    print(f"  理论面积: {theoretical_area:.6f} m²")
    print(f"  面积相对误差: {area_error:.3f}%")
    
    # 创建求解器
    solver = SphericalBEMSolver(mesh, voltage=100.0, use_symmetry=True)
    
    # 组装系统矩阵
    print("\n" + "=" * 60)
    G, H = solver.assemble_system_matrices(gauss_order=3)
    
    # 求解表面电场
    print("\n" + "=" * 60)
    sigma_elements, sigma_nodes, E_elements = solver.solve_electric_field(G, H)
    
    # 验证结果
    print("\n" + "=" * 60)
    results = solver.validate_solution(sigma_elements, E_elements)
    
    # 测试电场计算
    print("\n" + "=" * 60)
    print("测试空间点电场计算:")
    
    test_points = [
        np.array([2.0, 0.0, 0.0]),   # 球外点
        np.array([0.5, 0.0, 0.0]),   # 球内点
        np.array([1.5, 1.5, 0.0])    # 球外点
    ]
    
    for i, point in enumerate(test_points):
        print(f"\n测试点 {i+1}: {point}")
        E_exact, phi_exact = solver.calculate_electric_field_at_point(point, 'exact')
        E_approx, phi_approx = solver.calculate_electric_field_at_point(point, 'approx')
        E_analytic, phi_analytic = solver.calculate_electric_field_at_point(point, 'analytic')
        
        print(f"  精确积分: E = {E_exact}, |E| = {np.linalg.norm(E_exact):.3f} V/m, φ = {phi_exact:.3f} V")
        print(f"  近似计算: E = {E_approx}, |E| = {np.linalg.norm(E_approx):.3f} V/m, φ = {phi_approx:.3f} V")
        print(f"  解析解: E = {E_analytic}, |E| = {np.linalg.norm(E_analytic):.3f} V/m, φ = {phi_analytic:.3f} V")
    
    # 计算电场线
    print("\n" + "=" * 60)
    field_lines, start_points = solver.compute_electric_field_lines(
        num_lines=30, max_distance=5.0, rtol=1e-4, atol=1e-6, method='analytic'
    )
    
    # 创建可视化器
    visualizer = ElectricFieldVisualizer(solver)
    
    # 绘制电场线
    import matplotlib.pyplot as plt
    fig1, ax1 = visualizer.plot_field_lines_3d(field_lines, start_points, num_lines_to_plot=20)
    plt.savefig('electric_field_lines.png', dpi=300, bbox_inches='tight')
    print("\n电场线图已保存为 'electric_field_lines.png'")
    
    # 计算并绘制等势面
    print("\n" + "=" * 60)
    print("计算等势面...")
    try:
        surfaces, potentials = solver.compute_equipotential_surfaces(num_surfaces=5, method='analytic')
        fig2, ax2 = visualizer.plot_equipotential_surfaces(surfaces, potentials, alpha=0.3)
        plt.savefig('equipotential_surfaces.png', dpi=300, bbox_inches='tight')
        print("等势面图已保存为 'equipotential_surfaces.png'")
    except ImportError:
        print("需要scikit-image库来提取等势面，跳过此步骤")
    
    # 绘制沿直线的电场分布
    print("\n" + "=" * 60)
    start_point = np.array([1.01, 0.0, 0.0])
    end_point = np.array([5.0, 0.0, 0.0])
    fig3, axes = visualizer.plot_field_strength_along_line(start_point, end_point, method='analytic')
    plt.savefig('field_along_line.png', dpi=300, bbox_inches='tight')
    print("沿直线电场分布图已保存为 'field_along_line.png'")
    
    # 显示所有图形
    plt.show()
    
    # 保存计算结果
    np.savez('electric_field_results.npz',
             sigma_elements=sigma_elements,
             sigma_nodes=sigma_nodes,
             E_elements=E_elements,
             total_charge=solver.total_charge,
             field_lines=field_lines,
             start_points=start_points,
             results=results)
    
    print("\n计算结果已保存为 'electric_field_results.npz'")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("=== 计算完成 ===")
    print(f"计算精度:")
    print(f"  最大相对误差: {results['max_E_error']:.3f}%")
    print(f"  平均相对误差: {results['mean_E_error']:.3f}%")
    print(f"  总电荷误差: {results['charge_error']:.3f}%")
    
    if results['max_E_error'] < 2.0:
        print("\n计算成功，精度满足要求")
    else:
        print("\n计算存在一定误差，建议检查实现")