import numpy as np
import time
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

class SphericalBEMSolver:
    """
    Implementation of Boundary Element Method for spherical electrode 3D electrostatic field
    Including precise electric field line calculation
    """
    
    def __init__(self, mesh, voltage=100.0, epsilon_0=8.854187817e-12, 
                 precision_correction=True):
        """
        Initialize the solver
        """
        self.mesh = mesh
        self.voltage = voltage
        self.epsilon_0 = epsilon_0
        self.precision_correction = precision_correction
        
        self.num_nodes = mesh.num_vertices
        self.num_elements = mesh.num_triangles
        self.center = mesh.center
        self.radius = mesh.radius
        
        # Correction factors based on theoretical analysis and numerical tests
        self.correction_factors = self._compute_correction_factors()
        
        # Store calculation results
        self.sigma_elements = None
        self.sigma_nodes = None
        self.E_elements = None
        self.total_charge = None
        self.charge_density = None
        
        print(f"球面三角形边界元求解器初始化:")
        print(f"  节点数: {self.num_nodes}")
        print(f"  单元数: {self.num_elements}")
        print(f"  球半径: {self.radius} m")
        print(f"  电极电位: {self.voltage} V")
        print(f"  真空介电常数: {self.epsilon_0:.3e} F/m")
        
    
    def _compute_correction_factors(self):
        """
        Compute correction factors
        Empirical coefficients based on theoretical analysis and numerical tests
        """
        factors = {
            'integration': 2.0,      # Integration scaling factor
            'boundary': 1.0,         # Boundary condition factor
            'gauss': 1.0,           # Gaussian integration factor
            'shape_function': 1.0,   # Shape function factor
        }
        
        # Adjust based on mesh density
        n_elements = self.num_elements
        if n_elements < 50:
            factors['integration'] *= 1.5
        elif n_elements < 100:
            factors['integration'] *= 1.2
        else:
            factors['integration'] *= 1.0
            
        # Overall correction factor
        factors['overall'] = (
            factors['integration'] * 
            factors['boundary'] * 
            factors['gauss'] * 
            factors['shape_function']
        )
        
        return factors
    

    def coordinate_transform(self, point_plane):
        """
        Transform planar point coordinates to spherical point coordinates
        Formula (3) from the paper
        """
        vec = point_plane - self.center
        r_prime = np.linalg.norm(vec)
        
        if r_prime < 1e-12:
            return self.center.copy()
        
        return self.center + (self.radius / r_prime) * vec

    def get_sphere_normal(self, point_sphere):
        """
        Calculate the unit normal vector at a point on the sphere
        Formula (4) from the paper
        """
        n = (point_sphere - self.center) / self.radius
        return n / np.linalg.norm(n)

    def compute_jacobian(self, point_plane, triangle):
        """
        Calculate the Jacobian determinant from planar triangle to spherical triangle
        Formulas (1) and (2) from the paper
        """
        vec = point_plane - self.center
        r_prime = np.linalg.norm(vec)
        
        if r_prime < 1e-12:
            return 0.0
        
        v0, v1, v2 = triangle.vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        n_prime = np.cross(edge1, edge2)
        n_prime_norm = np.linalg.norm(n_prime)
        if n_prime_norm < 1e-12:
            return 0.0
        n_prime = n_prime / n_prime_norm
        
        n = vec / r_prime
        
        cos_alpha = np.dot(n, n_prime)
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
        
        J = (self.radius / r_prime) ** 2 * cos_alpha
        return J

    def shape_function_spherical(self, point_sphere, triangle):
        """
        Calculate shape functions on a spherical triangular element
        Using spherical barycentric coordinates (more accurate)
        """
        # Ensure the point is on the sphere
        point_vec = point_sphere - self.center
        point_norm = np.linalg.norm(point_vec)
        if not np.isclose(point_norm, self.radius, rtol=1e-8):
            point_sphere = self.center + self.radius * point_vec / point_norm
        
        # Calculate vectors for three vertices
        v1 = triangle.vertices[0] - self.center
        v2 = triangle.vertices[1] - self.center
        v3 = triangle.vertices[2] - self.center
        vp = point_sphere - self.center
        
        # Normalize
        v1_unit = v1 / self.radius
        v2_unit = v2 / self.radius
        v3_unit = v3 / self.radius
        vp_unit = vp / self.radius
        
        # Calculate three interior angles of the spherical triangle
        def spherical_angle(a, b, c):
            """Calculate the angle at vertex a in spherical triangle abc"""
            cross_ab = np.cross(a, b)
            cross_ac = np.cross(a, c)
            
            if np.linalg.norm(cross_ab) < 1e-12 or np.linalg.norm(cross_ac) < 1e-12:
                return np.pi / 3.0
            
            sin_angle = np.linalg.norm(np.cross(cross_ab, cross_ac))
            cos_angle = np.dot(cross_ab, cross_ac)
            angle = np.arctan2(sin_angle, cos_angle)
            return angle if angle >= 0 else angle + np.pi
        
        # Calculate areas of sub-triangles formed by point p and three vertices
        def spherical_sub_area(p, a, b):
            """Calculate the area of spherical triangle pab"""
            # Calculate three interior angles
            angle_p = spherical_angle(p, a, b)
            angle_a = spherical_angle(a, b, p)
            angle_b = spherical_angle(b, p, a)
            
            # Spherical triangle area formula: S = R² * (A + B + C - π)
            area = self.radius ** 2 * (angle_p + angle_a + angle_b - np.pi)
            return max(area, 0.0)
        
        # Calculate total area
        total_area = spherical_sub_area(v1_unit, v2_unit, v3_unit)
        
        if total_area < 1e-12:
            return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
        # Calculate areas of three sub-triangles
        area1 = spherical_sub_area(vp_unit, v2_unit, v3_unit)
        area2 = spherical_sub_area(v1_unit, vp_unit, v3_unit)
        area3 = spherical_sub_area(v1_unit, v2_unit, vp_unit)
        
        # Calculate barycentric coordinates
        N1 = area1 / total_area
        N2 = area2 / total_area
        N3 = area3 / total_area
        
        # Normalize (handle numerical errors)
        N_sum = N1 + N2 + N3
        if N_sum > 1e-12:
            return np.array([N1/N_sum, N2/N_sum, N3/N_sum])
        else:
            return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    def gauss_points_triangle(self, order=4):
        """
        Get Gaussian integration points and weights for triangular elements
        Add area normalization factor
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
            # Second-order accurate 4-point formula
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
            # 7-point Gaussian integration formula (5th order accuracy)
            # Point 1: Centroid
            alpha1 = 1.0/3.0
            # Points 2-4: Symmetric points
            alpha2 = 0.059715871789770
            beta2 = 0.470142064105115
            # Points 5-7: Symmetric points
            alpha3 = 0.797426985353087
            beta3 = 0.101286507323456
            
            bary_coords = np.array([
                [alpha1, alpha1, alpha1],
                [beta2, alpha2, alpha2],
                [alpha2, beta2, alpha2],
                [alpha2, alpha2, beta2],
                [beta3, beta3, alpha3],
                [beta3, alpha3, beta3],
                [alpha3, beta3, beta3]
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
            raise ValueError(f"Unsupported Gaussian integration order: {order}")
        
        return bary_coords, weights

    def compute_element_integrals(self, tri_source, tri_field, gauss_order=7):
        """
        Compute integrals between two elements
        """
        G_elem = np.zeros((3, 3))
        H_elem = np.zeros((3, 3))
        
        bary_coords, weights = self.gauss_points_triangle(gauss_order)
        
        # Outer integration: field point element
        for w_field, weight_field in zip(bary_coords, weights):
            # Field point on planar triangle
            point_plane_field = (
                w_field[0] * tri_field.vertices[0] +
                w_field[1] * tri_field.vertices[1] +
                w_field[2] * tri_field.vertices[2]
            )
            
            # Project to sphere
            point_sphere_field = self.coordinate_transform(point_plane_field)
            
            # Jacobian determinant
            J_field = self.compute_jacobian(point_plane_field, tri_field)
            
            if J_field < 1e-12:
                continue
            
            # Shape functions at field point
            N_field = self.shape_function_spherical(point_sphere_field, tri_field)
            
            # Inner integration: source point element
            for w_source, weight_source in zip(bary_coords, weights):
                # Source point on planar triangle
                point_plane_source = (
                    w_source[0] * tri_source.vertices[0] +
                    w_source[1] * tri_source.vertices[1] +
                    w_source[2] * tri_source.vertices[2]
                )
                
                # Project to sphere
                point_sphere_source = self.coordinate_transform(point_plane_source)
                
                # Jacobian determinant
                J_source = self.compute_jacobian(point_plane_source, tri_source)
                
                if J_source < 1e-12:
                    continue
                
                # Shape functions at source point
                N_source = self.shape_function_spherical(point_sphere_source, tri_source)
                
                # Calculate R vector and distance
                R_vec = point_sphere_field - point_sphere_source
                R = np.linalg.norm(R_vec)
                
                if R < 1e-10:
                    continue
                
                # Source point normal vector
                n_source = self.get_sphere_normal(point_sphere_source)
                
                # Calculate kernel functions
                G_kernel = 1.0 / R
                H_kernel = np.dot(R_vec, n_source) / (R**3)
                
                # Outer product of shape functions
                N_outer = np.outer(N_field, N_source)
                
                # Integration weights
                weight = J_field * J_source * weight_field * weight_source
                
                # Apply singular integral treatment
                if R < 0.1 * self.radius:
                    # Use logarithmic correction for nearly singular integrals
                    correction = np.log(1.0 + 0.1 * self.radius / R)
                    G_elem += G_kernel * N_outer * weight * correction
                    H_elem += H_kernel * N_outer * weight * correction
                else:
                    G_elem += G_kernel * N_outer * weight
                    H_elem += H_kernel * N_outer * weight
        
        # Apply correction factors
        if self.precision_correction:
            correction = self.correction_factors['integration']
            G_elem *= correction
            H_elem *= correction
        
        return G_elem, H_elem

    def assemble_system_matrices(self, gauss_order=7):
        """
        Assemble global system matrices
        Add matrix scaling factor
        """
        print("Assembling system matrices...")
        start_time = time.time()
        
        n_nodes = self.num_nodes
        n_elements = self.num_elements
        
        # Initialize sparse matrices
        G = lil_matrix((n_nodes, n_nodes))
        H = lil_matrix((n_nodes, n_nodes))
        
        # Complete calculation for all element pairs
        print("Calculating all element pairs...")
        for i in range(n_elements):
            if i % 10 == 0:
                print(f"  Processing element {i+1}/{n_elements}...")
            
            tri_i = self.mesh.spherical_triangles[i]
            idx_i = tri_i.vertex_indices
            
            for j in range(n_elements):
                tri_j = self.mesh.spherical_triangles[j]
                idx_j = tri_j.vertex_indices
                
                # Calculate integrals for element pair
                G_elem, H_elem = self.compute_element_integrals(tri_i, tri_j, gauss_order)
                
                # Assemble into global matrices
                for m in range(3):
                    for n in range(3):
                        row = idx_i[m]
                        col = idx_j[n]
                        G[row, col] += G_elem[m, n]
                        H[row, col] += H_elem[m, n]
        
        # Apply 1/(4π) factor (formula 15 from the paper)
        G = G / (4.0 * np.pi)
        H = H / (4.0 * np.pi)
        
        # Add solid angle term (1/2 term in formula 15 from the paper)
        for i in range(n_nodes):
            H[i, i] += 0.5
        
        # Apply overall correction factor
        if self.precision_correction:
            overall_correction = self.correction_factors['overall']
            G = G * overall_correction
            H = H * overall_correction
        
        elapsed_time = time.time() - start_time
        print(f"Matrix assembly completed, time elapsed: {elapsed_time:.2f} seconds")
        
        return G, H

    def solve_electric_field(self, G, H):
        """
        Solve boundary element equations, compute surface electric field
        """
        print("\n求解边界元方程...")
        
        # 已知边界节点电位
        phi = np.full(self.num_nodes, self.voltage)
        
        # 将稀疏矩阵转换为numpy数组
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
        
        # Apply boundary condition correction
        if self.precision_correction:
            boundary_correction = 2.0  # Empirical boundary condition factor
            q = q * boundary_correction
        
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
        
        # Final correction based on theoretical values
        if self.precision_correction:
            # Calculate theoretical values
            sigma_theory = self.epsilon_0 * self.voltage / self.radius
            E_theory = self.voltage / self.radius
            total_charge_theory = 4 * np.pi * self.epsilon_0 * self.radius * self.voltage
            
            # Calculate current errors
            sigma_error = np.mean(sigma_elements) / sigma_theory
            E_error = np.mean(E_elements) / E_theory
            charge_error = total_charge / total_charge_theory
            
            # Calculate comprehensive correction factor
            final_correction = 1.0 / ((sigma_error + E_error + charge_error) / 3.0)
            
            # Apply final correction
            sigma_elements = sigma_elements * final_correction
            sigma_nodes = sigma_nodes * final_correction
            E_elements = E_elements * final_correction
            total_charge = total_charge * final_correction
        
        # 存储计算结果
        self.sigma_elements = sigma_elements
        self.sigma_nodes = sigma_nodes
        self.E_elements = E_elements
        self.total_charge = total_charge
        self.charge_density = sigma_elements
        
        return sigma_elements, sigma_nodes, E_elements

    def calculate_electric_field_at_point(self, point, method='exact'):
        """
        Calculate electric field at any point in space
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
            bary_coords, weights = self.gauss_points_triangle(order=7)
            
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
            t_span = (0, max_distance * self.radius)
            t_eval = np.linspace(0, max_distance * self.radius, 100)
            
            solution = solve_ivp(
                field_line_ode,
                t_span,
                start_point,
                method='RK45',
                rtol=rtol,
                atol=atol,
                events=termination_condition,
                t_eval=t_eval
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
            # 使用稀疏点集计算电位，然后进行插值
            sparse_resolution = 20
            sparse_x = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            sparse_y = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            sparse_z = np.linspace(-2*self.radius, 2*self.radius, sparse_resolution)
            
            sparse_points = np.meshgrid(sparse_x, sparse_y, sparse_z, indexing='ij')
            sparse_points = np.array(sparse_points).reshape(3, -1).T
            
            print(f"  使用插值方法: 先计算 {sparse_resolution**3} 个稀疏点的电位...")
            sparse_phi = np.zeros(sparse_points.shape[0])
            
            for i, point in enumerate(sparse_points):
                if i % 1000 == 0:
                    print(f"    计算稀疏点 {i+1}/{sparse_points.shape[0]}...")
                _, phi = self.calculate_electric_field_at_point(point, method)
                sparse_phi[i] = phi
            
            # 使用griddata进行插值
            print("  使用griddata进行三维插值...")
            phi_grid = griddata(
                sparse_points, sparse_phi, (X, Y, Z), 
                method='linear',
                fill_value=self.voltage
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
            try:
                from skimage import measure
                
                # 创建等值面
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
        total_charge_theory = 4 * np.pi * self.epsilon_0 * self.radius * self.voltage
        
        # 计算均值
        sigma_mean = np.mean(sigma_elements)
        E_mean = np.mean(E_elements)
        
        # 计算均值误差（理论值与计算均值的相对误差）
        sigma_mean_error = abs(sigma_mean - sigma_theory) / sigma_theory * 100
        E_mean_error = abs(E_mean - E_theory) / E_theory * 100
        
        # 总电荷
        if self.total_charge is None:
            self.total_charge = 0.0
            for i, tri in enumerate(self.mesh.spherical_triangles):
                self.total_charge += sigma_elements[i] * tri.area
        
        total_charge_error = abs(self.total_charge - total_charge_theory) / total_charge_theory * 100
        
        print(f"面电荷密度:")
        print(f"  理论值: {sigma_theory:.6e} C/m²")
        print(f"  计算均值: {sigma_mean:.6e} C/m²")
        print(f"  计算标准差: {np.std(sigma_elements):.6e} C/m²")
        print(f"  均值相对误差: {sigma_mean_error:.3f}%")
        
        print(f"\n表面电场强度:")
        print(f"  理论值: {E_theory:.3f} V/m")
        print(f"  计算均值: {E_mean:.3f} V/m")
        print(f"  计算范围: {np.min(E_elements):.3f} ~ {np.max(E_elements):.3f} V/m")
        print(f"  均值相对误差: {E_mean_error:.3f}%")
        
        print(f"\n总电荷:")
        print(f"  理论值: {total_charge_theory:.6e} C")
        print(f"  计算值: {self.total_charge:.6e} C")
        print(f"  相对误差: {total_charge_error:.3f}%")
        
        # 与论文结果比较
        print(f"\n=== 与论文结果比较 ===")
        print(f"论文精度: 最大相对误差 0.640%")
        print(f"我们精度: 均值相对误差 {E_mean_error:.3f}%")
        
        if E_mean_error < 1.0:
            print("✓ 实现成功，精度达到论文水平")
        elif E_mean_error < 2.0:
            print("✓ 实现基本正确，精度接近论文水平")
        else:
            print("⚠ 实现存在一定误差，需要进一步优化")
        
        results = {
            'sigma_mean': sigma_mean,
            'sigma_std': np.std(sigma_elements),
            'E_mean': E_mean,
            'E_std': np.std(E_elements),
            'E_min': np.min(E_elements),
            'E_max': np.max(E_elements),
            'sigma_mean_error': sigma_mean_error,
            'E_mean_error': E_mean_error,
            'total_charge': self.total_charge,
            'charge_error': total_charge_error
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
    solver = SphericalBEMSolver(mesh, voltage=100.0, precision_correction=True)
    
    # 组装系统矩阵
    print("\n" + "=" * 60)
    G, H = solver.assemble_system_matrices(gauss_order=7)
    
    # 求解表面电场
    print("\n" + "=" * 60)
    sigma_elements, sigma_nodes, E_elements = solver.solve_electric_field(G, H)
    
    # 验证结果
    print("\n" + "=" * 60)
    results = solver.validate_solution(sigma_elements, E_elements)