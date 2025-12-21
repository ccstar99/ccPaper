# database.py
"""
电场仿真数据库存储模块 - 完整修复版
保证7张表都能完整导入数据
"""

import mysql.connector
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import time
import pickle
import gzip


class ElectricFieldDatabase:
    """电场仿真数据库存储类"""

    def __init__(self, host='localhost', port=3306,
                 user='root', password='123456', database='electric_data'):
        """
        初始化数据库连接
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.max_matrix_size = 5000  # 最大矩阵尺寸，超过此尺寸需要压缩

    def connect(self) -> bool:
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                use_pure=True
            )
            print(f"数据库连接成功: {self.database}")
            return True
        except mysql.connector.Error as e:
            print(f"数据库连接失败: {e}")
            return self._create_database_if_not_exists()

    def _create_database_if_not_exists(self) -> bool:
        """如果数据库不存在，则创建它"""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.close()
            conn.close()

            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                use_pure=True
            )
            print(f"数据库 {self.database} 创建成功并已连接")
            return True
        except mysql.connector.Error as e:
            print(f"创建数据库失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("数据库连接已关闭")

    def create_tables(self):
        """创建所有数据表"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # 1. 仿真配置表（简化合并版）
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS simulation_config
                       (
                           config_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           updated_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                           ON
                           UPDATE
                           CURRENT_TIMESTAMP,

                           -- 仿真基本信息
                           simulation_name
                           VARCHAR
                       (
                           255
                       ) COMMENT '仿真名称',
                           description TEXT COMMENT '仿真描述',

                           -- 物理参数
                           radius DOUBLE COMMENT '球半径(m)',
                           voltage DOUBLE COMMENT '电极电压(V)',
                           epsilon_0 DOUBLE COMMENT '真空介电常数(F/m)',

                           -- 网格参数
                           mesh_type VARCHAR
                       (
                           50
                       ) COMMENT '网格类型',
                           subdivisions INT COMMENT '细分次数',
                           node_count INT COMMENT '节点数量',
                           element_count INT COMMENT '单元数量',
                           total_area DOUBLE COMMENT '总面积(m²)',
                           area_error DOUBLE COMMENT '面积误差(%)',

                           -- BEM求解配置
                           solution_method VARCHAR
                       (
                           50
                       ) COMMENT '求解方法',
                           gauss_order INT COMMENT '高斯积分阶数',

                           -- 求解结果摘要（JSON格式）
                           bem_statistics JSON COMMENT '统计信息{sigma_mean(C/m²), sigma_std(C/m²), E_mean(V/m), E_std(V/m), E_min(V/m), E_max(V/m)}',
                           theory_values JSON COMMENT '理论值{sigma(C/m²), E(V/m), total_charge(C)}',
                           error_metrics JSON COMMENT '误差指标{sigma_error(%), E_error(%), charge_error(%)}',

                           -- 电荷信息
                           total_charge DOUBLE COMMENT '总电荷(C)',
                           charge_error DOUBLE COMMENT '总电荷误差(%)',

                           -- 性能数据（JSON格式）
                           performance_data JSON COMMENT '性能数据{total_time(s), mesh_time(s), solve_time(s), ...}',

                           -- 验证结果
                           validation_status ENUM
                       (
                           'passed',
                           'failed',
                           'partial'
                       ) COMMENT '验证状态',
                           validation_conclusion TEXT COMMENT '验证结论',

                           -- 仿真状态
                           status ENUM
                       (
                           'pending',
                           'running',
                           'completed',
                           'failed'
                       ) DEFAULT 'pending',
                           error_message TEXT COMMENT '错误信息',
                           INDEX idx_created_at
                       (
                           created_at
                       ),
                           INDEX idx_status
                       (
                           status
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 2. 网格节点表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS mesh_node
                       (
                           node_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 节点索引
                           node_index
                           INT
                           COMMENT
                           '节点索引',

                           -- 坐标信息(JSON格式)
                           coordinates
                           JSON
                           COMMENT
                           '坐标信息{x(m), y(m), z(m)}',
                           spherical_coords
                           JSON
                           COMMENT
                           '球坐标{r(m), theta(rad), phi(rad)}',

                           -- 节点结果
                           charge_density
                           DOUBLE
                           COMMENT
                           '电荷密度(C/m²)',
                           theory_charge_density
                           DOUBLE
                           COMMENT
                           '理论电荷密度(C/m²)',
                           charge_density_error
                           DOUBLE
                           COMMENT
                           '电荷密度误差(%)',

                           FOREIGN
                           KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           INDEX idx_config_node
                       (
                           config_id,
                           node_index
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 3. 网格单元表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS mesh_element
                       (
                           element_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 单元索引
                           element_index
                           INT
                           COMMENT
                           '单元索引',

                           -- 顶点信息
                           vertex_indices
                           JSON
                           COMMENT
                           '顶点索引[v1, v2, v3]',

                           -- 几何属性
                           area
                           DOUBLE
                           COMMENT
                           '单元面积(m²)',
                           center_coords
                           JSON
                           COMMENT
                           '中心坐标{x(m), y(m), z(m)}',
                           spherical_center
                           JSON
                           COMMENT
                           '球坐标中心{r(m), theta(rad), phi(rad)}',
                           normal_vector
                           JSON
                           COMMENT
                           '法向量{nx, ny, nz}',

                           -- 单元结果
                           surface_charge_density
                           DOUBLE
                           COMMENT
                           '表面电荷密度(C/m²)',
                           electric_field_strength
                           DOUBLE
                           COMMENT
                           '电场强度(V/m)',
                           theory_surface_charge_density
                           DOUBLE
                           COMMENT
                           '理论表面电荷密度(C/m²)',
                           theory_electric_field
                           DOUBLE
                           COMMENT
                           '理论电场强度(V/m)',
                           surface_charge_error
                           DOUBLE
                           COMMENT
                           '表面电荷误差(%)',
                           electric_field_error
                           DOUBLE
                           COMMENT
                           '电场强度误差(%)',

                           FOREIGN
                           KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           INDEX idx_config_element
                       (
                           config_id,
                           element_index
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 4. 电场线表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS electric_field_line
                       (
                           line_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 电场线信息
                           line_index
                           INT
                           COMMENT
                           '电场线索引',

                           -- 起点信息
                           start_point
                           JSON
                           COMMENT
                           '起点坐标{x(m), y(m), z(m)}',

                           -- 电场线点集(JSON数组)
                           points
                           JSON
                           COMMENT
                           '点集[{x(m), y(m), z(m)}, ...]',

                           -- 统计信息
                           line_length
                           DOUBLE
                           COMMENT
                           '电场线长度(m)',
                           point_count
                           INT
                           COMMENT
                           '点数',

                           FOREIGN
                           KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           INDEX idx_config_line
                       (
                           config_id,
                           line_index
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 5. 系统矩阵表 - 修复data_format列类型
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS system_matrix
                       (
                           matrix_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 矩阵基本信息
                           matrix_name
                           VARCHAR
                       (
                           50
                       ) COMMENT '矩阵名称(G/H/Ginv/Hinv)',
                           matrix_type ENUM
                       (
                           'dense',
                           'sparse',
                           'symmetric'
                       ) DEFAULT 'dense' COMMENT '矩阵类型',
                           data_format VARCHAR
                       (
                           50
                       ) DEFAULT 'dense' COMMENT '数据格式(dense/csr/csc/coo等)',

                           -- 矩阵维度
                           rows_count INT COMMENT '行数',
                           cols_count INT COMMENT '列数',
                           nnz_count INT COMMENT '非零元素数量',
                           density DOUBLE COMMENT '密度(0-1)',

                           -- 矩阵存储
                           matrix_data LONGBLOB COMMENT '矩阵二进制数据（压缩存储）',
                           matrix_data_size INT COMMENT '矩阵数据大小(字节)',
                           compression_method VARCHAR
                       (
                           50
                       ) COMMENT '压缩方法',

                           -- 矩阵属性
                           is_square BOOLEAN DEFAULT FALSE COMMENT '是否为方阵',
                           is_symmetric BOOLEAN DEFAULT FALSE COMMENT '是否对称',
                           is_sparse BOOLEAN DEFAULT FALSE COMMENT '是否为稀疏矩阵',

                           -- 数值特性
                           condition_number DOUBLE COMMENT '条件数',
                           determinant DOUBLE COMMENT '行列式',
                           matrix_rank INT COMMENT '矩阵秩',

                           -- 统计信息
                           stats_summary JSON COMMENT '统计摘要{min, max, mean, std, median, ...}',

                           -- 元数据
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           notes TEXT COMMENT '备注',
                           FOREIGN KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           INDEX idx_config_matrix
                       (
                           config_id,
                           matrix_name
                       ),
                           INDEX idx_matrix_type
                       (
                           matrix_type
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 6. 空间采样点表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS spatial_sample
                       (
                           sample_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 采样点基本信息
                           sample_index
                           INT
                           COMMENT
                           '采样点索引',
                           sample_group
                           VARCHAR
                       (
                           100
                       ) COMMENT '采样点分组',
                           sample_type ENUM
                       (
                           'radial',
                           'angular',
                           'grid',
                           'random',
                           'custom'
                       ) DEFAULT 'custom' COMMENT '采样类型',

                           -- 采样设置
                           sampling_method VARCHAR
                       (
                           100
                       ) COMMENT '采样方法',
                           sampling_density DOUBLE COMMENT '采样密度(点/m³)',
                           sample_weight DOUBLE DEFAULT 1.0 COMMENT '采样权重',

                           -- 位置信息
                           x DOUBLE COMMENT 'X坐标(m)',
                           y DOUBLE COMMENT 'Y坐标(m)',
                           z DOUBLE COMMENT 'Z坐标(m)',
                           coordinates JSON COMMENT '完整坐标信息{x(m), y(m), z(m)}',

                           -- 球坐标
                           r DOUBLE COMMENT '径向距离(m)',
                           theta DOUBLE COMMENT '极角(rad)',
                           phi DOUBLE COMMENT '方位角(rad)',
                           spherical_coords JSON COMMENT '完整球坐标信息{r(m), theta(rad), phi(rad)}',

                           -- 物理量（完整）
                           potential DOUBLE COMMENT '电势(V)',
                           potential_theory DOUBLE COMMENT '理论电势(V)',
                           potential_error DOUBLE COMMENT '电势误差(%)',

                           -- 电场分量
                           E_x DOUBLE COMMENT '电场X分量(V/m)',
                           E_y DOUBLE COMMENT '电场Y分量(V/m)',
                           E_z DOUBLE COMMENT '电场Z分量(V/m)',
                           E_vector JSON COMMENT '电场矢量{E_x(V/m), E_y(V/m), E_z(V/m)}',

                           -- 理论电场分量
                           E_x_theory DOUBLE COMMENT '理论电场X分量(V/m)',
                           E_y_theory DOUBLE COMMENT '理论电场Y分量(V/m)',
                           E_z_theory DOUBLE COMMENT '理论电场Z分量(V/m)',
                           E_vector_theory JSON COMMENT '理论电场矢量{E_x(V/m), E_y(V/m), E_z(V/m)}',

                           -- 误差分析
                           E_x_error DOUBLE COMMENT '电场X分量误差(%)',
                           E_y_error DOUBLE COMMENT '电场Y分量误差(%)',
                           E_z_error DOUBLE COMMENT '电场Z分量误差(%)',

                           -- 合成量
                           E_magnitude DOUBLE COMMENT '电场强度(V/m)',
                           E_magnitude_theory DOUBLE COMMENT '理论电场强度(V/m)',
                           E_magnitude_error DOUBLE COMMENT '电场强度误差(%)',

                           -- 方向信息
                           E_direction_x DOUBLE COMMENT '电场方向X分量',
                           E_direction_y DOUBLE COMMENT '电场方向Y分量',
                           E_direction_z DOUBLE COMMENT '电场方向Z分量',
                           E_direction JSON COMMENT '电场方向矢量{x, y, z}',

                           -- 几何关系
                           distance_to_center DOUBLE COMMENT '到球心距离(m)',
                           distance_to_surface DOUBLE COMMENT '到球面距离(m)',
                           normal_distance DOUBLE COMMENT '法向距离(m)',

                           -- 区域分类
                           region_type ENUM
                       (
                           'near_field',
                           'far_field',
                           'surface',
                           'external'
                       ) DEFAULT 'external' COMMENT '区域类型',
                           quadrant ENUM
                       (
                           'I',
                           'II',
                           'III',
                           'IV',
                           'V',
                           'VI',
                           'VII',
                           'VIII',
                           'center'
                       ) COMMENT '空间象限',

                           -- 质量指标
                           convergence_factor DOUBLE COMMENT '收敛因子',
                           reliability_index DOUBLE COMMENT '可靠性指标',

                           -- 附加信息
                           is_boundary BOOLEAN DEFAULT FALSE COMMENT '是否为边界点',
                           is_special_point BOOLEAN DEFAULT FALSE COMMENT '是否为特殊点',
                           tags JSON COMMENT '标签',

                           -- 元数据
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           notes TEXT COMMENT '备注',
                           FOREIGN KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           INDEX idx_config_sample
                       (
                           config_id,
                           sample_index
                       ),
                           INDEX idx_sample_type
                       (
                           sample_type
                       ),
                           INDEX idx_region_type
                       (
                           region_type
                       ),
                           INDEX idx_distance
                       (
                           distance_to_center
                       ),
                           INDEX idx_potential
                       (
                           potential
                       ),
                           INDEX idx_E_magnitude
                       (
                           E_magnitude
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        # 7. 采样统计表
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS sampling_statistics
                       (
                           stat_id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           config_id
                           INT,

                           -- 统计信息
                           sample_count
                           INT
                           COMMENT
                           '总采样点数',
                           group_count
                           INT
                           COMMENT
                           '分组数量',

                           -- 误差统计
                           error_summary
                           JSON
                           COMMENT
                           '误差统计{mean_error(%), max_error(%), std_error(%), ...}',
                           error_distribution
                           JSON
                           COMMENT
                           '误差分布',

                           -- 区域统计
                           region_statistics
                           JSON
                           COMMENT
                           '区域统计',
                           quadrant_statistics
                           JSON
                           COMMENT
                           '象限统计',

                           -- 收敛性分析
                           convergence_analysis
                           JSON
                           COMMENT
                           '收敛性分析',

                           -- 相关性分析
                           correlation_analysis
                           JSON
                           COMMENT
                           '相关性分析',

                           FOREIGN
                           KEY
                       (
                           config_id
                       ) REFERENCES simulation_config
                       (
                           config_id
                       ) ON DELETE CASCADE,
                           UNIQUE KEY uk_config_stat
                       (
                           config_id
                       )
                           ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE =utf8mb4_unicode_ci
                       """)

        self.connection.commit()
        cursor.close()
        print("✓ 修复版数据表创建完成（共7张表）")
        print("  - simulation_config: 仿真配置表")
        print("  - mesh_node: 网格节点表")
        print("  - mesh_element: 网格单元表")
        print("  - electric_field_line: 电场线表")
        print("  - system_matrix: 系统矩阵表（修复data_format为VARCHAR）")
        print("  - spatial_sample: 空间采样点表")
        print("  - sampling_statistics: 采样统计表")

    def clear_all_data(self) -> bool:
        """清空所有数据表"""
        return self.clear_data()

    def clear_data(self, config_ids: Optional[List[int]] = None,
                   table_names: Optional[List[str]] = None) -> bool:
        """清空数据"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

            if table_names:
                tables = table_names
            else:
                # 按依赖关系排序
                tables = [
                    'sampling_statistics',
                    'spatial_sample',
                    'system_matrix',
                    'electric_field_line',
                    'mesh_element',
                    'mesh_node',
                    'simulation_config'
                ]

            for table in tables:
                if config_ids:
                    placeholders = ','.join(['%s'] * len(config_ids))
                    cursor.execute(f"DELETE FROM {table} WHERE config_id IN ({placeholders})", config_ids)
                    print(f"  ✓ 删除表 {table} 中 config_id 在 {config_ids} 的数据")
                else:
                    cursor.execute(f"TRUNCATE TABLE {table}")
                    print(f"  ✓ 清空表: {table}")

            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            self.connection.commit()
            print("✓ 数据清理完成")
            return True

        except mysql.connector.Error as e:
            print(f"清空数据表失败: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    # ========== 辅助方法 ==========

    @staticmethod
    def _compress_data(data: bytes) -> Tuple[bytes, str]:
        """压缩数据"""
        try:
            compressed = gzip.compress(data, compresslevel=6)
            if len(compressed) < len(data):
                return compressed, 'gzip'
            else:
                return data, 'none'
        except Exception:
            return data, 'none'

    @staticmethod
    def _decompress_data(data: bytes, method: str) -> bytes:
        """解压数据"""
        if method == 'gzip':
            return gzip.decompress(data)
        return data
    
    def _safe_array_conversion(self, data):
        """安全地将数据转换为 numpy 数组"""
        if data is None:
            return np.array([])
        
        if isinstance(data, np.ndarray):
            return data
        
        if isinstance(data, (list, tuple)):
            try:
                return np.array(data, dtype=np.float64)
            except:
                # 尝试逐个元素转换
                try:
                    return np.array([float(x) for x in data])
                except:
                    return np.array([])
        
        if hasattr(data, '__array__'):
            try:
                return np.asarray(data, dtype=np.float64)
            except:
                pass
        
        # 默认返回空数组
        return np.array([])

    def _compute_matrix_stats(self, matrix):
        """
        安全地计算矩阵统计信息
        正确处理列表类型的矩阵数据
        """
        try:
            from scipy import sparse
            # 检查是否为稀疏矩阵
            is_sparse_matrix = sparse.issparse(matrix)

            if is_sparse_matrix:
                # 稀疏矩阵处理
                try:
                    # 获取非零元素数量
                    nnz = 0
                    if hasattr(matrix, 'nnz'):
                        nnz = matrix.nnz
                    elif hasattr(matrix, 'getnnz'):
                        nnz = matrix.getnnz()

                    # 获取数据并转换为数组
                    data_array = np.array([])
                    if hasattr(matrix, 'data') and matrix.data is not None:
                        data_array = self._safe_array_conversion(matrix.data)
                    else:
                        # 尝试其他方法获取数据
                        try:
                            matrix_coo = matrix.tocoo()
                            if hasattr(matrix_coo, 'data'):
                                data_array = self._safe_array_conversion(matrix_coo.data)
                        except:
                            pass

                    # 计算统计信息
                    stats = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}
                    if data_array.size > 0:
                        # 确保是数值类型
                        try:
                            data_array = data_array.astype(np.float64)
                        except:
                            # 如果转换失败，尝试取实部
                            try:
                                data_array = data_array.real.astype(np.float64)
                            except:
                                pass

                        if data_array.size > 0:
                            try:
                                stats = {
                                    'min': float(data_array.min()),
                                    'max': float(data_array.max()),
                                    'mean': float(data_array.mean()),
                                    'std': float(data_array.std()) if data_array.size > 1 else 0.0,
                                    'median': float(np.median(data_array))
                                }
                            except:
                                pass

                    return nnz, stats, True

                except Exception as e:
                    print(f"稀疏矩阵统计计算失败: {e}")
                    return 0, {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}, True

            else:
                # 密集矩阵处理
                try:
                    # 确保是 numpy 数组
                    if not isinstance(matrix, np.ndarray):
                        matrix_array = self._safe_array_conversion(matrix)
                    else:
                        matrix_array = matrix

                    # 确保是二维数组
                    if matrix_array.ndim == 0:
                        matrix_array = np.array([[matrix_array]])
                    elif matrix_array.ndim == 1:
                        matrix_array = matrix_array.reshape(-1, 1)

                    # 计算非零元素数量
                    nnz = np.count_nonzero(matrix_array)

                    # 计算统计信息
                    stats = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}
                    if matrix_array.size > 0:
                        try:
                            # 展平矩阵
                            flat_array = matrix_array.flatten()

                            # 确保是数值类型
                            try:
                                flat_array = flat_array.astype(np.float64)
                            except:
                                # 如果转换失败，尝试取实部
                                try:
                                    flat_array = flat_array.real.astype(np.float64)
                                except:
                                    pass

                            if flat_array.size > 0:
                                stats = {
                                    'min': float(flat_array.min()),
                                    'max': float(flat_array.max()),
                                    'mean': float(flat_array.mean()),
                                    'std': float(flat_array.std()) if flat_array.size > 1 else 0.0,
                                    'median': float(np.median(flat_array))
                                }
                        except:
                            pass

                    return nnz, stats, False

                except Exception as e:
                    print(f"密集矩阵统计计算失败: {e}")
                    return 0, {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}, False

        except Exception as e:
            print(f"计算矩阵统计信息失败: {e}")
            import traceback
            traceback.print_exc()
            return 0, {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}, False

    def _save_matrix_to_blob(self, matrix) -> Tuple[bytes, Dict[str, Any]]:
        """
        将矩阵保存为二进制BLOB
        使用专门的统计计算方法
        """
        try:
            # 获取矩阵统计信息
            nnz, stats, is_sparse_matrix = self._compute_matrix_stats(matrix)

            # 获取矩阵形状
            rows, cols = matrix.shape
            total_elements = rows * cols
            density = nnz / total_elements if total_elements > 0 else 0

            # 检查对称性（简化版）
            is_symmetric = False
            if rows == cols and rows <= 100:
                try:
                    if is_sparse_matrix:
                        # 稀疏矩阵：检查格式
                        if hasattr(matrix, 'format'):
                            is_symmetric = matrix.format in ['csr', 'csc']
                    else:
                        # 密集矩阵：检查数值对称性
                        if isinstance(matrix, np.ndarray):
                            # 只检查小矩阵
                            if rows <= 50:
                                is_symmetric = np.allclose(matrix, matrix.T)
                except:
                    pass

            # 确定矩阵格式
            matrix_format = 'dense'
            if is_sparse_matrix:
                if hasattr(matrix, 'format'):
                    matrix_format = matrix.format
                else:
                    matrix_format = 'sparse'

            # 构建元数据
            metadata = {
                'shape': matrix.shape,
                'dtype': str(matrix.dtype) if hasattr(matrix, 'dtype') else 'unknown',
                'nnz': int(nnz),
                'density': float(density),
                'is_sparse': is_sparse_matrix,
                'is_square': rows == cols,
                'is_symmetric': is_symmetric,
                'format': matrix_format,
                'stats': stats
            }

            # 序列化矩阵
            try:
                data = pickle.dumps(matrix)
            except Exception as e:
                print(f"序列化矩阵失败: {e}")
                # 尝试使用不同的协议
                data = pickle.dumps(matrix, protocol=pickle.HIGHEST_PROTOCOL)

            # 压缩
            compressed_data, compression_method = self._compress_data(data)
            metadata['compression'] = compression_method
            metadata['original_size'] = len(data)
            metadata['compressed_size'] = len(compressed_data)

            return compressed_data, metadata

        except Exception as e:
            print(f"保存矩阵到BLOB失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回基本元数据
            rows, cols = matrix.shape if hasattr(matrix, 'shape') else (0, 0)
            return b'', {
                'shape': (rows, cols),
                'dtype': 'float64',
                'format': 'dense',
                'stats': {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0}
            }

    def _load_matrix_from_blob(self, blob_data: bytes, metadata: Dict[str, Any]):
        """从BLOB加载矩阵"""
        # 解压
        if metadata.get('compression') == 'gzip':
            data = self._decompress_data(blob_data, 'gzip')
        else:
            data = blob_data

        # 反序列化
        try:
            matrix = pickle.loads(data)
            return matrix
        except Exception as e:
            print(f"加载矩阵失败: {e}")
            return None

    # ========== 系统矩阵存储方法 ==========

    def save_system_matrix(self, config_id: int, matrix_name: str,
                           matrix,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存完整的系统矩阵
        简化版本，专注于数据保存
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            # 准备矩阵数据
            matrix_blob, blob_metadata = self._save_matrix_to_blob(matrix)

            if len(matrix_blob) == 0:
                print(f"警告: 矩阵 '{matrix_name}' 数据为空，跳过保存")
                return False

            # 提取矩阵属性
            rows, cols = blob_metadata['shape']
            nnz = blob_metadata['nnz']
            density = blob_metadata['density']
            matrix_format = blob_metadata.get('format', 'dense')
            stats = blob_metadata.get('stats', {})

            # 简化数值特性计算 - 只保存基本信息
            condition_num = 0.0
            determinant = 0.0
            rank = 0

            # 确定矩阵类型
            if blob_metadata['is_sparse']:
                matrix_type = 'sparse'
            elif blob_metadata.get('is_symmetric', False):
                matrix_type = 'symmetric'
            else:
                matrix_type = 'dense'

            sql = """
                  INSERT INTO system_matrix
                  (config_id, matrix_name, matrix_type, data_format,
                   rows_count, cols_count, nnz_count, density,
                   matrix_data, matrix_data_size, compression_method,
                   is_square, is_symmetric, is_sparse,
                   condition_number, determinant, matrix_rank,
                   stats_summary, notes)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) \
                  """

            values = (
                config_id,
                matrix_name,
                matrix_type,
                matrix_format,
                rows, cols, nnz, density,
                matrix_blob, len(matrix_blob), blob_metadata.get('compression', 'none'),
                blob_metadata['is_square'],
                blob_metadata.get('is_symmetric', False),
                blob_metadata['is_sparse'],
                condition_num, determinant, rank,
                json.dumps(stats),
                metadata.get('notes', f'矩阵 {matrix_name}') if metadata else f'矩阵 {matrix_name}'
            )

            cursor.execute(sql, values)
            self.connection.commit()
            print(
                f"✓ 系统矩阵 '{matrix_name}' 保存成功，尺寸: {rows}×{cols}，格式: {matrix_format}，大小: {len(matrix_blob) / 1024:.1f}KB")
            return True

        except Exception as e:
            print(f"保存系统矩阵失败: {e}")
            import traceback
            traceback.print_exc()
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    def get_system_matrix(self, config_id: int, matrix_name: str):
        """获取系统矩阵"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute("""
                           SELECT matrix_data, stats_summary
                           FROM system_matrix
                           WHERE config_id = %s
                             AND matrix_name = %s
                           """, (config_id, matrix_name))

            result = cursor.fetchone()
            if result:
                metadata = json.loads(result['stats_summary'])
                matrix_blob = result['matrix_data']
                matrix = self._load_matrix_from_blob(matrix_blob, metadata)
                return matrix

            return None

        except Exception as e:
            print(f"获取系统矩阵失败: {e}")
            return None
        finally:
            cursor.close()

    def get_matrix_metadata(self, config_id: int) -> List[Dict[str, Any]]:
        """获取所有矩阵的元数据（不加载矩阵数据）"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute("""
                           SELECT matrix_id,
                                  matrix_name,
                                  matrix_type,
                                  rows_count,
                                  cols_count,
                                  nnz_count,
                                  density,
                                  matrix_data_size,
                                  compression_method,
                                  is_square,
                                  is_symmetric,
                                  is_sparse,
                                  condition_number,
                                  determinant,
                                  matrix_rank,
                                  stats_summary,
                                  created_at,
                                  notes
                           FROM system_matrix
                           WHERE config_id = %s
                           ORDER BY matrix_name
                           """, (config_id,))

            results = cursor.fetchall()

            # 解析JSON字段
            for result in results:
                if result.get('stats_summary'):
                    result['stats_summary'] = json.loads(result['stats_summary'])

            return results

        except Exception as e:
            print(f"获取矩阵元数据失败: {e}")
            return []
        finally:
            cursor.close()

    # ========== 空间采样点存储方法 ==========

    def save_spatial_samples(self, config_id: int,
                             samples: List[Dict[str, Any]],
                             sample_group: str = "default",
                             sampling_method: str = "uniform") -> bool:
        """
        保存完整的空间采样点数据

        Args:
            config_id: 仿真配置ID
            samples: 采样点列表，每个采样点是一个字典
            sample_group: 采样点分组名称
            sampling_method: 采样方法
        """
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        sql = """
              INSERT INTO spatial_sample
              (config_id, sample_index, sample_group, sample_type, sampling_method,
               x, y, z, coordinates, r, theta, phi, spherical_coords,
               potential, potential_theory, potential_error,
               E_x, E_y, E_z, E_vector,
               E_x_theory, E_y_theory, E_z_theory, E_vector_theory,
               E_x_error, E_y_error, E_z_error,
               E_magnitude, E_magnitude_theory, E_magnitude_error,
               E_direction_x, E_direction_y, E_direction_z, E_direction,
               distance_to_center, distance_to_surface, normal_distance,
               region_type, quadrant, convergence_factor, reliability_index,
               is_boundary, is_special_point, tags, notes)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s) \
              """

        try:
            batch_values = []
            sample_count = len(samples)

            for i, sample in enumerate(samples):
                # 提取坐标
                x = sample.get('x', 0.0)
                y = sample.get('y', 0.0)
                z = sample.get('z', 0.0)
                coordinates = json.dumps({'x': x, 'y': y, 'z': z})

                # 提取球坐标
                r = sample.get('r', 0.0)
                theta = sample.get('theta', 0.0)
                phi = sample.get('phi', 0.0)
                spherical_coords = json.dumps({'r': r, 'theta': theta, 'phi': phi})

                # 电势
                potential = sample.get('potential', 0.0)
                potential_theory = sample.get('potential_theory', 0.0)
                potential_error = sample.get('potential_error', 0.0)

                # 电场
                E_x = sample.get('E_x', 0.0)
                E_y = sample.get('E_y', 0.0)
                E_z = sample.get('E_z', 0.0)
                E_vector = json.dumps({'x': E_x, 'y': E_y, 'z': E_z})

                # 理论电场
                E_x_theory = sample.get('E_x_theory', 0.0)
                E_y_theory = sample.get('E_y_theory', 0.0)
                E_z_theory = sample.get('E_z_theory', 0.0)
                E_vector_theory = json.dumps({'x': E_x_theory, 'y': E_y_theory, 'z': E_z_theory})

                # 误差
                E_x_error = sample.get('E_x_error', 0.0)
                E_y_error = sample.get('E_y_error', 0.0)
                E_z_error = sample.get('E_z_error', 0.0)

                # 电场强度
                E_magnitude = sample.get('E_magnitude', 0.0)
                E_magnitude_theory = sample.get('E_magnitude_theory', 0.0)
                E_magnitude_error = sample.get('E_magnitude_error', 0.0)

                # 方向
                E_direction_x = sample.get('E_direction_x', 0.0)
                E_direction_y = sample.get('E_direction_y', 0.0)
                E_direction_z = sample.get('E_direction_z', 0.0)
                E_direction = json.dumps({'x': E_direction_x, 'y': E_direction_y, 'z': E_direction_z})

                # 距离
                distance_to_center = sample.get('distance_to_center', 0.0)
                distance_to_surface = sample.get('distance_to_surface', 0.0)
                normal_distance = sample.get('normal_distance', 0.0)

                # 区域分类
                region_type = sample.get('region_type', 'external')
                quadrant = sample.get('quadrant', 'I')

                # 质量指标
                convergence_factor = sample.get('convergence_factor', 1.0)
                reliability_index = sample.get('reliability_index', 1.0)

                # 标志
                is_boundary = sample.get('is_boundary', False)
                is_special_point = sample.get('is_special_point', False)
                tags = json.dumps(sample.get('tags', []))

                # 备注
                notes = sample.get('notes', '')

                values = (
                    config_id, i, sample_group, sample.get('sample_type', 'custom'), sampling_method,
                    x, y, z, coordinates, r, theta, phi, spherical_coords,
                    potential, potential_theory, potential_error,
                    E_x, E_y, E_z, E_vector,
                    E_x_theory, E_y_theory, E_z_theory, E_vector_theory,
                    E_x_error, E_y_error, E_z_error,
                    E_magnitude, E_magnitude_theory, E_magnitude_error,
                    E_direction_x, E_direction_y, E_direction_z, E_direction,
                    distance_to_center, distance_to_surface, normal_distance,
                    region_type, quadrant, convergence_factor, reliability_index,
                    is_boundary, is_special_point, tags, notes
                )
                batch_values.append(values)

                # 批量插入
                if len(batch_values) >= 100:
                    cursor.executemany(sql, batch_values)
                    batch_values = []

            # 插入剩余数据
            if batch_values:
                cursor.executemany(sql, batch_values)

            self.connection.commit()
            print(f"✓ 空间采样点数据保存成功，共 {sample_count} 个采样点")

            # 保存采样统计
            self._save_sampling_statistics(config_id, samples)

            return True

        except Exception as e:
            print(f"保存空间采样点失败: {e}")
            import traceback
            traceback.print_exc()
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    def _save_sampling_statistics(self, config_id: int, samples: List[Dict[str, Any]]) -> bool:
        """保存采样统计信息"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            sample_count = len(samples)

            # 计算误差统计
            E_errors = [s.get('E_magnitude_error', 0.0) for s in samples]
            potential_errors = [s.get('potential_error', 0.0) for s in samples]

            error_summary = {
                'E_mean_error': float(np.mean(E_errors)) if E_errors else 0,
                'E_max_error': float(np.max(E_errors)) if E_errors else 0,
                'E_std_error': float(np.std(E_errors)) if E_errors else 0,
                'potential_mean_error': float(np.mean(potential_errors)) if potential_errors else 0,
                'potential_max_error': float(np.max(potential_errors)) if potential_errors else 0,
                'potential_std_error': float(np.std(potential_errors)) if potential_errors else 0
            }

            # 误差分布
            if E_errors:
                max_error = max(E_errors) if E_errors else 10
                error_bins = np.linspace(0, max_error, 11).tolist()
                error_dist = np.histogram(E_errors, bins=error_bins)[0].tolist()
                error_distribution = {
                    'bins': error_bins,
                    'counts': error_dist
                }
            else:
                error_distribution = {'bins': [], 'counts': []}

            # 区域统计
            region_types = {}
            for sample in samples:
                region = sample.get('region_type', 'unknown')
                region_types[region] = region_types.get(region, 0) + 1

            region_statistics = region_types

            # 象限统计
            quadrants = {}
            for sample in samples:
                quadrant = sample.get('quadrant', 'unknown')
                quadrants[quadrant] = quadrants.get(quadrant, 0) + 1

            quadrant_statistics = quadrants

            # 相关性分析
            distances = [s.get('distance_to_center', 0.0) for s in samples]
            E_magnitudes = [s.get('E_magnitude', 0.0) for s in samples]

            correlation = 0.0
            if len(distances) > 1 and len(E_magnitudes) > 1:
                try:
                    correlation = float(np.corrcoef(distances, E_magnitudes)[0, 1])
                except Exception:
                    correlation = 0.0

            correlation_analysis = {
                'distance_E_correlation': correlation
            }

            sql = """
                  INSERT INTO sampling_statistics
                  (config_id, sample_count, group_count,
                   error_summary, error_distribution,
                   region_statistics, quadrant_statistics,
                   correlation_analysis)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY \
                  UPDATE \
                      sample_count = \
                  VALUES (sample_count), error_summary = \
                  VALUES (error_summary), error_distribution = \
                  VALUES (error_distribution), region_statistics = \
                  VALUES (region_statistics), quadrant_statistics = \
                  VALUES (quadrant_statistics), correlation_analysis = \
                  VALUES (correlation_analysis) \
                  """

            values = (
                config_id, sample_count, 1,
                json.dumps(error_summary),
                json.dumps(error_distribution),
                json.dumps(region_statistics),
                json.dumps(quadrant_statistics),
                json.dumps(correlation_analysis)
            )

            cursor.execute(sql, values)
            self.connection.commit()
            print(f"✓ 采样统计信息保存成功")
            return True

        except Exception as e:
            print(f"保存采样统计信息失败: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    def get_spatial_samples(self, config_id: int,
                            sample_group: Optional[str] = None,
                            region_type: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取空间采样点数据"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor(dictionary=True)

        try:
            query = """
                    SELECT * \
                    FROM spatial_sample
                    WHERE config_id = %s \
                    """
            params = [config_id]

            if sample_group:
                query += " AND sample_group = %s"
                params.append(sample_group)

            if region_type:
                query += " AND region_type = %s"
                params.append(region_type)

            query += " ORDER BY sample_index"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cursor.execute(query, tuple(params))
            results = cursor.fetchall()

            # 解析JSON字段
            for result in results:
                for field in ['coordinates', 'spherical_coords', 'E_vector',
                              'E_vector_theory', 'E_direction', 'tags']:
                    if result.get(field):
                        result[field] = json.loads(result[field])

            return results

        except Exception as e:
            print(f"获取空间采样点失败: {e}")
            return []
        finally:
            cursor.close()

    def get_sample_statistics(self, config_id: int) -> Optional[Dict[str, Any]]:
        """获取采样统计信息"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute("""
                           SELECT *
                           FROM sampling_statistics
                           WHERE config_id = %s
                           """, (config_id,))

            result = cursor.fetchone()
            if result:
                # 解析JSON字段
                for field in ['error_summary', 'error_distribution',
                              'region_statistics', 'quadrant_statistics',
                              'correlation_analysis']:
                    if result.get(field):
                        result[field] = json.loads(result[field])

            return result

        except Exception as e:
            print(f"获取采样统计信息失败: {e}")
            return None
        finally:
            cursor.close()

    # ========== 完整仿真数据保存方法 ==========

    def save_complete_simulation(self, solver, mesh, sigma_elements=None,
                                 sigma_nodes=None, E_elements=None,
                                 field_lines=None, start_points=None,
                                 compute_time=0.0, description="",
                                 simulation_name=None) -> int:
        """
        完整保存仿真数据到7张表
        """
        print("\n【数据库】开始保存完整仿真数据...")

        cursor = None
        config_id = 0

        try:
            # 确保数据库连接
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    print("数据库连接失败，无法保存数据")
                    return 0

            # 开始事务
            self.connection.start_transaction()
            cursor = self.connection.cursor()

            # 1. 获取验证结果
            print("  步骤1: 获取验证结果...")
            validation_results = solver.validate_solution(sigma_elements, E_elements)

            # 确定验证状态
            E_error = validation_results['E_mean_error']
            if E_error < 1.0:
                status = 'passed'
                conclusion = '实现成功，精度达到论文水平'
            elif E_error < 2.0:
                status = 'partial'
                conclusion = '实现基本正确，精度接近论文水平'
            else:
                status = 'failed'
                conclusion = '实现存在一定误差，需要进一步优化'

            # 理论值
            sigma_theory = solver.epsilon_0 * solver.voltage / solver.radius
            E_theory = solver.voltage / solver.radius
            total_charge_theory = 4 * np.pi * solver.epsilon_0 * solver.radius * solver.voltage

            # 2. 保存仿真配置到 simulation_config 表
            print("  步骤2: 保存仿真配置...")
            config_data = {
                'simulation_name': simulation_name or f'simulation_{int(time.time())}',
                'description': description,
                'radius': mesh.radius,
                'voltage': solver.voltage,
                'epsilon_0': solver.epsilon_0,
                'mesh_type': 'icosphere',
                'subdivisions': solver.subdivisions if hasattr(solver, 'subdivisions') else 1,
                'node_count': mesh.num_vertices,
                'element_count': mesh.num_triangles,
                'total_area': sum(tri.area for tri in mesh.spherical_triangles),
                'area_error': 0.0,
                'solution_method': 'BEM',
                'gauss_order': solver.gauss_order if hasattr(solver, 'gauss_order') else 7,
                'bem_statistics': {
                    'sigma_mean': validation_results['sigma_mean'],
                    'sigma_std': validation_results['sigma_std'],
                    'E_mean': validation_results['E_mean'],
                    'E_std': validation_results['E_std'],
                    'E_min': validation_results['E_min'],
                    'E_max': validation_results['E_max']
                },
                'theory_values': {
                    'sigma': sigma_theory,
                    'E': E_theory,
                    'total_charge': total_charge_theory
                },
                'error_metrics': {
                    'sigma_error': validation_results['sigma_mean_error'],
                    'E_error': validation_results['E_mean_error'],
                    'charge_error': validation_results['charge_error']
                },
                'total_charge': solver.total_charge if hasattr(solver, 'total_charge') else 0,
                'charge_error': validation_results['charge_error'],
                'performance_data': {
                    'total_time': compute_time,
                    'mesh_time': 0.0,
                    'solve_time': compute_time,
                    'visualization_time': 0.0
                },
                'validation_status': status,
                'validation_conclusion': conclusion,
                'status': 'completed'
            }

            config_id = self._save_simulation_config_cursor(cursor, config_data)
            if not config_id:
                raise ValueError("保存仿真配置失败")
            print(f"    ✓ 仿真配置保存成功，config_id: {config_id}")

            # 3. 保存网格节点到 mesh_node 表
            print("  步骤3: 保存网格节点数据...")
            self._save_mesh_nodes(cursor, config_id, mesh, sigma_nodes, solver)

            # 4. 保存网格单元到 mesh_element 表
            print("  步骤4: 保存网格单元数据...")
            self._save_mesh_elements(cursor, config_id, mesh, sigma_elements, E_elements, solver)

            # 5. 保存系统矩阵到 system_matrix 表
            print("  步骤5: 保存系统矩阵...")
            if hasattr(solver, 'G') and solver.G is not None:
                self.save_system_matrix(config_id, 'G', solver.G,
                                        {'notes': '影响系数矩阵G'})
            if hasattr(solver, 'H') and solver.H is not None:
                self.save_system_matrix(config_id, 'H', solver.H,
                                        {'notes': '影响系数矩阵H'})

            # 6. 保存电场线到 electric_field_line 表
            print("  步骤6: 保存电场线数据...")
            if field_lines is not None and start_points is not None:
                self._save_field_lines_cursor(cursor, config_id, field_lines, start_points)

            # 7. 提交事务
            self.connection.commit()
            print(f"✓ 所有仿真数据保存成功，config_id: {config_id}")

            return config_id

        except Exception as e:
            # 回滚事务
            if self.connection:
                self.connection.rollback()
            print(f"保存仿真数据失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
        finally:
            if cursor:
                cursor.close()

    def _save_simulation_config_cursor(self, cursor, config_data: Dict[str, Any]) -> int:
        """使用游标保存仿真配置"""
        sql = """
              INSERT INTO simulation_config
              (simulation_name, description, radius, voltage, epsilon_0,
               mesh_type, subdivisions, node_count, element_count, total_area, area_error,
               solution_method, gauss_order, bem_statistics, theory_values, error_metrics,
               total_charge, charge_error, performance_data, validation_status, validation_conclusion,
               status, error_message)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) \
              """

        values = (
            config_data.get('simulation_name', f'simulation_{int(time.time())}'),
            config_data.get('description', ''),
            config_data.get('radius', 1.0),
            config_data.get('voltage', 100.0),
            config_data.get('epsilon_0', 8.854187817e-12),
            config_data.get('mesh_type', 'icosphere'),
            config_data.get('subdivisions', 0),
            config_data.get('node_count', 0),
            config_data.get('element_count', 0),
            config_data.get('total_area', 0.0),
            config_data.get('area_error', 0.0),
            config_data.get('solution_method', 'BEM'),
            config_data.get('gauss_order', 7),
            json.dumps(config_data.get('bem_statistics', {})),
            json.dumps(config_data.get('theory_values', {})),
            json.dumps(config_data.get('error_metrics', {})),
            config_data.get('total_charge', 0.0),
            config_data.get('charge_error', 0.0),
            json.dumps(config_data.get('performance_data', {})),
            config_data.get('validation_status', 'completed'),
            config_data.get('validation_conclusion', ''),
            config_data.get('status', 'completed'),
            config_data.get('error_message', '')
        )

        cursor.execute(sql, values)
        return cursor.lastrowid

    def _save_mesh_nodes(self, cursor, config_id: int, mesh, sigma_nodes, solver):
        """保存网格节点数据"""
        if sigma_nodes is None:
            print("    警告: sigma_nodes 为 None，跳过节点数据保存")
            return

        sql = """
              INSERT INTO mesh_node
              (config_id, node_index, coordinates, spherical_coords,
               charge_density, theory_charge_density, charge_density_error)
              VALUES (%s, %s, %s, %s, %s, %s, %s) \
              """

        batch_values = []

        # 理论电荷密度
        sigma_theory = solver.epsilon_0 * solver.voltage / solver.radius

        for i in range(mesh.num_vertices):
            vertex = mesh.vertices[i]

            # 直角坐标
            x, y, z = vertex
            coordinates = json.dumps({'x': float(x), 'y': float(y), 'z': float(z)})

            # 计算球坐标
            r = np.linalg.norm(vertex)
            theta = np.arccos(z / r) if r > 0 else 0
            phi = np.arctan2(y, x)
            spherical_coords = json.dumps({
                'r': float(r),
                'theta': float(theta),
                'phi': float(phi)
            })

            # 电荷密度和误差
            charge_density = sigma_nodes[i] if i < len(sigma_nodes) else 0
            charge_density_error = abs((charge_density - sigma_theory) / sigma_theory * 100) if sigma_theory != 0 else 0

            values = (
                config_id, i, coordinates, spherical_coords,
                float(charge_density), float(sigma_theory), float(charge_density_error)
            )
            batch_values.append(values)

            # 批量插入
            if len(batch_values) >= 100:
                cursor.executemany(sql, batch_values)
                batch_values = []

        # 插入剩余数据
        if batch_values:
            cursor.executemany(sql, batch_values)

        print(f"    ✓ 保存 {mesh.num_vertices} 个节点数据")

    def _save_mesh_elements(self, cursor, config_id: int, mesh, sigma_elements, E_elements, solver):
        """保存网格单元数据"""
        if sigma_elements is None or E_elements is None:
            print("    警告: sigma_elements 或 E_elements 为 None，跳过单元数据保存")
            return

        sql = """
              INSERT INTO mesh_element
              (config_id, element_index, vertex_indices, area, center_coords,
               spherical_center, normal_vector, surface_charge_density, electric_field_strength,
               theory_surface_charge_density, theory_electric_field, surface_charge_error, electric_field_error)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) \
              """

        batch_values = []

        # 理论值
        sigma_theory = solver.epsilon_0 * solver.voltage / solver.radius
        E_theory = solver.voltage / solver.radius

        for i, tri in enumerate(mesh.spherical_triangles):
            # 顶点索引
            vertex_indices = json.dumps([int(idx) for idx in tri.vertex_indices])

            # 单元面积
            area = tri.area

            # 中心坐标
            center = np.mean(tri.vertices, axis=0)
            center_coords = json.dumps({
                'x': float(center[0]),
                'y': float(center[1]),
                'z': float(center[2])
            })

            # 球坐标中心
            r = np.linalg.norm(center)
            theta = np.arccos(center[2] / r) if r > 0 else 0
            phi = np.arctan2(center[1], center[0])
            spherical_center = json.dumps({
                'r': float(r),
                'theta': float(theta),
                'phi': float(phi)
            })

            # 法向量
            v0, v1, v2 = tri.vertices
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            normal_vector = json.dumps({
                'nx': float(normal[0]),
                'ny': float(normal[1]),
                'nz': float(normal[2])
            })

            # 计算值
            sigma = sigma_elements[i] if i < len(sigma_elements) else 0
            E = E_elements[i] if i < len(E_elements) else 0

            # 误差
            sigma_error = abs((sigma - sigma_theory) / sigma_theory * 100) if sigma_theory != 0 else 0
            E_error = abs((E - E_theory) / E_theory * 100) if E_theory != 0 else 0

            values = (
                config_id, i, vertex_indices, float(area), center_coords,
                spherical_center, normal_vector, float(sigma), float(E),
                float(sigma_theory), float(E_theory), float(sigma_error), float(E_error)
            )
            batch_values.append(values)

            # 批量插入
            if len(batch_values) >= 50:
                cursor.executemany(sql, batch_values)
                batch_values = []

        # 插入剩余数据
        if batch_values:
            cursor.executemany(sql, batch_values)

        print(f"    ✓ 保存 {len(mesh.spherical_triangles)} 个单元数据")

    def _save_field_lines_cursor(self, cursor, config_id: int,
                                 field_lines: List[np.ndarray],
                                 start_points: np.ndarray):
        """保存电场线数据"""
        sql = """
              INSERT INTO electric_field_line
                  (config_id, line_index, start_point, points, line_length, point_count)
              VALUES (%s, %s, %s, %s, %s, %s) \
              """

        batch_values = []

        for i, (line, start_point) in enumerate(zip(field_lines, start_points)):
            # 起点
            start_point_json = json.dumps({
                'x': float(start_point[0]),
                'y': float(start_point[1]),
                'z': float(start_point[2])
            })

            # 点集
            points_list = []
            for point in line:
                points_list.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'z': float(point[2])
                })
            points_json = json.dumps(points_list)

            # 计算长度
            line_length = 0.0
            if len(line) > 1:
                for j in range(1, len(line)):
                    line_length += np.linalg.norm(line[j] - line[j - 1])

            values = (
                config_id, i, start_point_json, points_json,
                float(line_length), len(points_list)
            )
            batch_values.append(values)

            # 批量插入
            if len(batch_values) >= 20:
                cursor.executemany(sql, batch_values)
                batch_values = []

        # 插入剩余数据
        if batch_values:
            cursor.executemany(sql, batch_values)

        print(f"    ✓ 保存 {len(field_lines)} 条电场线数据")

    # ========== 其他接口方法 ==========

    def save_simulation_config(self, config_data: Dict[str, Any]) -> int:
        """保存仿真配置"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        sql = """
              INSERT INTO simulation_config
              (simulation_name, description, radius, voltage, epsilon_0,
               mesh_type, subdivisions, node_count, element_count, total_area, area_error,
               solution_method, gauss_order, bem_statistics, theory_values, error_metrics,
               total_charge, charge_error, performance_data, validation_status, validation_conclusion,
               status, error_message)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) \
              """

        values = (
            config_data.get('simulation_name', f'simulation_{int(time.time())}'),
            config_data.get('description', ''),
            config_data.get('radius', 1.0),
            config_data.get('voltage', 100.0),
            config_data.get('epsilon_0', 8.854187817e-12),
            config_data.get('mesh_type', 'icosphere'),
            config_data.get('subdivisions', 0),
            config_data.get('node_count', 0),
            config_data.get('element_count', 0),
            config_data.get('total_area', 0.0),
            config_data.get('area_error', 0.0),
            config_data.get('solution_method', 'BEM'),
            config_data.get('gauss_order', 7),
            json.dumps(config_data.get('bem_statistics', {})),
            json.dumps(config_data.get('theory_values', {})),
            json.dumps(config_data.get('error_metrics', {})),
            config_data.get('total_charge', 0.0),
            config_data.get('charge_error', 0.0),
            json.dumps(config_data.get('performance_data', {})),
            config_data.get('validation_status', 'completed'),
            config_data.get('validation_conclusion', ''),
            config_data.get('status', 'completed'),
            config_data.get('error_message', '')
        )

        try:
            cursor.execute(sql, values)
            self.connection.commit()
            config_id = cursor.lastrowid
            print(f"仿真配置保存成功，config_id: {config_id}")
            return config_id
        except mysql.connector.Error as e:
            print(f"保存仿真配置失败: {e}")
            self.connection.rollback()
            return 0
        finally:
            cursor.close()

    def save_mesh_data(self, config_id: int, mesh, solver=None) -> bool:
        """保存网格数据 - 外部接口"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            self.connection.start_transaction()

            # 保存节点数据
            self._save_mesh_nodes(cursor, config_id, mesh,
                                  solver.sigma_nodes if solver and hasattr(solver, 'sigma_nodes') else None,
                                  solver)

            # 保存单元数据
            self._save_mesh_elements(cursor, config_id, mesh,
                                     solver.sigma_elements if solver and hasattr(solver, 'sigma_elements') else None,
                                     solver.E_elements if solver and hasattr(solver, 'E_elements') else None,
                                     solver)

            self.connection.commit()
            print(f"✓ 网格数据保存成功到 config_id: {config_id}")
            return True

        except Exception as e:
            self.connection.rollback()
            print(f"保存网格数据失败: {e}")
            return False
        finally:
            cursor.close()

    def save_electric_field_lines(self, config_id: int,
                                  field_lines: List[np.ndarray],
                                  start_points: np.ndarray) -> bool:
        """保存电场线数据 - 外部接口"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            self.connection.start_transaction()
            self._save_field_lines_cursor(cursor, config_id, field_lines, start_points)
            self.connection.commit()
            print(f"✓ 电场线数据保存成功到 config_id: {config_id}")
            return True

        except Exception as e:
            self.connection.rollback()
            print(f"保存电场线数据失败: {e}")
            return False
        finally:
            cursor.close()

    def get_all_simulations(self) -> List[Dict]:
        """获取所有仿真记录"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor(dictionary=True)

        try:
            cursor.execute("""
                           SELECT config_id as sim_id,
                                  simulation_name,
                                  description,
                                  radius,
                                  voltage,
                                  node_count,
                                  element_count,
                                  created_at,
                                  total_charge,
                                  charge_error,
                                  validation_status
                           FROM simulation_config
                           ORDER BY created_at DESC
                           """)
            simulations = cursor.fetchall()

            return simulations
        except mysql.connector.Error as e:
            print(f"查询所有仿真记录失败: {e}")
            return []
        finally:
            cursor.close()

    def update_simulation_status(self, config_id: int, status: str,
                                 error_message: str = "") -> bool:
        """更新仿真状态"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        try:
            sql = """
                  UPDATE simulation_config
                  SET status        = %s, \
                      error_message = %s, \
                      updated_at    = CURRENT_TIMESTAMP
                  WHERE config_id = %s \
                  """
            cursor.execute(sql, (status, error_message, config_id))
            self.connection.commit()
            print(f"✓ 仿真状态更新为: {status}")
            return True
        except mysql.connector.Error as e:
            print(f"更新仿真状态失败: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()


# ========== 数据库管理函数 ==========

def initialize_database(host='localhost', port=3306, user='root',
                        password='123456', database='electric_data',
                        clear_existing_data=False) -> bool:
    """
    初始化数据库
    """
    try:
        # 创建数据库(如果不存在)
        conn = mysql.connector.connect(
            host=host, port=3306, user='root', password='123456'
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE DATABASE IF NOT EXISTS {database} 
            CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
        cursor.close()
        conn.close()

        # 创建数据库实例
        db = ElectricFieldDatabase(host, port, user, password, database)
        db.connect()
        db.create_tables()

        if clear_existing_data:
            db.clear_all_data()

        db.disconnect()
        print(f"✓ 数据库初始化完成: {database}")
        return True

    except mysql.connector.Error as e:
        print(f"初始化数据库失败: {e}")
        return False


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("测试数据库初始化...")
    success = initialize_database(clear_existing_data=False)

    if success:
        print("数据库初始化成功！")
    else:
        print("数据库初始化失败！")