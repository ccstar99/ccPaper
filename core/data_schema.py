# core/data_schema.py
"""
核心数据契约定义模块
本模块定义了整个项目中所有模块必须遵守的统一数据格式。
任何模块间的数据交换都必须符合此处定义的契约。

设计原则：
1. 3D优先：所有空间坐标默认为3D，2D场景视为z=0的特例
2. 不可变性：字段一旦创建，不应修改（通过类型系统约束）
3. 类型安全：使用TypedDict和现代Python类型注解
4. 扩展性：通过metadata字段支持未来扩展
"""

from typing import TypedDict, Optional, Literal, Any
import numpy as np
from numpy.typing import NDArray


# ============================================================================ #
# 基础数据类型定义
# ============================================================================ #

class Charge(TypedDict):
    """
    电荷定义契约

    所有电荷必须统一为3D空间表示，2D计算时z坐标设为0.0。
    这消除了旧架构中2D/3D数据格式不一致的问题。

    示例：
    {
        'position': (0.0, 1.0, 0.0),  # x=0, y=1, z=0
        'value': 1.5e-9               # +1.5 nC
    }
    """
    position: tuple[float, float, float]
    value: float


class ModelParameters(TypedDict, total=False):
    """
    物理模型参数配置

    total=False表示所有字段可选，支持不同模型的灵活配置。
    各模型应只使用自己需要的字段。
    """
    # 通用参数
    model_type: Literal["point", "line", "ring", "disk", "bem_sphere"]
    dimension: Literal["2D", "3D"]

    # 几何参数
    radius: float  # 圆环/圆盘/球体半径
    length: float  # 线电荷长度
    divisions: int  # BEM网格细分等级

    # 电荷参数
    total_charge: float  # 总电荷量
    charge_density: float  # 电荷密度（线/面）
    voltage: float  # BEM边界电压

    # 位置参数
    center: tuple[float, float, float]  # 几何中心


# ============================================================================ #
# 核心解算结果契约
# ============================================================================ #

class FieldSolution(TypedDict):
    """
    场解算结果 - 核心数据契约

    所有物理模型的compute_field()方法必须返回此格式。
    确保visualization、ml、ui模块有统一的数据接口。

    字段规范：
    - points: 观察点坐标，必须是(N, 3)形状的np.ndarray
    - vectors: 电场向量，与points一一对应，(N, 3)
    - potentials: 可选的电位标量场，(N,)
    - charges: 源电荷列表，用于结果标注和反向查询
    - metadata: 任意元数据，用于记录计算信息
    """
    points: "NDArray[np.float64]"  # 形状: (N, 3)
    vectors: "NDArray[np.float64]"  # 形状: (N, 3)
    potentials: Optional["NDArray[np.float64]"]  # 形状: (N,)
    charges: list[Charge]  # 源电荷列表
    metadata: dict[str, Any]  # 计算元数据


class BEMSolution(TypedDict):
    """
    BEM专用解算结果

    由于TypedDict继承在某些类型检查器中支持不完善，
    这里显式重新定义所有字段以确保类型安全。
    """
    # FieldSolution 基础字段
    points: "NDArray[np.float64]"  # 形状: (N, 3)
    vectors: "NDArray[np.float64]"  # 形状: (N, 3)
    potentials: Optional["NDArray[np.float64]"]  # 形状: (N,)
    charges: list[Charge]  # 源电荷列表
    metadata: dict[str, Any]  # 计算元数据

    # BEM 特有字段
    vertices: "NDArray[np.float64]"  # 形状: (M, 3)
    triangles: "NDArray[np.int32]"  # 形状: (K, 3)
    vertex_potentials: "NDArray[np.float64]"  # 形状: (M,)
    surface_charges: Optional["NDArray[np.float64]"]  # 形状: (M,)


# ============================================================================ #
# 可视化输入契约
# ============================================================================ #

class VisualizationConfig(TypedDict, total=False):
    """
    可视化配置参数

    控制绘图行为的可选参数集合。
    """
    backend: Literal["matplotlib", "plotly"]  # 后端选择
    show_charges: bool  # 显示电荷
    show_vectors: bool  # 显示电场向量
    show_streamlines: bool  # 显示电场线（与show_field_lines功能相同）
    show_field_lines: bool  # 是否显示电场线（为兼容性添加）
    show_contours: bool  # 显示等势线
    colormap: str  # 颜色映射名称
    vector_scale: float  # 向量缩放因子
    resolution: int  # 采样分辨率


class AnimationConfig(TypedDict):
    """
    动画配置参数
    """
    frame_count: int  # 总帧数
    interval_ms: int  # 帧间隔（毫秒）
    repeat: bool  # 是否循环播放
    save_path: Optional[str]  # 保存路径


# ============================================================================ #
# 性能监控数据契约
# ============================================================================ #

class PerformanceMetrics(TypedDict):
    """
    性能监控指标

    用于performance模块记录和UI展示。
    """
    execution_time_s: float  # 执行时间（秒）
    memory_usage_mb: Optional[float]  # 内存使用（MB）
    n_points: int  # 计算点数
    n_charges: int  # 电荷数量
    timestamp: float  # 时间戳
    function_name: str  # 函数名


# ============================================================================ #
# 验证结果契约
# ============================================================================ #

class ValidationResult(TypedDict):
    """
    验证结果格式

    所有validate函数的标准返回值。
    """
    is_valid: bool
    message: str
    detail: Optional[dict[str, Any]]


# ============================================================================ #
# 工厂函数：创建标准空对象
# ============================================================================ #

def create_empty_solution(dimension: Literal["2D", "3D"] = "2D") -> FieldSolution:
    """
    创建空的场解对象，用于初始化或错误处理

    Args:
        dimension: "2D"或"3D"，决定points的第三列
    """
    return {
        'points': np.empty((0, 3), dtype=np.float64),
        'vectors': np.empty((0, 3), dtype=np.float64),
        'potentials': None,
        'charges': [],
        'metadata': {
            'dimension': dimension,
            'status': 'empty',
            'created_by': 'data_schema.create_empty_solution'
        }
    }


def create_bem_empty_solution() -> BEMSolution:
    """
    创建空的BEM解对象
    """
    # 显式创建所有字段，确保类型检查器能识别
    solution_dict = {
        'points': np.zeros((0, 3), dtype=np.float64),
        'vectors': np.zeros((0, 3), dtype=np.float64),
        'potentials': None,
        'charges': [],
        'metadata': {
            'dimension': '3D',
            'status': 'empty',
            'created_by': 'data_schema.create_bem_empty_solution'
        },
        'vertices': np.zeros((0, 3), dtype=np.float64),
        'triangles': np.zeros((0, 3), dtype=np.int32),
        'vertex_potentials': np.zeros(0, dtype=np.float64),
        'surface_charges': None
    }

    # 使用类型断言强制转换为 BEMSolution 类型
    from typing import cast
    return cast(BEMSolution, solution_dict)