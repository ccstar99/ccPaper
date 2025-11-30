# configs/__init__.py
"""
宇宙配置加载系统

提供类型安全的配置访问，支持环境变量覆盖和用户自定义配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 配置存储
_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}

# 配置文件路径
_CONFIG_DIR = Path(__file__).parent
_CONFIG_FILES = {
    'physics': _CONFIG_DIR / 'physics.yaml',
    'ui': _CONFIG_DIR / 'ui.yaml',
    'engine': _CONFIG_DIR / 'engine.yaml'
}

# 用户配置目录
_USER_CONFIG_DIR = Path.home() / '.cosmic_field'
_USER_CONFIG_FILE = _USER_CONFIG_DIR / 'config.yaml'


@dataclass
class ConfigSource:
    """配置源信息"""
    name: str
    path: Path
    exists: bool
    type: str = "file"


class ConfigValidationError(Exception):
    """配置验证异常"""
    pass


def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        file_path: YAML文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML格式错误
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"配置文件为空: {file_path}")
            return {}

        logger.debug(f"配置已加载: {file_path.name}")
        return config

    except FileNotFoundError:
        logger.error(f"配置文件未找到: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {file_path} - {e}")
        raise


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典

    用override中的值覆盖base中的值，保留未覆盖的项

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    merged = base.copy()

    for key, value in override.items():
        if (key in merged and
                isinstance(merged[key], dict) and
                isinstance(value, dict)):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_config(config_type: str, reload: bool = False) -> Dict[str, Any]:
    """
    获取指定类型的配置

    Args:
        config_type: 配置类型 ('physics', 'ui', 'engine')
        reload: 是否强制重新加载（跳过缓存）

    Returns:
        配置字典

    Raises:
        ValueError: 配置类型不支持
        FileNotFoundError: 默认配置文件缺失
    """
    if config_type not in _CONFIG_FILES:
        raise ValueError(
            f"不支持的配置类型: {config_type}. "
            f"可选类型: {list(_CONFIG_FILES.keys())}"
        )

    # 从缓存返回（除非强制重载）
    if not reload and config_type in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_type]

    # 加载默认配置
    config_path = _CONFIG_FILES[config_type]
    if not config_path.exists():
        raise FileNotFoundError(
            f"默认配置文件缺失: {config_path}. "
            f"请检查安装完整性。"
        )

    config = load_yaml_config(config_path)

    # 合并用户自定义配置（如果存在）
    if _USER_CONFIG_FILE.exists():
        try:
            user_config_all = load_yaml_config(_USER_CONFIG_FILE)
            user_config = user_config_all.get(config_type, {})

            if user_config:
                config = merge_configs(config, user_config)
                logger.info(f"合并用户配置: {config_type}")
        except Exception as e:
            logger.warning(f"用户配置加载失败: {e}")

    # 应用环境变量覆盖（最高优先级）
    config = _apply_environment_overrides(config_type, config)

    # 验证配置
    _validate_configuration(config_type, config)

    # 缓存配置
    _CONFIG_CACHE[config_type] = config

    return config


def _apply_environment_overrides(config_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用环境变量覆盖

    环境变量命名规则：
    - COSMIC_FIELD_{TYPE}_{KEY}_{SUBKEY}=value

    例如：
    - COSMIC_FIELD_PHYSICS_CONSTANTS_COULOMB_CONSTANT=9.0e9
    - COSMIC_FIELD_UI_VISUALIZATION_BACKEND=matplotlib
    """
    prefix = f"COSMIC_FIELD_{config_type.upper()}_"
    overrides = {}

    for env_key, env_value in os.environ.items():
        if env_key.startswith(prefix):
            # 解析嵌套键
            key_parts = env_key[len(prefix):].lower().split('_')

            # 构建嵌套字典
            current = overrides
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # 转换值类型
            current[key_parts[-1]] = _parse_environment_value(env_value)

    if overrides:
        config = merge_configs(config, overrides)
        logger.info(f"应用环境变量覆盖: {config_type}")

    return config


def _parse_environment_value(value: str) -> Union[str, float, int, bool]:
    """
    转换环境变量字符串到适当类型

    优先级：
    1. bool (true/false/yes/no)
    2. int (纯数字)
    3. float (科学计数法或小数)
    4. str (原始字符串)
    """
    value_lower = value.lower().strip()

    # bool
    if value_lower in ('true', 'yes', '1'):
        return True
    if value_lower in ('false', 'no', '0'):
        return False

    # int
    try:
        return int(value)
    except ValueError:
        pass

    # float
    try:
        return float(value)
    except ValueError:
        pass

    # str
    return value


def _validate_configuration(config_type: str, config: Dict[str, Any]) -> None:
    """
    验证配置有效性

    根据配置类型执行不同的验证规则
    """
    validators = {
        'physics': _validate_physics_config,
        'ui': _validate_interface_config,
        'engine': _validate_engine_config
    }

    if validator := validators.get(config_type):
        validator(config)


def _validate_physics_config(config: Dict[str, Any]) -> None:
    """
    验证物理配置

    检查：
    - 基本物理常数为正数
    - 参数范围合理
    """
    constants = config.get('constants', {})

    # 检查必需常数
    required_constants = ['coulomb_constant', 'epsilon_0', 'mu_0']
    for const_name in required_constants:
        if const_name not in constants:
            raise ConfigValidationError(f"缺少必需物理常数: {const_name}")

        value = constants[const_name]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigValidationError(f"物理常数必须为正数: {const_name} = {value}")

    # 验证参数范围
    ranges = config.get('parameter_ranges', {})
    for param, bounds in ranges.items():
        if 'min' not in bounds or 'max' not in bounds:
            raise ConfigValidationError(f"参数范围缺少min/max: {param}")

        if bounds['min'] >= bounds['max']:
            raise ConfigValidationError(
                f"参数范围无效: {param}.min ({bounds['min']}) >= "
                f"{param}.max ({bounds['max']})"
            )


def _validate_interface_config(config: Dict[str, Any]) -> None:
    """
    验证界面配置

    检查：
    - 默认值在有效范围内
    - 颜色映射名称有效
    """
    computation = config.get('computation', {})
    visualization = config.get('visualization', {})

    # 验证网格大小
    grid_size = computation.get('default_grid_size', 50)
    if not isinstance(grid_size, int) or grid_size < 10 or grid_size > 200:
        raise ConfigValidationError(
            f"grid_size无效: {grid_size} (应为10-200的整数)"
        )

    # 验证后端
    backend = visualization.get('backend', 'plotly')
    supported_backends = ['matplotlib', 'plotly']
    if backend not in supported_backends:
        raise ConfigValidationError(
            f"不支持的后端: {backend}. 可选: {supported_backends}"
        )

    # 验证颜色映射
    colormap = visualization.get('colormap', 'viridis')
    valid_colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'hot', 'cool', 'coolwarm', 'bwr', 'seismic'
    ]
    if colormap not in valid_colormaps:
        logger.warning(f"未知的颜色映射: {colormap}")


def _validate_engine_config(config: Dict[str, Any]) -> None:
    """
    验证引擎配置

    检查：
    - ML参数有效性
    - 缓存设置合理
    """
    ml_config = config.get('machine_learning', {})

    if ml_config.get('enabled', False):
        # 验证加速器类型
        accelerator = ml_config.get('accelerator', 'idw')
        supported_accelerators = ['idw', 'rbf', 'surrogate']
        if accelerator not in supported_accelerators:
            raise ConfigValidationError(
                f"不支持的ML加速器: {accelerator}. "
                f"可选: {supported_accelerators}"
            )

        # 验证采样数
        n_samples = ml_config.get('training', {}).get('max_samples', 5000)
        if not isinstance(n_samples, int) or n_samples < 100:
            raise ConfigValidationError(
                f"训练点数无效: {n_samples} (应 >= 100)"
            )


def create_user_config() -> None:
    """
    创建默认用户配置文件

    在用户目录生成初始配置模板
    """
    _USER_CONFIG_DIR.mkdir(exist_ok=True)

    if _USER_CONFIG_FILE.exists():
        logger.info(f"用户配置已存在: {_USER_CONFIG_FILE}")
        return

    # 默认用户配置模板
    user_template = {
        'ui': {
            'visualization': {
                'backend': 'plotly',
                'colormap': 'viridis',
                'show_performance': True
            },
            'computation': {
                'default_grid_size': 60
            }
        },
        'engine': {
            'machine_learning': {
                'enabled': False,
                'accelerator': 'idw'
            }
        }
    }

    try:
        with open(_USER_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(user_template, f, allow_unicode=True, sort_keys=False)

        logger.info(f"已创建默认用户配置: {_USER_CONFIG_FILE}")
        print(f"用户配置文件已创建: {_USER_CONFIG_FILE}")
        print("你可以编辑此文件来自定义系统行为")

    except Exception as e:
        logger.error(f"创建用户配置失败: {e}")
        raise


def list_config_sources() -> List[ConfigSource]:
    """
    列出所有配置源

    Returns:
        配置源信息列表
    """
    sources = []

    # 默认配置源
    for name, path in _CONFIG_FILES.items():
        sources.append(ConfigSource(
            name=f"{name}_default",
            path=path,
            exists=path.exists(),
            type="file"
        ))

    # 用户配置源
    sources.append(ConfigSource(
        name="user_config",
        path=_USER_CONFIG_FILE,
        exists=_USER_CONFIG_FILE.exists(),
        type="file"
    ))

    # 环境变量源
    sources.append(ConfigSource(
        name="environment_variables",
        path=Path("COSMIC_FIELD_*"),
        exists=True,
        type="env"
    ))

    return sources


def clear_config_cache() -> None:
    """清除配置缓存"""
    _CONFIG_CACHE.clear()
    logger.debug("配置缓存已清除")


def get_config_value(config_type: str, key_path: str, default: Any = None) -> Any:
    """
    获取配置值

    Args:
        config_type: 配置类型
        key_path: 键路径，用点号分隔 (如 'constants.coulomb_constant')
        default: 默认值（如果键不存在）

    Returns:
        配置值
    """
    config = get_config(config_type)

    keys = key_path.split('.')
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def set_config_value(config_type: str, key_path: str, value: Any) -> None:
    """
    设置配置值（仅影响当前运行实例）

    Args:
        config_type: 配置类型
        key_path: 键路径，用点号分隔
        value: 要设置的值
    """
    if config_type not in _CONFIG_CACHE:
        get_config(config_type)  # 确保配置已加载

    config = _CONFIG_CACHE[config_type]
    keys = key_path.split('.')
    current = config

    # 遍历到最后一个键的父级
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # 设置值
    current[keys[-1]] = value


# ============================================================================ #
# 默认配置内容（内嵌，确保文件缺失时仍可运行）
# ============================================================================ #

def _ensure_default_configs() -> None:
    """确保默认配置文件存在"""
    for name, path in _CONFIG_FILES.items():
        if not path.exists():
            logger.warning(f"默认配置缺失: {path.name}，正在创建...")

            # 根据名称选择默认内容
            if name == 'physics':
                content = _DEFAULT_PHYSICS_CONFIG
            elif name == 'ui':
                content = _DEFAULT_UI_CONFIG
            elif name == 'engine':
                content = _DEFAULT_ENGINE_CONFIG
            else:
                continue

            try:
                path.parent.mkdir(exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"已创建默认配置: {path.name}")
            except Exception as e:
                logger.error(f"创建配置失败: {e}")


_DEFAULT_PHYSICS_CONFIG = """# 物理宇宙常数配置

constants:
  coulomb_constant: 8.99e9        # 库仑力常数 (N·m²/C²)
  epsilon_0: 8.854187817e-12      # 真空介电常数 (F/m)  
  mu_0: 1.2566370614e-6           # 真空磁导率 (H/m)
  speed_of_light: 2.99792458e8    # 光速 (m/s)

parameter_ranges:
  charge_value: {min: -1e-6, max: 1e-6, unit: C}
  line_charge_density: {min: -1e-6, max: 1e-6, unit: C/m}
  ring_charge: {min: -1e-6, max: 1e-6, unit: C}
  radius: {min: 1e-6, max: 1000.0, unit: m}
  length: {min: 1e-6, max: 1000.0, unit: m}
  voltage: {min: -1e6, max: 1e6, unit: V}

tolerances:
  singular_distance: 1e-12
  convergence: 1e-6
  max_field_threshold: 1e15

integration:
  default_points: 50
  gauss_legendre_order: 7
  singular_subdivision: 8
"""

_DEFAULT_UI_CONFIG = """# 星云界面配置

app:
  title: "宇宙静电场仿真"
  layout: "wide"
  initial_sidebar_state: "expanded"

computation:
  default_grid_size: 50
  min_grid_size: 10
  max_grid_size: 200
  grid_step: 10
  default_charges: 1
  min_charges: 1
  max_charges: 20

visualization:
  backend: "plotly"
  show_vectors: true
  show_contours: true
  show_streamlines: false
  show_charges: true
  colormap: "viridis"
  vector_scale: 1.0

performance:
  warn_time_threshold: 5.0
  max_memory_mb: 2048
  cache_ttl_seconds: 1800
"""

_DEFAULT_ENGINE_CONFIG = """# 量子引擎配置

engine:
  validation_level: "strict"
  cache:
    enabled: true
    max_size_mb: 512
    ttl_seconds: 1800

machine_learning:
  enabled: false
  accelerator: "idw"
  idw: {n_neighbors: 8, power: 2.0, min_distance: 1e-9}
  rbf: {kernel: "thin_plate_spline", epsilon: 1.0, smoothing: 0.0}
  surrogate: {max_training_points: 5000, default_kernel: "rbf", refinement_threshold: 0.1}
  training: {min_samples: 100, max_samples: 5000, adaptive_sampling: true, high_gradient_boost: 2.0}

parallel:
  enabled: false
  max_workers: 4
  backend: "threading"

bem:
  default_divisions: 2
  min_triangle_quality: 0.1
  gauss_points: 7
  singular_tolerance: 1e-12
"""

# 初始化时检查配置
_ensure_default_configs()


# ============================================================================ #
# 单元测试
# ============================================================================ #

def test_config_loading() -> None:
    """测试配置加载"""
    # 加载所有配置
    physics = get_config('physics')
    ui = get_config('ui')
    engine = get_config('engine')

    assert 'constants' in physics
    assert 'coulomb_constant' in physics['constants']
    assert physics['constants']['coulomb_constant'] == 8.99e9

    assert 'visualization' in ui
    assert 'backend' in ui['visualization']

    assert 'machine_learning' in engine

    logger.info("配置加载测试通过")


def test_config_merge() -> None:
    """测试配置合并"""
    base = {'a': {'b': 1, 'c': 2}, 'd': 3}
    override = {'a': {'b': 10}, 'd': 4}

    merged = merge_configs(base, override)

    assert merged['a']['b'] == 10  # 被覆盖
    assert merged['a']['c'] == 2  # 保留
    assert merged['d'] == 4  # 被覆盖

    logger.info("配置合并测试通过")


def test_environment_override() -> None:
    """测试环境变量覆盖"""
    import os

    # 设置测试环境变量
    os.environ['COSMIC_FIELD_PHYSICS_CONSTANTS_COULOMB_CONSTANT'] = '9.0e9'

    # 重新加载物理配置
    if 'physics' in _CONFIG_CACHE:
        del _CONFIG_CACHE['physics']

    physics = get_config('physics', reload=True)

    # 检查值是否被覆盖
    assert physics['constants']['coulomb_constant'] == 9.0e9

    # 清理
    del os.environ['COSMIC_FIELD_PHYSICS_CONSTANTS_COULOMB_CONSTANT']
    del _CONFIG_CACHE['physics']

    logger.info("环境变量覆盖测试通过")


def test_config_validation() -> None:
    """测试配置验证"""
    # 测试无效物理配置
    invalid_physics = {
        'constants': {
            'coulomb_constant': -1  # 负数，无效
        }
    }

    try:
        _validate_physics_config(invalid_physics)
        assert False, "应抛出ConfigValidationError"
    except ConfigValidationError as e:
        assert "正数" in str(e)

    # 测试无效UI配置
    invalid_ui = {
        'computation': {
            'default_grid_size': 5  # 小于最小值
        }
    }

    try:
        _validate_interface_config(invalid_ui)
        assert False, "应抛出ConfigValidationError"
    except ConfigValidationError as e:
        assert "grid_size无效" in str(e)

    logger.info("配置验证测试通过")


def test_user_config_creation() -> None:
    """测试用户配置创建"""
    # 删除用户配置（如果存在）
    if _USER_CONFIG_FILE.exists():
        _USER_CONFIG_FILE.unlink()

    # 创建默认配置
    create_user_config()

    assert _USER_CONFIG_FILE.exists()

    # 验证内容
    user_config = load_yaml_config(_USER_CONFIG_FILE)
    assert 'ui' in user_config

    logger.info("用户配置创建测试通过")


def test_config_value_access() -> None:
    """测试配置值访问"""
    value = get_config_value('physics', 'constants.coulomb_constant')
    assert value == 8.99e9

    # 测试默认值
    value = get_config_value('physics', 'nonexistent.key', 'default')
    assert value == 'default'

    logger.info("配置值访问测试通过")


def run_all_tests() -> None:
    """运行所有配置模块测试"""
    logger.info("开始运行配置模块单元测试")

    test_config_loading()
    test_config_merge()
    test_environment_override()
    test_config_validation()
    test_user_config_creation()
    test_config_value_access()

    logger.info("所有配置模块单元测试通过")


def display_config_sources() -> None:
    """显示所有配置源信息"""
    print("=" * 60)
    print("宇宙场配置系统")
    print("=" * 60)

    sources = list_config_sources()

    print("\n配置源:")
    for source in sources:
        status = "存在" if source.exists else "缺失"
        print(f"  • {source.name}: {source.path} [{status}]")

    print(f"\n环境变量覆盖:")
    print(f"  前缀: COSMIC_FIELD_*")
    print(f"  示例: COSMIC_FIELD_PHYSICS_CONSTANTS_COULOMB_CONSTANT=9.0e9")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_all_tests()
    elif "--list" in sys.argv:
        display_config_sources()
    elif "--create-user-config" in sys.argv:
        create_user_config()
    else:
        print(__doc__)
        print("\n运行选项:")
        print("  python configs/__init__.py --test           # 运行单元测试")
        print("  python configs/__init__.py --list           # 列出配置源")
        print("  python configs/__init__.py --create-user-config  # 创建用户配置")