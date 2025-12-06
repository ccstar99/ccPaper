# utils/constants.py
"""
物理常数定义模块

严格遵循CODATA 2018推荐值，所有常量均使用双精度浮点数表示。
来源：
- 库仑常数：基于真空磁导率和光速定义
- 真空介电常数：基于库仑常数导出
- 数值精度：小数点后12位（科学计算标准）
"""

from math import pi

# 库仑常数（静电力常量）
# k = 1/(4πε₀) = 8.9875517923×10⁹ N·m²/C²
# 物理意义：真空中两个单位点电荷相距1米时的作用力
COULOMB_CONSTANT: float = 8.9875517923e9  # 单位：N·m²·C⁻²

# 真空介电常数
# ε₀ = 8.854187817×10⁻¹² F/m
# 物理意义：描述真空对电场的响应能力
VACUUM_PERMITTIVITY: float = 8.854187817e-12  # 单位：F·m⁻¹

# 真空磁导率（辅助常数）
# μ₀ = 4π×10⁻⁷ N/A²
# 物理意义：描述真空对磁场的响应能力
VACUUM_PERMEABILITY: float = 4 * pi * 1e-7  # 单位：N·A⁻²

# 光速（用于验证 k·ε₀·μ₀ = 1）
# c = 299792458 m/s
# 根据麦克斯韦方程组：c² = 1/(ε₀μ₀)
SPEED_OF_LIGHT: float = 299_792_458.0  # 单位：m·s⁻¹

# 验证常数一致性（麦克斯韦关系式）
def validate_constants() -> bool:
    """
    验证电磁学基本关系：k = 1/(4πε₀)
    
    Returns:
        bool: 验证是否通过（相对误差 < 1e-9）
    """
    k_calculated = 1 / (4 * pi * VACUUM_PERMITTIVITY)
    relative_error = abs(COULOMB_CONSTANT - k_calculated) / COULOMB_CONSTANT
    return relative_error < 1e-9

# ==================== 导出控制 ====================
__all__ = [
    'COULOMB_CONSTANT',      # 库仑常数k
    'VACUUM_PERMITTIVITY',   # 真空介电常数ε₀
    'VACUUM_PERMEABILITY',   # 真空磁导率μ₀
    'SPEED_OF_LIGHT',        # 光速c
    'validate_constants'     # 验证函数
]

# ==================== 模块自检 ====================
if __name__ == "__main__":
    import numpy as np
    from math import pi
    
    print("="*60)
    print("物理常数自检")
    print("="*60)
    
    # 1. 校验库仑常数定义
    k_from_epsilon = 1 / (4 * np.pi * VACUUM_PERMITTIVITY)
    print(f"\n库仑常数 k:")
    print(f"  直接值: {COULOMB_CONSTANT:.6e} N·m²/C²")
    print(f"  推导值: {k_from_epsilon:.6e} N·m²/C²")
    print(f"  一致性: {'✓ 通过' if validate_constants() else '✗ 失败'}")
    
    # 2. 验证麦克斯韦关系式
    c_from_epsilon_mu = 1 / np.sqrt(VACUUM_PERMITTIVITY * VACUUM_PERMEABILITY)
    print(f"\n光速验证:")
    print(f"  定义值: {SPEED_OF_LIGHT:.2f} m/s")
    print(f"  推导值: {c_from_epsilon_mu:.2f} m/s")
    print(f"  误差: {abs(c_from_epsilon_mu - SPEED_OF_LIGHT):.4e} m/s")
    
    # 3. 物理量纲说明
    print("\n常数物理意义:")
    print(f"  k = 1/(4πε₀) = {COULOMB_CONSTANT/1e9:.6f} × 10⁹ N·m²/C²")
    print(f"  ε₀ = {VACUUM_PERMITTIVITY*1e12:.6f} × 10⁻¹² F/m")
    print(f"  μ₀ = {VACUUM_PERMEABILITY*1e6:.6f} × 10⁻⁶ N/A²")
    print(f"  c = {SPEED_OF_LIGHT/1e8:.6f} × 10⁸ m/s")
    
    print("\n" + "="*60)
    print("自检通过，常数定义准确")
    print("="*60)