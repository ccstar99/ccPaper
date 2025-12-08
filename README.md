# *RUNNING CC* 静电场可视化平台
## 项目概述
*RUNNING CC* 是一个基于Python开发的静电场可视化平台，采用现代化的Web界面和高性能计算引擎，支持多种电荷类型的电场计算与可视化。该平台旨在为物理教学和科研提供直观、高效的静电场分析工具。

# 静电场可视化平台

## 项目简介
基于Python的静电场可视化平台，支持2D/3D电场强度、电势分布和电场线可视化，提供现代Web界面和高效计算。

## 核心功能
- **多电荷类型**：点电荷、无限长线电荷、圆环电荷
- **计算模块**：电场强度、电势、自适应步长电场线追踪
- **可视化**：2D/3D电场图、电势等高线、宇宙风格渲染
- **交互界面**：实时参数调整、动态主题切换

## 技术栈
- **计算**：NumPy, SciPy
- **可视化**：Matplotlib, Plotly, PyVista
- **Web框架**：Streamlit

## 安装运行
### 快速启动
1. 运行 `run_app.bat` (Windows)
2. 或执行：`python -m streamlit run ui/app.py`

### 依赖安装
```bash
pip install numpy scipy matplotlib streamlit plotly pyvista
## 项目结构

```
├── core/                    # 核心计算模块
│   ├── field_calculator.py  # 电场强度计算器
│   ├── potential_calculator.py  # 电势计算器
│   └── field_line_tracer.py  # 电场线追踪器
├── physics/                 # 物理模型
│   ├── point.py             # 点电荷模型
│   ├── line.py              # 线电荷模型
│   └── ring.py              # 圆环电荷模型
├── ui/                      # 用户界面
│   └── app.py               # Streamlit应用主程序
├── utils/                   # 工具函数
│   ├── constants.py         # 物理常数
│   └── geometry.py          # 几何计算工具
├── requirements.txt         # 项目依赖
└── run_app.bat             # 启动脚本
```

## 代码示例
### 基本使用示例
```python
from core.field_calculator import FieldCalculator
from physics.point import PointCharge
import numpy as np

# 创建电荷系统
charges = [
    PointCharge(q=1e-6, position=[0, 0, 0]),  # 正电荷
    PointCharge(q=-1e-6, position=[1, 0, 0])   # 负电荷
]

# 初始化场计算器
calculator = FieldCalculator(charges)
# 计算电场强度
test_point = np.array([0.5, 0, 0])
E = calculator.electric_field(test_point)
print(f"电场强度: {E}")
print(f"场强大小: {np.linalg.norm(E):.6e} N/C")
```

## 扩展开发

### 添加新的电荷类型

1. 在 `physics/` 目录下创建新的电荷模型文件
2. 实现 `ChargeProtocol` 协议接口：
   ```python
   class NewChargeType:
       def electric_field(self, points: np.ndarray) -> np.ndarray:
           # 实现电场计算逻辑
           pass
       
       @property
       def q(self) -> float:
           # 返回电荷量
           pass
   ```
3. 在 `field_calculator.py` 中导入并使用新的电荷类型

## 技术亮点

1. **现代化Python开发**：
   - 使用Type Hints提高代码可读性和可维护性
   - 采用Protocol类实现面向接口编程
   - 利用NumPy的矢量化计算提升性能

2. **模块化设计**：
   - 清晰的分层架构
   - 低耦合的组件设计
   - 易于扩展和维护

3. **用户友好界面**：
   - 直观的Web界面
   - 实时交互反馈
   - 动态主题切换

4. **高性能计算**：
   - 矢量化计算避免Python循环开销
   - 自适应算法优化计算效率
   - 内存高效的数组操作

## 应用场景
- **物理教学**：直观展示电场分布和叠加原理
- **科研分析**：快速构建和分析复杂电荷系统
- **工程设计**：电场仿真和优化
- **学习研究**：理解静电场的基本概念和规律

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 邮箱：[2409185982qq.com]

## 更新日志

### v1.0.0 (2025-12-6)
- 初始版本发布
- 支持点电荷、线电荷、圆环电荷
- 实现2D/3D可视化
- 自适应电场线追踪算法
- 基于Streamlit的Web界面
