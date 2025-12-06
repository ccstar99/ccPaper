# *RUNNING CC* 静电场可视化平台

## 项目概述

*RUNNING CC* 是一个基于Python开发的静电场可视化平台，采用现代化的Web界面和高性能计算引擎，支持多种电荷类型的电场计算与可视化。该平台旨在为物理教学和科研提供直观、高效的静电场分析工具。

### 核心特性

- **多电荷类型支持**：点电荷、无限长线电荷、圆环电荷
- **高性能计算**：基于NumPy的矢量化计算，支持大量电荷叠加
- **丰富的可视化**：2D/3D电场强度分布、电势分布、电场线轨迹
- **交互式界面**：基于Streamlit的Web应用，支持实时参数调整
- **动态主题**：根据时间自动切换主题风格
- **自适应电场线追踪**：智能调整步长的电场线生成算法

## 技术栈

- **核心计算**：Python 3.8+, NumPy, SciPy
- **可视化**：Matplotlib, Plotly
- **Web框架**：Streamlit

## 安装与运行

### 环境要求

- Python 3.8 或更高版本
- 支持的操作系统：Windows, macOS, Linux

### 快速启动（Windows）

1. 下载或克隆项目到本地
2. 双击运行 `run_app.bat` 脚本
3. 脚本将自动：
   - 检查Python环境
   - 创建并激活虚拟环境
   - 安装所有依赖
   - 启动Web应用
4. 浏览器将自动打开应用地址（默认：http://localhost:8506）

### 手动安装（所有平台）

1. 克隆或下载项目
2. 进入项目根目录
3. 创建虚拟环境：
   ```bash
   python -m venv venv
   ```
4. 激活虚拟环境：
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
5. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
6. 启动应用：
   ```bash
   python -m streamlit run ui/app.py
   ```

## 功能介绍

### 1. 电荷系统构建

平台支持三种基本电荷类型，可通过界面灵活配置：

- **点电荷**：可设置电荷量、位置、显示半径
- **无限长线电荷**：可设置线密度、位置
- **圆环电荷**：可设置电荷量、半径、位置

### 2. 电场计算

基于电场叠加原理，平台支持：

- 多电荷系统的总电场强度计算
- 电势分布计算
- 电场线轨迹追踪（自适应步长算法）

### 3. 可视化功能

- **2D电场强度分布**：热力图显示场强大小
- **3D电场矢量场**：箭头图显示场强方向和大小
- **电势等高线**：等值线图显示电势分布
- **电场线轨迹**：流线图显示电场线

### 4. 性能优化

- 矢量化计算：相比传统标量循环，性能提升显著
- 自适应算法：根据场强自动调整计算步长
- 内存优化：高效的数组操作，支持大规模计算

## 使用指南

### 基本操作流程

1. **启动应用**：运行 `run_app.bat` 或手动启动
2. **配置电荷**：在左侧面板添加和配置电荷
3. **选择计算类型**：电场强度、电势或电场线
4. **调整可视化参数**：分辨率、颜色映射、视图角度等
5. **查看结果**：在主面板查看可视化结果
6. **保存结果**：可下载图像或数据

### 高级功能

- **多视图切换**：2D/3D视图无缝切换
- **参数联动**：修改电荷参数后实时更新结果
- **性能监控**：查看计算时间和资源占用
- **主题切换**：自动或手动切换界面主题

## 核心算法说明

### 1. 电场叠加原理

```
E_total = Σ E_i
```

其中，`E_i` 是第i个电荷在空间某点产生的电场强度，遵循库仑定律：

```
E_i = k·q_i·r̂ / r²
```

- `k`：库仑常数（8.9875517923 × 10⁹ N·m²/C²）
- `q_i`：第i个电荷的电荷量
- `r`：电荷到计算点的距离
- `r̂`：从电荷指向计算点的单位矢量

### 2. 电势计算

电势遵循标量叠加原理：

```
V_total = Σ V_i
```

- 点电荷电势：`V = k·q / r`
- 线电荷电势：基于积分计算
- 圆环电荷电势：基于解析公式

### 3. 电场线追踪

采用自适应步长的欧拉积分算法：

1. 从起始点出发
2. 计算当前点的电场方向
3. 沿电场方向移动一小步（步长自适应调整）
4. 重复步骤2-3，直到达到终止条件

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

### 性能对比

平台内置了性能对比测试，展示矢量化计算的优势：

```python
from core.field_calculator import performance_comparison

# 运行性能对比测试
performance_comparison()
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

### 扩展可视化功能

1. 在 `ui/app.py` 中添加新的可视化组件
2. 利用Matplotlib或Plotly的API实现新的可视化效果
3. 集成到Streamlit应用的界面中

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

---

*RUNNING CC* - 让静电场可视化更简单、更高效！⚡
