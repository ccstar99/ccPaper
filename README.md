# *RUNNING CC* 可视化平台
## 项目概述
采用现代化的Web界面和高性能计算引擎，支持多种电荷类型的电场计算与可视化。该平台旨在为物理教学和科研提供直观、高效的静电场分析工具。
基于Python的静电场可视化平台，支持2D/3D电场强度、电势分布和电场线可视化，提供现代Web界面和高效计算。

## 核心功能
- **多电荷类型**：点电荷、无限长线电荷、圆环电荷、边界元法球型电荷
- **计算模块**：电场强度、电势、自适应步长电场线追踪
- **可视化**：2D/3D电场图、电势等高线、宇宙风格渲染
- 
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
