# ui/app.py
"""
*RUNNING CC* 静电场可视化平台
保留所有祝福语、动态主题和UI设计，集成新的核心算法
"""
import streamlit as st
import numpy as np
import sys
import os
import logging
import time
from datetime import datetime
import traceback
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from core.field_calculator import FieldCalculator
from core.potential_calculator import PotentialCalculator
from core.field_line_tracer import AdaptiveFieldLineTracer
import sys
import os

from physics.point import PointCharge
from physics.line import LineCharge
from physics.ring import RingCharge

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置（必须在任何Streamlit代码之前）
st.set_page_config(
    page_title="*RUNNING CC* ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ElectroFieldApp:
    """主应用控制器"""
    
    def __init__(self):
        self._initialize_session_state()
        # 暂时不初始化可视化器，在需要时根据电荷列表初始化
        self.visualizer = None
        
    def _initialize_session_state(self):
        """初始化会话状态"""
        defaults = {
            'current_solution': None,
            'performance_history': [],
            'ui_config': {
                'theme': self._get_current_theme(),
                'last_model': 'point_charge',
                'current_view': 'main_viz'
            },
            'user_prefs': {
                'show_tutorial': True,
                'animation_speed': 1.0
            },
            'blessing_index': 0,
            'last_blessing_update': time.time()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _get_current_theme(self) -> str:
        """动态主题"""
        hour = datetime.now().hour
        if 6 <= hour < 12: return "morning"
        elif 12 <= hour < 18: return "daylight"
        elif 18 <= hour < 22: return "evening"
        else: return "night"
    
    def _get_seasonal_blessings(self) -> list:
        """保留您的完整祝福语系统"""
        now = datetime.now()
        month, hour = now.month, now.hour
        
        # 季节判断
        if month in [12, 1, 2]: season = "冬"
        elif month in [3, 4, 5]: season = "春"
        elif month in [6, 7, 8]: season = "夏"
        else: season = "秋"
        
        # 时间段判断
        if 5 <= hour < 8:
            base = [
                f"晨光熹微，{season}日静电场探索开始",
                f"{season}晨清爽，电场计算正当时",
                "朝霞映照，电磁奥秘待你发现",
                f"{season}晨微风，电场研究启新程"
            ]
        elif 8 <= hour < 12:
            base = [
                f"{season}日上午好，电场仿真之旅启程",
                f"阳光{season}日，静电场分析正当时",
                f"{season}日明媚，电磁世界任你遨游",
                f"{season}日温暖，电场计算更精准"
            ]
        elif 12 <= hour < 14:
            base = [f"{season}日午安，电场计算伴你同行", "午间时光，探索电磁场的奇妙"]
        elif 14 <= hour < 18:
            base = [f"{season}日下午好，静电场研究继续", f"{season}日斜阳，电场分析正深入"]
        elif 18 <= hour < 22:
            base = [f"{season}日傍晚，在暮色中研究电磁场", "暮色降临，电场计算更显深邃"]
        else:
            base = ["深夜静谧，在星空下研究电磁奥秘", "万籁俱寂，电场计算正当时"]
        
        # 特殊季节祝福
        if season == "秋":
            base.extend(["秋风送爽，静电场研究正当时", "秋叶飘零，电磁场中寻规律"])
        
        # 通用祝福
        base.extend(["静电场中探真理，电磁世界任遨游", "电场分布显规律，电磁研究正深入"])
        return base
    
    def _get_current_blessing(self) -> str:
        """获取当前祝福语"""
        current_time = time.time()
        if current_time - st.session_state['last_blessing_update'] > 10:
            blessings = self._get_seasonal_blessings()
            idx = st.session_state['blessing_index'] + 1
            st.session_state['blessing_index'] = idx % len(blessings)
            st.session_state['last_blessing_update'] = current_time
            return blessings[idx % len(blessings)]
        else:
            blessings = self._get_seasonal_blessings()
            return blessings[st.session_state['blessing_index'] % len(blessings)]
    
    def render_sidebar(self) -> Dict[str, Any]:
        """渲染侧边栏"""
        with st.sidebar:
            st.markdown("""
            <style>
            .sidebar-title {
                font-size: 1.8rem; font-weight: bold;
                background: linear-gradient(45deg, #8B5CF6, #6366F1);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="sidebar-title"><em>RUNNING CC</em></div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; color: #6B7280;"><em>RUNNING ccElectrons</em></div>', unsafe_allow_html=True)
            
            now = datetime.now()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; 
                        border-radius: 10px; color: white; text-align: center;">
                <div>{now.strftime('%H:%M:%S')}</div>
                <div>{now.strftime('%Y-%m-%d')}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
            # 模型选择
            st.subheader("物理模型")
            model_type = st.selectbox(
                "选择仿真模型",
                ["point_charge", "line_charge", "ring_charge", "dipole"],
                format_func=lambda x: {
                    'point_charge': '点电荷模型',
                    'line_charge': '线电荷模型',
                    'ring_charge': '圆环电荷模型',
                    'dipole': '电偶极子'
                }[x],
                index=0
            )
            
            # 参数配置
            st.subheader("模型参数")
            params = self._render_parameter_panel(model_type)
            
            # 可视化设置
            st.subheader("可视化配置")
            viz_config = self._render_visualization_settings()
            
            # 计算按钮
            st.markdown("---")
            calculate_btn = st.button("开始计算", type="primary", width='stretch')
            
            return {
                'model_type': model_type,
                **params,
                'calculate_requested': calculate_btn,
                'viz_config': viz_config
            }
    
    def _render_parameter_panel(self, model_type: str) -> Dict[str, Any]:
        """参数面板"""
        params = {}
        
        if model_type == 'point_charge':
            n_charges = st.slider("电荷数量", 1, 5, 2)
            charges = []
            for i in range(n_charges):
                with st.expander(f"电荷 {i+1}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        x = st.slider(f"X (m)", -3.0, 3.0, -1.0 + i*2.0, key=f"q{i}_x")
                        y = st.slider(f"Y (m)", -3.0, 3.0, 0.0, key=f"q{i}_y")
                        z = st.slider(f"Z (m)", -3.0, 3.0, 0.0, key=f"q{i}_z")
                    with col2:
                        q = st.number_input(f"电量 (C)", value=1e-9 * (-1 if i%2 else 1), format="%.2e", key=f"q{i}_val")
                    charges.append({'position': (x, y, z), 'value': q})
            params['charges'] = charges
            
        elif model_type == 'line_charge':
            col1, col2 = st.columns(2)
            with col1:
                params['lambda_val'] = st.number_input("线电荷密度 (C/m)", value=1e-9, format="%.2e")
            with col2:
                params['position'] = (st.slider("位置 X", -2.0, 2.0, 0.0), st.slider("位置 Y", -2.0, 2.0, 0.0))
            
        elif model_type == 'ring_charge':
            col1, col2 = st.columns(2)
            with col1:
                params['q'] = st.number_input("总电荷量 (C)", value=1e-6, format="%.2e")
                params['radius'] = st.slider("圆环半径 (m)", 0.5, 3.0, 1.0)
            with col2:
                params['position'] = (st.slider("中心 X", -2.0, 2.0, 0.0), st.slider("中心 Y", -2.0, 2.0, 0.0), 0.0)
            
        elif model_type == 'dipole':
            col1, col2 = st.columns(2)
            with col1:
                separation = st.slider("偶极间距 (m)", 0.1, 2.0, 1.0)
                q = st.number_input("电荷大小 (C)", value=1e-9, format="%.2e")
            with col2:
                orientation = st.selectbox("方向", ['horizontal', 'vertical'])
            
            if orientation == 'horizontal':
                charges = [
                    {'position': (-separation/2, 0, 0), 'value': q},
                    {'position': (separation/2, 0, 0), 'value': -q}
                ]
            else:
                charges = [
                    {'position': (0, -separation/2, 0), 'value': q},
                    {'position': (0, separation/2, 0), 'value': -q}
                ]
            params['charges'] = charges
        
        # 网格分辨率
        params['grid_size'] = st.slider("网格分辨率", 40, 150, 80, step=10)
        params['boundary'] = st.slider("计算边界", 2.0, 10.0, 5.0)
        
        return params
    
    def _render_visualization_settings(self) -> Dict[str, Any]:
        """可视化设置"""
        return {
            'n_field_lines': st.slider("电场线数量", 60, 200, 120),
            'show_arrows': st.checkbox("显示箭头", value=True),
            'arrow_spacing': st.slider("箭头间隔", 5, 30, 10)
        }
    
    def render_main_content(self, params: Dict[str, Any]):
        """主内容区"""
        # 动态祝福语
        st.markdown(f"""
        <div style="font-style: italic; color: #8B5CF6; text-align: center; margin-bottom: 1rem;">
            {self._get_current_blessing()}
        </div>
        """, unsafe_allow_html=True)
        
        # 标题
        st.markdown("""
        <style>
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        .main-title {
            font-size: 2.8rem; font-weight: bold;
            background: linear-gradient(45deg, #8B5CF6, #6366F1, #EC4899, #F97316, #F59E0B, #10B981, #3B82F6);
            background-size: 400% 400%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradient-animation 20s ease infinite;
        }
        .subtitle {
            font-size: 2.2rem; /* 比主标题小一号 */
            background: linear-gradient(45deg, #8B5CF6, #6366F1, #EC4899, #F97316, #F59E0B, #10B981, #3B82F6);
            background-size: 400% 400%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradient-animation 20s ease infinite;
        }
        .title-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            margin-bottom: 1rem;
        }
        .github-link {
            color: #8B5CF6;
            text-decoration: none;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }
        </style>
        <div class="title-container">
            <div><em class="main-title">RUNNING CC</em> <span class="subtitle">静电场仿真</span></div>
            <a href="https://github.com/ccstar99/ccPaper" target="_blank" class="github-link">
                cc's GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        if not params['calculate_requested']:
            st.info("请在左侧配置参数，点击“开始计算”")
            return
        
        # 执行计算
        try:
            with st.spinner("正在计算电场分布..."):
                start_time = time.time()
                
                # 1. 构建电荷对象
                charges = self._build_charges(params)
                
                # 2. 生成观测点
                observation_points = self._generate_grid(params)
                
                # 3. 计算电场和电势
                field_calc = FieldCalculator(charges)
                potential_calc = PotentialCalculator(charges)
                
                E = field_calc.electric_field(observation_points)
                V = potential_calc.potential(observation_points)
                
                # 4. 追踪电场线
                tracer = AdaptiveFieldLineTracer(charges, dim=3, max_iter=15000)
                field_lines = tracer.trace_all_field_lines()
                
                # 5. 渲染结果
                self._render_results(charges, observation_points, E, V, field_lines, params)
                
                # 性能记录
                compute_time = time.time() - start_time
                st.session_state['performance_history'].append({
                    'time': compute_time,
                    'n_lines': len(field_lines),
                    'n_points': len(observation_points)
                })
                st.success(f"计算完成！耗时 {compute_time:.2f}秒")
                
        except Exception as e:
            st.error(f"计算失败: {str(e)}")
            st.code(traceback.format_exc())
    
    def _build_charges(self, params: Dict[str, Any]) -> List:
        """构建电荷对象"""
        charges = []
        model_type = params['model_type']
        
        if model_type == 'point_charge' or model_type == 'dipole':
            for c in params['charges']:
                q, pos = c['value'], c['position']
                charges.append(PointCharge(q=q, position=pos, radius=max(0.05, abs(q)*1e5)))
                
        elif model_type == 'line_charge':
            pos = params['position']
            charges.append(LineCharge(lambda_val=params['lambda_val'], 
                                    position=pos, radius=0.05))
            
        elif model_type == 'ring_charge':
            pos = params['position']
            charges.append(RingCharge(q=params['q'], radius=params['radius'], 
                                    center_position=pos))
        
        return charges
    
    def _generate_grid(self, params: Dict[str, Any]) -> np.ndarray:
        """生成观测网格"""
        boundary = params['boundary']
        grid_size = params['grid_size']
        
        x = np.linspace(-boundary, boundary, grid_size)
        y = np.linspace(-boundary, boundary, grid_size)
        X, Y = np.meshgrid(x, y)
        
        return np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
    def _render_results(self, charges: List, points: np.ndarray, 
                       E: np.ndarray, V: np.ndarray, 
                       field_lines: List[np.ndarray], params: Dict[str, Any]):
        """渲染结果"""
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import numpy as np
        
        # 标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "电场线2D", "电场线3D", "电荷分布", "电势3D"
        ])
        
        with tab1:
            st.subheader("电场线2D")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制电场线（从3D转为2D）
            for line in field_lines:
                # 只取X-Y平面
                line_2d = line[:, :2]
                ax.plot(line_2d[:, 0], line_2d[:, 1], linewidth=0.5, alpha=0.7)
                
                # 添加箭头
                if params['viz_config']['show_arrows'] and len(line_2d) >= 20:
                    import matplotlib.patches as mpatches
                    import numpy as np
                    
                    # 根据线条长度动态确定箭头数量
                    arrow_spacing = max(15, len(line_2d) // 4)
                    arrow_indices = range(arrow_spacing, len(line_2d) - arrow_spacing, arrow_spacing)
                    
                    for idx in arrow_indices:
                        if idx >= len(line_2d)-1 or idx <= 0:
                            continue
                        
                        # 计算箭头位置和方向
                        pos = line_2d[idx]
                        
                        # 使用前向差分计算切线方向（电场方向）
                        if idx < len(line_2d) - 1:
                            direction = line_2d[idx+1] - line_2d[idx]
                        else:
                            direction = line_2d[idx] - line_2d[idx-1]
                        
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm == 0:
                            continue
                            
                        direction = direction / direction_norm
                        
                        # 创建小箭头
                        arrow = mpatches.FancyArrowPatch(
                            pos - direction * 0.08,  # 箭头起点稍后退
                            pos + direction * 0.02,  # 箭头终点稍前进
                            arrowstyle='->',  # 标准指向型箭头
                            mutation_scale=8,  # 箭头尺寸
                            mutation_aspect=1.5,  # 保持箭头比例
                            color='black',
                            linewidth=0.5,  # 箭头线宽
                            zorder=3,
                            alpha=0.9  # 稍微透明
                        )
                        ax.add_patch(arrow)
            
            # 绘制电荷
            for charge in charges:
                # 根据电荷类型获取电荷量符号
                if hasattr(charge, 'q'):
                    is_positive = charge.q > 0
                elif hasattr(charge, 'lambda_val'):
                    is_positive = charge.lambda_val > 0
                else:
                    is_positive = True
                color = 'red' if is_positive else 'blue'
                
                if hasattr(charge, 'get_ring_visualization_points'):
                    # 圆环电荷：绘制圆圈
                    ring_points = charge.get_ring_visualization_points()
                    ax.plot(ring_points[:, 0], ring_points[:, 1], 
                            linewidth=2, c=color)
                elif hasattr(charge, 'get_line_axis_points'):
                    # 线电荷：绘制线段
                    line_points = charge.get_line_axis_points()
                    ax.plot(line_points[:, 0], line_points[:, 1], 
                            linewidth=2, c=color)
                else:
                    # 点电荷：绘制点
                    ax.scatter(charge.position[0], charge.position[1], 
                              s=100, c=color, edgecolors='black')
            
            ax.set_aspect('equal')
            ax.set_xlim(-params['boundary'], params['boundary'])
            ax.set_ylim(-params['boundary'], params['boundary'])
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('2D 电场线分布')
            st.pyplot(fig, width='stretch')
        
        with tab2:
            st.subheader("电场线3D ")
            fig = go.Figure()
            
            # 绘制电场线
            import numpy as np
            apple_colors = ['#ff3b30', '#007aff', '#8e8e93', '#ff9500', '#34c759']
            for line in field_lines:
                # 随机选择苹果风格颜色
                color = np.random.choice(apple_colors)
                fig.add_trace(go.Scatter3d(
                    x=line[:, 0],
                    y=line[:, 1],
                    z=line[:, 2],
                    mode='lines',
                    line=dict(width=1, color=color),
                    opacity=0.7
                ))
            
            # 绘制电荷
            for charge in charges:
                # 根据电荷类型获取电荷量符号
                if hasattr(charge, 'q'):
                    is_positive = charge.q > 0
                elif hasattr(charge, 'lambda_val'):
                    is_positive = charge.lambda_val > 0
                else:
                    is_positive = True
                color = 'red' if is_positive else 'blue'
                
                # 确保位置是3D的，2D电荷默认z=0
                pos = charge.position
                z_val = pos[2] if len(pos) > 2 else 0.0
                
                # 根据电荷类型绘制不同的可视化
                if hasattr(charge, 'get_line_axis_points'):
                    # 线电荷：绘制线段
                    line_points = charge.get_line_axis_points()
                    fig.add_trace(go.Scatter3d(
                        x=line_points[:, 0],
                        y=line_points[:, 1],
                        z=line_points[:, 2],
                        mode='lines',
                        line=dict(width=4, color=color),
                        opacity=0.8,
                        name=f'线电荷 ({charge.lambda_val:.2e} C/m)'
                    ))
                elif hasattr(charge, 'get_ring_visualization_points'):
                    # 圆环电荷：绘制圆圈
                    ring_points = charge.get_ring_visualization_points()
                    fig.add_trace(go.Scatter3d(
                        x=ring_points[:, 0],
                        y=ring_points[:, 1],
                        z=ring_points[:, 2],
                        mode='lines',
                        line=dict(width=4, color=color),
                        opacity=0.8,
                        name=f'圆环电荷 ({charge.q:.2e} C)'
                    ))
                else:
                    # 点电荷：绘制点
                    fig.add_trace(go.Scatter3d(
                        x=[pos[0]],
                        y=[pos[1]],
                        z=[z_val],
                        mode='markers',
                        marker=dict(size=8, color=color, opacity=1)
                    ))
            
            boundary = params['boundary']
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-boundary, boundary]),
                    yaxis=dict(range=[-boundary, boundary]),
                    zaxis=dict(range=[-boundary, boundary]),
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)'
                ),
                title='3D 电场线分布'
            )
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader("电荷分布")
            fig = go.Figure()
            
            # 绘制电荷
            for charge in charges:
                # 根据电荷类型获取电荷量符号
                if hasattr(charge, 'q'):
                    is_positive = charge.q > 0
                elif hasattr(charge, 'lambda_val'):
                    is_positive = charge.lambda_val > 0
                else:
                    is_positive = True
                color = 'red' if is_positive else 'blue'
                
                # 确保位置是3D的，2D电荷默认z=0
                pos = charge.position
                z_val = pos[2] if len(pos) > 2 else 0.0
                
                # 根据电荷类型绘制不同的可视化
                if hasattr(charge, 'get_line_axis_points'):
                    # 线电荷：绘制线段
                    line_points = charge.get_line_axis_points()
                    fig.add_trace(go.Scatter3d(
                        x=line_points[:, 0],
                        y=line_points[:, 1],
                        z=line_points[:, 2],
                        mode='lines',
                        line=dict(width=6, color=color),
                        opacity=1,
                        name=f'线电荷 ({charge.lambda_val:.2e} C/m)'
                    ))
                elif hasattr(charge, 'get_ring_visualization_points'):
                    # 圆环电荷：绘制圆圈
                    ring_points = charge.get_ring_visualization_points()
                    fig.add_trace(go.Scatter3d(
                        x=ring_points[:, 0],
                        y=ring_points[:, 1],
                        z=ring_points[:, 2],
                        mode='lines',
                        line=dict(width=6, color=color),
                        opacity=1,
                        name=f'圆环电荷 ({charge.q:.2e} C)'
                    ))
                else:
                    # 点电荷：绘制点
                    fig.add_trace(go.Scatter3d(
                        x=[pos[0]],
                        y=[pos[1]],
                        z=[z_val],
                        mode='markers',
                        marker=dict(size=12, color=color, opacity=1)
                    ))
            
            boundary = params['boundary']
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-boundary, boundary]),
                    yaxis=dict(range=[-boundary, boundary]),
                    zaxis=dict(range=[-boundary, boundary]),
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)'
                ),
                title='电荷分布'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("电势3D")
            # 重塑电势数据为网格
            grid_size = params['grid_size']
            V_grid = V.reshape(grid_size, grid_size)
            x = np.linspace(-params['boundary'], params['boundary'], grid_size)
            y = np.linspace(-params['boundary'], params['boundary'], grid_size)
            X, Y = np.meshgrid(x, y)
            
            fig = go.Figure(data=[go.Surface(
                z=V_grid,
                x=X,
                y=Y,
                colorscale='viridis',
                opacity=0.8
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='电势 (V)',
                    xaxis=dict(range=[-params['boundary'], params['boundary']]),
                    yaxis=dict(range=[-params['boundary'], params['boundary']])
                ),
                title='2D 电势分布 (3D 表面图)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """运行应用"""
        params = self.render_sidebar()
        self.render_main_content(params)


if __name__ == "__main__":
    app = ElectroFieldApp()
    app.run()