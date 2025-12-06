# visualization/field_lines_3d.py
import plotly.graph_objects as go
import numpy as np
from typing import List

class CosmosFieldLines3D:
    """
    宇宙风格3D电场线可视化
    
    设计特性：
        - 3D电场线带箭头
        - 颜色表示场强/电势
        - 发光效果 + 动画
        - 交互式电荷信息
    """
    
    COSMOS_COLORS = {
        'positive': '#ff3366',
        'negative': '#3399ff',
        'field_line': '#4cc9f0',
        'arrow': '#ffd700',
        'background': '#0b0b1f',
        'grid': '#1e2a47'
    }
    
    def __init__(self, field_lines: List[np.ndarray],
                 charges: List,
                 show_arrows: bool = True,
                 arrow_density: float = 0.2,
                 tube_radius: float = 0.02,
                 opacity: float = 0.8,
                 color_by_potential: bool = True):
        """
        Args:
            field_lines: 3D电场线点列表
            charges: 电荷对象列表
            arrow_density: 箭头密度（0-1）
        """
        self.field_lines = field_lines
        self.charges = charges
        self.show_arrows = show_arrows
        self.arrow_density = arrow_density
        self.tube_radius = tube_radius
        self.opacity = opacity
        self.color_by_potential = color_by_potential
    
    def _calculate_line_colors(self, line: np.ndarray, 
                              potential_calculator=None) -> List[str]:
        """计算线上每个点的颜色"""
        if not self.color_by_potential or potential_calculator is None:
            # 固定颜色
            return [self.COSMOS_COLORS['field_line']] * len(line)
        
        # 根据电势计算颜色
        V = potential_calculator.potential(line, absolute=True)
        V_min, V_max = np.min(V), np.max(V)
        
        colors = []
        for v in V:
            if v > 0:
                # 正电势：红色系
                norm = v / max(V_max, 1e-12)
                r = 255
                g = int(51 * (1 - norm))
                b = int(102 * (1 - norm))
            else:
                # 负电势：蓝色系
                norm = abs(v) / max(abs(V_min), 1e-12)
                r = int(51 * (1 - norm))
                g = int(102 * (1 - norm))
                b = 255
            
            colors.append(f'rgb({r},{g},{b})')
        
        return colors
    
    def create_field_line_traces(self, potential_calculator=None) -> List[go.Scatter3d]:
        """创建电场线3D轨迹"""
        traces = []
        
        for i, line in enumerate(self.field_lines):
            if len(line) < 10:
                continue
            
            # 计算颜色
            colors = self._calculate_line_colors(line, potential_calculator)
            
            # 创建主轨迹（带颜色渐变）
            trace = go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(
                    color=colors,
                    width=6,
                ),
                opacity=self.opacity,
                name=f'电场线 {i+1}',
                showlegend=False,
                hoverinfo='text',
                hovertext=self._create_hover_text(line, i)
            )
            traces.append(trace)
            
            # 添加箭头
            if self.show_arrows:
                arrow_traces = self._create_arrow_traces(line)
                traces.extend(arrow_traces)
        
        return traces
    
    def _create_arrow_traces(self, line: np.ndarray) -> List[go.Cone]:
        """创建箭头轨迹"""
        arrows = []
        
        # 计算线上等间距的点
        n_arrows = max(1, int(len(line) * self.arrow_density))
        indices = np.linspace(0, len(line)-2, n_arrows, dtype=int)
        
        for idx in indices:
            if idx >= len(line) - 1:
                continue
            
            # 计算切线方向
            p0 = line[max(0, idx-1)]
            p1 = line[idx]
            p2 = line[min(len(line)-1, idx+1)]
            
            direction = (p2 - p0)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            
            # 创建锥形箭头
            arrow = go.Cone(
                x=[p1[0]],
                y=[p1[1]],
                z=[p1[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                sizemode='scaled',
                sizeref=0.15,
                colorscale=[[0, self.COSMOS_COLORS['arrow']], 
                           [1, self.COSMOS_COLORS['arrow']]],
                showscale=False,
                anchor='tail',
                hoverinfo='skip'
            )
            arrows.append(arrow)
        
        return arrows
    
    def create_charge_traces(self) -> List[go.Scatter3d]:
        """创建电荷轨迹"""
        traces = []
        
        for charge in self.charges:
            if hasattr(charge, 'position'):
                pos = charge.position
                q = getattr(charge, 'q', 0)
                
                color = self.COSMOS_COLORS['positive'] if q > 0 else self.COSMOS_COLORS['negative']
                
                trace = go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=color,
                        opacity=1.0,
                        line=dict(color='white', width=2)
                    ),
                    name=f'电荷: {q:.2e}C',
                    hoverinfo='text',
                    hovertext=f"""
                    <b>电荷量</b>: {q:.2e} C<br>
                    <b>位置</b>: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})
                    """
                )
                traces.append(trace)
        
        return traces
    
    def _create_hover_text(self, line: np.ndarray, line_idx: int) -> str:
        """创建悬停文本"""
        length = len(line)
        start = line[0]
        end = line[-1]
        
        return f"""
        <b>电场线 #{line_idx+1}</b><br>
        <b>长度</b>: {length} 个点<br>
        <b>起点</b>: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})<br>
        <b>终点</b>: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})<br>
        <b>总长度</b>: {np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)):.2f} m
        """
    
    def create_layout(self, title: str = "宇宙电场线") -> go.Layout:
        """创建宇宙风格布局"""
        return go.Layout(
            title=dict(
                text=title,
                font=dict(size=28, color='white', family='Arial Black'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title='X (m)',
                    gridcolor=self.COSMOS_COLORS['grid'],
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                yaxis=dict(
                    title='Y (m)',
                    gridcolor=self.COSMOS_COLORS['grid'],
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                zaxis=dict(
                    title='Z (m)',
                    gridcolor=self.COSMOS_COLORS['grid'],
                    gridwidth=1,
                    backgroundcolor=self.COSMOS_COLORS['background'],
                    color='white'
                ),
                bgcolor=self.COSMOS_COLORS['background'],
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            paper_bgcolor=self.COSMOS_COLORS['background'],
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(11, 11, 31, 0.8)',
                bordercolor=self.COSMOS_COLORS['grid'],
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="播放动画",
                    method="animate",
                    args=[None, dict(frame=dict(duration=500, redraw=True), 
                                     fromcurrent=True, mode='immediate')]
                )]
            )]
        )
    
    def plot(self, potential_calculator=None,
             title: str = "宇宙电场线",
             show_animation: bool = True) -> go.Figure:
        """生成3D电场线图"""
        # 收集所有轨迹
        all_traces = []
        
        # 添加电场线
        field_line_traces = self.create_field_line_traces(potential_calculator)
        all_traces.extend(field_line_traces)
        
        # 添加电荷
        charge_traces = self.create_charge_traces()
        all_traces.extend(charge_traces)
        
        # 创建布局
        layout = self.create_layout(title)
        
        # 创建图形
        fig = go.Figure(data=all_traces, layout=layout)
        
        # 添加动画帧（可选）
        if show_animation and len(self.field_lines) > 0:
            frames = self._create_animation_frames()
            fig.frames = frames
        
        # 添加统计信息标注
        total_lines = len(self.field_lines)
        total_points = sum(len(line) for line in self.field_lines)
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"<b>统计信息</b><br>电场线: {total_lines} 条<br>总点数: {total_points}",
            showarrow=False,
            font=dict(color='white', size=12),
            bgcolor='rgba(11, 11, 31, 0.8)',
            bordercolor=self.COSMOS_COLORS['grid'],
            borderwidth=1,
            borderpad=10
        )
        
        return fig
    
    def _create_animation_frames(self) -> List[go.Frame]:
        """创建动画帧（电场线生长效果）"""
        frames = []
        max_length = max(len(line) for line in self.field_lines)
        
        # 创建20个帧
        for frame_idx in range(20):
            step = max_length // 20
            current_max = min((frame_idx + 1) * step, max_length)
            
            frame_traces = []
            
            # 创建部分电场线
            for line in self.field_lines:
                if len(line) > current_max:
                    partial_line = line[:current_max]
                else:
                    partial_line = line
                
                if len(partial_line) < 2:
                    continue
                
                trace = go.Scatter3d(
                    x=partial_line[:, 0],
                    y=partial_line[:, 1],
                    z=partial_line[:, 2],
                    mode='lines',
                    line=dict(
                        color=self.COSMOS_COLORS['field_line'],
                        width=6,
                    ),
                    opacity=0.7,
                    showlegend=False
                )
                frame_traces.append(trace)
            
            # 添加电荷
            for charge in self.charges:
                if hasattr(charge, 'position'):
                    pos = charge.position
                    q = getattr(charge, 'q', 0)
                    color = self.COSMOS_COLORS['positive'] if q > 0 else self.COSMOS_COLORS['negative']
                    
                    trace = go.Scatter3d(
                        x=[pos[0]],
                        y=[pos[1]],
                        z=[pos[2]],
                        mode='markers',
                        marker=dict(size=15, color=color),
                        showlegend=False
                    )
                    frame_traces.append(trace)
            
            frame = go.Frame(data=frame_traces, 
                           name=f'frame_{frame_idx}')
            frames.append(frame)
        
        return frames