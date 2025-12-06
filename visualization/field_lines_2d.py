# visualization/field_lines_2d.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Tuple, Optional

class AppleStyleFieldLines2D:
    """
    苹果风格2D电场线可视化
    
    设计特性：
        - 简洁现代设计（苹果风格）
        - 平滑的电场线 + 箭头标注
        - 颜色表示场强大小
        - 电荷位置清晰标记
        - 响应式布局
    """
    
    # 苹果风格配色方案
    APPLE_COLORS = {
        'background': '#f5f5f7',
        'grid': '#e0e0e0',
        'positive': '#ff3b30',    # 苹果红
        'negative': '#007aff',    # 苹果蓝
        'neutral': '#8e8e93',     # 苹果灰
        'ring': '#ff9500',        # 苹果橙
        'field_line': '#34c759',  # 苹果绿
        'text': '#1d1d1f'
    }
    
    def __init__(self, field_lines: List[np.ndarray],
                 charges: List,
                 bounds: Tuple[float, float, float, float] = (-2, 2, -2, 2),
                 line_width: float = 1.5,
                 arrow_scale: float = 1.0,
                 color_by_field_strength: bool = True):
        """
        Args:
            field_lines: 电场线点列表
            charges: 电荷对象列表
            bounds: (x_min, x_max, y_min, y_max)
        """
        self.field_lines = field_lines
        self.charges = charges
        self.bounds = bounds
        self.line_width = line_width
        self.arrow_scale = arrow_scale
        self.color_by_field_strength = color_by_field_strength
        
    def _create_figure(self, figsize: Tuple[int, int] = (12, 10)):
        """创建苹果风格的图形"""
        plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # 设置苹果风格背景
        fig.patch.set_facecolor(self.APPLE_COLORS['background'])
        ax.set_facecolor('white')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=14, color=self.APPLE_COLORS['text'], 
                     fontfamily='SF Pro Text')
        ax.set_ylabel('Y (m)', fontsize=14, color=self.APPLE_COLORS['text'],
                     fontfamily='SF Pro Text')
        
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        
        # 网格线
        ax.grid(True, color=self.APPLE_COLORS['grid'], linewidth=0.5, alpha=0.5)
        
        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.APPLE_COLORS['grid'])
        ax.spines['bottom'].set_color(self.APPLE_COLORS['grid'])
        
        return fig, ax
    
    def _calculate_field_strength_color(self, line: np.ndarray, 
                                       field_calculator) -> np.ndarray:
        """根据场强计算线条颜色渐变"""
        if not self.color_by_field_strength:
            return np.full((len(line), 3), 
                          self._hex_to_rgb(self.APPLE_COLORS['field_line']))
        
        # 计算线上每个点的场强
        E_mag = np.linalg.norm(field_calculator.electric_field(line), axis=1)
        
        # 对数归一化（处理大动态范围）
        log_E = np.log10(E_mag + 1e-12)
        norm = (log_E - np.min(log_E)) / (np.max(log_E) - np.min(log_E) + 1e-12)
        
        # 创建绿色系渐变
        colors = []
        for n in norm:
            # 从浅绿到深绿
            r = int(52 * (1 - n) + 0 * n)
            g = int(199 * (1 - n) + 105 * n)
            b = int(89 * (1 - n) + 56 * n)
            colors.append((r/255, g/255, b/255))
        
        return np.array(colors)
    
    def _plot_charges(self, ax):
        """绘制电荷位置"""
        for charge in self.charges:
            if hasattr(charge, 'position'):
                pos = charge.position[:2]  # 取XY平面
                q = getattr(charge, 'q', 0)
                
                if q > 0:
                    color = self.APPLE_COLORS['positive']
                    marker = 'o'
                    label = f'+{abs(q):.1e}C'
                elif q < 0:
                    color = self.APPLE_COLORS['negative']
                    marker = 'o'
                    label = f'-{abs(q):.1e}C'
                else:
                    color = self.APPLE_COLORS['neutral']
                    marker = 's'
                    label = 'Neutral'
                
                # 点大小与电荷量成正比
                size = 100 + 50 * np.log10(abs(q) + 1e-12)
                
                ax.scatter(pos[0], pos[1], 
                          s=size, 
                          c=color, 
                          marker=marker,
                          edgecolors='white',
                          linewidths=2,
                          alpha=0.8,
                          label=label,
                          zorder=5)
                
                # 添加文本标注
                ax.text(pos[0], pos[1] + 0.2, 
                       f'{label}\n({pos[0]:.1f},{pos[1]:.1f})',
                       ha='center', va='bottom',
                       fontsize=9,
                       fontfamily='SF Pro Text',
                       color=self.APPLE_COLORS['text'])
    
    def _plot_field_lines_with_arrows(self, ax, field_calculator):
        """绘制带箭头的电场线"""
        for i, line in enumerate(self.field_lines):
            if len(line) < 10:  # 太短的线不画
                continue
            
            # 计算颜色
            colors = self._calculate_field_strength_color(line, field_calculator)
            
            # 创建线段集合
            segments = np.array([line[:-1], line[1:]]).transpose(1, 0, 2)
            lc = LineCollection(segments, 
                               colors=colors[:-1],  # 每段颜色取起点颜色
                               linewidths=self.line_width,
                               alpha=0.8,
                               zorder=2)
            ax.add_collection(lc)
            
            # 添加小箭头（在线条上均匀分布）
            if len(line) >= 20:
                # 根据线条长度动态确定箭头数量（每15-20个点一个箭头）
                arrow_spacing = max(15, len(line) // 4)
                arrow_indices = range(arrow_spacing, len(line) - arrow_spacing, arrow_spacing)
                
                for idx in arrow_indices:
                    if idx >= len(line)-1 or idx <= 0:
                        continue
                    
                    # 计算箭头位置和方向
                    pos = line[idx]
                    
                    # 使用前向差分计算切线方向（电场方向）
                    if idx < len(line) - 1:
                        direction = line[idx+1] - line[idx]
                    else:
                        direction = line[idx] - line[idx-1]
                    
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm == 0:
                        continue
                        
                    direction = direction / direction_norm
                    
                    # 根据电场方向和电荷类型选择浅色三原色调色板
                    # 为了简单实现，我们使用基于场强方向的浅色三原色
                    # 可以根据需要扩展为根据电荷来源计算颜色
                    
                    # 定义浅色三原色
                    light_red = '#ffcccc'    # 浅红色
                    light_blue = '#cce5ff'   # 浅蓝色
                    light_yellow = '#ffffcc' # 浅黄色
                    
                    # 根据场线方向选择颜色
                    # 使用方向角来决定颜色
                    angle = np.arctan2(direction[1], direction[0])
                    if -np.pi/3 < angle <= np.pi/3:
                        arrow_color = light_red    # 右向箭头使用浅红
                    elif np.pi/3 < angle <= np.pi:  # 上向箭头使用浅黄
                        arrow_color = light_yellow
                    else:
                        arrow_color = light_blue   # 左/下向箭头使用浅蓝
                    
                    # 创建标准指向型小箭头
                    arrow = mpatches.FancyArrowPatch(
                        pos - direction * 0.08 * self.arrow_scale,  # 箭头起点稍后退
                        pos + direction * 0.02 * self.arrow_scale,  # 箭头终点稍前进
                        arrowstyle='->',  # 标准指向型箭头
                        mutation_scale=8 * self.arrow_scale,  # 缩小箭头尺寸
                        mutation_aspect=1.5,  # 保持箭头比例
                        color=arrow_color,
                        linewidth=self.line_width * 0.8,  # 比场线稍细
                        zorder=3,
                        alpha=0.9  # 稍微透明
                    )
                    ax.add_patch(arrow)
    
    def _add_field_strength_contour(self, ax, field_calculator):
        """添加场强等值线"""
        x_min, x_max, y_min, y_max = self.bounds
        
        # 创建网格
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        
        points = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
        E_mag = field_calculator.field_magnitude(points)
        E_mag_grid = E_mag.reshape(X.shape)
        
        # 对数等值线（处理大动态范围）
        log_E = np.log10(E_mag_grid + 1e-12)
        
        # 绘制等值线
        contour = ax.contour(X, Y, log_E, 
                            levels=10,
                            colors=self.APPLE_COLORS['grid'],
                            linewidths=0.5,
                            alpha=0.3,
                            zorder=1)
        
        # 添加等值线标注
        ax.clabel(contour, inline=True, fontsize=8,
                 fmt=lambda x: f'10^{{{x:.1f}}} N/C',
                 colors=self.APPLE_COLORS['text'])
    
    def _create_legend(self, ax):
        """创建苹果风格的图例"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=self.APPLE_COLORS['positive'],
                  markersize=10, label='正电荷 (+Q)'),
            Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=self.APPLE_COLORS['negative'],
                  markersize=10, label='负电荷 (-Q)'),
            Line2D([0], [0], color=self.APPLE_COLORS['field_line'],
                  linewidth=self.line_width, label='电场线')
        ]
        
        ax.legend(handles=legend_elements,
                 loc='upper right',
                 frameon=True,
                 framealpha=0.9,
                 edgecolor=self.APPLE_COLORS['grid'],
                 facecolor='white',
                 fontsize=11,
                 fontfamily='SF Pro Text')
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """十六进制颜色转RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    def plot(self, field_calculator, 
             title: str = "电场线分布",
             save_path: Optional[str] = None) -> plt.Figure:
        """生成2D电场线图"""
        fig, ax = self._create_figure()
        
        # 添加场强等值线背景
        self._add_field_strength_contour(ax, field_calculator)
        
        # 绘制电场线
        self._plot_field_lines_with_arrows(ax, field_calculator)
        
        # 绘制电荷
        self._plot_charges(ax)
        
        # 添加图例
        self._create_legend(ax)
        
        # 设置标题
        ax.set_title(title, 
                    fontsize=24, 
                    fontweight='semibold',
                    fontfamily='SF Pro Display',
                    color=self.APPLE_COLORS['text'],
                    pad=20)
        
        # 添加统计信息
        total_lines = len(self.field_lines)
        avg_length = np.mean([len(line) for line in self.field_lines])
        
        info_text = f"电场线总数: {total_lines}\n平均长度: {avg_length:.1f} points"
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='SF Pro Text',
                color=self.APPLE_COLORS['text'],
                verticalalignment='top',
                bbox=dict(boxstyle='round', 
                         facecolor='white', 
                         alpha=0.9,
                         edgecolor=self.APPLE_COLORS['grid']))
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
        
        return fig