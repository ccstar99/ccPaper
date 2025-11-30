# utils/performance.py
"""
性能监控和缓存管理模块

提供应用性能跟踪、资源监控和缓存优化功能。
"""

import time
import logging
import os
import sys
from typing import Dict, Any, Optional, List, Callable
import threading
import gc

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    性能监控器
    
    跟踪代码执行时间、内存使用和系统资源。
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: List[Dict[str, Any]] = []
        
    def start_timer(self, name: str) -> None:
        """开始计时
        
        Args:
            name: 计时器名称
        """
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时
        
        Args:
            name: 计时器名称
            
        Returns:
            执行时间（秒）
        """
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            # 存储指标
            if name not in self.metrics:
                self.metrics[name] = {'times': [], 'count': 0}
            
            self.metrics[name]['times'].append(elapsed)
            self.metrics[name]['count'] += 1
            
            return elapsed
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况
        
        Returns:
            内存使用统计（MB）
        """
        # 简单替代方案，使用sys模块提供基本内存信息
        # 注意：这不是精确的物理内存使用量，但提供了基本参考
        try:
            # 尝试使用Python内置的内存使用估计
            if hasattr(sys, 'getsizeof'):
                # 这是一个简化的估计，可能不准确
                return {
                    'rss': 0.0,  # 物理内存 - 这里设为0作为占位符
                    'vms': 0.0   # 虚拟内存 - 这里设为0作为占位符
                }
            else:
                return {
                    'rss': 0.0,
                    'vms': 0.0
                }
        except Exception as e:
            logger.warning(f"获取内存使用失败: {e}")
            return {
                'rss': 0.0,
                'vms': 0.0
            }
    
    def log_system_metrics(self) -> None:
        """记录系统指标"""
        metrics = {
            'timestamp': time.time(),
            'memory': self.get_memory_usage(),
            'cpu_percent': 0.0,  # CPU使用率 - 设为0作为占位符
            'threads': threading.active_count()
        }
        self.system_metrics.append(metrics)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            性能统计摘要
        """
        summary = {}
        
        for name, data in self.metrics.items():
            summary[name] = {
                'avg_time': sum(data['times']) / len(data['times']),
                'min_time': min(data['times']),
                'max_time': max(data['times']),
                'count': data['count']
            }
        
        return summary
    
    def clear(self) -> None:
        """清除所有性能数据"""
        self.start_times.clear()
        self.metrics.clear()
        self.system_metrics.clear()


class CacheManager:
    """
    缓存管理器
    
    提供内存缓存功能，支持过期时间和大小限制。
    """
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        """初始化缓存管理器
        
        Args:
            max_size: 最大缓存项数量
            default_ttl: 默认过期时间（秒）
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查缓存项是否过期
        
        Args:
            entry: 缓存项
            
        Returns:
            是否过期
        """
        if 'expires_at' in entry:
            return time.time() > entry['expires_at']
        return False
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存项"""
        expired_keys = [k for k, v in self.cache.items() if self._is_expired(v)]
        for key in expired_keys:
            del self.cache[key]
    
    def _ensure_capacity(self) -> None:
        """确保缓存不超过最大容量"""
        if len(self.cache) >= self.max_size:
            # 简单的LRU实现：删除最旧的项
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get('created_at', 0))
            del self.cache[oldest_key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示使用默认值
        """
        self._cleanup_expired()
        self._ensure_capacity()
        
        entry = {
            'value': value,
            'created_at': time.time()
        }
        
        if ttl is not None or self.default_ttl > 0:
            entry['expires_at'] = time.time() + (ttl or self.default_ttl)
        
        self.cache[key] = entry
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期返回None
        """
        self._cleanup_expired()
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        if self._is_expired(entry):
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return entry['value']
    
    def delete(self, key: str) -> None:
        """删除缓存项
        
        Args:
            key: 缓存键
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class ProgressTracker:
    """
    进度跟踪器
    
    用于跟踪长时间运行任务的进度。
    """
    
    def __init__(self, total_steps: int = 100, description: str = "Processing"):
        """初始化进度跟踪器
        
        Args:
            total_steps: 总步骤数
            description: 任务描述
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.update_times: List[float] = []
    
    def update(self, steps: int = 1) -> None:
        """更新进度
        
        Args:
            steps: 完成的步骤数
        """
        self.current_step = min(self.current_step + steps, self.total_steps)
        self.update_times.append(time.time())
    
    def get_progress(self) -> float:
        """获取当前进度百分比
        
        Returns:
            进度百分比（0-100）
        """
        return (self.current_step / self.total_steps) * 100
    
    def get_estimated_time_remaining(self) -> float:
        """估算剩余时间
        
        Returns:
            估计剩余时间（秒）
        """
        if self.current_step == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        rate = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        
        return remaining_steps / rate if rate > 0 else 0.0
    
    def reset(self) -> None:
        """重置进度跟踪器"""
        self.current_step = 0
        self.start_time = time.time()
        self.update_times.clear()
    
    def __str__(self) -> str:
        """返回进度字符串表示"""
        progress = self.get_progress()
        eta = self.get_estimated_time_remaining()
        
        return f"{self.description}: {progress:.1f}% - ETA: {eta:.1f}s"
