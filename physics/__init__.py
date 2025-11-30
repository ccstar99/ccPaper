"""
物理模型包
包含各种静电学物理模型的实现
"""

from .point_charge import (
    PointChargeModel,
    DipoleModel,
    ElectrostaticModels,
    IntegrationProcess,
    BoundaryElementSolver
)

__all__ = [
    'PointChargeModel',
    'DipoleModel',
    'ElectrostaticModels',
    'IntegrationProcess',
    'BoundaryElementSolver'
]