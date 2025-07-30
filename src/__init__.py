"""
HNAF-Jose: Hybrid Normalized Advantage Function Implementation

This package contains the implementation of the Hybrid Normalized Advantage Function (HNAF)
with corrected NAF and user functions integration.

Author: Eden Rochman
Date: July 2024
"""

from .naf_corrected import CorrectedOptimizationFunctions
from .hnaf_stable import StableHNAF, train_stable_hnaf
from .optimization_functions import OptimizationFunctions

__version__ = "1.0.0"
__author__ = "Eden Rochman"
__email__ = "eden@example.com"

__all__ = [
    "CorrectedOptimizationFunctions",
    "StableHNAF", 
    "train_stable_hnaf",
    "OptimizationFunctions"
] 