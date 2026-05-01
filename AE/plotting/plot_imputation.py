"""
Plotting functions for imputation analysis (pi/mu values across essential vs non-essential genes).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from scipy.stats import mannwhitneyu

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Utils.plot_config import COLORS


def calculate_stats(essential: np.ndarray, nonessential: np.ndarray) -> Dict[str, float]:
    """Calculate Mann-Whitney U test between essential and non-essential values."""
    ess_clean = essential[~np.isnan(essential)]
    non_clean = nonessential[~np.isnan(nonessential)]
    
    if len(ess_clean) == 0 or len(non_clean) == 0:
        return {'mann_whitney_pvalue': np.nan}
    
    _, p_value = mannwhitneyu(ess_clean, non_clean, alternative='two-sided')
    return {'mann_whitney_pvalue': p_value}


def create_boxplot(ax, essential: np.ndarray, nonessential: np.ndarray, 
                   title: str, show_stats: bool = True, ylabel: str = 'π value') -> None:
    """Create boxplot comparing essential vs non-essential genes."""
    bp = ax.boxplot([essential, nonessential], labels=['Essential', 'Non-Essential'],
                     patch_artist=True, widths=0.6, medianprops=dict(color='#666666', linewidth=1.5))
    
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if show_stats and len(essential) > 0 and len(nonessential) > 0:
        p = calculate_stats(essential, nonessential)['mann_whitney_pvalue']
        if not np.isnan(p):
            if p < 0.001:
                p_text = 'Mann-Whitney U: p < 0.001***'
            elif p < 0.01:
                p_text = f'Mann-Whitney U: p = {p:.3f}**'
            elif p < 0.05:
                p_text = f'Mann-Whitney U: p = {p:.3f}*'
            else:
                p_text = f'Mann-Whitney U: p = {p:.3f} ns'
            
            ax.text(0.5, 0.98, p_text, transform=ax.transAxes, ha='center', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))


