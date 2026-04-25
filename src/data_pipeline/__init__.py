"""
数据管道模块
"""

from .load_data import load_full_data, load_local_data, load_data_streaming
from .clean_data import clean_data, save_clean_data
from .eda import (
    plot_age_distribution,
    plot_profession_distribution,
    plot_department_distribution,
    plot_gender_distribution,
    plot_education_distribution,
    plot_marital_distribution,
    plot_age_by_occupation,
    plot_dashboard,
    print_summary_stats,
)