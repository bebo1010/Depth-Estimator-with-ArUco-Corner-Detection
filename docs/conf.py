# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Depth-Estimator-with-ArUco-Corners-Detection'
copyright = '2025, I-Lung Chang'
author = 'I-Lung Chang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',    # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',    # Add links to highlighted source code
    'sphinx.ext.intersphinx', # Link to other projects' documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for latex output ----------------------------
latex_engine = "xelatex"
latex_use_xindy = False
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
    \usepackage{xeCJK}  % 支援中文
    \setCJKmainfont[BoldFont=Microsoft JhengHei, ItalicFont=Microsoft JhengHei]{Microsoft JhengHei}
    \setCJKsansfont{Microsoft JhengHei}
    \setCJKmonofont{Microsoft JhengHei}
    \usepackage[utf8]{inputenc}
    \usepackage{geometry}
    \geometry{a4paper, margin=2.5cm}  % 調整邊距
    \usepackage[breakall]{truncate}  % 自動斷行
    \onecolumn  % 強制單列模式
    \usepackage{longtable}  % 支援長表
    \usepackage{booktabs}  % 改進表格樣式（可選）
    \usepackage{makeidx}
    \usepackage[columns=1]{idxlayout}
    ''',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
