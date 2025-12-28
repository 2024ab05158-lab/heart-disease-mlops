"""
Pytest configuration file for the heart-disease-mlops project.
This file is automatically loaded by pytest and sets up the Python path.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

