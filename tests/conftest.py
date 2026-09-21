"""
Pytest configuration file for the SAR dataset tests.
"""
import sys
from pathlib import Path

# Add the project root to Python path so we can import sarenv modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
