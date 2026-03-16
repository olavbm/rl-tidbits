"""Test configuration - ensure jax_boids is importable."""

import sys
from pathlib import Path

# Add project root to sys.path so tests can import jax_boids
sys.path.insert(0, str(Path(__file__).parent.parent))
