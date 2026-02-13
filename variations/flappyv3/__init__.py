"""Flappy v3 env. Build the C extension: cd variations/flappyv3 && make"""

try:
    from . import binding
except ImportError:
    binding = None

from .curriculum import FlappyCurriculum, compute_difficulty, curriculum_env_creator

__all__ = ["binding", "compute_difficulty", "FlappyCurriculum", "curriculum_env_creator"]
