"""Curly-succotash: first RL env for PufferLib."""

import warnings

# Suppress Gym deprecation message from pufferlib's dependencies (we only use Gymnasium)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from curly_succotash.env import SamplePufferEnv, make_gymnasium_env

__all__ = ["SamplePufferEnv", "make_gymnasium_env"]
