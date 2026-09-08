"""FitCoach backend application package."""

from .config import load_environment


load_environment()

__all__ = ["load_environment"]
