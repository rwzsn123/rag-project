"""Helpers for running scripts directly from the scripts directory."""
import os
import sys


def ensure_project_root():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
