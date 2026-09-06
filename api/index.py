"""Vercel entry point for the IRISCOPE Flask application."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app import app  # noqa: E402
