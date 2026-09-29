#!/usr/bin/env python3
"""Актуальное имя; probe_three.py остаётся совместимым CLI, без лимита 3."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('probe_three.py')),run_name='__main__')
