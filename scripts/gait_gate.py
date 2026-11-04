#!/usr/bin/env python3
"""Gait quality gate CLI. Implementation: k1_utils.gait_gate.  Usage: gait_gate.py <trace.npz> [--strict]"""
import sys

from k1_utils.gait_gate import main

if __name__ == "__main__":
    sys.exit(main())
