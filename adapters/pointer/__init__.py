"""Pointer/cursor detection -- see adapters/pointer/cursor_detector.py and
docs/pointer.md.

Detecting a cursor already baked into recorded video pixels is a small
classical-CV motion-detection problem, not a live-OS-cursor-tracking
problem (pynput/pyautogui-style tools don't apply here, they read the live
OS cursor, not pixels). Implemented with three-frame differencing behind
the PointerAdapter contract in core/interfaces.py -- see
docs/component-decisions.md for why this beat template matching, optical
flow, and ML.
"""
