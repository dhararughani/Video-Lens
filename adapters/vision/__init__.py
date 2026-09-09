"""Vision/multimodal adapter -- see adapters/vision/claude_vision.py and
docs/vision.md.

Frame description/understanding calls out to the Claude API (a hosted
multimodal LLM), per the CPU-first hardware philosophy -- no local VLM
weights or GPU requirement. Implements VisionAdapter in core/interfaces.py.
"""
