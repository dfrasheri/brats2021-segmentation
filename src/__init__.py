"""BraTS 2021 segmentation.

Deliberately light: submodules are imported explicitly (`from src.data import ...`)
so that importing config never drags in torch, sklearn or nibabel.
"""
__version__ = "0.2.0"
