"""Minimal stand-in for DGL: HoMMI only uses `dgl.geometry.farthest_point_sampler`.
(DGL wheels for recent torch are no longer downloadable from data.dgl.ai, HTTP 403.)"""
from . import geometry  # noqa: F401
