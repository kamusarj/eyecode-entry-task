"""Euclidean signed distance field of the scene (stand-in for nvblox in MoManipVLA Eq. 5).

The paper builds the ESDF with nvblox from RGB-D (real world) or from the 3D object models (OVMM).
We follow the OVMM variant: voxelize the scene furniture, run an exact Euclidean distance transform,
and query it with trilinear interpolation. The target object and the target receptacle (bowl) are
excluded from the ESDF, as the paper excludes the target-object mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from kinematics import yaw_rot


class ESDF:
    def __init__(self, furniture, room_half=3.5, height=2.0, res=0.025, **_):
        self.res = res
        self.origin = np.array([-room_half, -room_half, 0.0])
        shape = np.ceil(np.array([2 * room_half, 2 * room_half, height]) / res).astype(int) + 1
        occ = np.zeros(shape, bool)
        for f in furniture:
            if not f.collidable:
                continue
            R = yaw_rot(f.yaw)
            ext = np.abs(R) @ f.half                            # world AABB half extents
            lo = np.clip(np.floor((f.center - ext - self.origin) / res).astype(int), 0, shape - 1)
            hi = np.clip(np.ceil((f.center + ext - self.origin) / res).astype(int) + 1, 0, shape)
            axes = [self.origin[i] + res * np.arange(lo[i], hi[i]) for i in range(3)]
            P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).astype(np.float32)
            local = (P - f.center) @ R                          # world -> box frame (R^T p == p @ R)
            occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= np.all(np.abs(local) <= f.half + 1e-6, axis=-1)
        # signed distance: positive outside obstacles, negative inside
        outside = ndimage.distance_transform_edt(~occ) * res
        inside = ndimage.distance_transform_edt(occ) * res
        self.sdf = (outside - inside).astype(np.float32)
        self.shape = np.array(shape)

    def query(self, pts):
        """Trilinear interpolation; points outside the grid are treated as free (distance 1 m)."""
        g = (pts - self.origin) / self.res
        i0 = np.floor(g).astype(int)
        f = g - i0
        valid = np.all((i0 >= 0) & (i0 < self.shape - 1), axis=1)
        out = np.ones(len(pts), np.float32)
        if not valid.any():
            return out
        i0, f = i0[valid], f[valid]
        s = self.sdf
        acc = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    acc = acc + w * s[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        out[valid] = acc
        return out


class BoxSDF:
    """Exact signed distance to the scene boxes (ESDF "constructed with 3D object models", paper Sec. 4.1).
    Same query() interface as the voxel ESDF; the voxel version is kept as the nvblox-like ablation."""

    def __init__(self, furniture, **_):
        boxes = [f for f in furniture if f.collidable]
        self.c = np.stack([f.center for f in boxes])                       # (K,3)
        self.R = np.stack([yaw_rot(f.yaw) for f in boxes])                 # (K,3,3)
        self.h = np.stack([f.half for f in boxes])                         # (K,3)

    def query(self, pts):
        local = np.einsum("kji,nkj->nki", self.R, pts[:, None, :] - self.c[None])   # R^T (p - c)
        q = np.abs(local) - self.h[None]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(q.max(-1), 0.0)
        return (outside + inside).min(-1).astype(np.float32)
