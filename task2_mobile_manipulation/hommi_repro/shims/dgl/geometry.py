import torch


@torch.no_grad()
def farthest_point_sampler(pos: torch.Tensor, npoints: int, start_idx: int = 0) -> torch.Tensor:
    """Same contract as dgl.geometry.farthest_point_sampler: pos (B, N, C) -> indices (B, npoints)."""
    B, N, _ = pos.shape
    pos = pos.float()
    idx = torch.zeros(B, npoints, dtype=torch.long, device=pos.device)
    dist = torch.full((B, N), float("inf"), device=pos.device)
    farthest = torch.full((B,), int(start_idx), dtype=torch.long, device=pos.device)
    batch = torch.arange(B, device=pos.device)
    for i in range(npoints):
        idx[:, i] = farthest
        centroid = pos[batch, farthest].unsqueeze(1)
        dist = torch.minimum(dist, ((pos - centroid) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return idx
