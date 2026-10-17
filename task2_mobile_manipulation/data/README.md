# Bundled review data

`review_samples/hommi_validation_6.npz` contains one middle sample from each of the six held-out Laundry demonstrations. Its `metadata.json` records selection indices and action indexing. These samples support a small inference check; they do not reproduce the historical 395-sample aggregate by themselves.

`wbc_laundry30/` contains the official 30-episode dataset plan plus only `camera_head_lookatpoint` and `episode_ends` from the processed replay buffer. This is sufficient to replay the recorded WBC experiment, including the measured look-at depths. It omits images and cannot train a policy.

`vla_demos/waypoint_head.pt` is the trained small waypoint head. Generate its training data with `momanipvla_repro/vla_policy.py gen`.

The full 3.75 GB HoMMI replay buffer and raw recordings are reconstructed using the pinned manifest and processing commands in the task README. Original data: [real-stanford/hommi](https://huggingface.co/datasets/real-stanford/hommi). See the dataset's terms for reuse.
