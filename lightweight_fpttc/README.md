# Lightweight FP-TTC

Two ways of making FP-TTC lighter, one folder each. Everything else in FP-TTC stays as released (`fpttc_kitti` weights, frozen).

| Folder | What is removed | Status |
|---|---|---|
| `removing_decoder/` | the decoder and the matching-clue attention, replaced by Conv3 (3 conv layers) | built and measured: 6.83 M parameters, Mid error 68.9 (full model 54.7) |
| `removing_flow_affine/` | also the flow + affine stage (the query no longer comes from it) | two versions built and measured (A keeps `scale_conv`: 6.17 M parameters; B drops it: 5.43 M); Mid error 69.0 / 68.9 as is, 64.6 / 65.2 with Conv3 retrained (full 54.7); see `results/versions_summary.md`, pictures in `output/kitti/` |

`removing_flow_affine/` reuses `light_model.py`, the Conv3 weights and the KITTI cache of `removing_decoder/`.
Run the scripts from the project root with the project `.venv`; each file's header says how.
