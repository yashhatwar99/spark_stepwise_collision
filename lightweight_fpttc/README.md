# Lightweight FP-TTC

Two ways of making FP-TTC lighter, one folder each. Everything else in FP-TTC stays as released (`fpttc_kitti` weights, frozen). Only Conv3, the small replacement for the decoder, is trained, and only on KITTI.

| Folder | What is removed | Results (Mid error on 40 held-out KITTI pairs, lower is better; full model 54.7) |
|---|---|---|
| `removing_decoder_only/` | the decoder and the matching-clue attention, replaced by Conv3 (3 conv layers). Flow + affine is kept | 6.83 M parameters. Conv3 trained on KITTI truth: 68.9. Conv3 distilled (`distillation/`): 61.7 |
| `removing_flow_affine_decoder/` | the decoder (as above) **and** the flow + affine stage (the scale encoder's query now comes from the image features) | two versions, A keeps `scale_conv` (6.17 M parameters), B drops it (5.43 M). Conv3 as it is: 69.0 / 68.9. Trained on KITTI truth: 64.6 / 65.2. Distilled (A): 61.7 |

## Subfolders

- `removing_decoder_only/distillation/`: Conv3 copies the full model's scale map (no KITTI truth). Mid error 61.7. Has its own `results/` and `output/kitti/` pictures.
- `removing_flow_affine_decoder/distillation_only/`, `distill_then_kitti/`, `distill_more_frames/`: three ways of teaching Conv3 by distillation on version A (61.7, 62.1, and 62.4 on 553 pairs). Each has `results/` and `output/kitti/` pictures.
- `removing_flow_affine_decoder/results/versions_summary.md`: parameters, speed, accuracy and per-car TTC for versions A and B.

Every new method also gets its output pictures on KITTI frames in its own `output/kitti/` folder (not committed).

`removing_flow_affine_decoder/` reuses `light_model.py`, the Conv3 weights and the KITTI cache of `removing_decoder_only/`.
Run the scripts from the project root with the project `.venv`; each file's header says how.
