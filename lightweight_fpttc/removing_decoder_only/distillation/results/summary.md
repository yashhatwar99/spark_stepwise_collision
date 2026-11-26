# Decoder removed only, Conv3 taught by distillation

Flow + affine is kept. Conv3 copies the full model's scale map on the 140 KITTI training pairs. No KITTI truth is used in training.

Mid error on the 40 held-out KITTI images (lower is better):

| | Mid error |
|---|---|
| Full FP-TTC | 54.7 |
| Decoder removed, Conv3 trained on KITTI truth (removing_decoder_only) | 68.9 |
| **Decoder removed, Conv3 distilled (this folder)** | **61.7** |

Cars only 118.5, background only 50.8. Other 160 images: 48.6; all 200: 51.3.

- KITTI cars, held-out 40: scale error 0.70%, TTC error (middle) 0.20 s, cars under 3 s found 94%, precision 94%
- KITTI cars, all 200: scale error 0.51%, TTC error (middle) 0.14 s, cars under 3 s found 96%, precision 97%

Speed and size are those of the decoder-removed model (same architecture, only Conv3's weights differ): see `../results/summary.md`.
