# Version A, Conv3 trained by distillation on more KITTI frames

Conv3 copies the full model's scale map on 553 pairs (140 KITTI scene flow training pairs + 413 raw drive pairs). No KITTI truth is used in training.

Mid error on the 40 held-out KITTI images (lower is better):

| | Mid error |
|---|---|
| Full FP-TTC | 54.7 |
| Decoder removed (part 1) | 68.9 |
| Version A, Conv3 as it is | 69.0 |
| Version A, Conv3 trained directly on KITTI truth | 64.6 |
| **Version A, this method (distillation on more frames)** | **62.4** |

Cars only 123.4, background only 50.0.

| | all 40 held-out | other 160 | all 200 |
|---|---|---|---|
| Mid error, this method | 62.4 | 51.8 | 53.9 |

- KITTI cars, held-out 40: scale error 0.69%, TTC error (middle) 0.32 s, cars under 3 s found 94%, precision 94%
- KITTI cars, all 200: scale error 0.62%, TTC error (middle) 0.16 s, cars under 3 s found 96%, precision 97%
