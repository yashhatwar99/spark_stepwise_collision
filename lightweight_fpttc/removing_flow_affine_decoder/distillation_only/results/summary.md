# Version A, Conv3 trained by distillation only

Conv3 copies the full model's scale map (every pixel). No KITTI truth is used in training.

Mid error on the 40 held-out KITTI images (lower is better):

| | Mid error |
|---|---|
| Full FP-TTC | 54.7 |
| Decoder removed (part 1) | 68.9 |
| Version A, Conv3 as it is | 69.0 |
| Version A, Conv3 trained directly on KITTI truth | 64.6 |
| **Version A, this method (distillation only)** | **61.7** |

Cars only 118.1, background only 50.7.

| | all 40 held-out | other 160 | all 200 |
|---|---|---|---|
| Mid error, this method | 61.7 | 48.6 | 51.2 |

- KITTI cars, held-out 40: scale error 0.65%, TTC error (middle) 0.20 s, cars under 3 s found 94%, precision 94%
- KITTI cars, all 200: scale error 0.51%, TTC error (middle) 0.14 s, cars under 3 s found 96%, precision 97%
