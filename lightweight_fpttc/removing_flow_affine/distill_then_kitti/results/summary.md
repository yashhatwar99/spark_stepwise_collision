# Version A, Conv3 trained by distillation first, then on KITTI truth

Stage 1: copy the full model's scale map. Stage 2: continue with KITTI's true scale. Mid error after stage 1 alone: 62.1.

Mid error on the 40 held-out KITTI images (lower is better):

| | Mid error |
|---|---|
| Full FP-TTC | 54.7 |
| Decoder removed (part 1) | 68.9 |
| Version A, Conv3 as it is | 69.0 |
| Version A, Conv3 trained directly on KITTI truth | 64.6 |
| **Version A, this method (distillation, then KITTI truth)** | **62.1** |

Cars only 119.3, background only 51.0.

| | all 40 held-out | other 160 | all 200 |
|---|---|---|---|
| Mid error, this method | 62.1 | 48.9 | 51.6 |

- KITTI cars, held-out 40: scale error 0.69%, TTC error (middle) 0.25 s, cars under 3 s found 94%, precision 94%
- KITTI cars, all 200: scale error 0.51%, TTC error (middle) 0.14 s, cars under 3 s found 96%, precision 97%
