# Full FP-TTC against light FP-TTC (new decoder)

Everything else is the same as the full model (frozen, released `fpttc_kitti` weights). Only the decoder is replaced by Conv3 on Concat(SF_t, x_t), trained on KITTI.

| | Full FP-TTC | Light FP-TTC | Change |
|---|---|---|---|
| Parameters | 12.44 M | 6.83 M | -5.60 M (-45%), better |
| Model size (32-bit) | 50 MB | 27 MB | -22 MB (-45%), better |
| Time per pair, GPU | 292 ms | 247 ms | -45 ms (-15%), better |
| Pairs per second, GPU | 3.4 | 4.0 | +0.6 (+18%), better |
| Time per pair, CPU | 6.7 s | 5.1 s | -1.6 s (-24%), better |
| Peak GPU memory | 0.74 GB | 0.74 GB | -0.00 GB (-0%), same |
| Mid error, 40 held-out images (lower is better) | 54.7 | 68.9 | +14.2 (+26%), worse |
| Mid error, other 160 images (lower is better) | 42.7 | 55.9 | +13.2 (+31%), worse |
| Mid error, all 200 images (lower is better) | 45.1 | 58.5 | +13.4 (+30%), worse |
| Mid error on cars, held-out | 99.0 | 135.2 | +36.3 (+37%), worse |
| Mid error on background, held-out | 45.7 | 54.3 | +8.6 (+19%), worse |
| Per-car scale error, KITTI cars, held out 40 (85 cars) | 0.56% | 0.79% | +0.23% (+41%), worse |
| Per-car TTC error (middle), KITTI cars, held out 40 (85 cars) | 0.27 s | 0.21 s | -0.06 s (-23%), better |
| Cars under 3 s found, KITTI cars, held out 40 (85 cars) | 94% | 94% | +0 pts, same |
| Precision, KITTI cars, held out 40 (85 cars) | 94% | 94% | +0 pts, same |
| Per-car scale error, KITTI cars, all 200 (425 cars) | 0.43% | 0.56% | +0.14% (+33%), worse |
| Per-car TTC error (middle), KITTI cars, all 200 (425 cars) | 0.13 s | 0.13 s | +0.00 s (+2%), worse |
| Cars under 3 s found, KITTI cars, all 200 (425 cars) | 96% | 97% | +1 pts, better |
| Precision, KITTI cars, all 200 (425 cars) | 98% | 97% | -1 pts, worse |
| Per-car scale error, nuScenes cars (1418 cars) | 0.34% | 0.34% | -0.01% (-2%), better |
| Per-car TTC error (middle), nuScenes cars (1418 cars) | 0.94 s | 1.04 s | +0.10 s (+11%), worse |
| Cars under 3 s found, nuScenes cars (1418 cars) | 87% | 84% | -4 pts, worse |
| Precision, nuScenes cars (1418 cars) | 80% | 86% | +7 pts, better |

## Parameters by part (full model)

| Part | Parameters | In the light model |
|---|---|---|
| backbone CNN | 1,196,576 | kept, frozen |
| backbone transformer | 3,150,336 | kept, frozen |
| flow + affine (matching) | 667,921 | kept, frozen |
| scale encoder | 1,448,864 | kept, frozen |
| matching-clue attention | 328,448 | removed |
| decoder + upsample | 5,642,896 | removed |
| Conv3 (new) | 369,409 | added, trained |
