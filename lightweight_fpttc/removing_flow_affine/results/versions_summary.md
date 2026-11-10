# Light FP-TTC without flow + affine: version A and version B

- **A**: only flow + affine removed. The encoder's own `scale_conv` stays and makes the query from the image features.
- **B**: flow + affine and `scale_conv` removed. The query is one fixed constant.
- *asis*: Conv3 from `removing_decoder`, not trained again. *retrained*: Conv3 trained again on this version's SF_t (encoder frozen).

## Size and speed

| | Full FP-TTC | Decoder removed | A | B |
|---|---|---|---|---|
| Parameters | 12.44 M | 6.83 M | 6.17 M | 5.43 M |
| Time per pair, GPU | 282 ms | 241 ms | 222 ms | 219 ms |
| Time per pair, CPU | 5.2 s | 4.7 s | 3.9 s | 3.8 s |
| Peak GPU memory | 0.72 GB | 0.72 GB | 1.16 GB | 1.11 GB |

SF_t difference from the full model: A 0.51 %, B 0.51 %.

## Accuracy (Mid error, lower is better)

| | Full | Decoder removed | A asis | A retrained | B asis | B retrained |
|---|---|---|---|---|---|---|
| KITTI held-out 40 | 54.7 | 68.9 | 69.0 | 64.6 | 68.9 | 65.2 |
| KITTI other 160 | 42.7 | 55.9 | 55.9 | 48.5 | 55.9 | 49.3 |
| KITTI all 200 | 45.1 | 58.5 | 58.6 | 51.7 | 58.5 | 52.5 |

## Per-car TTC (median method)

| | Full | A asis | A retrained | B asis | B retrained |
|---|---|---|---|---|---|
| KITTI cars, held-out 40: scale error | 0.56% | 0.79% | 0.69% | 0.79% | 0.68% |
| KITTI cars, held-out 40: TTC error (middle) | 0.27 s | 0.21 s | 0.24 s | 0.21 s | 0.25 s |
| KITTI cars, held-out 40: cars under 3 s found | 94% | 94% | 94% | 94% | 94% |
| KITTI cars, held-out 40: precision | 94% | 94% | 94% | 94% | 94% |
| KITTI cars, all 200: scale error | 0.43% | 0.56% | 0.46% | 0.56% | 0.46% |
| KITTI cars, all 200: TTC error (middle) | 0.13 s | 0.13 s | 0.12 s | 0.13 s | 0.12 s |
| KITTI cars, all 200: cars under 3 s found | 96% | 97% | 96% | 97% | 95% |
| KITTI cars, all 200: precision | 98% | 97% | 98% | 97% | 98% |
| nuScenes cars: scale error | 0.34% | 0.34% | 0.35% | 0.34% | 0.35% |
| nuScenes cars: TTC error (middle) | 0.94 s | 1.03 s | 0.95 s | 1.03 s | 0.96 s |
| nuScenes cars: cars under 3 s found | 87% | 84% | 87% | 84% | 87% |
| nuScenes cars: precision | 80% | 86% | 85% | 86% | 85% |
