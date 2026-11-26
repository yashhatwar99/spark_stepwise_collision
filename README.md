# Single-camera collision prediction (BTP)

Can one ordinary camera tell a driver how long until a car in front becomes a collision? This repository is a bachelor thesis project that builds up the answer in stages, measures every stage against real ground truth, and ends with **FP-TTC** (a neural network that predicts time-to-collision from two camera frames), per-car TTC readouts from it, and a **lightweight version** of it.

Everything runs on a laptop (RTX 3050, 4 GB of video memory). Datasets, model weights and the FP-TTC code are not in the repo; see [Setup](#setup).

## What is in here

| Part | Folder | What it does |
|---|---|---|
| Step-by-step pipeline | [`src/`](src/) | The first build, one script per stage: YOLO detection, tracking, ground-plane distance, Kalman smoothing, TTC and collision risk. Kept as it was. |
| Rebuild with measurements | [`src2/`](src2/) | The same ideas rewritten with one config file and evaluated against nuScenes ground truth (detection, LiDAR-verified 3D boxes, tracking, stereo, and a collision warning from 2D boxes alone). |
| 3D boxes and prediction | [`src3/`](src3/) | Camera-only 3D boxes (Deep3DBox idea), tracking over nuScenes sweeps (about 10 frames per second), predicted future boxes and a check of them against where the vehicles really went. |
| FP-TTC evaluation | [`fpttc_results/`](fpttc_results/) | Runs the released FP-TTC model on KITTI and nuScenes: accuracy, speed, figures and a results site. |
| Per-car TTC | [`per_car_ttc/`](per_car_ttc/) | Turns FP-TTC's per-pixel scale map into one TTC per car: a median method, a small learned head, and comparison pages. |
| **Lightweight FP-TTC** | [`lightweight_fpttc/`](lightweight_fpttc/) | Makes FP-TTC smaller and faster and measures what it costs in accuracy. See below. |
| Tools | [`page_tools/`](page_tools/), [`presentation/`](presentation/), [`paper_search/`](paper_search/), [`sumo/`](sumo/) | Edit-mode snippet for the results pages, slide images, a search over the papers, a synthetic SUMO scenario. |
| Sample pictures | [`results/`](results/) | One example image per pipeline stage. |

`papers.txt` lists the papers read for the project and `limitations.txt` gives a plain-language summary of each paper's limits. `workflow.txt` lists the published workflow and results pages.

## Lightweight FP-TTC: the main recent result

FP-TTC predicts a *scale* `s` for every pixel: how much an object's distance changes between two frames 0.1 s apart. TTC then follows from `s`. The full model has 12.4 M parameters. We replaced its heaviest parts and measured the trade.

- **Decoder removed:** the decoder (6 M parameters) is replaced by three small convolution layers (Conv3).
- **Flow + affine removed:** the matching stage is dropped as well. It turns out the scale encoder barely uses what that stage gives it, so nothing is lost. Two versions: **A** keeps the encoder's own `scale_conv` layer; **B** also drops it.
- **Conv3 taught by distillation:** Conv3 learns to copy the full model's scale map. This beat learning only from KITTI's truth.

Measured on 40 KITTI pairs the light model never trained on (Mid error is the paper's metric, lower is better):

| | Full FP-TTC | Version A, distilled |
|---|---|---|
| Parameters | 12.44 M | 6.17 M (-50%) |
| Time per pair, GPU | 282 ms | 222 ms (-21%) |
| Time per pair, CPU | 5.2 s | 3.9 s (-25%) |
| Mid error | 54.7 | 61.7 (+13%) |

Per-car TTC stays close to the full model on KITTI cars, and the direct-trained versions match it on nuScenes (about 87% of cars under 3 s are found). The authors report their light model as about 15% faster and about 13% less accurate; these figures are from our earlier reading of their paper and should be re-checked before quoting. Details, tables for every method and the code are in [`lightweight_fpttc/`](lightweight_fpttc/).

## Per-car TTC in short

On the full FP-TTC output, taking the median of the TTC map inside each car's box gives a per-car TTC with a median scale error of 0.56% on KITTI's held-out cars. A small learned head and the comparison against the median are in [`per_car_ttc/`](per_car_ttc/). The nuScenes check uses 1,418 cars with true TTC computed from 3D boxes and the ego pose (nuScenes has no 2D labels).

## Setup

1. **Python environment.** Create a virtual environment (the scripts assume `.venv/` in the project root) and install the packages the scripts import (PyTorch with CUDA, OpenCV, NumPy, pandas, matplotlib, ultralytics, nuscenes-devkit).
2. **FP-TTC code and weights.** Clone https://github.com/LChanglin/FP-TTC into `fpttc/` (it has its own git and is ignored here) and download the released weights it points to (`fpttc_kitti`, `fpttc_mix`).
3. **Data.** Follow [`data/README.md`](data/README.md): nuScenes v1.0-mini, KITTI scene flow 2015 (training pairs and the multiview frames), and optionally KITTI raw.
4. **Run from the project root.** Each script's header says what it needs and how to run it. For example:

```
python lightweight_fpttc/removing_decoder_only/evaluate_light.py          # full vs decoder-removed model
python lightweight_fpttc/removing_flow_affine_decoder/evaluate_versions.py   # versions A and B
python per_car_ttc/median/check_on_kitti.py                          # median per-car TTC on KITTI
```

The light-model scripts need cached features of KITTI that are not committed (1 to 3 GB each); the first training script builds them.

## How the results were measured

- **KITTI split.** Scene numbers divisible by 5 are the 40 held-out test scenes, numbers ending in 1 are 20 "quiz" scenes used only to pick the best training round, and the other 140 are for training. Only KITTI is used for training; nuScenes is test-only.
- **Mid error.** `mean |log(predicted s) - log(true s)| x 10,000` over pixels where KITTI has a true scale.
- **Timing.** One KITTI pair at 640x320, the model alone, middle of many runs, on the laptop GPU and CPU. GPU times move a few milliseconds between runs.
- **Extra frames for distillation** come from KITTI raw drive 0009. That drive contains KITTI scene flow scenes 7 to 12, so the frames near the held-out and quiz scenes were removed (`lightweight_fpttc/removing_flow_affine_decoder/distill_more_frames/check_overlap.py`).

## Known limits

- **The frozen full model may have seen the "held-out" scenes.** The released `fpttc_kitti` weights were trained by the authors on KITTI; we have not checked whether their training included our 200 scenes. Only the light model's Conv3 is certain to be unseen. nuScenes is unseen by everything.
- **The light model is less accurate on cars than on background.** Its Mid error on cars is higher than the full model's, and more training frames did not close the gap.
- **nuScenes mini has no crashes**, so it tests perception and TTC accuracy, not crash prediction itself.
- **Small samples.** The 85 held-out KITTI cars are few, and training noise moves results by a couple of points between runs.

## Credits

FP-TTC: *FP-TTC: Fast Prediction of Time-to-Collision Using Monocular Images*, Li et al., IEEE TCSVT 2025. Datasets: KITTI, nuScenes (their licences apply and the data are not redistributed here). Detection uses YOLOv8 (Ultralytics).
