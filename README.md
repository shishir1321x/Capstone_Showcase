# Unsupervised Domain Adaptation for CT → MRI Organ Segmentation

A 3D U-Net trained only on CT loses more than half its accuracy when you point it at MRI. This repo closes most of that gap at inference time, using no MRI labels, no retraining, and no second network.

**0.495 → 0.721 mean Dice on MRI**, against an in-domain oracle of ~0.80. That recovers **74% of the domain gap** for the cost of a forward pass.

Built on [AMOS-22](https://amos22.grand-challenge.org/) and [MONAI](https://monai.io/), and it runs on one consumer GPU.

---

## The problem

CT intensities are calibrated: Hounsfield units mean the same thing on every scanner, with air near −1000. MRI intensities are arbitrary and depend on the sequence, the scanner, and the vendor. A network trained on CT learns to associate each organ with a CT-specific intensity signature that simply is not present in MRI.

The result is domain shift. Our CT model scores ~0.80 Dice in-domain and 0.495 on MRI, and the gap does not close with longer training. The two validation curves separate inside the first fifty epochs and never reconverge, so this is a distribution mismatch rather than an undertrained model.

Annotating a fresh MRI dataset would fix it, but expert annotation is the most expensive resource in medical imaging. Hence adaptation.

## The method

Two steps, both applied at inference, neither touching a learned weight.

**1. Percentile intensity normalization.** The fixed `[-175, 250]` HU window that suits CT maps most MRI voxels to a meaningless clipped value. Instead, rescale each MRI volume by its own 0.5 and 99.5 intensity percentiles into `[0, 1]`:

```
x̂ = clip((x − q₀.₅(x)) / (q₉₉.₅(x) − q₀.₅(x)), 0, 1)
```

Percentiles rather than min/max, so a handful of outlier voxels cannot distort the mapping. Same principle nnU-Net uses.

**2. Adaptive Batch Normalization (AdaBN).** Percentile scaling fixes the input; the internal features are still calibrated to CT. AdaBN freezes every weight and bias and pushes unlabelled MRI patches forward so each BatchNorm layer re-estimates its running mean and variance from target data. No gradients, no learning, only the normalization statistics move.

This is why the backbone uses BatchNorm rather than InstanceNorm. InstanceNorm keeps no running statistics, so there would be nothing for AdaBN to recalibrate.

## Results

Mean Dice over the 13 organs annotated in the AMOS-22 MRI subset. No stage uses a target label.

| Stage | Setting | Mean Dice | Labels | Gap closed |
|:--|:--|:--:|:--:|:--:|
| 1 | Naive transfer | 0.495 | No | — |
| 2 | Percentile normalization | 0.642 | No | 48.2% |
| 3 | Percentile + AdaBN | **0.721** | No | **74.1%** |
| 4 | Oracle (in-domain CT) | ~0.80 | n/a | — |

Gap closed is `(s_DA − s_B) / (s_O − s_B)`, the M3DA protocol. It is measured against an oracle with the same architecture and training budget, so it says how much of *this* model's lost accuracy comes back, not that the absolute numbers are state of the art.

### Per organ

| Organ | Naive | Percentile | +AdaBN | Oracle | Total gain |
|:--|:--:|:--:|:--:|:--:|:--:|
| Spleen | 0.750 | 0.796 | 0.855 | 0.947 | +0.105 |
| Right kidney | 0.707 | 0.765 | 0.825 | 0.936 | +0.118 |
| Left kidney | 0.727 | 0.782 | 0.748 | 0.769 | +0.021 |
| Gallbladder | 0.173 | 0.410 | 0.468 | 0.679 | +0.295 |
| Esophagus | 0.391 | 0.437 | 0.496 | 0.716 | +0.105 |
| Liver | 0.769 | 0.861 | 0.920 | 0.960 | +0.151 |
| Stomach | 0.533 | 0.730 | 0.789 | 0.904 | +0.256 |
| Aorta | 0.630 | 0.628 | 0.687 | 0.907 | +0.057 |
| Postcava (IVC) | 0.483 | 0.499 | 0.558 | 0.849 | +0.075 |
| Pancreas | 0.589 | 0.734 | 0.793 | 0.834 | +0.204 |
| Right adrenal gland | 0.266 | 0.360 | 0.419 | 0.670 | +0.153 |
| Left adrenal gland | 0.343 | 0.574 | 0.586 | 0.607 | +0.243 |
| Duodenum | 0.321 | 0.445 | 0.504 | 0.741 | +0.183 |

Normalization helps most where the two modalities look least alike. The gallbladder more than doubles; the aorta, already transferring well, barely moves. AdaBN then improves 12 of the 13 organs, the left kidney being the one small exception (0.782 → 0.748, already near its own 0.769 oracle).

The ceiling is visible in the last two columns. Small, thin, low-contrast structures keep a large residual gap — the right adrenal gland reaches 0.419 against an oracle of 0.670 — because their failure is structural, not just a matter of intensity.

### Ablation

Same frozen checkpoint in every row, so the differences come from the adaptation alone.

| Percentile norm. | AdaBN | Mean MRI Dice |
|:--:|:--:|:--:|
| — | — | 0.495 |
| — | ✓ | *not yet reported* |
| ✓ | — | 0.642 |
| ✓ | ✓ | **0.721** |

<!-- TODO: fill in the AdaBN-only configuration. -->

## Setup

```bash
git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt
```

Tested on Python 3.12 with PyTorch, MONAI, NiBabel and NumPy, on a single NVIDIA RTX 3090. Mixed precision is on by default, so 3D training fits in consumer VRAM.

## Data

Get AMOS-22 from the [official challenge page](https://amos22.grand-challenge.org/) and point the config at it. The split follows the AMOS convention: case IDs ≤ 500 are CT, above 500 are MRI.

Two of the 15 classes, bladder and prostate/uterus, are not annotated in the released MRI subset, so all cross-modality numbers here are over the remaining 13 organs.

## Usage

Three stages, each runnable on its own.

```bash
# 1. Resample every volume to 1.5 × 1.5 × 2.0 mm. Run once; the output is reused.
python <preprocess_script>.py

# 2. Train the source U-Net on CT. 300 epochs, best-validation checkpoint kept.
python <train_script>.py

# 3. Evaluate naive / percentile / +AdaBN / oracle from that one checkpoint.
python <evaluate_script>.py
```

<!-- TODO: replace the three placeholders above with the real filenames and flags. -->

Because adaptation happens only at inference, all four benchmark settings come from a single set of trained weights. Any difference between them is the adaptation, not training variance.

## Configuration

| Parameter | Value | Role |
|:--|:--|:--|
| `SEED` | 42 | Fixed random seed |
| `CT_MAX_ID` | 500 | IDs ≤ 500 are CT, above are MRI |
| `NUM_CLASSES` | 16 | 15 organs plus background |
| `SPACING` | (1.5, 1.5, 2.0) mm | Common resampled voxel spacing |
| `CT_WINDOW` | (−175, 250) HU | Soft-tissue window for source CT |
| `PATCH` | (96, 96, 96) | Foreground-biased training patch |
| `CHANNELS` | (32, 64, 128, 256, 320) | U-Net channel widths |
| `STRIDES` | (2, 2, 2, 2) | Downsampling strides |
| `BATCH` | 2 | Patches per step |
| `LR` | 2e-4 | AdamW learning rate |
| `NUM_EPOCHS` | 300 | Source training length |
| `VAL_INTERVAL` | 5 | Epochs between validation passes |
| `CACHE_RATE` | 0.0 | No in-memory caching, bounds host RAM |

Loss is Dice + cross-entropy. Optimizer is AdamW. Full volumes are segmented by sliding-window inference over overlapping 96³ patches.

## What this does not do

- **One shift, one dataset.** CT → MRI on AMOS-22 only. The reverse direction is untested.
- **Dice only.** No boundary-based metrics (95th-percentile Hausdorff, average symmetric surface distance), so we cannot claim the recovered segmentations are boundary-accurate, which is what matters clinically.
- **Short schedule.** Training is shorter than the full-length schedules in the original benchmarks. The relative trends hold; the absolute numbers sit below the state of the art.
- **A hard ceiling.** ~0.08 mean Dice of residual gap remains. Statistical alignment cannot fix organs whose failure is structural.
- **CycleGAN did not work here.** A 2D CycleGAN produced anatomically faithful CT → MRI translations, but the 2D segmenter trained on them under-fit at our data scale and never beat the naive baseline. The bottleneck was the segmenter, not the translation.

## Where it goes next

1. **Graph neural networks** to model inter-organ topology and supply the anatomical prior that intensity correction cannot.
2. **3D CycleGAN** with a stronger segmentation backbone, since the translations themselves were sound.
3. **Full AMOS-22** plus histogram matching and bias-field correction for the acquisition-level variation percentile scaling leaves behind.
4. **Head-to-head comparison** against SIFA, entropy minimisation, Fourier domain adaptation and gradient-based test-time adaptation on the same benchmark, with boundary-aware metrics throughout.

## Built on

- Ji et al., *AMOS: A Large-Scale Abdominal Multi-Organ Benchmark* (2022)
- Li et al., *Adaptive Batch Normalization for Practical Domain Adaptation* (2018)
- Çiçek et al., *3D U-Net* (2016) and Ronneberger et al., *U-Net* (2015)
- Isensee et al., *nnU-Net* (2021), whose percentile clipping is the direct precedent for step 1
- Shirokikh et al., *M3DA* (2025), whose protocol we report against
- Cardoso et al., *MONAI* (2022)

Full bibliography in the report.

## Citing

<!-- TODO: update once the report has a permanent link or DOI. -->

```bibtex
@misc{amos22_uda,
  title  = {Unsupervised Domain Adaptation for Medical Image Segmentation on AMOS-22},
  author = {Alam, MD. Shaiful and Ratul, Nahid Sarwar and
            Hossain, MD. Jarif Mehtab and Ahmad, Abu},
  year   = {2026},
  note   = {Senior Design Project, North South University},
  url    = {<repo-url>}
}
```

## Authors

MD. Shaiful Alam · Nahid Sarwar Ratul · MD. Jarif Mehtab Hossain · Abu Ahmad

Supervised by Dr. Mohammad Abdul Qayum, Department of Electrical and Computer Engineering, North South University.

## License

<!-- TODO: pick a license, add a LICENSE file, and name it here. -->

AMOS-22 carries its own licence. Check it before redistributing any data or derived weights.
