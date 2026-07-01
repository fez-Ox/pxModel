# Backbone Comparison Results

**Generated:** 2026-06-30 13:39:33

## Configuration

| Parameter | Value |
|---|---|
| Image size | 224 |
| Batch size | 32 |
| Epochs (phase 1) | 10 |
| Epochs (phase 2) | 25 |
| Dropout | 0.3 |

## Summary

| Backbone | Params | Size (MB) | Mean F1 | Inf (ms) | Time | Ckpt (MB) |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 4.34M | 16.5 | 0.7945 | 0.2 | 2m 40s | 16.8 |
| efficientnet_b3 | 11.09M | 42.3 | 0.7791 | 0.4 | 2m 46s | 42.9 |
| efficientnet_b7 | 64.44M | 245.8 | 0.7963 | 1.2 | 3m 3s | 247.4 |
| mobilenet_v3_large | 3.22M | 12.3 | 0.7903 | 0.1 | 2m 44s | 12.5 |
| convnext_tiny | 28.02M | 106.9 | 0.8387 | 0.5 | 2m 49s | 106.9 |
| convnext_small | 49.65M | 189.4 | 0.8549 | 0.8 | 2m 2s | 189.5 |
| convnext_base | 87.83M | 335.0 | 0.8789 | 1.2 | 2m 28s | 335.2 |

## Per-backbone Details

### efficientnet_b0

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9189 | 0.9273 | 0.9231 |
| plastic_wrap | 0.3696 | 0.7727 | 0.5000 |
| sealed | 0.9120 | 0.8906 | 0.9012 |
| open | 0.8000 | 0.9157 | 0.8539 |

### efficientnet_b3

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9020 | 0.8364 | 0.8679 |
| plastic_wrap | 0.4571 | 0.7273 | 0.5614 |
| sealed | 0.9016 | 0.8594 | 0.8800 |
| open | 0.8072 | 0.8072 | 0.8072 |

### efficientnet_b7

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9519 | 0.9000 | 0.9252 |
| plastic_wrap | 0.4706 | 0.7273 | 0.5714 |
| sealed | 0.9381 | 0.8281 | 0.8797 |
| open | 0.7400 | 0.8916 | 0.8087 |

### mobilenet_v3_large

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9333 | 0.8909 | 0.9116 |
| plastic_wrap | 0.4444 | 0.7273 | 0.5517 |
| sealed | 0.8992 | 0.8359 | 0.8664 |
| open | 0.7789 | 0.8916 | 0.8315 |

### convnext_tiny

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9444 | 0.9273 | 0.9358 |
| plastic_wrap | 0.6818 | 0.6818 | 0.6818 |
| sealed | 0.9237 | 0.8516 | 0.8862 |
| open | 0.7857 | 0.9277 | 0.8508 |

### convnext_small

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9528 | 0.9182 | 0.9352 |
| plastic_wrap | 0.6818 | 0.6818 | 0.6818 |
| sealed | 0.9496 | 0.8828 | 0.9150 |
| open | 0.8316 | 0.9518 | 0.8876 |

### convnext_base

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.9459 | 0.9545 | 0.9502 |
| plastic_wrap | 0.6667 | 0.8182 | 0.7347 |
| sealed | 0.9825 | 0.8750 | 0.9256 |
| open | 0.8438 | 0.9759 | 0.9050 |
