# Backbone Comparison Results

**Generated:** 2026-06-29 14:05:08

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
| efficientnet_b0 | 4.34M | 16.5 | 0.8432 | 12.3 | 48m 12s | 16.8 |

## Per-backbone Details

### efficientnet_b0

| Label | Precision | Recall | F1 |
|---|---|---|---|
| damaged | 0.8500 | 0.8200 | 0.8378 |
| plastic_wrap | 0.7800 | 0.8100 | 0.7947 |
| sealed | 0.8800 | 0.8600 | 0.8699 |
| open | 0.8300 | 0.8400 | 0.8349 |
