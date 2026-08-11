# 门控边权重汇总 (causal / spurious / independent)

| 变体 | 边类型 | 均值 | 中位数 | std | 条数 |
|------|--------|------|--------|-----|------|
| pl192_full_v2 | causal | 0.8039 | 0.9027 | 0.2403 | 4 |
| pl192_full_v2 | spurious | 0.9999 | 0.9999 | 0.0000 | 16 |
| pl192_full_v2 | independent | 0.9999 | 1.0000 | 0.0001 | 44 |
| pl192_full_v2_fixed | causal | 0.9999 | 0.9999 | 0.0000 | 4 |
| pl192_full_v2_fixed | spurious | 0.9999 | 0.9999 | 0.0000 | 16 |
| pl192_full_v2_fixed | independent | 0.9999 | 0.9999 | 0.0000 | 44 |
| pl96_full_v2 | causal | 0.9809 | 0.9930 | 0.0242 | 4 |
| pl96_full_v2 | spurious | 0.9999 | 1.0000 | 0.0001 | 16 |
| pl96_full_v2 | independent | 0.9999 | 1.0000 | 0.0001 | 44 |
| pl96_full_v2_fixed | causal | 0.9999 | 0.9999 | 0.0001 | 4 |
| pl96_full_v2_fixed | spurious | 0.9999 | 0.9999 | 0.0001 | 16 |
| pl96_full_v2_fixed | independent | 0.9999 | 1.0000 | 0.0000 | 44 |

## 分离度 (causal_mean - spurious_mean)

- pl192_full_v2: -0.1960
- pl192_full_v2_fixed: -0.0000
- pl96_full_v2: -0.0190
- pl96_full_v2_fixed: -0.0000
