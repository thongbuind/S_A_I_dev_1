# Benchmark Results — SAI 100M

- **Samples per dataset**: 100
- **Device**: mps:0
- **Date**: 2026-06-19 03:18

---

## 1. VMLU

| Metric | Value |
|--------|-------|
| Accuracy | 0.00% (0/0) |
| Skipped | 1000 mẫu (< 2 lựa chọn hợp lệ hoặc gold không hợp lệ) |
| Dataset | `tridm/VMLU` / test |
| Method | Logit-based (argmax token A/B/C/D) |


## 2. MMLU-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | 22.00% (22/100) |
| Dataset | `alexandrainst/m_mmlu` (vi) / test |
| Method | Logit-based (argmax A/B/C/D tokens) |


## 3. GSM8K-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | 2.00% (2/100) |
| Skipped | 0 mẫu (không tìm được gold number) |
| Dataset | `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated / train` |
| Method | Generate → extract last number → exact match |
| Note | Gold = số cuối trong response_vi gốc của dataset |


## 4. XNLI-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | 35.00% (35/100) |
| Dataset | `facebook/xnli` (vi) / test |
| Labels | entailment / neutral / contradiction |
| Method | Logit-based (argmax A/B/C tokens) |


