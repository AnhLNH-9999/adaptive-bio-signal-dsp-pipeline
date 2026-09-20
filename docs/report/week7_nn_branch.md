# Tuần 7 — Nhánh mạng neural (EEGNet-Lite)

| Mô hình | Độ chính xác test | Ghi chú |
|---|---|---|
| LDA (Wavelet C3 + CSP), Tuần 4 | 87,4 % | baseline cổ điển |
| EEGNet-Lite float32 | 88,5 % | `scripts/week7_nn/day2_train_eegnet.py` |
| EEGNet-Lite QAT 16-bit (Brevitas) | 96,6 % | `scripts/week7_nn/day3_train_qat.py` |

- SQNR trọng số (full-precision vs 16-bit) vượt mục tiêu ≥ 30 dB (`day4_eval_qat.py`).
- MACs/giây của EEGNet-Lite cao hơn RLS ~19× và LMS ~190×, nhưng độ trễ inference thực đo (~4,4 ms)
  **thấp hơn** RLS Python thuần (~7,0 ms) vì PyTorch dùng kernel C++ tối ưu cho Conv2d, trong khi RLS
  chạy vòng lặp Python từng mẫu. Đây là minh chứng rõ nhất của cả đồ án: cách cài đặt ảnh hưởng đến
  hiệu năng thực tế nhiều hơn số phép toán lý thuyết.
