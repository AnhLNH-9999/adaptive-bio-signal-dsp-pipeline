# Tuần 5 — Fixed-point Q15

## `float_to_q15()` / `q15_to_float()`
- Chuẩn Q15: scale = 2¹⁵ = 32768, clamp về [−1, 32767/32768] **trước** khi nhân để tránh tràn int16
  (1,0 × 32768 = 32768 vượt giới hạn 32767).
- Unit test biên (−1, 0, gần 1) PASS. Sai số làm tròn ~1,24e−05, khớp lý thuyết 2⁻¹⁵.

## `lms_filter_q15()`
- Mô phỏng đúng phép tính phần cứng DSP: mỗi tích Q15 × Q15 = Q30, dịch phải 15 bit để về Q15;
  dùng int64 làm trung gian; bão hòa (clip) về [−32768, 32767] sau mỗi bước cộng dồn.
- Chạy trên tín hiệu Tuần 1 (sine 10 Hz, μ = 0,05, N = 4), lưu cả bản float và Q15.

## SQNR lần đầu (chưa chuẩn hóa)
- **SQNR = 2,38 dB**, thấp hơn nhiều mục tiêu 30 dB.
- Nguyên nhân: tín hiệu d (biên độ tối đa 2,2136) vượt khoảng [−1, 1) của Q15 nên bị bão hòa nghiêm
  trọng. Bằng chứng: giá trị 32767 lặp lại trong `e_q15` ngay từ đầu.

## Chuẩn hóa + SQNR cuối cùng
- Chuẩn hóa `d = d / (max|d| × 1,05)` → biên độ tối đa 0,9524 (chừa 5% để tránh chạm biên).
- **SQNR sau chuẩn hóa = 50,31 dB**, đạt mục tiêu ≥ 30 dB.
- Cải thiện ~48 dB chỉ từ một thao tác chuẩn hóa: vấn đề Q15 nằm ở bước chuẩn bị dữ liệu đầu vào,
  không phải bản thân thuật toán lượng tử hóa.

Tái tạo: `python scripts/week5_fixedpoint/day3_run_lms_q15.py && python scripts/week5_fixedpoint/day4_sqnr.py`.
