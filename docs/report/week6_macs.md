# Tuần 6 — Phân tích MACs/giây (RLS vs LMS)

## Công thức lý thuyết
- LMS = 2N MAC/mẫu (tích vô hướng + cập nhật trọng số, mỗi bước O(N)).
- RLS = 4N² MAC/mẫu (hai phép toán ma trận N×N chiếm ưu thế: `pi = P @ x` và cập nhật P bằng outer
  product, mỗi phép ước lượng ~2N²).
- Với N = 8, fs = 250 Hz: LMS 16 MAC/mẫu → 4.000 MACs/s; RLS 256 MAC/mẫu → 64.000 MACs/s.
  **Tỷ lệ lý thuyết RLS/LMS = 16,0×.**

## Đếm MAC thực tế trong code
- LMS: 16 MAC/mẫu, **khớp** lý thuyết.
- RLS: 160 MAC/mẫu, **không khớp** 256. Đếm theo nghĩa "1 MAC = 1 nhân-cộng", mỗi phép ma trận N×N
  chỉ N² (không nhân đôi) → 2N² = 128, cộng 4 phép O(N) (denom, k, w, y) mà lý thuyết bỏ qua →
  128 + 32 = 160. Với N nhỏ, các phép O(N) chiếm ~20% tổng, không hề nhỏ.

## Bảng tổng hợp (N = 8, fs = 250 Hz)

| Chỉ tiêu | LMS | RLS | Tỷ lệ RLS/LMS |
|---|---|---|---|
| MAC/mẫu (lý thuyết) | 16 | 256 | 16,0× |
| MAC/mẫu (đếm thực tế) | 16 | 160 | 10,0× |
| MACs/giây (lý thuyết) | 4.000 | 64.000 | 16,0× |
| MACs/giây (đếm thực tế) | 4.000 | 40.000 | 10,0× |
| Thời gian chạy Python (ms), Tuần 2 | 12,23 | 55,41 | 4,5× |
| Thời gian chạy Cython (ms), Tuần 2 | 0,11 | 0,79 | 7,2× |

## Nhận xét
Tỷ lệ RLS/LMS giảm dần qua ba góc nhìn: 16,0× (lý thuyết) → 10,0× (đếm thực tế) → 4,5–7,2× (đo thời
gian). Công thức 4N²/2N là ước lượng bậc độ lớn, phóng đại chênh lệch vì bỏ qua các phép O(N); thời
gian chạy thực đo còn chịu overhead Python/thư viện chứ không chỉ phản ánh số phép toán.

**Kết luận (Mục tiêu 2 của proposal):** MACs/giây hữu ích để ước lượng nhanh độ phức tạp nhưng không
thay thế được đo đạc thực tế trên nền tảng phần cứng đích. Với FPGA/MCU, **LMS an toàn hơn** về cả tốc
độ lẫn bộ nhớ (không cần lưu ma trận P N×N); **RLS chỉ nên dùng khi thực sự cần hội tụ nhanh** và phần
cứng đủ mạnh.

Tái tạo: `python scripts/week6_macs/day{1,2,3,4,5}_*.py`.
