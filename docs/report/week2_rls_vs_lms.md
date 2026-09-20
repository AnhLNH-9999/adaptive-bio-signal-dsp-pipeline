# Tuần 2 — So sánh RLS và LMS

## Tóm tắt

Hai bộ lọc thích nghi RLS và LMS được cài đặt trên cùng một giao diện hàm
`filter(d, x_ref, tham_so, N) -> (e, w)` để đảm bảo so sánh công bằng trên cùng dữ liệu đầu vào.

**Ảnh hưởng của forgetting factor λ** (tín hiệu sine 10 Hz tổng hợp, N = 4): λ càng gần 1 cho sai số
xác lập càng thấp. λ = 0,9 gây mất ổn định nghiêm trọng (RMS cuối tăng vọt). Đây là hiện tượng
*covariance wind-up* kinh điển của RLS: tín hiệu tham chiếu chỉ là một sóng sine đơn tần nên không đủ
đa dạng để kích thích mọi hướng của ma trận hiệp phương sai P, khiến P nổ dần theo các hướng không
được kích thích khi λ nhỏ. Vì vậy λ = 0,99 được chọn làm giá trị khuyến nghị.

**Trên kênh EEG thật** (BNCI2014_001, kênh C3, nhiễu điện lưới 50 Hz tiêm nhân tạo với biên độ gấp đôi
độ lệch chuẩn tín hiệu): RLS (λ = 0,99) hội tụ nhanh hơn rõ rệt trong khoảng 250 mẫu đầu, LMS
(μ = 0,01) cần nhiều mẫu hơn. Từ khoảng mẫu 300 trở đi hai đường sai số trùng khít, tức với bài toán
loại bỏ một tần số đơn, hiệu năng xác lập của hai thuật toán gần như tương đương.

**Chi phí tính toán**: đổi lại tốc độ hội tụ, RLS đắt hơn đáng kể. Đo trên cùng máy (5000 mẫu, N = 8),
RLS Python thuần chậm hơn LMS ~4,5 lần; bản Cython tỷ lệ giãn thành ~7,2 lần. Cả hai gần khớp N = 8,
đúng với độ phức tạp O(N²) mỗi mẫu của RLS so với O(N) của LMS. Cython tăng tốc cả hai (LMS ~111×,
RLS ~70×) nhưng không thay đổi tỷ lệ tương đối vốn xuất phát từ bản thân thuật toán.

**Kết luận**: RLS phù hợp khi cần hội tụ nhanh và λ được chọn cẩn thận; LMS đơn giản, rẻ hơn mỗi mẫu và
ổn định số học hơn vì không cập nhật ma trận N×N. Với bài toán khử nhiễu EEG chỉ cần loại 1–2 tần số
cố định, cả hai đều khả thi. Ưu tiên LMS hoặc RLS-Cython khi cần thời gian thực, ưu tiên RLS khi
artifact thay đổi nhanh theo thời gian.

## Bảng 1 — RLS: RMS theo λ (sine 10 Hz tổng hợp, N = 4)

| λ | RMS 10% đầu | RMS 10% cuối |
|---|---|---|
| 0,90 | 0,3052 | 0,4876 ⚠️ mất ổn định (covariance wind-up) |
| 0,95 | 0,3041 | 0,1187 |
| 0,99 | 0,3034 | 0,1163 |
| 0,995 | 0,3033 | 0,1157 |
| 1,00 | 0,3032 | 0,1150 |

## Bảng 2 — RLS vs LMS trên cùng tín hiệu (N = 4)

| Thuật toán | Tham số | RMS 10% đầu | RMS 10% cuối |
|---|---|---|---|
| LMS | μ = 0,05 | 0,8138 | 0,1261 |
| RLS | λ = 0,99 | 0,3034 | 0,1163 |

## Bảng 3 — Thời gian chạy (5000 mẫu, N = 8, cùng máy)

| Bộ lọc | Thời gian (ms) | So với Python thuần |
|---|---|---|
| LMS (Python thuần) | 12,23 | 1× |
| LMS (Cython) | 0,11 | ~111× nhanh hơn |
| RLS (Python thuần) | 55,41 | 1× |
| RLS (Cython) | 0,79 | ~70× nhanh hơn |

Tỷ lệ RLS/LMS: Python thuần ≈ 4,5×; Cython ≈ 7,2×.

Tái tạo: `python scripts/week2_rls/lambda_sweep.py` và `python scripts/week2_rls/benchmark_all.py`.
