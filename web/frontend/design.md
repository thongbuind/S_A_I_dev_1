# Hệ Thống Thiết Kế SAI (Design System)

Tài liệu này quy định các tiêu chuẩn thiết kế cho giao diện web **SAI (Powered by Nidai)**. Giao diện được xây dựng dựa trên phong cách tối giản, hiện đại, sử dụng hiệu ứng kính (Glassmorphism) và tông màu tím không gian (Deep Purple) làm chủ đạo.

## 1. Triết lý Thiết kế
- **Tính Hiện đại:** Sử dụng các dải màu gradient và hiệu ứng đổ bóng mềm mại.
- **Tính Tập trung:** Nội dung trò chuyện là trung tâm, các thành phần điều hướng (sidebar) có độ tương phản thấp hơn để giảm xao nhãng.
- **Hiệu ứng Kính (Glassmorphism):** Sử dụng độ trong suốt (`rgba`) và `backdrop-filter: blur` để tạo chiều sâu.

## 2. Kiến trúc Theme

Dark theme là hệ quy chiếu chính. Theme chỉ cung cấp **bảng màu RGB thuần**;
không theme nào được tự định nghĩa opacity, gradient, blur, border, shadow,
spacing, sizing, animation hoặc behavior của component.

### 2.1. Màu Nền & Bề Mặt (Background & Surface)
| Biến palette | RGB | Mô tả |
| :--- | :--- | :--- |
| `--color-bg` | `14 10 26` | Màu nền chính |
| `--color-bg-alt` | `19 13 36` | Màu nền phụ |
| `--color-surface` | `14 10 26` | Sắc nền cho surface |
| `--color-surface-elevated` | `10 6 22` | Sắc nền cho island/dropdown |

### 2.2. Màu Nhấn (Accent Colors)
| Biến palette | RGB | Mô tả |
| :--- | :--- | :--- |
| `--color-accent` | `124 58 237` | Màu thương hiệu chính |
| `--color-accent-deep` | `75 0 130` | Màu sâu cho gradient |
| `--color-accent-soft` | `167 139 250` | Màu accent mềm |
| `--color-shadow` | `75 0 130` | Màu dùng trong shadow recipe |

### 2.3. Màu Văn Bản (Typography Colors)
| Biến palette | RGB | Mô tả |
| :--- | :--- | :--- |
| `--color-text-primary` | `232 231 252` | Chữ chính |
| `--color-text-secondary` | `232 231 252` | Sắc gốc cho chữ phụ |
| `--color-text-muted` | `94 82 120` | Chữ ghi chú và placeholder |

### 2.4. Kiến trúc token

```text
theme.css
 └── palette RGB: background, surface, text, brand, semantic, particle

ui-foundation.css
 └── material recipe: blur, saturation và fallback dùng chung

landing.css / chat/style.css
 └── component recipe duy nhất: opacity, gradient, border, shadow và state
```

Mọi màu có alpha phải được tạo bên ngoài palette bằng cú pháp
`rgb(var(--color-name) / alpha)`. Nhờ vậy một theme mới chỉ thay các RGB
channel; vật liệu và behavior của component được giữ nguyên.

`theme.js` là nơi duy nhất đọc và ghi khóa `sai-theme`. Script phải được tải
trong `<head>` trước stylesheet của trang để tránh nháy sai theme khi tải trang.

## 3. Kiểu Chữ (Typography)
- **Font chính:** `DM Sans`, `ui-sans-serif`, system-ui.
- **Font tiêu đề nghệ thuật:** `Playfair Display` (cho số liệu) hoặc `Cormorant Garamond` (cho trích dẫn).
- **Trọng lượng (Weight):**
  - **Light (300):** Dùng cho mô tả dài, trích dẫn.
  - **Regular (400):** Văn bản nội dung chat.
  - **Medium/Semi-Bold (500/600):** Tiêu đề nhỏ, nút bấm, tên Model.
  - **Bold (700/800):** Tiêu đề chính (Hero Title), Logo.

## 4. Các Thành Phần Giao Diện (Components)

### 4.1. Khung Chat (Chat Bubbles)
- **User Message:**
    - **Nền:** Gradient chéo (`135deg`) từ `rgba(124,58,237,0.15)` đến `rgba(75,0,130,0.22)`.
    - **Viền:** `1px solid rgba(124,58,237,0.28)`.
    - **Đổ bóng:** `inset 0 1px 0 rgba(232,231,252,0.07)` (Tạo độ nổi khối nhẹ).
- **AI Message:**
    - **Nền:** `rgba(124,58,237,0.08)` kết hợp với `backdrop-filter: blur(12px)`.
    - **Viền:** `1px solid rgba(124,58,237,0.18)`.
    - **Chữ:** Căn đều (justify) để tạo cảm giác chuyên nghiệp.

### 4.2. Thanh Nhập Liệu (Input Box)
- **Cấu trúc:** Một `input-wrapper` chứa `textarea` và các nút chức năng.
- **Nền:** `linear-gradient(135deg, rgba(124,58,237,0.12), rgba(75,0,130,0.18)), rgba(18,10,35,0.72)`.
- **Focus:** Khi người dùng nhập liệu, `box-shadow` sẽ chuyển sang màu `--purple-glow` để báo hiệu trạng thái hoạt động.

### 4.3. Nút Bấm (Buttons)
- **Primary (Bắt đầu ngay):** Sử dụng gradient rực rỡ, bo góc tròn hoàn toàn (pill-shaped), có đổ bóng phát sáng (`--btn-primary-shadow`).
- **Secondary (Khám phá):** Nền tối mờ, viền tím nhạt, đổi màu chữ sang trắng khi hover.

## 5. Quy tắc bắt buộc cho mọi Theme

1. **Một element, một material recipe:** Light, Dark và theme mở rộng phải dùng
   cùng số lớp background, opacity, blur, saturation, border và shadow geometry.
2. **Glass surface:** Panel lớn dùng blur `20px`; bubble/control dùng `12px`;
   elevated surface dùng `24px`; navbar dùng `28px`. Theme không được override.
3. **Bảng kiến trúc:** Wrapper luôn là kính; header, total và hover chỉ là các
   lớp accent trong suốt phủ lên cùng wrapper.
4. **Chat bubble:** User và AI bubble giữ nguyên gradient/layering, alpha,
   border, shadow và blur. Theme chỉ đổi RGB channel.
5. **Border geometry:** Width, radius và alpha là recipe của component, không
   nằm trong bảng màu.
6. **Animation và behavior:** Tốc độ, quỹ đạo, spacing, breakpoint và
   interaction không phụ thuộc theme. Canvas đọc màu từ CSS nhưng giữ một alpha
   range duy nhất.
7. **Component selectors:** Không tạo visual override bằng
   `[data-theme="light"] .component`. Selector theme ngoài `theme.css` chỉ được
   dùng cho state như icon của nút chuyển theme.
8. **Theme mới:** Thêm palette RGB trong `theme.css`; không sửa component CSS
   hoặc thêm nhánh màu/opacity trong JavaScript.

## 6. Hiệu ứng & Tương tác
- **Reveal Animation:** Các phần tử xuất hiện khi cuộn trang với độ trễ (delay) khác nhau để tạo nhịp điệu.
- **Sidebar Transition:** Sử dụng `cubic-bezier(0.34, 1.56, 0.64, 1)` cho cảm giác đàn hồi khi đóng/mở.
- **Theme Toggle:** Chuyển đổi mượt mà bằng `transition: all 0.3s ease`.
