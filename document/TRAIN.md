# Huấn luyện mô hình sinh văn bản theo 2 giai đoạn: Pre-train và Fine-tune

## Mục tiêu

Huấn luyện một mô hình sinh văn bản (text generation model) theo hai giai đoạn:

1. **Pre-train**: Dạy mô hình hiểu ngôn ngữ nói chung, học mối liên hệ giữa các từ trong câu.
2. **Fine-tune**: Dạy mô hình trả lời đúng yêu cầu đầu vào theo định dạng yêu cầu → phản hồi.

---

## Giai đoạn 1: Pre-train

### Mục đích:
- Huấn luyện mô hình dạng autoregressive để học cách **đoán từ kế tiếp** trong câu.

### Cấu trúc dữ liệu:

Giả sử có một câu: "Tôi đang bật đèn phòng khách"

#### Input: Tôi đang bật đèn phòng khách [EOS]

**Lưu ý**:
- Target là input dịch trái 1 bước.
- BOS không tính loss.
- Loss chỉ tính từ token đầu tiên của target đến trước [EOS].

---

## Giai đoạn 2: Fine-tune

### Mục đích:
- Dạy mô hình phản hồi đúng câu hỏi (request → response).
- Học cấu trúc định dạng: `[REQ] → [RES]`.

### Cấu trúc dữ liệu:

Giả sử có cặp dữ liệu: 
Request: Bật quạt phòng ngủ lên mức 2
Response: Đã bật quạt phòng ngủ lên mức 2

#### Input: [BOS] Bật quạt phòng ngủ lên mức 2 [SEP] Đã bật quạt phòng ngủ lên mức 2
#### Target: [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [SEP] Đã bật quạt phòng ngủ lên mức 2 [EOS]

> Số lượng `[PAD]` đúng bằng số token trong phần request (kể cả dấu câu).  
> Thêm `[PAD] [SEP]` để target khớp chiều dài với input.  
> Các `[PAD]` không được tính loss → bạn cần `loss_mask`.

---

## Mục tiêu huấn luyện

- **Giai đoạn 1**: Mô hình học ngữ pháp, ngữ nghĩa, cấu trúc tiếng Việt.
- **Giai đoạn 2**: Mô hình học "khi gặp câu hỏi thì nên sinh phản hồi nào".

---

## Tóm tắt lại

| Giai đoạn | Input                                | Target                                     | Tính loss ở đâu?                          |
|----------|--------------------------------------|--------------------------------------------|-------------------------------------------|
| Pre-train | `[BOS] A B C D`                     | `A B C D [EOS]`                            | Từ `A` đến `EOS`                          |
| Fine-tune | `[BOS] REQ [SEP] RES`               | `[PAD]*len(REQ) + [PAD] + RES + [EOS]`    | Từ token sau `[SEP]` đến trước `[EOS]`   |

---

## Tips kỹ thuật khi implement:

- Sử dụng `attention_mask` để che `[PAD]` ở input.
- Sử dụng `loss_mask` (hoặc `sample_weight`) để che `[PAD]` ở target.
- Nếu dùng HuggingFace, có thể dùng `labels = -100` cho các vị trí không tính loss.
