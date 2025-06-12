# *Build a Pre-trained Language Model (LM) from scratch with Transformer architecture*

---

# I. Problem and issue

1. Mô hình đang gặp tình trạng **Catastrophic Forgetting**, hiện đang tìm hiểu cách giải quyết.

**[UPDATE]**: Đã phần nào giải quyết được tình trạng **Catastrophic Forgetting**, nhưng chưa triệt để, cần nâng cấp thêm.

2. **Vấn đề cách huấn luyện**
   - **Mô tả vấn đề:**
     - **Dữ liệu huấn luyện:** Chỉ có câu: *"Cho tới nay, bánh mì vẫn là món ăn phổ biến được yêu thích của người Việt Nam"*.
     - **Hạn chế:** Khi input là *"bánh mì"*, mô hình không thể sinh ra câu *"bánh mì vẫn là món ăn phổ biến..."* mà thay vào đó sinh ra một câu khác không liên quan.
     - **Dự đoán nguyên nhân:** Mô hình phụ thuộc quá nhiều vào **vị trí của từ** trong dữ liệu huấn luyện, dẫn đến việc mặc định từ *"bánh mì"* không thể đứng đầu câu.
   - **Cách huấn luyện hiện tại:**
     - Với dữ liệu **pre-train**:
       - Input: `[BOS] + seq + [SEP] + seq`
       - Target: `seq + [SEP] + seq + [EOS]`
     - Với dữ liệu **fine-tune**:
       - Input: `[BOS] + req + [SEP] + res`
       - Target: `req + [SEP] + res + [EOS]`
   - **Hướng xử lý:** Thay đổi cách huấn luyện để giảm pos bias.

3. Chưa thể tối ưu không gian lưu trữ data huấn luyện (sẽ giải quyết sau).

---

# II. Introduction

## 1. **Data**

### a, Pre-train data:

*Crawl từ wikipedia theo các chủ đề, tách thành từng câu một. Chỉ lấy những câu có độ dài `5 < x < 100`. Dùng VNCoreNLP để tách token, tạo ra vocab.txt. Duyệt qua vocab, nếu từ nào có tần suất xuất hiện dưới 10 lần thì xoá khỏi vocab.txt. Duyệt lại data, câu nào có chứa từ đã bị xoá thì xoá câu đó luôn. Mục đích là vì mô hình còn nhỏ, nên sẽ ưu tiên huấn luyện những từ ngữ thông dụng nhất.*

- **Lịch sử:** Chú trọng vào lịch sử Việt Nam qua các thời kì và các nhân vật nổi tiếng.

- **Ẩm thực:** Ẩm thực Việt Nam.

- **Thể thao:** Định nghĩa những môn thể thao phổ biến, 1 vài cá nhân, tập thể.

- **Thiết bị gia dụng:** Những thiết bị gia dụng thường gặp trong nhà.

- **Công nghệ:**

- **Thời tiết:**

- **Giao thông:** Những phương tiện giao thông.

- **Giáo dục:**

- **Gia đình:** Tập trung vào định nghĩa những mối quan hệ giữa các thành viên trong gia đình.

### b, Fine-tune data: Tự viết :))

## 2. **Architecture**

### a, DecoderBlock

- **Multi-heads Attention:** Use causal_mask.

- **Feed Forward:** Sử dụng hàm kích hoạt `relu`.

### b, Model

- *`pos_embedding` hơi đơn giản, sẽ nâng cấp.*

```python
self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)

self.decoder_blocks = [
    DecoderBlock(d_model, num_heads, ff_dim, dropout)
    for _ in range(num_layers)
]
self.dropout_layer = layers.Dropout(dropout)
self.final_layer = layers.Dense(vocab_size, activation="softmax")
```

```python
seq_len = tf.shape(inputs)[1]
positions = tf.range(start=0, limit=seq_len, delta=1)
x = self.token_embedding(inputs) + self.pos_embedding(positions)

x = self.dropout_layer(x, training=training)

for block in self.decoder_blocks:
    x = block(x, training=training)

return self.final_layer(x)
```

## 3. **Pre-train** 

*(Đây là cách làm hiện tại, đang tìm hiểu cách khác để có hiệu quả tốt hơn)*

**Mục đích:** Huấn luyện mô hình dạng autoregressive để học cách đoán từ kế tiếp trong câu.

**Cấu trúc dữ liệu:** Giả sử có một câu: "Tôi đang học".

**Input:** [BOS] + tôi + đang + học + [SEP] + tôi + đang + học

**Target:** tôi + đang + học + [SEP] + tôi + đang + học + [EOS]

*(Như đã nói ở trên, chuẩn bị data như thế này chưa ổn)*

**Các vấn đề hiện tại:**

- **Mô hình quá phụ thuộc vào vị trí từ (position bias):** Nguyên nhân là vì dùng absolute positional encoding (sin-cos) hoặc data chưa được xáo trộn đủ. Giải pháp: Dùng relative positional encoding.

- **Mô hình học vẹt (memorization):** Ví dụ req là "Đinh Tiên Hoàng" thì res sẽ là "đinh bộ lĩnh lên_ngôi hoàng_đế" (lấy luôn một câu trong data). Tin vui là mô hình đã hiểu được Đinh Tiên Hoàng là Đinh Bộ Lĩnh (maybe :))) Nhưng haizzz, vấn đề là mô hình học vẹt 100%.

- Trong 1 diễn biến khác, nếu req là "việt nam" thì res sẽ là "việt_nam được yêu thích của người việt_nam là cà_phê được yêu thích của người việt_nam , đặc_biệt là giới sinh_viên và người việt_nam , đặc_biệt là giới sinh_viên và người lao_động". Ngược lại với bên trên, lần này tin vui là mô hình không copy nguyên câu từ data mà cố gắng sinh câu mới, cho thấy khả năng generalization sơ khai, nhưng nhược điểm nhỏ là ngữ nghĩa lủng củng (cái này thì có thể khắc phục được bằng cách mở rộng data). 

**Evaluation:**
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** `adam`
