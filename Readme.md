### Mô hình đang gặp trình trạng Catastrophic Forgetting, đang tìm hiểu cách giải quyết.
### [UPDATE] Đã phần nào giải quyết được trình trạng Catastrophic Forgetting, nhưng cần nâng cấp thêm.

---

## Ưu tiên giải quyết
### Vấn đề: 
- Data huấn luyện chỉ có : \"Cho tới nay, bánh mì vẫn là món ăn phổ biến được yêu thích của người Việt Nam\"
- Nếu input là "bánh mì" thì mô hình không thể dự đoán được những từ tiếp theo một cách chính xác, dự đoán lí do là vì cách học [BOS] + text + [SEP] + text + [EOS], vì từ bánh mì chưa bao giờ đứng đầu câu cả, nên mô hình không dự đoán chính xác được. Hướng xử lí, với mỗi dòng data, tạo ra nhiều data nữa bằng cách dịch phải dần dần, nhưng thách thức là đến cuối câu có thể tạo ra data vô nghĩa. 
- Nếu input là "bánh mì" thì mô hình sẽ không thể gen ra "\bánh mì vẫn là món ăn phổ biến...\" mà sẽ gen ra 1 câu khác không liên quan.
- **Dự đoán lí do:** Mô hình đang dựa quá nhiều và vị trí của từ trong data huấn luyện, khiến cho mô hình mặc định từ đấy không thể đứng đầu câu.
- **Cách huấn luyện hiện tại:**
    - **Với pre_train data**:
        - inp = [BOS] + sequence + [SEP] + sequence
        - tgt = sequence + [SEP] + sequence + [EOS]
    - **Với fine_tune data**:
        - inp = [BOS] + req_ids + [SEP] + res_ids
        - tgt = req_ids + [SEP] + res_ids + [EOS]
    
---

### Chưa thể tối ưu không gian lưu trữ data huấn luyện (giải quyết sau).
