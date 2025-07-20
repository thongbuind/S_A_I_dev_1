# Build a Pre-trained Language Model (LM) from scratch with Transformer architecture

## Thang điểm đánh giá bài kiểm tra tạo câu:
- **10**: Câu hay tuyệt đối.
- **9**: Câu đúng cấu trúc, đúng nội dung, khá ổn.
- **6-8**: Câu đúng cấu trúc, nội dung tạm đúng nhưng chung chung quá, không cụ thể hoặc nội dung chỉ gần đúng.
- **5**: Câu đúng cấu trúc, nhưng sai ngữ nghĩa, nội dung.
- **2-4**: Câu sai cấu trúc, nhưng nội dung vẫn có gì đó liên quan nhẹ
- **1**: Câu không liên quan, sai cấu trúc, các từ không có sự liên quan.

---

## Test pre-train

*Mô hình mới pre-train, chưa fine-tune nên chỉ có thể sinh văn bản dựa trên đầu vào, không yêu cầu tính chính xác của thông tin, chỉ xét cấu trúc và ngữ nghĩa.*

### Test 1
**Req:** bánh mì 

**Res:** bánh_mì không_chỉ là một món ăn mà_còn là một phần của văn_hoá ẩm_thực việt_nam , thể_hiện sự sáng_tạo và sáng_tạo của người đầu_bếp .

**Nhận xét:** Đầu vào đơn giản nên đã gen ra nửa đầu câu mượt, tuy nhiên đó là lấy nguyên từ dataset. Nửa sau thể hiện sự sáng tạo khi đã gen mới, mặc dù bị lặp từ "sáng tạo" nhưng ngữ nghĩa rất ổn.

**Điểm:** 8

---

### Test 2
**Req:** bánh mì có nguồn gốc từ 

**Res:** bánh_mì có nguồn_gốc từ các vùng miền nam , bao_gồm nhiều loại hoa và các loại hoa khác nhau , mỗi loại đều có hương_vị riêng_biệt .

**Nhận xét:** Đầu vào này yêu cầu từ nối tiếp là một danh từ, mô hình đã gen đúng. Nửa sau, mô hình đã sáng tạo, tuy đúng cấu trúc "bao gồm" + 2 danh từ nhưng ngữ nghĩa không liên quan đến bánh mì cho lắm.

**Điểm:** 7

---

### Test 3
**Req:** việt nam 

**Res:** việt_nam là một quốc_gia có nền kinh_tế mạnh_mẽ , với nhiều quốc_gia khác nhau , bao_gồm cả lào và việt_nam .

**Nhận xét:** Đầu vào cũng đơn giản, nhưng khác với câu bánh mì ở trên, mô hình đã sáng tạo, và nội dung đúng, hay. Nửa sau hoàn toàn sai.

**Điểm:** nửa đầu 10, nửa sau 2, tổng kết 6

---

### Test 4
**Req:** việt nam sở hữu 

**Res:** việt_nam sở_hữu nhiều quốc_gia đã trở_thành một trong những quốc_gia có ảnh_hưởng lớn đến nền kinh_tế và xã_hội của đất_nước .

**Nhận xét:** Có phần nào hiểu về cấu trúc câu rồi, nhưng chưa phân biệt được các danh từ với nhau, sai ngữ nghĩa.

**Điểm:** 5

---

### Test 5
**Req:** phở 

**Res:** phở thìn lò_đúc được biết đến với nhiều món ăn độc_đáo , từ ẩm_thực đến các món ăn truyền_thống , tạo nên một món ăn hấp_dẫn .

**Nhận xét:** 1/3 đầu câu sáng tạo, hay, 10 điểm. 1/3 tiếp theo cũng rất hay, có liên kết với đoạn đầu. 1/3 câu cuối đúng cấu trúc, nhưng nó nên là "tạo nên một menu hấp dẫn" thì mới đúng, hoặc là "tạo nên nhiều món ăn hấp dẫn".

**Điểm:** 9,5

---

### Test 6
**Req:** buổi sáng người việt nam thường ăn 

**Res:** buổi sáng người việt_nam thường ăn kèm với những món ăn nhẹ và được chế_biến thành các món ăn khác nhau , tạo nên sự đa_dạng trong ẩm_thực việt_nam .

**Nhận xét:** Đầu vào yêu cầu tên một món ăn (danh từ), thực ra thì kết quả cũng phần nào hiểu được, chỉ là có phần hơi lệch thôi. Nói hay thì không nhưng nói dở thì cũng không đúng.
Xét về cục bộ từng đoạn thì ổn: "thường ăn kèm với những", "được chế_biến thành các món ăn khác nhau , tạo nên sự đa_dạng", nhưng về tổng quan toàn câu thì lại chưa hay.

**Điểm:** 7

---

### Test 7
**Req:** đám mây 

**Res:** đám mây là một trong những nguyên_tố nhẹ hàng_đầu thế_giới , nổi_bật với khả_năng di_chuyển linh_hoạt và khả_năng di_chuyển linh_hoạt .

**Nhận xét:** Đầu vào là một danh từ đơn giản, mô hình đã gen ra vị ngữ đúng cấu trúc, nhưng vẫn bị lặp từ, sai ngữ nghĩa.

**Điểm:** 4,5

---

### Test 8
**Req:** Đinh Tiên Hoàng lên ngôi 

**Res:** đinh tiên hoàng lên_ngôi , một trong những triều_đại nổi_bật của triều nguyễn , đã trở_thành một trong những nhân_vật quan_trọng trong lịch_sử việt_nam .

**Nhận xét:** Đề bài yêu cầu gen ra một chữ "hoàng đế", hoặc "vua", là một chức danh gì đó nhưng mô hình lại chọn dấu phẩy ngắt ý. Phần sau thì khá ổn đúng cấu trúc ngữ nghĩa (hơi sai chỗ từ "triều đại"), cũng phần nào hiểu được nhưng chưa hay.

**Điểm:** 5

---

### Test 9
**Req:** lê thái tổ có miếu hiệu 

**Res:** lê_thái tổ có miếu_hiệu nổi_bật với những công_trình kiến_trúc độc_đáo , phản_ánh sự giao_thoa văn_hoá giữa các nền văn_minh khác nhau .

**Nhận xét:** Input đã có cấu trúc chủ ngữ + vị ngữ và còn khuyết ở cuối, nhưng mô hình lại nghĩ đầu vào là chủ ngữ/danh từ và thêm vị ngữ đằng sau. Về cấu trúc thì 1 điểm còn ngữ nghĩa thì 5 điểm, hay nhưng không liên quan.

**Điểm:** 3

---

### Test 10
**Req:** công thức 1 

**Res:** công_thức 1 là một trong những giải đua xe nổi_tiếng nhất thế_giới , nổi_bật với những thành_tích ấn_tượng và phong_cách lái_xe xuất_sắc .

**Nhận xét:** Nửa câu đầu đúng, lấy trong dataset, 9 điểm. Ở nửa sau, mô hình đã biết chủ ngữ là một danh từ nhưng lại nhầm lẫn với danh từ chỉ người dẫn đến sai nghĩa, nhưng cấu trúc thì oke, 7 điểm.

**Điểm:** 8

---

### Test 11
**Req:** sáng hôm ấy 

**Res:** hôm ấy là một trong những nhân_vật quan_trọng trong lịch_sử việt_nam , nổi_bật với những thành_tựu trong việc xây_dựng đất_nước .

**Nhận xét:** Sai cấu trúc, ngữ nghĩa không liên quan.

**Điểm:** 1,5

---

### Test 12
**Req:** sau khi ăn xong, chúng tôi đi 

**Res:** sau khi ăn xong , chúng_tôi đi kèm với nhiều loại iv , tạo nên một bức tranh đa_dạng trong văn_hoá việt_nam .

**Điểm:** 1

---

### Test 13
**Req:** mặc dù 

**Res:** mặc_dù không có con sông lớn , nhưng cũng có nhiều loại khác nhau , nhưng cũng đã trở_thành một phần không_thể thiếu trong việc phát_triển kinh_tế của đất_nước .

**Nhận xét:** Đúng cấu trúc "mặc dù-nhưng", ngữ nghĩa không liên quan.

**Điểm:** 5

---

### Test 14
**Req:** bởi vì trời mưa, 

**Res:** bởi_vì trời mưa , nơi mà các hạt trở_thành một phần quan_trọng trong văn_hoá và lịch_sử của việt_nam .

**Điểm:** 1
