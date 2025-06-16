import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data.script.tokenizer import tokenize, detokenize, vocab, max_seq_len

# Lấy đường dẫn tuyệt đối đến file vocab.txt dựa trên vị trí file hiện tại
current_file = Path(__file__).resolve()

model_path = project_root / "model" / "s_a_i.keras"
model = models.load_model(model_path)

infor = {}
infor_dir = current_file.parent.parent / "config" / "infor.json"
with open(infor_dir, "r", encoding="utf-8") as f:
    infor = json.load(f)


# ============================
# Hàm Tạo Phản Hồi Cá Nhân Hóa
# ============================
def generate_response(sentence, max_new_tokens=32, infor=infor):
    """Tạo phản hồi dựa trên thông tin cá nhân - AUTOREGRESSIVE GENERATION"""
    req_tokens = tokenize(sentence)
    
    # Bắt đầu với [BOS] + request + [SEP]
    current_sequence = [vocab["[BOS]"]] + req_tokens + [vocab["[SEP]"]]
    
    # Generate từng token một
    for step in range(max_new_tokens):
        # Pad sequence để fit model
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(
            [current_sequence], maxlen=max_seq_len, padding='post'
        )
        
        # Predict next token
        preds = model.predict(padded_input, verbose=0)
        
        # Thử cả 2 cách lấy prediction
        # Cách 1: lấy ở vị trí cuối sequence thật
        pos1 = len(current_sequence) - 1
        if pos1 < preds.shape[1]:
            next_token_probs1 = preds[0, pos1, :]
            next_token1 = np.argmax(next_token_probs1)
            
            # Cách 2: lấy ở vị trí cuối cùng (có thể là padding)
            # next_token_probs2 = preds[0, -1, :]
            # next_token2 = np.argmax(next_token_probs2)
            
            # Sử dụng method 1 trước
            next_token = next_token1
        else:
            next_token = vocab["[EOS]"]
        
        # Dừng nếu gặp EOS hoặc PAD
        if next_token == vocab["[EOS]"] or next_token == vocab["[PAD]"]:
            break
            
        current_sequence.append(next_token)
        
        # Tránh sequence quá dài
        if len(current_sequence) >= max_seq_len:
            break
    
    # Trích xuất phần response (sau [SEP])
    sep_position = len([vocab["[BOS]"]] + req_tokens + [vocab["[SEP]"]])
    response_tokens = current_sequence[sep_position:]

    return detokenize(response_tokens, infor)

# ================
# Kiểm Tra Mô Hình
# ================

print("\n=== Test pre-train ===")
print("Req: bánh mì. \nRes: ", generate_response("bánh mì"))
print("Req: việt nam. \nRes: ", generate_response("việt nam"))
print("Req: phở. \nRes: ", generate_response("phở"))
print("Req: đám mây. \nRes: ", generate_response("đám mây"))
print("Req: Đinh Tiên Hoàng. \nRes: ", generate_response("Đinh Tiên Hoàng"))
print("Req: lê thái tổ. \nRes: ", generate_response("lê thái tổ"))
print("Req: công thức 1. \nRes: ", generate_response("công thức 1"))

print("=== Test fine-tune ===")
print("Req: chào. \nRes: ", generate_response("chào"))
print("Req: bạn tên là gì. \nRes: ", generate_response("bạn tên là gì"))
print("Req: bạn mấy tuổi. \nRes: ", generate_response("bạn mấy tuổi"))
print("Req: bạn sinh ngày mấy. \nRes: ", generate_response("bạn sinh ngày mấy"))
print("Req: người tạo ra bạn tên là gì. \nRes: ", generate_response("người tạo ra bạn tên là gì"))
print("Req: hãy tự giới thiệu bản thân. \nRes: ", generate_response("hãy tự giới thiệu bản thân"))
