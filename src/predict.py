import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
from vncorenlp import VnCoreNLP
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Lấy đường dẫn tuyệt đối đến file vocab.txt dựa trên vị trí file hiện tại
current_file = Path(__file__).resolve()

model_path = project_root / "model" / "s_a_i.keras"
model = models.load_model(model_path)

infor = {}
infor_dir = current_file.parent.parent / "config" / "infor.json"
with open(infor_dir, "r", encoding="utf-8") as f:
    infor = json.load(f)

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# Đọc vocab
vocab = {}
vocab_path = current_file.parent.parent / "data" / "vocab.txt"
with open(vocab_path, "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

idx2word = {i: w for w, i in vocab.items()}

VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

def tokenize(sentence):
    """Chuyển đổi câu thành token số, sử dụng VnCoreNLP để tách từ tiếng Việt"""
    word_segments = annotator.tokenize(sentence.lower())
    words = [word for segment in word_segments for word in segment]
    tokens = [vocab.get(w, vocab["[UNK]"]) for w in words]
    return tokens

def detokenize(tokens, infor=None):
    """Chuyển token số về câu văn bản, thay thế token đặc biệt nếu cần"""
    special_tokens = {0, 1, 2, 3, 4, 5, 6}  # PAD, UNK, BOS, EOS, SEP
    words = []
    for t in tokens:
        if t in special_tokens or t not in idx2word:
            continue
        word = idx2word[t]
        if infor and word in infor:
            words.append(infor[word])
        else:
            words.append(word)
    return " ".join(words)


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
print("Req: phở là \nRes: ", generate_response("phở là"))
print("Req: lê thái tổ \nRes: ", generate_response("lê thái tổ"))
print("Req: để nấu một bát bún \nRes: ", generate_response("để nấu một bát bún"))
print("Req: đinh tiên hoàng lên ngôi \nRes: ", generate_response("đinh tiên hoàng lên ngôi"))
print("Req: bánh mì \nRes: ", generate_response("bánh mì"))
print("Req: sau buổi chiều hôm ấy \nRes: ", generate_response("sau buổi chiều hôm ấy"))
print("Req: khi trời mưa, \nRes: ", generate_response("khi trời mưa,"))
print("Req: vào năm 2023, \nRes: ", generate_response("vào năm 2023,"))
print("Req: anh ấy đến đón tôi đi \nRes: ", generate_response("anh ấy đến đón tôi đi"))
print("Req: mặc dù \nRes: ", generate_response("mặc dù"))

print("=== Test fine-tune ===")
print("Req: chào \nRes: ", generate_response("chào"))
print("Req: bạn tên là gì \nRes: ", generate_response("bạn tên là gì"))
print("Req: bạn mấy tuổi \nRes: ", generate_response("bạn mấy tuổi"))
print("Req: bạn sinh ngày mấy \nRes: ", generate_response("bạn sinh ngày mấy"))
print("Req: người tạo ra bạn tên là gì \nRes: ", generate_response("người tạo ra bạn tên là gì"))
print("Req: hãy tự giới thiệu bản thân \nRes: ", generate_response("hãy tự giới thiệu bản thân"))


'''
║ Epoch:   0, Batch:   1/217, Loss: 9.9134 ║
║ Epoch:   0, Batch: 101/217, Loss: 5.0778 ║
║ Epoch:   0, Batch: 201/217, Loss: 3.5451 ║
║ Epoch:   0, Batch: 217/217, Loss: 3.3904 ║
║ Epoch:   1, Batch:   1/217, Loss: 3.3907 ║
║ Epoch:   1, Batch: 101/217, Loss: 2.8936 ║
║ Epoch:   1, Batch: 201/217, Loss: 2.5849 ║
║ Epoch:   1, Batch: 217/217, Loss: 2.5348 ║
║ Epoch:   2, Batch:   1/217, Loss: 2.5356 ║
║ Epoch:   2, Batch: 101/217, Loss: 2.3568 ║
║ Epoch:   2, Batch: 201/217, Loss: 2.2107 ║
║ Epoch:   2, Batch: 217/217, Loss: 2.1833 ║
║ Epoch:   3, Batch:   1/217, Loss: 2.1838 ║
║ Epoch:   3, Batch: 101/217, Loss: 2.0841 ║
║ Epoch:   3, Batch: 201/217, Loss: 1.9941 ║
║ Epoch:   3, Batch: 217/217, Loss: 1.9759 ║
║ Epoch:   4, Batch:   1/217, Loss: 1.9763 ║
║ Epoch:   4, Batch: 101/217, Loss: 1.9105 ║
║ Epoch:   4, Batch: 201/217, Loss: 1.8468 ║
║ Epoch:   4, Batch: 217/217, Loss: 1.8333 ║
║ Epoch:   5, Batch:   1/217, Loss: 1.8337 ║
║ Epoch:   5, Batch: 101/217, Loss: 1.7846 ║
║ Epoch:   5, Batch: 201/217, Loss: 1.7344 ║
║ Epoch:   5, Batch: 217/217, Loss: 1.7237 ║
║ Epoch:   6, Batch:   1/217, Loss: 1.7240 ║
║ Epoch:   6, Batch: 101/217, Loss: 1.6837 ║
║ Epoch:   6, Batch: 201/217, Loss: 1.6407 ║
║ Epoch:   6, Batch: 217/217, Loss: 1.6316 ║
║ Epoch:   7, Batch:   1/217, Loss: 1.6318 ║
║ Epoch:   7, Batch: 101/217, Loss: 1.5965 ║
║ Epoch:   7, Batch: 201/217, Loss: 1.5580 ║
║ Epoch:   7, Batch: 217/217, Loss: 1.5501 ║
║ Epoch:   8, Batch:   1/217, Loss: 1.5503 ║
║ Epoch:   8, Batch: 101/217, Loss: 1.5185 ║
║ Epoch:   8, Batch: 201/217, Loss: 1.4839 ║
║ Epoch:   8, Batch: 217/217, Loss: 1.4769 ║
║ Epoch:   9, Batch:   1/217, Loss: 1.4771 ║
║ Epoch:   9, Batch: 101/217, Loss: 1.4488 ║
║ Epoch:   9, Batch: 201/217, Loss: 1.4179 ║
║ Epoch:   9, Batch: 217/217, Loss: 1.4118 ║
║ Epoch:  10, Batch:   1/217, Loss: 1.4119 ║
║ Epoch:  10, Batch: 101/217, Loss: 1.3864 ║
║ Epoch:  10, Batch: 201/217, Loss: 1.3588 ║
║ Epoch:  10, Batch: 217/217, Loss: 1.3534 ║
║ Epoch:  11, Batch:   1/217, Loss: 1.3535 ║
║ Epoch:  11, Batch: 101/217, Loss: 1.3304 ║
║ Epoch:  11, Batch: 201/217, Loss: 1.3057 ║
║ Epoch:  11, Batch: 217/217, Loss: 1.3009 ║
║ Epoch:  12, Batch:   1/217, Loss: 1.3010 ║
║ Epoch:  12, Batch: 101/217, Loss: 1.2803 ║
║ Epoch:  12, Batch: 201/217, Loss: 1.2582 ║
║ Epoch:  12, Batch: 217/217, Loss: 1.2539 ║
║ Epoch:  13, Batch:   1/217, Loss: 1.2539 ║
║ Epoch:  13, Batch: 101/217, Loss: 1.2351 ║
║ Epoch:  13, Batch: 201/217, Loss: 1.2153 ║
║ Epoch:  13, Batch: 217/217, Loss: 1.2114 ║
║ Epoch:  14, Batch:   1/217, Loss: 1.2115 ║
║ Epoch:  14, Batch: 101/217, Loss: 1.1946 ║
║ Epoch:  14, Batch: 201/217, Loss: 1.1767 ║
║ Epoch:  14, Batch: 217/217, Loss: 1.1732 ║
║ Epoch:  15, Batch:   1/217, Loss: 1.1733 ║
║ Epoch:  15, Batch: 101/217, Loss: 1.1578 ║
║ Epoch:  15, Batch: 201/217, Loss: 1.1416 ║
║ Epoch:  15, Batch: 217/217, Loss: 1.1384 ║
║ Epoch:  16, Batch:   1/217, Loss: 1.1384 ║
║ Epoch:  16, Batch: 101/217, Loss: 1.1246 ║
║ Epoch:  16, Batch: 201/217, Loss: 1.1100 ║
║ Epoch:  16, Batch: 217/217, Loss: 1.1071 ║
║ Epoch:  17, Batch:   1/217, Loss: 1.1072 ║
║ Epoch:  17, Batch: 101/217, Loss: 1.0943 ║
║ Epoch:  17, Batch: 201/217, Loss: 1.0809 ║
║ Epoch:  17, Batch: 217/217, Loss: 1.0782 ║
║ Epoch:  18, Batch:   1/217, Loss: 1.0783 ║
║ Epoch:  18, Batch: 101/217, Loss: 1.0665 ║
║ Epoch:  18, Batch: 201/217, Loss: 1.0541 ║
║ Epoch:  18, Batch: 217/217, Loss: 1.0517 ║
║ Epoch:  19, Batch:   1/217, Loss: 1.0517 ║
║ Epoch:  19, Batch: 101/217, Loss: 1.0408 ║
║ Epoch:  19, Batch: 201/217, Loss: 1.0295 ║
║ Epoch:  19, Batch: 217/217, Loss: 1.0273 ║
║ Epoch:  20, Batch:   1/217, Loss: 1.0273 ║
║ Epoch:  20, Batch: 101/217, Loss: 1.0173 ║
║ Epoch:  20, Batch: 201/217, Loss: 1.0068 ║
║ Epoch:  20, Batch: 217/217, Loss: 1.0047 ║
║ Epoch:  21, Batch:   1/217, Loss: 1.0047 ║
║ Epoch:  21, Batch: 101/217, Loss: 0.9955 ║
║ Epoch:  21, Batch: 201/217, Loss: 0.9858 ║
║ Epoch:  21, Batch: 217/217, Loss: 0.9839 ║
║ Epoch:  22, Batch:   1/217, Loss: 0.9839 ║
║ Epoch:  22, Batch: 101/217, Loss: 0.9753 ║
║ Epoch:  22, Batch: 201/217, Loss: 0.9664 ║
║ Epoch:  22, Batch: 217/217, Loss: 0.9646 ║
║ Epoch:  23, Batch:   1/217, Loss: 0.9646 ║
║ Epoch:  23, Batch: 101/217, Loss: 0.9567 ║
║ Epoch:  23, Batch: 201/217, Loss: 0.9484 ║
║ Epoch:  23, Batch: 217/217, Loss: 0.9467 ║
║ Epoch:  24, Batch:   1/217, Loss: 0.9467 ║
║ Epoch:  24, Batch: 101/217, Loss: 0.9394 ║
║ Epoch:  24, Batch: 201/217, Loss: 0.9316 ║
║ Epoch:  24, Batch: 217/217, Loss: 0.9300 ║
║ Epoch:  25, Batch:   1/217, Loss: 0.9300 ║
║ Epoch:  25, Batch: 101/217, Loss: 0.9231 ║
║ Epoch:  25, Batch: 201/217, Loss: 0.9159 ║
║ Epoch:  25, Batch: 217/217, Loss: 0.9144 ║
║ Epoch:  26, Batch:   1/217, Loss: 0.9144 ║
║ Epoch:  26, Batch: 101/217, Loss: 0.9079 ║
║ Epoch:  26, Batch: 201/217, Loss: 0.9011 ║
║ Epoch:  26, Batch: 217/217, Loss: 0.8997 ║
║ Epoch:  27, Batch:   1/217, Loss: 0.8997 ║
║ Epoch:  27, Batch: 101/217, Loss: 0.8938 ║
║ Epoch:  27, Batch: 201/217, Loss: 0.8874 ║
║ Epoch:  27, Batch: 217/217, Loss: 0.8861 ║
║ Epoch:  28, Batch:   1/217, Loss: 0.8861 ║
║ Epoch:  28, Batch: 101/217, Loss: 0.8804 ║
║ Epoch:  28, Batch: 201/217, Loss: 0.8744 ║
║ Epoch:  28, Batch: 217/217, Loss: 0.8731 ║
║ Epoch:  29, Batch:   1/217, Loss: 0.8731 ║
║ Epoch:  29, Batch: 101/217, Loss: 0.8678 ║
║ Epoch:  29, Batch: 201/217, Loss: 0.8621 ║
║ Epoch:  29, Batch: 217/217, Loss: 0.8609 ║
║ Epoch:  30, Batch:   1/217, Loss: 0.8610 ║
║ Epoch:  30, Batch: 101/217, Loss: 0.8559 ║
║ Epoch:  30, Batch: 201/217, Loss: 0.8506 ║
║ Epoch:  30, Batch: 217/217, Loss: 0.8495 ║
║ Epoch:  31, Batch:   1/217, Loss: 0.8495 ║
║ Epoch:  31, Batch: 101/217, Loss: 0.8448 ║
║ Epoch:  31, Batch: 201/217, Loss: 0.8398 ║
║ Epoch:  31, Batch: 217/217, Loss: 0.8387 ║
║ Epoch:  32, Batch:   1/217, Loss: 0.8387 ║
║ Epoch:  32, Batch: 101/217, Loss: 0.8342 ║
║ Epoch:  32, Batch: 201/217, Loss: 0.8294 ║
║ Epoch:  32, Batch: 217/217, Loss: 0.8284 ║
║ Epoch:  33, Batch:   1/217, Loss: 0.8284 ║
║ Epoch:  33, Batch: 101/217, Loss: 0.8241 ║
║ Epoch:  33, Batch: 201/217, Loss: 0.8196 ║
║ Epoch:  33, Batch: 217/217, Loss: 0.8186 ║
║ Epoch:  34, Batch:   1/217, Loss: 0.8186 ║
║ Epoch:  34, Batch: 101/217, Loss: 0.8145 ║
║ Epoch:  34, Batch: 201/217, Loss: 0.8102 ║
║ Epoch:  34, Batch: 217/217, Loss: 0.8093 ║
║ Epoch:  35, Batch:   1/217, Loss: 0.8093 ║
║ Epoch:  35, Batch: 101/217, Loss: 0.8054 ║
║ Epoch:  35, Batch: 201/217, Loss: 0.8013 ║
║ Epoch:  35, Batch: 217/217, Loss: 0.8005 ║
║ Epoch:  36, Batch:   1/217, Loss: 0.8005 ║
║ Epoch:  36, Batch: 101/217, Loss: 0.7968 ║
║ Epoch:  36, Batch: 201/217, Loss: 0.7929 ║
║ Epoch:  36, Batch: 217/217, Loss: 0.7920 ║
║ Epoch:  37, Batch:   1/217, Loss: 0.7921 ║
║ Epoch:  37, Batch: 101/217, Loss: 0.7885 ║
║ Epoch:  37, Batch: 201/217, Loss: 0.7848 ║
║ Epoch:  37, Batch: 217/217, Loss: 0.7840 ║
║ Epoch:  38, Batch:   1/217, Loss: 0.7840 ║
║ Epoch:  38, Batch: 101/217, Loss: 0.7807 ║
║ Epoch:  38, Batch: 201/217, Loss: 0.7771 ║
║ Epoch:  38, Batch: 217/217, Loss: 0.7763 ║
║ Epoch:  39, Batch:   1/217, Loss: 0.7763 ║
║ Epoch:  39, Batch: 101/217, Loss: 0.7731 ║
║ Epoch:  39, Batch: 201/217, Loss: 0.7697 ║
║ Epoch:  39, Batch: 217/217, Loss: 0.7690 ║
║ Epoch:  40, Batch:   1/217, Loss: 0.7690 ║
║ Epoch:  40, Batch: 101/217, Loss: 0.7659 ║
║ Epoch:  40, Batch: 201/217, Loss: 0.7627 ║
║ Epoch:  40, Batch: 217/217, Loss: 0.7620 ║
║ Epoch:  41, Batch:   1/217, Loss: 0.7620 ║
║ Epoch:  41, Batch: 101/217, Loss: 0.7590 ║
║ Epoch:  41, Batch: 201/217, Loss: 0.7559 ║
║ Epoch:  41, Batch: 217/217, Loss: 0.7552 ║
║ Epoch:  42, Batch:   1/217, Loss: 0.7552 ║
║ Epoch:  42, Batch: 101/217, Loss: 0.7524 ║
║ Epoch:  42, Batch: 201/217, Loss: 0.7494 ║
║ Epoch:  42, Batch: 217/217, Loss: 0.7488 ║
║ Epoch:  43, Batch:   1/217, Loss: 0.7488 ║
║ Epoch:  43, Batch: 101/217, Loss: 0.7461 ║
║ Epoch:  43, Batch: 201/217, Loss: 0.7432 ║
║ Epoch:  43, Batch: 217/217, Loss: 0.7426 ║
║ Epoch:  44, Batch:   1/217, Loss: 0.7426 ║
║ Epoch:  44, Batch: 101/217, Loss: 0.7400 ║
║ Epoch:  44, Batch: 201/217, Loss: 0.7373 ║
║ Epoch:  44, Batch: 217/217, Loss: 0.7366 ║
║ Epoch:  45, Batch:   1/217, Loss: 0.7367 ║
║ Epoch:  45, Batch: 101/217, Loss: 0.7342 ║
║ Epoch:  45, Batch: 201/217, Loss: 0.7315 ║
║ Epoch:  45, Batch: 217/217, Loss: 0.7309 ║
║ Epoch:  46, Batch:   1/217, Loss: 0.7309 ║
║ Epoch:  46, Batch: 101/217, Loss: 0.7285 ║
║ Epoch:  46, Batch: 201/217, Loss: 0.7259 ║
║ Epoch:  46, Batch: 217/217, Loss: 0.7254 ║
║ Epoch:  47, Batch:   1/217, Loss: 0.7254 ║
║ Epoch:  47, Batch: 101/217, Loss: 0.7231 ║
║ Epoch:  47, Batch: 201/217, Loss: 0.7206 ║
║ Epoch:  47, Batch: 217/217, Loss: 0.7201 ║
║ Epoch:  48, Batch:   1/217, Loss: 0.7201 ║
║ Epoch:  48, Batch: 101/217, Loss: 0.7179 ║
║ Epoch:  48, Batch: 201/217, Loss: 0.7155 ║
║ Epoch:  48, Batch: 217/217, Loss: 0.7150 ║
║ Epoch:  49, Batch:   1/217, Loss: 0.7150 ║
║ Epoch:  49, Batch: 101/217, Loss: 0.7129 ║
║ Epoch:  49, Batch: 201/217, Loss: 0.7106 ║
║ Epoch:  49, Batch: 217/217, Loss: 0.7101 ║
╚═════════════════════════════════════════╝
'''