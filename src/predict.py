import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
from vncorenlp import VnCoreNLP
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

current_file = Path(__file__).resolve()

model_path = project_root / "model" / "s_a_i.keras"
model = models.load_model(model_path)

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# Đọc vocab
vocab = {}
vocab_path = current_file.parent.parent/ "data" / "vocab.txt"
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

def generate_response(sentence, max_new_tokens=32, infor=None):
    """Tạo phản hồi dựa trên thông tin cá nhân - AUTOREGRESSIVE GENERATION"""
    req_tokens = tokenize(sentence)
    
    # Bắt đầu với [BOS] + request
    current_sequence = [vocab["[BOS]"]] + req_tokens
    
    # Generate từng token một
    for step in range(max_new_tokens):
        # Pad sequence để fit model
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(
            [current_sequence], maxlen=max_seq_len, padding='post', dtype='int32'
        )
        
        # Predict next token
        preds = model(padded_input, training=False)
        
        # Lấy token ở vị trí cuối sequence thật
        pos1 = len(current_sequence) - 1
        if pos1 < preds.shape[1]:
            next_token_probs1 = preds[0, pos1, :]
            next_token1 = np.argmax(next_token_probs1)
            next_token = int(next_token1)  # Ép kiểu thành int
        else:
            next_token = vocab["[EOS]"]
        
        # Dừng nếu gặp EOS hoặc PAD
        if next_token == vocab["[EOS]"] or next_token == vocab["[PAD]"]:
            break
            
        current_sequence.append(next_token)
        
        # Tránh sequence quá dài
        if len(current_sequence) >= max_seq_len:
            break
    
    # Trả về toàn bộ sequence (bỏ [BOS])
    return detokenize(current_sequence[1:], infor)

# ================
# Kiểm Tra Mô Hình
# ================

print("\n=== Test pre-train ===")
print("Req: bánh mì \nRes: ", generate_response("bánh mì"))
print("Req: bánh mì có nguồn gốc từ \nRes: ", generate_response("bánh mì có nguồn gốc từ"))
print("Req: việt nam \nRes: ", generate_response("việt nam"))
print("Req: việt nam sở hữu \nRes: ", generate_response("việt nam sở hữu"))
print("Req: phở \nRes: ", generate_response("phở"))
print("Req: buổi sáng người việt nam thường ăn \nRes: ", generate_response("buổi sáng người việt nam thường ăn"))
print("Req: đám mây \nRes: ", generate_response("đám mây"))
print("Req: Đinh Tiên Hoàng lên ngôi \nRes: ", generate_response("Đinh Tiên Hoàng lên ngôi"))
print("Req: lê thái tổ có miếu hiệu \nRes: ", generate_response("lê thái tổ có miếu hiệu"))
print("Req: công thức 1 \nRes: ", generate_response("công thức 1"))
print("Req: sáng hôm ấy \nRes: ", generate_response("ng hôm ấy"))
print("Req: sau khi ăn xong, chúng tôi đi \nRes: ", generate_response("sau khi ăn xong, chúng tôi đi"))
print("Req: mặc dù \nRes: ", generate_response("mặc dù"))
print("Req: bởi vì trời mưa, \nRes: ", generate_response("bởi vì trời mưa,"))

