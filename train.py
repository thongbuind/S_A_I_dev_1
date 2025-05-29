import tensorflow as tf
import numpy as np
import json
from vncorenlp import VnCoreNLP
from model import Model
import csv

with open('config.json', 'r') as f:
    config = json.load(f)

max_seq_len=config['max_seq_len']
d_model=config['d_model']
num_heads=config['num_heads']
num_layers=config['num_layers']
ff_dim=config['ff_dim']
dropout=config['dropout']

infor = {}
with open("infor.json", "r", encoding="utf-8") as f:
    infor = json.load(f)

vocab = {}
with open("vocab.txt", "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

# Mở rộng từ điển với tên từ infor
for word in infor["nametoken"].split():
    if word not in vocab:
        vocab[word] = len(vocab)
for word in infor["roletoken"].split():
    if word not in vocab:
        vocab[word] = len(vocab)

idx2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

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

def load_pretrain_dataset(file_path):
    """Tải dữ liệu pre-train từ CSV 1 cột: mỗi dòng là một câu"""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 1:
                dataset.append(row[0].strip())
    return dataset

def load_finetune_dataset(file_path):
    """Tải dữ liệu fine-tune từ CSV 2 cột: input, output"""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                dataset.append((row[0].strip(), row[1].strip()))
    return dataset

# ===========================
# 3. Chuẩn hóa dữ liệu huấn luyện - FIXED
# ===========================
def prepare_pretrain_data(data):
    X, Y = [], []
    for sentence in data:
        tokens = tokenize(sentence)
        if len(tokens) < 2:
            continue 
        tokens = tokens + [vocab["[EOS]"]]
        inp_ids = [vocab["[BOS]"]] + tokens[:-1]
        out_ids = tokens
        X.append(inp_ids)
        Y.append(out_ids)
    # FIX: Sử dụng max_len cố định
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_len, padding='post')
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_seq_len, padding='post')
    return X, Y, max_seq_len

def prepare_finetune_data(data):
    X, Y = [], []
    for req, res in data:
        req_ids = tokenize(req)
        res_ids = tokenize(res) + [vocab["[EOS]"]]
        
        # Đơn giản hơn: chỉ train trên response part
        inp = [vocab["[BOS]"]] + req_ids + [vocab["[SEP]"]] + res_ids
        tgt = req_ids + [vocab["[SEP]"]] + res_ids + [vocab["[EOS]"]]
        
        X.append(inp)
        Y.append(tgt)
    
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_len, padding='post')
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_seq_len, padding='post')
    return X, Y, max_seq_len

# ===========================
# 4. Huấn luyện
# ===========================
model = Model(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, dropout)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# --- Pre-train ---
print("=== Bắt đầu pre-train ===")
pretrain_data = load_pretrain_dataset("pre_train.csv")
train_X, train_Y, MAX_LEN = prepare_pretrain_data(pretrain_data)  # FIX: Lưu MAX_LEN

for epoch in range(1000):
    loss = model.train_on_batch(train_X, train_Y)
    if epoch % 100 == 0:
        print(f"[Pretrain] Epoch {epoch}, Loss: {loss:.4f}")

model.save_weights("pretrain.weights.h5")

# --- Fine-tune ---
print("\n=== Bắt đầu fine-tune ===")
fine_tune_data = load_finetune_dataset("fine_tune.csv")
train_X, train_Y, _ = prepare_finetune_data(fine_tune_data)  # FIX: Không cần lưu max_len

model.load_weights("pretrain.weights.h5")
for epoch in range(1000):
    loss = model.train_on_batch(train_X, train_Y)
    if epoch % 50 == 0:
        print(f"[Finetune] Epoch {epoch}, Loss: {loss:.4f}")

model.save_weights("finetune.weights.h5")

# ===========================
# 5. Hàm Tạo Phản Hồi Cá Nhân Hóa - FIXED
# ===========================
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
            next_token_probs2 = preds[0, -1, :]
            next_token2 = np.argmax(next_token_probs2)
            
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

# ===========================
# 6. Kiểm Tra Mô Hình
# ===========================
print("Thử nghiệm chào hỏi:", generate_response("chào"))
print("bạn tên gì:", generate_response("bạn tên gì"))
print("bạn tên là gì:", generate_response("bạn tên là gì"))
print("vai trò của bạn:", generate_response("vai trò của bạn"))
print("bạn mấy tuổi:", generate_response("bạn mấy tuổi"))
