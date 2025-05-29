import tensorflow as tf
import numpy as np
import json
from vncorenlp import VnCoreNLP
import csv
from pathlib import Path

# Lấy đường dẫn tuyệt đối đến file vocab.txt dựa trên vị trí file hiện tại
current_file = Path(__file__).resolve()

config_dir = current_file.parent.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# Đọc vocab
vocab = {}
vocab_path = current_file.parent.parent / "vocab.txt"
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

    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_seq_len, padding='post')
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_seq_len, padding='post')
    return X, Y

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
    return X, Y


raw_dir = current_file.parent.parent / "raw"
pretrain_data = load_pretrain_dataset(raw_dir / "pre_train.csv")
pre_train_X, pre_train_Y = prepare_pretrain_data(pretrain_data) 
fine_tune_data = load_finetune_dataset(raw_dir / "fine_tune.csv")
fine_tune_X, fine_tune_Y = prepare_finetune_data(fine_tune_data)

np.set_printoptions(threshold=np.inf)

data_tokenized_dir = current_file.parent.parent / "processed" / "data_tokenized.py"
with open(data_tokenized_dir, "w", encoding="utf-8") as f:
    f.write("import numpy as np\n\n")
    
    f.write(f"pre_train_X = np.array({repr(pre_train_X.tolist())})\n\n")
    f.write(f"pre_train_Y = np.array({repr(pre_train_Y.tolist())})\n\n")
    
    f.write(f"fine_tune_X = np.array({repr(fine_tune_X.tolist())})\n\n")
    f.write(f"fine_tune_Y = np.array({repr(fine_tune_Y.tolist())})\n")

