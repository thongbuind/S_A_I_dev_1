import pandas as pd
from pathlib import Path
from vncorenlp import VnCoreNLP
import json
import csv
from pathlib import Path

# Khởi tạo VnCoreNLP
VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

# Lấy đường dẫn gốc của file hiện tại
current_file = Path(__file__).resolve()
data_dir = current_file.parent.parent      # data/
raw_dir = data_dir / "raw"                 # data/raw/

# Tạo đường dẫn đầy đủ đến file CSV
file1_path = raw_dir / "pre_train.csv"
file2_path = raw_dir / "fine_tune.csv"

# Đọc dữ liệu từ cả hai file
file1 = pd.read_csv(file1_path, encoding="utf-8")
file2 = pd.read_csv(file2_path, encoding="utf-8")

# Gộp dữ liệu từ cả hai file
texts1 = file1.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()
texts2 = file2.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()
texts = texts1 + texts2  # Kết hợp danh sách câu từ cả hai file

# Tạo tập từ vựng
vocab = set()

for sentence in texts:
    sentence = sentence.lower()
    result = annotator.tokenize(sentence)
    for word_list in result:
        vocab.update(word_list)

# Thêm các token đặc biệt
special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]
sorted_vocab = special_tokens + sorted(vocab)  # Đặc biệt ở đầu, từ thường theo sau

# Gán index cho từ vựng
word_to_id = {word: idx for idx, word in enumerate(sorted_vocab)}

infor = {}
infor_dir = current_file.parent.parent.parent / "config" / "infor.json"
with open(infor_dir, "r", encoding="utf-8") as f:
    infor = json.load(f)

# Mở rộng từ điển với tên từ infor
for word in infor["nametoken"].split():
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)
for word in infor["agetoken"].split():
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)
for word in infor["birthdaytoken"].split():
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)
for word in infor["creatornametoken"].split():
    if word not in word_to_id:
        word_to_id[word] = len(word_to_id)

# Thư mục chứa file createvocab.py
current_file = Path(__file__).resolve()
data_dir = current_file.parent.parent
vocab_path = data_dir / "vocab.txt"

# Lưu vào vocab.txt
with open(vocab_path, "w", encoding="utf-8") as f:
    for word, idx in word_to_id.items():
        f.write(f"{word}\t{idx}\n")

print("✅ Đã tách từ, thêm token đặc biệt và lưu vào vocab.txt từ cả hai file thành công!")

# Lấy đường dẫn tuyệt đối đến file vocab.txt dựa trên vị trí file hiện tại
current_file = Path(__file__).resolve()
config_dir = current_file.parent.parent.parent / "config" / "config.json"

# Đọc config.json
with open(config_dir, "r", encoding="utf-8") as f:
    config = json.load(f)

# Cập nhật vocab_size
vocab_size = len(word_to_id)
config["vocab_size"] = vocab_size

# Ghi lại file config.json
with open(config_dir, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"✅ Đã cập nhật 'vocab_size' = {vocab_size} vào config.json")