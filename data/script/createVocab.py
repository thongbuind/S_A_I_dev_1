import json
from pathlib import Path
from vncorenlp import VnCoreNLP
import pandas as pd

# Khởi tạo VnCoreNLP
VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

# Đường dẫn thư mục
current_file = Path(__file__).resolve()
data_dir = current_file.parent.parent
raw_dir = data_dir / "raw"

# ----------- ĐỌC FILE .json (file1 mới) -----------
file1_path = raw_dir / "pre_train.json"
with open(file1_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Lấy nội dung từ field "content" trong mỗi object
texts1 = []
for entry in json_data:
    if "content" in entry and isinstance(entry["content"], list):
        texts1.extend(entry["content"])

# ----------- ĐỌC FILE .csv (file2 cũ) -----------
file2_path = raw_dir / "fine_tune.csv"
file2 = pd.read_csv(file2_path, encoding="utf-8")
texts2 = file2.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()

# ----------- GHÉP TẤT CẢ TEXTS -----------
texts = texts1 + texts2

# ----------- TẠO VOCABULARY -----------
vocab = set()
for sentence in texts:
    sentence = sentence.lower()
    result = annotator.tokenize(sentence)
    for word_list in result:
        vocab.update(word_list)

special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]
sorted_vocab = special_tokens + sorted(vocab)
word_to_id = {word: idx for idx, word in enumerate(sorted_vocab)}

# ----------- THÊM TOKEN TỪ infor.json -----------
infor_dir = current_file.parent.parent.parent / "config" / "infor.json"
with open(infor_dir, "r", encoding="utf-8") as f:
    infor = json.load(f)

for key in ["nametoken", "agetoken", "birthdaytoken", "creatornametoken"]:
    for word in infor[key].split():
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

# ----------- LƯU vocab.txt -----------
vocab_path = data_dir / "vocab.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    for word, idx in word_to_id.items():
        f.write(f"{word}\t{idx}\n")

print("✅ Đã tách từ, thêm token đặc biệt và lưu vào vocab.txt từ JSON + CSV!")

# ----------- CẬP NHẬT config.json -----------
config_dir = current_file.parent.parent.parent / "config" / "config.json"
with open(config_dir, "r", encoding="utf-8") as f:
    config = json.load(f)

config["vocab_size"] = len(word_to_id)

with open(config_dir, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"✅ Đã cập nhật 'vocab_size' = {len(word_to_id)} vào config.json")