import json
from pathlib import Path
from vncorenlp import VnCoreNLP
import pandas as pd
import time

# Khởi tạo VnCoreNLP
VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"

def init_annotator():
    return VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

annotator = init_annotator()

# Đường dẫn thư mục
current_file = Path(__file__).resolve()
data_dir = current_file.parent.parent
raw_dir = data_dir / "raw"

# ----------- ĐỌC FILE .json (file1 mới) -----------
file1_path = raw_dir / "pre_train.json"
texts1 = []
with open(file1_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # Bỏ qua dòng trống
            texts1.append(line)

# ----------- ĐỌC FILE .csv (file2 cũ) -----------
file2_path = raw_dir / "fine_tune.csv"
file2 = pd.read_csv(file2_path, encoding="utf-8")
texts2 = file2.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()

# ----------- GHÉP TẤT CẢ TEXTS -----------
texts = texts1 + texts2

# ----------- TẠO VOCABULARY -----------
vocab = set()
batch_size = 50
max_retries = 3

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    print(f"Đang xử lý batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    for sentence in batch:
        sentence = sentence.lower()
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = annotator.tokenize(sentence)
                for word_list in result:
                    vocab.update(word_list)
                break  # Thành công, thoát khỏi vòng lặp retry
            except Exception as e:
                retry_count += 1
                print(f"Lỗi khi tokenize (lần thử {retry_count}): {e}")
                if retry_count < max_retries:
                    print("Đang khởi động lại VnCoreNLP...")
                    try:
                        annotator.close()
                    except:
                        pass
                    time.sleep(2)
                    annotator = init_annotator()
                    time.sleep(1)
                else:
                    print(f"Bỏ qua câu: {sentence[:50]}...")
    
    # Thêm delay giữa các batch
    time.sleep(0.1)

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