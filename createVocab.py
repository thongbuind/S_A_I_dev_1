import pandas as pd
from vncorenlp import VnCoreNLP

# Khởi tạo VnCoreNLP
VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

# Đọc dữ liệu từ cả hai file
file1 = pd.read_csv("pre_train.csv", encoding="utf-8")  # File thứ nhất
file2 = pd.read_csv("fine_tune.csv", encoding="utf-8")  # File thứ hai

# Gộp dữ liệu từ cả hai file
texts1 = file1.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()
texts2 = file2.astype(str).apply(lambda col: col.dropna().tolist()).values.flatten().tolist()
texts = texts1 + texts2  # Kết hợp danh sách câu từ cả hai file

# Tạo tập từ vựng
vocab = set()

for sentence in texts:
    result = annotator.tokenize(sentence)
    for word_list in result:
        vocab.update(word_list)

# Thêm các token đặc biệt
special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[BOS]", "[EOS]"]
sorted_vocab = special_tokens + sorted(vocab)  # Đặc biệt ở đầu, từ thường theo sau

# Gán index cho từ vựng
word_to_id = {word: idx for idx, word in enumerate(sorted_vocab)}

# Lưu vào vocab.txt
with open("vocab.txt", "w", encoding="utf-8") as f:
    for word, idx in word_to_id.items():
        f.write(f"{word}\t{idx}\n")

print("✅ Đã tách từ, thêm token đặc biệt và lưu vào vocab.txt từ cả hai file thành công!")