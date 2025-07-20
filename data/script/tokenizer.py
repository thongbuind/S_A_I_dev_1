import tensorflow as tf
import numpy as np
import json
from vncorenlp import VnCoreNLP
import csv
from pathlib import Path
import time

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
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        dataset = [item.strip() for item in json_data if isinstance(item, str) and item.strip()]
    return dataset

def load_finetune_dataset(file_path):
    """Tải dữ liệu fine-tune từ CSV 2 cột: input, output"""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                dataset.append((row[0].strip(), row[1].strip()))
    return dataset * 10

def prepare_data(pretrain_data, finetune_data, vocab, max_seq_len, pretrain_ratio=0.7, batch_size=50):
    """
    Chuẩn bị dữ liệu gộp từ pretrain_data và finetune_data, với định dạng thống nhất: 
    [BOS] + sequence + [SEP] + sequence + [EOS].
    
    Args:
        pretrain_data: List các câu tự do (ví dụ: ["Max Verstappen là tay đua F1", ...])
        finetune_data: List các cặp [câu hỏi, câu trả lời] (ví dụ: [["Bạn là ai?", "Tôi là S.A.I"], ...])
        vocab: Dictionary ánh xạ từ sang ID
        max_seq_len: Độ dài tối đa của chuỗi
        pretrain_ratio: Tỷ lệ dữ liệu pretrain trong tập gộp (mặc định 0.7)
        batch_size: Kích thước batch để xử lý tokenization (mặc định 50)
    
    Returns:
        X: Dữ liệu đầu vào (input IDs) - không padding
        Y: Dữ liệu mục tiêu (target IDs) - không padding
        lengths: List độ dài thực của từng sequence
    """
    X, Y, lengths = [], [], []
    max_retries = 3
    pretrain_samples = int(len(pretrain_data) * pretrain_ratio)
    
    for i in range(0, pretrain_samples, batch_size):
        batch_data = pretrain_data[i:i+batch_size]
        print(f"Đang xử lý batch pretrain {i//batch_size + 1}/{(pretrain_samples + batch_size - 1)//batch_size}")
        for sentence in batch_data:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    tokens = tokenize(sentence)
                    if len(tokens) < 2 or len(tokens) * 2 + 2 > max_seq_len:  # Kiểm tra độ dài để tránh vượt max_seq_len
                        break
                    # Lặp lại chuỗi tokens
                    sequence = tokens
                    # Tạo đầu vào và mục tiêu
                    inp = [vocab["[BOS]"]] + sequence
                    tgt = sequence + [vocab["[EOS]"]]
                    X.append(inp)
                    Y.append(tgt)
                    lengths.append(len(inp))  # Lưu độ dài thực
                    break  # Thành công, thoát khỏi vòng lặp retry
                except Exception as e:
                    retry_count += 1
                    print(f"Lỗi khi tokenize (lần thử {retry_count}): {e}")
                    if retry_count < max_retries:
                        print("Đang thử lại...")
                        time.sleep(1)
                    else:
                        print(f"Bỏ qua câu: {sentence[:50]}...")
        
        # Thêm delay giữa các batch
        time.sleep(0.1)

    # Xử lý dữ liệu fine-tune
    for i in range(0, len(finetune_data), batch_size):
        batch_data = finetune_data[i:i+batch_size]
        print(f"Đang xử lý batch finetune {i//batch_size + 1}/{(len(finetune_data) + batch_size - 1)//batch_size}")
        for req, res in batch_data:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    req_ids = tokenize(req)
                    res_ids = tokenize(res)
                    # Tạo đầu vào và mục tiêu
                    inp = [vocab["[BOS]"]] + req_ids + [vocab["[SEP]"]] + res_ids
                    tgt = req_ids + [vocab["[SEP]"]] + res_ids + [vocab["[EOS]"]]
                    X.append(inp)
                    Y.append(tgt)
                    lengths.append(len(inp))  # Lưu độ dài thực
                    break  # Thành công, thoát khỏi vòng lặp retry
                except Exception as e:
                    retry_count += 1
                    print(f"Lỗi khi tokenize (lần thử {retry_count}): {e}")
                    if retry_count < max_retries:
                        print("Đang thử lại...")
                        time.sleep(1)
                    else:
                        print(f"Bỏ qua cặp: {req[:50]}...")
        # Thêm delay giữa các batch
        time.sleep(0.1)
    return X, Y, lengths

# Tải và chuẩn bị dữ liệu
raw_dir = current_file.parent.parent / "raw"
pretrain_data = load_pretrain_dataset(raw_dir / "pre_train.json")
finetune_data = load_finetune_dataset(raw_dir / "fine_tune.csv")
X, Y, lengths = prepare_data(pretrain_data, finetune_data, vocab, max_seq_len, pretrain_ratio=1.0, batch_size=50)

np.set_printoptions(threshold=np.inf)

# Lưu dữ liệu gộp với thông tin lengths
data_tokenized_dir = current_file.parent.parent / "processed" / "data_tokenized.py"
with open(data_tokenized_dir, "w", encoding="utf-8") as f:
    f.write("import numpy as np\n\n")
    f.write(f"X = {repr(X)}\n\n")  # Lưu dưới dạng list
    f.write(f"Y = {repr(Y)}\n\n")  # Lưu dưới dạng list
    f.write(f"lengths = {repr(lengths)}\n")  # Lưu thông tin độ dài

print(f"Đã lưu dữ liệu dynamic padding vào: {data_tokenized_dir}")
print(f"Tổng số mẫu: {len(X)}")
print(f"Độ dài sequence trung bình: {np.mean(lengths):.2f}")
print(f"Độ dài sequence min/max: {min(lengths)}/{max(lengths)}")

# Gợi ý tích hợp EWC để ngăn catastrophic forgetting
"""
Để ngăn catastrophic forgetting, có thể tích hợp Elastic Weight Consolidation (EWC) trong quá trình huấn luyện:
1. Tính Fisher Information Matrix trên dữ liệu gộp để xác định các trọng số quan trọng.
2. Thêm penalty term vào hàm mất mát:
   L = L_combined + λ/2 * Σ(F_i * (θ_i - θ_i_initial)^2)
3. Ví dụ code EWC trong TensorFlow:

class EWC:
    def __init__(self, model, dataset, lambda_ewc=1.0):
        self.model = model
        self.dataset = dataset
        self.lambda_ewc = lambda_ewc
        self.params = {var.name: var for var in model.trainable_variables}
        self.means = {var.name: tf.identity(var) for var in model.trainable_variables}
        self.precision_matrices = self.compute_fisher()

    def compute_fisher(self):
        precision_matrices = {name: tf.zeros_like(param) for name, param in self.params.items()}
        for x, y in self.dataset:
            with tf.GradientTape() as tape:
                logits = self.model(x)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            for var, grad in zip(self.model.trainable_variables, gradients):
                precision_matrices[var.name] += tf.square(grad) / len(self.dataset)
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for var in model.trainable_variables:
            name = var.name
            loss += tf.reduce_sum(self.precision_matrices[name] * tf.square(var - self.means[name]))
        return self.lambda_ewc * loss

# Huấn luyện với EWC
model = tf.keras.Sequential([...])
ewc = EWC(model, tf.data.Dataset.from_tensor_slices((combined_X, combined_Y)).batch(32))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        loss += ewc.penalty(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(num_epochs):
    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((combined_X, combined_Y)).batch(32):
        loss = train_step(x_batch, y_batch)
"""
