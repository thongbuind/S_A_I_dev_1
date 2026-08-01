import json
import time
from pathlib import Path
import sentencepiece as spm

# Khởi động đồng hồ đo thời gian toàn bộ tiến trình
start_total_time = time.time()

# 1. Giữ nguyên logic tìm thư mục gốc tự động
current_file = Path(__file__).resolve()
project_root = current_file.parent

RAW_DATA_ABSOLUTE_PATH = Path("/Users/thongbui.nd/Documents/Thong Bui/dev_2/data/raw/pretrain_data.jsonl")

vocab_size = 10000
input_text_path = project_root / "spm_input.txt"

# =========================================================
# BƯỚC 1+2: ĐỌC, SÀN LỌC VÀ GHI THẲNG RA FILE TẠM
# =========================================================
print(f"🚀 Bước 1+2: Đang xử lý dữ liệu và ghi ra file tạm: {input_text_path.name}...")
start_step1 = time.time()

total_lines_read = 0
malformed_lines = 0
empty_lines = 0
valid_count = 0

# Đọc từ file hardcode tuyệt đối, ghi vào file tạm ở project_root
with open(RAW_DATA_ABSOLUTE_PATH, "r", encoding="utf-8") as fin, \
     open(input_text_path, "w", encoding="utf-8") as fout:

    for line in fin:
        total_lines_read += 1

        if total_lines_read % 100000 == 0:
            print(f"   [Tiến trình] Đã quét qua {total_lines_read:,} dòng dữ liệu thô...")

        line = line.strip()
        if not line:
            empty_lines += 1
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue

        if isinstance(obj, dict) and "text" in obj:
            text = obj["text"].strip()
            if text:
                fout.write(text.lower() + "\n")
                valid_count += 1
            else:
                empty_lines += 1

print(f"📊 Thống kê dữ liệu đã đọc:")
print(f"   - Tổng số dòng quét qua: {total_lines_read:,}")
print(f"   - Số dòng trống/không có text: {empty_lines:,}")
print(f"   - Số dòng lỗi định dạng JSON: {malformed_lines:,}")
print(f"   - Số dòng hợp lệ đưa vào huấn luyện: {valid_count:,}")
print(f"⏱️  Thời gian xử lý Bước 1+2: {time.time() - start_step1:.2f} giây\n")

# =========================================================
# BƯỚC 3: HUẤN LUYỆN SENTENCEPIECE
# =========================================================
print("🧠 Bước 3: Đang huấn luyện SentencePiece (Unigram)...")
print("   * Lưu ý: Thư viện SentencePiece có bộ log C++ nội bộ riêng, chuẩn bị hiển thị bên dưới... *")
start_step3 = time.time()

spm.SentencePieceTrainer.train(
    input=str(input_text_path),
    model_prefix=str(project_root / "tokenizer"),
    vocab_size=vocab_size,
    model_type="unigram",
    character_coverage=0.9999,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
    user_defined_symbols=["<|im_start|>", "<|im_end|>"],
    normalization_rule_name='identity',
    # split_by_whitespace=False,
    treat_whitespace_as_suffix=False,
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
    remove_extra_whitespaces=False,
    split_digits=True,
    byte_fallback=True,
    max_sentencepiece_length=8
)

print(f"\nThời gian huấn luyện SentencePiece: {time.time() - start_step3:.2f} giây\n")

# =========================================================
# HOÀN TẤT TIẾN TRÌNH
# =========================================================
print("🎉 Đã huấn luyện xong!")
print(f"📁 Xuất file thành công tại:")
print(f"   - Model: {project_root / 'tokenizer.model'}")
print(f"   - Vocab: {project_root / 'tokenizer.vocab'}")
print(f"🚀 Tổng thời gian thực hiện toàn bộ script: {time.time() - start_total_time:.2f} giây.")
