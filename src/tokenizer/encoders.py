import json
from pathlib import Path
import numpy as np
import sentencepiece as spm

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
config_file = project_root / "config" / "base.json"
data_dir = project_root / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

tokenizer_file = data_dir / "tokenizer.model"
sp = spm.SentencePieceProcessor()
sp.load(str(tokenizer_file))

BOS = sp.piece_to_id("[BOS]")
EOS = sp.piece_to_id("[EOS]")
IM_START = sp.piece_to_id("<|im_start|>")
IM_END = sp.piece_to_id("<|im_end|>")

def iter_text_jsonl(path, key="text"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and key in obj:
                text = obj[key].strip()
                if text:
                    yield text

def iter_sft_conversations(path):
    """Đọc file .json chứa 1 mảng các object {instruction, history: [{user, model}, ...]}."""
    with open(raw_dir / path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    for obj in data:
        if not isinstance(obj, dict):
            continue
        history = obj.get("history", [])
        if not history:
            continue
        yield obj

def _encode_pretrain(text):
    tokens = sp.encode(text.lower(), out_type=int)
    if len(tokens) < 2 or len(tokens) + 2 > max_seq_len:
        return None
    x = [BOS] + tokens
    y = tokens + [EOS]
    return x, y

def _encode_sft(sample):
    instruction = sample.get("instruction", "").strip()
    history = sample.get("history", [])
    if not history:
        return None

    all_tokens = []
    mask = []

    if instruction:
        instr_ids = (
            [IM_START]
            + sp.encode("system\n" + instruction.lower(), out_type=int)
            + [IM_END]
        )
        all_tokens += instr_ids
        mask += [0] * len(instr_ids)

    for turn in history:
        user_text = turn.get("user", "").strip()
        model_text = turn.get("model", "").strip()
        if not model_text:
            continue

        user_ids = (
            [IM_START]
            + sp.encode("user\n" + user_text.lower(), out_type=int)
            + [IM_END]
        )
        model_ids = (
            [IM_START]
            + sp.encode("model\n" + model_text.lower(), out_type=int)
            + [IM_END]
        )

        all_tokens += user_ids + model_ids
        mask += [0] * len(user_ids) + [1] * len(model_ids)

    if not all_tokens:
        return None

    input_ids = [BOS] + all_tokens + [EOS]
    if len(input_ids) > max_seq_len:
        return None

    target_ids = input_ids[1:]
    mask = mask + [1]

    assert len(mask) == len(target_ids), (
        f"Độ dài mask ({len(mask)}) khác target_ids ({len(target_ids)})"
    )
    return input_ids, target_ids, mask

def _two_pass_write(path, encoded_iter_fn, has_mask=False):
    """Logic 2-pass gốc, ghi ra .npz (giống bản đầu tiên)."""
    x_offsets = [0]
    y_offsets = [0]
    m_offsets = [0]
    lengths = []

    for idx, item in enumerate(encoded_iter_fn()):
        if idx % 100_000 == 0:
            print(f"  [pass 1] {idx:,} dòng...")
        if has_mask:
            x, y, m = item
        else:
            x, y = item
        x_offsets.append(x_offsets[-1] + len(x))
        y_offsets.append(y_offsets[-1] + len(y))
        if has_mask:
            m_offsets.append(m_offsets[-1] + len(m))
        lengths.append(len(x))

    n = len(lengths)
    if n == 0:
        return 0, 0

    total_x = x_offsets[-1]
    total_y = y_offsets[-1]

    x_flat = np.empty(total_x, dtype=np.int32)
    y_flat = np.empty(total_y, dtype=np.int32)
    if has_mask:
        m_flat = np.empty(m_offsets[-1], dtype=np.int8)

    x_off_arr = np.array(x_offsets, dtype=np.int64)
    y_off_arr = np.array(y_offsets, dtype=np.int64)
    len_arr = np.array(lengths, dtype=np.int32)
    if has_mask:
        m_off_arr = np.array(m_offsets, dtype=np.int64)

    for idx, item in enumerate(encoded_iter_fn()):
        if idx % 100_000 == 0:
            print(f"  [pass 2] {idx:,}/{n:,} dòng...")
        if has_mask:
            x, y, m = item
            m_flat[m_offsets[idx]: m_offsets[idx + 1]] = m
        else:
            x, y = item
        x_flat[x_offsets[idx]: x_offsets[idx + 1]] = x
        y_flat[y_offsets[idx]: y_offsets[idx + 1]] = y

    out = {
        "X_flat": x_flat, "X_offsets": x_off_arr,
        "Y_flat": y_flat, "Y_offsets": y_off_arr,
        "lengths": len_arr,
    }
    if has_mask:
        out["M_flat"] = m_flat
        out["M_offsets"] = m_off_arr

    np.savez_compressed(path, **out)
    return n, int(len_arr.sum())

def _write_manifest(manifest_path, shards):
    """Sổ mục lục các shard .npz để lúc load ghép lại thành 1 dataset logic."""
    manifest = {
        "shards": shards,
        "total_samples": sum(s["n_samples"] for s in shards),
        "total_tokens": sum(s["n_tokens"] for s in shards),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest

# ---- Pretrain: mỗi file pretrain_data_00X.jsonl -> 1 shard .npz riêng ----
pretrain_paths = sorted(raw_dir.glob("pretrain_data_*.jsonl"))
print(f"Pretrain - tìm thấy {len(pretrain_paths)} file shard...")

pretrain_shards = []
for p in pretrain_paths:
    shard_name = p.stem  # ví dụ: pretrain_data_001
    print(f"Pretrain shard {shard_name} - pass 1 & 2...")
    n, total = _two_pass_write(
        processed_dir / f"{shard_name}_ids.npz",
        lambda p=p: filter(None, (_encode_pretrain(t) for t in iter_text_jsonl(p))),
    )
    print(f"✅ {shard_name}: {n} samples | Tổng số token: {total:,}")
    pretrain_shards.append({
        "name": shard_name,
        "file": f"{shard_name}_ids.npz",
        "n_samples": n,
        "n_tokens": total,
    })

manifest = _write_manifest(processed_dir / "pretrain_manifest.json", pretrain_shards)
print(f"✅ Pretrain tổng: {manifest['total_samples']} samples | {manifest['total_tokens']:,} token | manifest: pretrain_manifest.json")

# cont_path = raw_dir / "continued_pretrain_data.jsonl"
# print("Continued Pretrain - pass 1 & 2...")
# n, total = _two_pass_write(processed_dir / "continued_pretrain_data_ids.npz", lambda: filter(None, (_encode_pretrain(t) for t in iter_text_jsonl(cont_path))))
# print(f"✅ Continued Pretrain: {n} samples | Tổng số token: {total:,}")

print("SFT1 - pass 1 & 2...")
n, total = _two_pass_write(processed_dir / "SFT1_data_ids.npz", lambda: filter(None, (_encode_sft(s) for s in iter_sft_conversations("SFT_1.json"))), has_mask=True)
print(f"✅ Đã lưu SFT1: {n} samples | Tổng số token: {total:,}")

# print("SFT2 - pass 1 & 2...")
# n, total = _two_pass_write(processed_dir / "SFT2_data_ids.npz", lambda: filter(None, (_encode_sft(s) for s in iter_sft_conversations("SFT_2.json"))), has_mask=True)
# print(f"✅ Đã lưu SFT2: {n} samples | Tổng số token: {total:,}")
