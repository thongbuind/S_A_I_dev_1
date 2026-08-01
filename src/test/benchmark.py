import os, re, sys, json, time, string, torch
from pathlib import Path
from collections import Counter

# ── Setup path ────────────────────────────────────────────────────────────────
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from tokenizers import Tokenizer
from model.TransformerModel import TransformerModel

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES   = 100
RESULT_FILE = current_file.parent / "benchmark_results.md"
OUTPUT_DIR  = current_file.parent / "benchmark_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
config_dir        = project_root / "config"
base_config_file  = config_dir / "base.json"
model_config_file = config_dir / "100M.json"
sft1_file         = project_root / "model" / "sft1_100M.pt"
data_dir          = project_root / "data"

with open(base_config_file)  as f: config = json.load(f)
with open(model_config_file) as f: config.update(json.load(f))

device = (torch.device('cuda')   if torch.cuda.is_available()
     else torch.device('mps:0')  if torch.backends.mps.is_available()
     else torch.device('cpu'))
print(f"🖥️  Device: {device}")

tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
vocab      = tokenizer.get_vocab()
BOS  = vocab["[BOS]"];  EOS  = vocab["[EOS]"]
PAD  = vocab["[PAD]"];  USER = vocab["<|user|>"]
SAI  = vocab["<|s.a.i|>"]

model = TransformerModel(
    config['vocab_size'], config['d_model'], config['num_heads'],
    config['num_layers'], config['ff_dim'], config['max_seq_len'],
    config['dropout']
)
model.load_state_dict(torch.load(sft1_file, map_location=device))
model.to(device).eval()
print(f"✅ Model loaded — {sum(p.numel() for p in model.parameters()):,} params")

# ── Debug: in token IDs của A/B/C/D ──────────────────────────────────────────
def _debug_option_tokens():
    print("\n🔍 Debug option tokens:")
    for letter in ("A", "B", "C", "D"):
        for variant in (f" {letter}", letter, f"({letter})", f"{letter}."):
            ids = tokenizer.encode(variant).ids
            print(f"   '{variant}' → ids={ids}")
    print()

_debug_option_tokens()

# ── Inference helpers ─────────────────────────────────────────────────────────

def build_input(user_input: str):
    ids = tokenizer.encode(" " + user_input).ids
    return [BOS, USER] + ids + [SAI]


def _find_option_token_ids(option_letters=("A", "B", "C", "D")):
    """
    Tìm token ID tốt nhất cho mỗi chữ cái lựa chọn.
    Thử nhiều variant để đảm bảo tìm được single token.
    Trả về dict: letter → token_id (hoặc None nếu không tìm được).
    """
    result = {}
    for letter in option_letters:
        found = None
        for variant in (f" {letter}", letter, f"({letter})", f"{letter}.", f" {letter}."):
            ids = tokenizer.encode(variant).ids
            if len(ids) == 1:
                found = ids[0]
                break
        result[letter] = found
    return result

# Cache token IDs
_OPTION_TOKEN_CACHE = {}

def get_option_logits(prompt_text: str, option_letters=("A", "B", "C", "D")):
    """
    Prefill với prompt, lấy logit của token A/B/C/D ở bước cuối.
    Nếu không tìm được single token → fallback generate + parse.
    Trả về: (pred_index: int, raw_logits: dict[letter→float])
    """
    cache_key = tuple(option_letters)
    if cache_key not in _OPTION_TOKEN_CACHE:
        _OPTION_TOKEN_CACHE[cache_key] = _find_option_token_ids(option_letters)
    token_ids = _OPTION_TOKEN_CACHE[cache_key]

    input_ids = build_input(prompt_text)
    t = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits, _ = model.prefill(t)   # (1, vocab)
    logits = logits[0]

    all_none = all(v is None for v in token_ids.values())

    if all_none:
        # Fallback: generate ngắn và parse chữ cái đầu
        raw_output = generate_text(prompt_text, max_new_tokens=10)
        parsed = raw_output.strip().upper()
        for i, letter in enumerate(option_letters):
            if parsed.startswith(letter):
                logits_dict = {l: None for l in option_letters}
                return i, logits_dict, raw_output
        return 0, {l: None for l in option_letters}, raw_output

    scores = []
    logits_dict = {}
    for letter in option_letters:
        tid = token_ids.get(letter)
        if tid is not None:
            val = logits[tid].item()
            scores.append(val)
            logits_dict[letter] = round(val, 4)
        else:
            scores.append(float('-inf'))
            logits_dict[letter] = None

    pred_idx = int(torch.tensor(scores).argmax().item())
    return pred_idx, logits_dict, None   # None = không dùng fallback generate


def generate_text(user_input: str, max_new_tokens: int = 150) -> str:
    return model.generate_response(
        user_input, tokenizer,
        max_new_tokens=max_new_tokens,
        beam_size=3,
        no_repeat_ngram=3,
        penalty=1.2,
        early_stop=True,
        patience=20,
    )

# ── Metric helpers ────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score_tokens(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_tokens) if pred_tokens else 0
    r = n_common / len(gold_tokens) if gold_tokens else 0
    return 2 * p * r / (p + r)


def rouge_l(pred: str, ref: str) -> float:
    pred_tok = pred.lower().split()
    ref_tok  = ref.lower().split()
    if not pred_tok or not ref_tok:
        return 0.0
    m, n = len(pred_tok), len(ref_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if pred_tok[i-1] == ref_tok[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / m; r = lcs / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def extract_number(text: str):
    nums = re.findall(r'-?\d+(?:[.,]\d+)?', text)
    if not nums:
        return None
    return nums[-1].replace(',', '')


def parse_choices_vmlu(item: dict):
    """
    Hỗ trợ nhiều schema VMLU:
    - choices: list (có thể < 4 phần tử → pad)
    - A, B, C, D: field riêng
    - option_A/B/C/D hoặc option_a/b/c/d
    """
    if 'choices' in item and isinstance(item['choices'], list):
        raw = item['choices']
        cleaned = []
        for c in raw:
            c = re.sub(r'^[A-D]\.\s*', '', str(c)).strip()
            cleaned.append(c)
        while len(cleaned) < 4:
            cleaned.append('')
        return cleaned

    if 'A' in item and 'B' in item:
        return [str(item.get(k, '')) for k in ('A', 'B', 'C', 'D')]

    if 'option_A' in item or 'option_a' in item:
        keys = [f'option_{k}' for k in ('A', 'B', 'C', 'D')]
        if keys[0] not in item:
            keys = [f'option_{k}' for k in ('a', 'b', 'c', 'd')]
        return [str(item.get(k, '')) for k in keys]

    return ['', '', '', '']


def parse_answer_vmlu(item: dict) -> str:
    """Lấy gold answer, chuẩn hóa về A/B/C/D."""
    raw = str(item.get('answer', item.get('Answer', item.get('label', 'A')))).strip()
    if raw in ('0', '1', '2', '3'):
        return 'ABCD'[int(raw)]
    return raw.upper()[:1]

# ── Markdown writer ───────────────────────────────────────────────────────────

def init_md():
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Results — SAI 100M\n\n")
        f.write(f"- **Samples per dataset**: {N_SAMPLES}\n")
        f.write(f"- **Device**: {device}\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")


def append_md(content: str):
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(content + "\n\n")


def save_outputs(name: str, rows: list):
    p = OUTPUT_DIR / f"{name}_outputs.json"
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"   💾 Outputs → {p}")

# ── BENCHMARK 1: VMLU ────────────────────────────────────────────────────────

def run_vmlu():
    print("\n📊 [1/7] VMLU ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("tridm/VMLU", split="test")
        print(f"   ✅ Loaded: tridm/VMLU/test — {len(ds)} samples")
    except Exception as e:
        print(f"   ❌ {e}")
        append_md("## 1. VMLU\n\n❌ Không tải được dataset.\n")
        return

    # Debug schema item đầu tiên
    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    rows = []
    correct = 0
    skipped = 0
    # FIX: tăng buffer lên N_SAMPLES * 10 để bù skip rate cao
    buffer_size = min(N_SAMPLES * 10, len(ds))
    ds_shuffled = ds.shuffle(seed=42).select(range(buffer_size))

    for item in ds_shuffled:
        if len(rows) >= N_SAMPLES:
            break

        q       = item.get('question', '')
        choices = parse_choices_vmlu(item)
        gold    = parse_answer_vmlu(item)

        valid_choices = [c for c in choices if c.strip()]
        if len(valid_choices) < 2 or gold not in 'ABCD':
            skipped += 1
            continue

        n_opts     = len(valid_choices)
        opt_letters = list("ABCD")[:n_opts]

        prompt = f"Câu hỏi: {q}\n"
        for letter, choice in zip(opt_letters, valid_choices):
            prompt += f"{letter}. {choice}\n"
        prompt += f"Chọn đáp án đúng ({'/'.join(opt_letters)}):"

        pred_idx, logits_dict, fallback_output = get_option_logits(prompt, tuple(opt_letters))
        pred_letter = opt_letters[pred_idx]
        is_correct  = (pred_letter == gold)
        if is_correct:
            correct += 1

        rows.append({
            "prompt":          prompt,
            "question":        q,
            "choices":         choices,
            "gold":            gold,
            "pred":            pred_letter,
            "correct":         is_correct,
            "logits":          logits_dict,
            "fallback_output": fallback_output,
        })

    total = len(rows)
    acc   = correct / total if total else 0
    print(f"   VMLU Accuracy: {correct}/{total} = {acc:.2%}  (skipped={skipped})")
    save_outputs("vmlu", rows)
    append_md(f"""## 1. VMLU

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.2%} ({correct}/{total}) |
| Skipped | {skipped} mẫu (< 2 lựa chọn hợp lệ hoặc gold không hợp lệ) |
| Dataset | `tridm/VMLU` / test |
| Method | Logit-based (argmax token A/B/C/D) |
""")

# ── BENCHMARK 2: MMLU-Vietnamese ─────────────────────────────────────────────

def run_mmlu_vi():
    print("\n📊 [2/7] MMLU-Vietnamese ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("alexandrainst/m_mmlu", "vi", split="test")
        print(f"   ✅ Loaded: alexandrainst/m_mmlu(vi)/test — {len(ds)} samples")
    except Exception as e:
        print(f"   ❌ {e}")
        append_md("## 2. MMLU-Vietnamese\n\n❌ Không tải được dataset.\n")
        return

    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    rows = []
    correct = 0
    ds_shuffled = ds.shuffle(seed=42).select(range(min(N_SAMPLES, len(ds))))

    for item in ds_shuffled:
        q       = item.get('instruction', '')
        choices = [str(item.get(k, '')) for k in ('option_a', 'option_b', 'option_c', 'option_d')]
        gold    = str(item.get('answer', 'A')).strip().upper()[:1]

        prompt = (f"Câu hỏi: {q}\n"
                  f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
                  f"Chọn đáp án đúng (A/B/C/D):")

        pred_idx, logits_dict, fallback_output = get_option_logits(prompt)
        pred_letter = "ABCD"[pred_idx]
        is_correct  = (pred_letter == gold)
        if is_correct:
            correct += 1

        rows.append({
            "prompt":          prompt,
            "question":        q,
            "choices":         choices,
            "gold":            gold,
            "pred":            pred_letter,
            "correct":         is_correct,
            "logits":          logits_dict,
            "fallback_output": fallback_output,
        })

    total = len(rows)
    acc   = correct / total if total else 0
    print(f"   MMLU-VI Accuracy: {correct}/{total} = {acc:.2%}")
    save_outputs("mmlu_vi", rows)
    append_md(f"""## 2. MMLU-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.2%} ({correct}/{total}) |
| Dataset | `alexandrainst/m_mmlu` (vi) / test |
| Method | Logit-based (argmax A/B/C/D tokens) |
""")

# ── BENCHMARK 3: GSM8K-Vietnamese ────────────────────────────────────────────

def run_gsm8k_vi():
    print("\n📊 [3/7] GSM8K-Vietnamese (MetaMath-VI) ...")
    from datasets import load_dataset

    CANDIDATES = [
        ("5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated", None, "train"),
        ("5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated", None, "train"),
    ]

    ds = None
    used_id = None
    for ds_id, cfg, split in CANDIDATES:
        try:
            ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
            used_id = f"{ds_id} / {split}"
            print(f"   ✅ Loaded: {used_id} — {len(ds)} samples, cols={ds.column_names}")
            break
        except Exception as e:
            print(f"   ⚠️  {ds_id}: {e}")

    if ds is None:
        append_md("## 3. GSM8K-Vietnamese\n\n❌ Không tải được dataset.\n")
        return

    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    rows = []
    correct = 0
    skipped = 0
    ds_shuffled = ds.shuffle(seed=42).select(range(min(N_SAMPLES * 3, len(ds))))

    for item in ds_shuffled:
        if len(rows) >= N_SAMPLES:
            break

        # FIX: đọc đúng column 'query_vi' và 'response_vi'
        q    = str(item.get('query_vi',    item.get('query',    item.get('instruction', item.get('input', ''))))).strip()
        resp = str(item.get('response_vi', item.get('response', item.get('output', '')))).strip()

        if not q or not resp:
            skipped += 1
            continue

        gold = extract_number(resp)
        if gold is None:
            skipped += 1
            continue

        prompt = f"Hãy giải bài toán sau và cho biết đáp án cuối cùng là số nào:\n{q}"
        raw_output = generate_text(prompt, max_new_tokens=200)
        pred       = extract_number(raw_output)
        is_correct = (pred is not None) and (pred == gold)
        if is_correct:
            correct += 1

        rows.append({
            "prompt":      prompt,
            "question":    q,
            "gold_answer": gold,
            "gold_full":   resp,
            "pred_number": pred,
            "raw_output":  raw_output,
            "correct":     is_correct,
        })

    total = len(rows)
    acc   = correct / total if total else 0
    print(f"   GSM8K-VI Accuracy: {correct}/{total} = {acc:.2%}  (skipped={skipped})")
    save_outputs("gsm8k_vi", rows)
    append_md(f"""## 3. GSM8K-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.2%} ({correct}/{total}) |
| Skipped | {skipped} mẫu (không tìm được gold number) |
| Dataset | `{used_id}` |
| Method | Generate → extract last number → exact match |
| Note | Gold = số cuối trong response_vi gốc của dataset |
""")

# ── BENCHMARK 4: XNLI-Vietnamese ─────────────────────────────────────────────

def run_xnli_vi():
    print("\n📊 [4/7] XNLI-Vietnamese ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("facebook/xnli", "vi", split="test")
        print(f"   ✅ Loaded: facebook/xnli(vi)/test — {len(ds)} samples")
    except Exception as e:
        print(f"   ❌ {e}")
        append_md("## 4. XNLI-Vietnamese\n\n❌ Không tải được dataset.\n")
        return

    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    options   = ("entailment", "neutral", "contradiction")

    rows = []
    correct = 0
    ds_shuffled = ds.shuffle(seed=42).select(range(min(N_SAMPLES, len(ds))))

    for item in ds_shuffled:
        premise    = str(item.get('premise', ''))
        hypothesis = str(item.get('hypothesis', ''))
        gold_int   = int(item.get('label', 0))
        gold       = label_map[gold_int]

        prompt = (f"Tiền đề: {premise}\n"
                  f"Giả thuyết: {hypothesis}\n"
                  f"Mối quan hệ giữa hai câu là:\n"
                  f"A. entailment (tiền đề kéo theo giả thuyết)\n"
                  f"B. neutral (không rõ quan hệ)\n"
                  f"C. contradiction (mâu thuẫn nhau)\n"
                  f"Chọn A, B, hoặc C:")

        pred_idx, logits_dict, fallback_output = get_option_logits(prompt, ("A", "B", "C"))
        pred       = options[pred_idx]
        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        rows.append({
            "prompt":          prompt,
            "premise":         premise,
            "hypothesis":      hypothesis,
            "gold":            gold,
            "pred":            pred,
            "correct":         is_correct,
            "logits":          logits_dict,
            "fallback_output": fallback_output,
        })

    total = len(rows)
    acc   = correct / total if total else 0
    print(f"   XNLI-VI Accuracy: {correct}/{total} = {acc:.2%}")
    save_outputs("xnli_vi", rows)
    append_md(f"""## 4. XNLI-Vietnamese

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.2%} ({correct}/{total}) |
| Dataset | `facebook/xnli` (vi) / test |
| Labels | entailment / neutral / contradiction |
| Method | Logit-based (argmax A/B/C tokens) |
""")

# ── BENCHMARK 5: XQuAD-Vietnamese ────────────────────────────────────────────

def run_xquad_vi():
    print("\n📊 [5/7] XQuAD-Vietnamese ...")
    from datasets import load_dataset

    try:
        ds = load_dataset("google/xquad", "xquad.vi", split="validation")
        print(f"   ✅ Loaded: google/xquad(xquad.vi)/validation — {len(ds)} samples")
    except Exception as e:
        print(f"   ❌ {e}")
        append_md("## 5. XQuAD-Vietnamese\n\n❌ Không tải được dataset.\n")
        return

    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    rows = []
    em_total = 0
    f1_total = 0.0
    ds_shuffled = ds.shuffle(seed=42).select(range(min(N_SAMPLES, len(ds))))

    for item in ds_shuffled:
        context  = item.get('context', '')[:800]
        question = item.get('question', '')
        ans_raw  = item.get('answers', {})
        if isinstance(ans_raw, dict):
            answers = ans_raw.get('text', [])
        else:
            answers = [str(ans_raw)]

        prompt = (f"Đọc đoạn văn sau và trả lời câu hỏi bằng tiếng Việt:\n\n"
                  f"Đoạn văn: {context}\n\n"
                  f"Câu hỏi: {question}\n"
                  f"Trả lời ngắn gọn:")

        raw_output = generate_text(prompt, max_new_tokens=80)
        em = max((int(exact_match(raw_output, a)) for a in answers), default=0)
        f1 = max((f1_score_tokens(raw_output, a) for a in answers), default=0.0)
        em_total += em
        f1_total += f1

        rows.append({
            "prompt":     prompt,
            "context":    context,
            "question":   question,
            "gold":       answers,
            "raw_output": raw_output,
            "em":         em,
            "f1":         round(f1, 4),
        })

    total  = len(rows)
    em_avg = em_total / total if total else 0
    f1_avg = f1_total / total if total else 0
    print(f"   XQuAD-VI EM: {em_avg:.2%}, F1: {f1_avg:.2%}")
    save_outputs("xquad_vi", rows)
    append_md(f"""## 5. XQuAD-Vietnamese

| Metric | Value |
|--------|-------|
| Exact Match | {em_avg:.2%} |
| Token F1 | {f1_avg:.2%} |
| Dataset | `google/xquad` (xquad.vi) / validation |
| Method | Generate → EM & F1 (SQuAD-style) |
""")

# ── BENCHMARK 6: UIT-VSFC Sentiment ──────────────────────────────────────────

def run_vsfc_sentiment():
    print("\n📊 [6/7] UIT-VSFC Sentiment ...")
    from datasets import load_dataset

    try:
        # FIX: thêm trust_remote_code=True
        ds = load_dataset(
            "uitnlp/vietnamese_students_feedback",
            split="test",
            trust_remote_code=True,
        )
        print(f"   ✅ Loaded: uitnlp/vietnamese_students_feedback/test — {len(ds)} samples")
    except Exception as e:
        print(f"   ❌ {e}")
        append_md("## 6. UIT-VSFC Sentiment\n\n❌ Không tải được dataset.\n")
        return

    first_item = dict(ds[0])
    print(f"   🔍 Schema sample[0]: { {k: str(v)[:80] for k, v in first_item.items()} }")

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    options   = ("negative", "neutral", "positive")

    rows = []
    correct = 0
    ds_shuffled = ds.shuffle(seed=42).select(range(min(N_SAMPLES, len(ds))))

    for item in ds_shuffled:
        sentence = str(item.get('sentence', ''))
        gold_int = int(item.get('sentiment', 1))
        gold     = label_map.get(gold_int, "neutral")

        prompt = (f"Đây là phản hồi của sinh viên về môn học:\n"
                  f"\"{sentence}\"\n\n"
                  f"Cảm xúc của câu này là:\n"
                  f"A. negative (tiêu cực)\n"
                  f"B. neutral (trung lập)\n"
                  f"C. positive (tích cực)\n"
                  f"Chọn A, B, hoặc C:")

        pred_idx, logits_dict, fallback_output = get_option_logits(prompt, ("A", "B", "C"))
        pred       = options[pred_idx]
        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        rows.append({
            "prompt":          prompt,
            "sentence":        sentence,
            "gold":            gold,
            "pred":            pred,
            "correct":         is_correct,
            "logits":          logits_dict,
            "fallback_output": fallback_output,
        })

    total = len(rows)
    acc   = correct / total if total else 0
    print(f"   VSFC Sentiment Accuracy: {correct}/{total} = {acc:.2%}")
    save_outputs("vsfc_sentiment", rows)
    append_md(f"""## 6. UIT-VSFC Sentiment

| Metric | Value |
|--------|-------|
| Accuracy | {acc:.2%} ({correct}/{total}) |
| Dataset | `uitnlp/vietnamese_students_feedback` / test |
| Labels | negative / neutral / positive |
| Method | Logit-based (argmax A/B/C tokens) |
| Note | Phân loại cảm xúc phản hồi sinh viên tiếng Việt |
""")

# ── BENCHMARK 7: MT-Bench-VI ──────────────────────────────────────────────────

MT_BENCH_QUESTIONS = [
    ("Writing",      "Hãy viết một bài thơ ngắn về mùa xuân ở Việt Nam."),
    ("Reasoning",    "Giải thích nguyên lý hoạt động của động cơ đốt trong bằng ngôn ngữ đơn giản."),
    ("Storytelling", "Hãy kể một câu chuyện ngắn về một chú mèo thông minh."),
    ("Science",      "Tại sao bầu trời có màu xanh? Giải thích cho học sinh lớp 5 hiểu."),
    ("Dialog",       "Viết một đoạn hội thoại giữa hai người bạn đang bàn về du lịch Đà Lạt."),
    ("Advice",       "Hãy đề xuất 5 thói quen tốt để nâng cao sức khỏe tâm thần."),
    ("Tech",         "Giải thích sự khác biệt giữa machine learning và deep learning."),
    ("Culture",      "Viết một đoạn mô tả ngắn về phở Hà Nội cho người nước ngoài."),
    ("Opinion",      "Tại sao việc đọc sách quan trọng trong thời đại số?"),
    ("Business",     "Hãy soạn một email xin lỗi khách hàng vì giao hàng trễ."),
]

def run_mt_bench_vi():
    print("\n📊 [7/7] MT-Bench-VI ...")
    rows = []
    for i, (category, q) in enumerate(MT_BENCH_QUESTIONS):
        print(f"   [{i+1}/{len(MT_BENCH_QUESTIONS)}] [{category}] {q[:55]}...")
        raw_output = generate_text(q, max_new_tokens=200)
        rows.append({
            "id":         i + 1,
            "category":   category,
            "prompt":     q,           # input gửi vào model
            "raw_output": raw_output,  # output thô từ model
        })
    save_outputs("mt_bench_vi", rows)

    md = "## 7. MT-Bench-VI\n\n"
    md += "_Đánh giá thủ công — xem output bên dưới hoặc trong `benchmark_outputs/`._\n\n"
    for r in rows:
        md += f"### [{r['category']}] Câu {r['id']}\n"
        md += f"**Q:** {r['prompt']}\n\n"
        md += f"**A:** {r['raw_output']}\n\n"
        md += "---\n\n"
    append_md(md)
    print(f"   ✅ MT-Bench-VI xong — {len(rows)} câu hỏi")

# ── Summary ───────────────────────────────────────────────────────────────────

def append_summary():
    append_md("""---

## Tóm tắt Dataset

| # | Benchmark | Dataset HuggingFace | Ngôn ngữ | Task |
|---|-----------|---------------------|----------|------|
| 1 | VMLU | `tridm/VMLU` | Tiếng Việt | Multiple choice (kiến thức tổng hợp) |
| 2 | MMLU-VI | `alexandrainst/m_mmlu` (vi) | Tiếng Việt | Multiple choice (kiến thức học thuật) |
| 3 | MetaMath-VI | `5CD-AI/Vietnamese-meta-math-*` | Tiếng Việt | Toán (số học) |
| 4 | XNLI-VI | `facebook/xnli` (vi) | Tiếng Việt | NLI 3 nhãn |
| 5 | XQuAD-VI | `google/xquad` (xquad.vi) | Tiếng Việt | Reading comprehension |
| 6 | UIT-VSFC | `uitnlp/vietnamese_students_feedback` | Tiếng Việt | Sentiment 3 nhãn |
| 7 | MT-Bench-VI | Hardcoded | Tiếng Việt | Open-ended generation |

## Cấu trúc Output JSON (mỗi mẫu)

| Field | Có trong | Mô tả |
|-------|----------|-------|
| `prompt` | Tất cả | Input thực tế gửi vào model |
| `raw_output` | Generate tasks (3,5,7) | Text generate thô từ model |
| `logits` | MC tasks (1,2,4,6) | Logit score từng lựa chọn A/B/C/D |
| `fallback_output` | MC tasks (1,2,4,6) | Output generate khi fallback (thường null) |
| `gold` | Tất cả | Nhãn/đáp án đúng |
| `pred` | Tất cả | Dự đoán của model |
| `correct` | Tất cả | True/False |

## Ghi chú

- Mô hình 100M tham số, SFT giai đoạn 1, chưa RLHF.
- **Logit-based** (trắc nghiệm): so logit single token đầu tiên của " A"/" B"/...
  - Nếu tokenizer không có single token → fallback generate + parse chữ đầu.
- **Generate** (tự do): beam search — beam=3, no_repeat_ngram=3, penalty=1.2.
- Output JSON đầy đủ trong `benchmark_outputs/` để review từng mẫu.
- Random seed=42 cho reproducibility.
""")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_md()
    print(f"\n🚀 Bắt đầu benchmark — {N_SAMPLES} mẫu/dataset\n{'='*60}")
    t0 = time.time()

    run_vmlu()
    run_mmlu_vi()
    run_gsm8k_vi()
    run_xnli_vi()
    run_xquad_vi()
    run_vsfc_sentiment()
    run_mt_bench_vi()
    append_summary()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ Xong! Thời gian: {elapsed/60:.1f} phút")
    print(f"📄 Kết quả: {RESULT_FILE}")
    print(f"📁 Outputs: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
