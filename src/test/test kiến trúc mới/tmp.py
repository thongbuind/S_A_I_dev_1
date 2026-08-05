import json

# ==========================
# Đọc file 35M
# ==========================
with open("/Users/thongbui.nd/Documents/Tài liệu - Nidai/Thong Bui/dev_2/src/test/test kiến trúc mới/sft1_35M.json", "r", encoding="utf-8") as f:
    data_35m = json.load(f)

# ==========================
# Đọc file 100M (jsonl)
# ==========================
data_100m = []
with open("/Users/thongbui.nd/Documents/Tài liệu - Nidai/Thong Bui/dev_2/src/test/test kiến trúc mới/sft1_100M.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data_100m.append(json.loads(line))

# ==========================
# Gộp theo input
# ==========================
merged = {}

# Thêm dữ liệu từ file 35M
for item in data_35m:
    inp = item["input"]

    merged[inp] = {
        "input": inp,
        "model_35M_old": item.get("old_model", ""),
        "model_35M_new": item.get("new_model", ""),
        "model_100M_old": ""
    }

# Thêm dữ liệu từ file 100M
for item in data_100m:
    inp = item["input"]

    if inp not in merged:
        merged[inp] = {
            "input": inp,
            "model_35M_old": "",
            "model_35M_new": "",
            "model_100M_old": item.get("output", "")
        }
    else:
        merged[inp]["model_100M_old"] = item.get("output", "")

# ==========================
# Xuất file JSON
# ==========================
result = list(merged.values())

with open("/Users/thongbui.nd/Documents/Tài liệu - Nidai/Thong Bui/dev_2/src/test/test kiến trúc mới/test.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Đã gộp {len(result)} mẫu vào merged.json")

