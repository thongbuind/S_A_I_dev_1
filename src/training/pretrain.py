from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import torch._dynamo
import logging
import json
import gc
import argparse
import math
from src.utils.utils import (get_step_lr_lambda, log_progress, load_checkpoint, save_checkpoint, estimate_training_flops, print_training_flops_summary, TflopsBenchmarker, KernelLogger)
from src.utils.chunked_loss import chunked_lm_loss
from src.data.Dataset import Dataset, split_train_val_test, load_data
from src.model.TransformerModel import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model size: 100M or 500M")
parser.add_argument(
    "--phase", type=str, required=True,
    choices=["pretrain", "pretrain_resume", "continued_pretrain", "continued_pretrain_resume", "full"],
    help="Training phase: pretrain | pretrain_resume | continued_pretrain | continued_pretrain_resume | full"
)
parser.add_argument(
    "--profile-kernels", action="store_true",
    help="Chỉ chạy 1 lần duy nhất, log tên kernel CPU/CUDA tại 1 step rồi thôi (dùng để debug performance)"
)
args = parser.parse_args()
torch._dynamo.config.cache_size_limit = 64
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)
model_size = args.model

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
config_dir = project_root / "config"
data_dir = project_root / "data"
model_dir = project_root / "model"
src_dir = project_root / "src"
base_config_file = config_dir / "base.json"
model_config_file = config_dir / f"{args.model}.json"
model_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
pretrain_tokenized_file = data_processed_dir / "pretrain_manifest.json"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_data_ids.npz"

pretrained_save_path = model_dir / f"pretrained_{model_size}.pt"
pretrained_ckpt_path = model_dir / f"pretrained_{model_size}.ckpt.pt"
continued_pretrained_save_path = model_dir / f"continued_pretrained_{model_size}.pt"
continued_pretrained_ckpt_path = model_dir / f"continued_pretrained_{model_size}.ckpt.pt"

# Peak VRAM cho logits mỗi chunk ~ LM_LOSS_CHUNK_SIZE * vocab_size * 4 bytes (fp32).
# Tune số này lên nếu còn dư VRAM (giảm số lần loop), giảm xuống nếu OOM.
LM_LOSS_CHUNK_SIZE = 4096


def train_loop(data_type, tokenized_file, epochs, learning_rate, weight_decay, num_workers, extra_file=None, model_save_path=None, resume_checkpoint_path=None, profile_kernels=False):
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                                BAT ĐAU LOAD DATA                                   ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    if extra_file is None:
        X, Y, lengths = load_data(data_type, tokenized_file)
    else:
        X, Y, lengths = load_data(data_type, tokenized_file, extra_file)

    X_train, Y_train, _, lengths_train, X_val, Y_val, _, lengths_val, X_test, Y_test, _, lengths_test = split_train_val_test(X, Y, None, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = Dataset.create_dataloader(X_train, Y_train, lengths_train, batch_size, max_seq_len, num_workers, shuffle=True)
    val_ds = Dataset.create_dataloader(X_val, Y_val, lengths_val, batch_size, max_seq_len, num_workers, shuffle=False)
    test_ds = Dataset.create_dataloader(X_test, Y_test, lengths_test, batch_size, max_seq_len, num_workers, shuffle=False)

    del X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test
    gc.collect()

    global optimizer
    # model_raw.parameters() -- KHÔNG dùng model.parameters() của wrapper compile để
    # tránh mọi rủi ro liên quan tới cách OptimizedModule proxy tham số ở các version
    # PyTorch khác nhau. model_raw và model (compiled) chia sẻ chung tensor tham số
    # nên optimizer.step() vẫn cập nhật đúng trọng số dùng trong forward compiled.
    optimizer = optim.AdamW(model_raw.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)

    total_steps = math.ceil(len(train_ds) / accumulation_steps) * epochs
    warmup_steps = total_steps // 10

    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0

    checkpoint_path = model_save_path.with_suffix(".ckpt.pt") if model_save_path is not None else None
    if resume_checkpoint_path is not None:
        if resume_checkpoint_path.exists():
            start_epoch, global_step, best_val_loss = load_checkpoint(
                resume_checkpoint_path, model_raw, optimizer, scheduler, device
            )
        else:
            log_progress(f"[WARNING] Checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")

    flops_info = estimate_training_flops(model=model_raw, num_layers=num_layers, max_seq_len=max_seq_len, d_model=d_model, epochs=epochs, batches_per_epoch=len(train_ds), batch_size=batch_size)
    print_training_flops_summary(flops_info, epochs)
    bench = TflopsBenchmarker(flops_per_token=flops_info["flops_per_token"], total_flops_needed=flops_info["total_flops"], total_batches_per_epoch=flops_info["total_batches_per_epoch"], epochs=epochs, device=device, warmup_steps=100, target_steps=500)
    kernel_logger = KernelLogger(enabled=profile_kernels, log_step=500)

    for epoch in range(start_epoch, epochs):
        model_raw.train()
        # Gom loss trên GPU (tensor), chỉ .item() khi thực sự cần in/log ->
        # tránh sync CPU-GPU mỗi micro-batch, kết quả loss cuối cùng không đổi.
        train_loss = torch.zeros((), device=device)
        batch_count = 0
        total_batches = len(train_ds)
        optimizer.zero_grad()

        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask, has_padding) in enumerate(train_ds):
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            _is_bench_step = bench.is_bench_step(epoch, start_epoch, batch_idx)
            if _is_bench_step:
                bench.on_step_begin(batch_idx, X_batch.shape[0] * X_batch.shape[1])

            _is_kernel_log = kernel_logger.should_log(epoch, start_epoch, batch_idx)
            if _is_kernel_log:
                kernel_logger.start()

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                # gọi qua `model_features` (bản đã torch.compile) để hưởng lợi fuse kernel,
                # DỪNG TRƯỚC lm_head -> chưa vật lý hóa logits full ở đây
                hidden = model_features(X_batch, attention_mask=attention_mask, has_padding=has_padding)
                hidden_flat = hidden.reshape(-1, hidden.size(-1))
                targets_flat = Y_batch.reshape(-1)
                weight_flat = sample_weight.reshape(-1)
                num_valid_tokens = weight_flat.sum()
                loss_sum = chunked_lm_loss(
                    hidden_flat, model_raw.lm_head.weight, targets_flat, weight_flat,
                    chunk_size=LM_LOSS_CHUNK_SIZE,
                )
                loss = loss_sum / (num_valid_tokens + 1e-8)

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            if _is_kernel_log:
                kernel_logger.stop_and_report(batch_idx, epoch, device)

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=1.0)
                optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            train_loss += loss.detach()
            batch_count += 1
            current_lr = optimizer.param_groups[0]['lr']

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = (train_loss / batch_count).item()
                print(f"\rEpoch {epoch+1}/{epochs} - Step {global_step}/{total_steps} - loss: {avg_loss:.4f} - lr: {current_lr:.2e}", end='')

            if _is_bench_step:
                bench.on_step_end(batch_idx)

            del X_batch, Y_batch, hidden, hidden_flat, loss, loss_sum, sample_weight, attention_mask

        print()

        train_loss = (train_loss / len(train_ds)).item()
        model_raw.eval()
        val_loss = torch.zeros((), device=device)

        with torch.no_grad():
            for X_batch, Y_batch, sample_weight, attention_mask, has_padding in val_ds:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                sample_weight = sample_weight.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    hidden = model_features(X_batch, attention_mask=attention_mask, has_padding=has_padding)
                    hidden_flat = hidden.reshape(-1, hidden.size(-1))
                    targets_flat = Y_batch.reshape(-1)
                    weight_flat = sample_weight.reshape(-1)
                    num_valid_tokens = weight_flat.sum()
                    loss_sum = chunked_lm_loss(
                        hidden_flat, model_raw.lm_head.weight, targets_flat, weight_flat,
                        chunk_size=LM_LOSS_CHUNK_SIZE,
                    )

                loss = loss_sum / (num_valid_tokens + 1e-8)
                val_loss += loss.detach()

                del X_batch, Y_batch, hidden, hidden_flat, loss, loss_sum, sample_weight, attention_mask

        val_loss = (val_loss / len(val_ds)).item()
        log_progress(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_save_path is not None:
                torch.save(model_raw.state_dict(), model_save_path)
                print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model to {model_save_path}")
            else:
                torch.save(model_raw.state_dict(), pretrained_save_path)
                print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model to default path")

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path, epoch, global_step,
                model_raw, optimizer, scheduler, best_val_loss
            )
            log_progress(f"Checkpoint saved → {checkpoint_path}")

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               DANH GIA TREN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    model_raw.eval()
    test_loss = torch.zeros((), device=device)

    with torch.no_grad():
        for X_batch, Y_batch, sample_weight, attention_mask, has_padding in test_ds:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                hidden = model_features(X_batch, attention_mask=attention_mask, has_padding=has_padding)
                hidden_flat = hidden.reshape(-1, hidden.size(-1))
                targets_flat = Y_batch.reshape(-1)
                weight_flat = sample_weight.reshape(-1)
                num_valid_tokens = weight_flat.sum()
                loss_sum = chunked_lm_loss(
                    hidden_flat, model_raw.lm_head.weight, targets_flat, weight_flat,
                    chunk_size=LM_LOSS_CHUNK_SIZE,
                )

            loss = loss_sum / (num_valid_tokens + 1e-8)
            test_loss += loss.detach()

            del X_batch, Y_batch, hidden, hidden_flat, loss, loss_sum, sample_weight, attention_mask

    test_loss = (test_loss / len(test_ds)).item()
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))

vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_kv_heads = config['num_kv_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
pretrain_epochs = config['pretrain_epochs']
continued_pretrain_epochs = config['continued_pretrain_epochs']
batch_size = config['batch_size']
train_ratio = config['train_ratio']
val_ratio = config['val_ratio']
pretrain_learning_rate = config['pretrain_learning_rate']
continued_pretrain_learning_rate = config['continued_pretrain_learning_rate']
accumulation_steps = config['accumulation_steps']
pretrain_weight_decay = config['pretrain_weight_decay']
continued_pretrain_weight_decay = config['continued_pretrain_weight_decay']
num_workers = config['num_workers']

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model_raw = TransformerModel(vocab_size, d_model, num_heads, num_kv_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
model = torch.compile(model_raw, dynamic=True)
model_features = torch.compile(model_raw.forward_features, dynamic=True)
optimizer = optim.AdamW(model_raw.parameters(), lr=pretrain_learning_rate, weight_decay=pretrain_weight_decay, fused=True)

print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BAT ĐAU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")

phase = args.phase

if phase == "pretrain":
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")

elif phase == "pretrain_resume":
    log_progress("Pretrain resume: khởi tạo model skeleton trước khi load checkpoint...")
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=pretrained_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")

elif phase == "continued_pretrain":
    log_progress("Load best model từ pretrain để tiếp tục training...")
    model_raw.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    model_raw.to(device)
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)
    log_progress(f"Reset optimizer với learning rate: {continued_pretrain_learning_rate}")

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")

elif phase == "continued_pretrain_resume":
    log_progress("Continued pretrain resume: khởi tạo model skeleton trước khi load checkpoint...")
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=continued_pretrained_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")

elif phase == "full":
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )

    log_progress("Đang load best model từ pretrain để tiếp tục training...")
    model_raw.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    model_raw.to(device)
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)
    log_progress(f"Reset optimizer với learning rate: {continued_pretrain_learning_rate}")

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )

    log_progress(f"Hoàn thành training!")
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
