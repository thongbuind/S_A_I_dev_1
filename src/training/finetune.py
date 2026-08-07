from pathlib import Path
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch._dynamo
import json
import gc
import argparse
import logging
import math
from src.utils.utils import (get_step_lr_lambda, freeze_layers, unfreeze_all_layers, log_progress, load_checkpoint, save_checkpoint, estimate_training_flops, print_training_flops_summary, TflopsBenchmarker, KernelLogger)
from src.utils.chunked_loss import chunked_lm_loss, chunked_sft_loss
from src.data.Dataset import Dataset, split_train_val_test, load_data
from src.training.PenaltyEngine import PenaltyEngine, WrongTokenMarginPenalty, WrongTokenEntropyPenalty, FocalOverconfidencePenalty
from src.model.TransformerModel import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model size: 35M, 100M or 500M")
parser.add_argument(
    "--phase", type=str, required=True,
    choices=["sft1", "sft2", "sft1_resume", "sft2_resume", "full"],
    help="Training phase: sft1 | sft2 | sft1_resume | sft2_resume | full"
)
parser.add_argument(
    "--profile-kernels", action="store_true",
    help="Profile kernel tại step 500 và lưu compact JSON"
)
parser.add_argument(
    "--limited-max-autotune", action="store_true",
    help="Bật max-autotune giới hạn cho forward_features (chỉ ATEN/TRITON)"
)
args = parser.parse_args()
torch._dynamo.config.cache_size_limit = 24
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
SFT1_data_ids_file = data_processed_dir / "SFT1_data_ids.npz"
SFT2_data_ids_file = data_processed_dir / "SFT2_data_ids.npz"

pretrained_save_path = model_dir / f"pretrained_{model_size}.pt"
sft1_save_path = model_dir / f"sft1_{model_size}.pt"
sft1_ckpt_path = model_dir / f"sft1_{model_size}.ckpt.pt"
sft2_save_path = model_dir / f"sft2_{model_size}.pt"
sft2_ckpt_path = model_dir / f"sft2_{model_size}.ckpt.pt"

# Peak logits mỗi chunk ~ LM_LOSS_CHUNK_SIZE * vocab_size * 4 bytes.
LM_LOSS_CHUNK_SIZE = 4096

def _build_val_test_loaders(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers):
    X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data)

    X_train, Y_train, mask_train, len_train, \
    X_val, Y_val, mask_val, len_val, \
    X_test, Y_test, mask_test, len_test = split_train_val_test(
        X, Y, loss_mask, lengths, train_ratio, val_ratio
    )

    val_ds = Dataset.create_dataloader(X_val, Y_val, len_val, batch_size, max_seq_len, num_workers, shuffle=False, loss_masks=mask_val)
    test_ds = Dataset.create_dataloader(X_test, Y_test, len_test, batch_size, max_seq_len, num_workers, shuffle=False, loss_masks=mask_test)

    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)

    del X_train, Y_train, mask_train, len_train
    del X_val, Y_val, mask_val, len_val
    del X_test, Y_test, mask_test, len_test
    del X, Y, loss_mask, lengths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_ds, test_ds, train_size, val_size, test_size

def _build_train_loader_epoch(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch=0):
    X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data, seed=epoch)

    X_train, Y_train, mask_train, len_train, \
    X_val, Y_val, mask_val, len_val, \
    X_test, Y_test, mask_test, len_test = split_train_val_test(
        X, Y, loss_mask, lengths, train_ratio, val_ratio
    )

    train_ds = Dataset.create_dataloader(X_train, Y_train, len_train, batch_size, max_seq_len, num_workers, shuffle=True, loss_masks=mask_train)

    del X_val, Y_val, mask_val, len_val
    del X_test, Y_test, mask_test, len_test
    del X_train, Y_train, mask_train, len_train
    del X, Y, loss_mask, lengths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_ds

def finetune(model_raw, optimizer, device, main_data, sub_data, num_epochs, model_save_path, train_ratio, val_ratio, batch_size, num_workers, phase_name, penalty_engine, resample_per_epoch=False, resume_checkpoint_path: Path = None, profile_kernels=False):
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║                               BẮT ĐẦU LOAD {phase_name.upper():<4} DATA                               ║")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    if not resample_per_epoch:
        X, Y, loss_mask, lengths = load_data(phase_name, main_data, sub_data)

        X_train, Y_train, mask_train, len_train, \
        X_val, Y_val, mask_val, len_val, \
        X_test, Y_test, mask_test, len_test = split_train_val_test(
            X, Y, loss_mask, lengths, train_ratio, val_ratio
        )

        log_progress(f"[{phase_name}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        train_ds = Dataset.create_dataloader(X_train, Y_train, len_train, batch_size, max_seq_len, num_workers, shuffle=True, loss_masks=mask_train)
        val_ds = Dataset.create_dataloader(X_val, Y_val, len_val, batch_size, max_seq_len, num_workers, shuffle=False, loss_masks=mask_val)
        test_ds = Dataset.create_dataloader(X_test, Y_test, len_test, batch_size, max_seq_len, num_workers, shuffle=False, loss_masks=mask_test)

        del X_train, Y_train, mask_train, len_train
        del X_val, Y_val, mask_val, len_val
        del X_test, Y_test, mask_test, len_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_steps = math.ceil(len(train_ds) / accumulation_steps) * num_epochs

    else:
        val_ds, test_ds, train_size, val_size, test_size = _build_val_test_loaders(
            phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers
        )
        log_progress(f"[{phase_name}] Train: ~{train_size}, Val: {val_size}, Test: {test_size}")

        train_ds = _build_train_loader_epoch(phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch=0)
        steps_per_epoch = len(train_ds)
        total_steps = math.ceil(steps_per_epoch / accumulation_steps) * num_epochs

    warmup_steps = int(total_steps * 0.15)
    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    log_progress(f"Step-based LR: warmup={warmup_steps} steps, total={total_steps} steps")

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    use_penalty = penalty_engine is not None and len(penalty_engine.rules) > 0
    checkpoint_path = model_save_path.with_suffix(".ckpt.pt")
    if resume_checkpoint_path is not None:
        if resume_checkpoint_path.exists():
            start_epoch, global_step, best_val_loss = load_checkpoint(
                resume_checkpoint_path, model_raw, optimizer, scheduler, device
            )
        else:
            log_progress(f"[WARNING] Checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")

    flops_info = estimate_training_flops(
        model=model_raw,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        d_model=d_model,
        epochs=num_epochs,
        batches_per_epoch=len(train_ds),
        batch_size=batch_size,
    )
    print_training_flops_summary(flops_info, num_epochs)
    bench = TflopsBenchmarker(
        flops_per_token=flops_info["flops_per_token"],
        total_flops_needed=flops_info["total_flops"],
        total_batches_per_epoch=flops_info["total_batches_per_epoch"],
        epochs=num_epochs,
        device=device,
        warmup_steps=100,
        target_steps=500,
    )
    kernel_logger = KernelLogger(enabled=profile_kernels, log_step=500)

    for epoch in range(start_epoch, num_epochs):

        if resample_per_epoch and epoch > 0:
            log_progress(f"[{phase_name}] Epoch {epoch+1}: Re-sampling train data...")
            train_ds = _build_train_loader_epoch(
                phase_name, main_data, sub_data, train_ratio, val_ratio, batch_size, num_workers, epoch
            )

        model_raw.train()
        train_loss = torch.zeros((), device=device)
        batch_count = 0
        total_batches = len(train_ds)
        penalty = torch.zeros((), device=device)

        optimizer.zero_grad(set_to_none=True)
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

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                hidden = model_features(
                    X_batch,
                    attention_mask=attention_mask,
                    has_padding=has_padding,
                )
                hidden_flat = hidden.reshape(-1, hidden.size(-1))
                inputs_flat = X_batch.reshape(-1)
                targets_flat = Y_batch.reshape(-1)
                weight_flat = sample_weight.reshape(-1)
                num_valid_tokens = weight_flat.sum()

                ce_sum, penalty_sum = chunked_sft_loss(
                    hidden_flat,
                    model_raw.lm_head.weight,
                    inputs_flat,
                    targets_flat,
                    weight_flat,
                    penalty_engine=penalty_engine if use_penalty else None,
                    chunk_size=LM_LOSS_CHUNK_SIZE,
                )
                ce_loss = ce_sum / (num_valid_tokens + 1e-8)
                penalty = penalty_sum / (num_valid_tokens + 1e-8)
                loss = ce_loss + penalty

            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            penalty_for_log = penalty.detach()

            if _is_kernel_log:
                kernel_logger.stop_and_report(batch_idx, epoch, device)

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            train_loss += loss.detach()
            batch_count += 1
            current_lr = optimizer.param_groups[0]["lr"]

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = (train_loss / batch_count).item()
                penalty_str = f" | penalty: {penalty_for_log.item():.4f}" if use_penalty else ""
                print(
                    f"\r{phase_name} | Epoch {epoch+1}/{num_epochs} "
                    f"- Step {global_step}/{total_steps} "
                    f"- loss: {avg_loss:.4f}{penalty_str} "
                    f"- lr: {current_lr:.2e}",
                    end=""
                )

            if _is_bench_step:
                bench.on_step_end(batch_idx)

            del X_batch, Y_batch, hidden, hidden_flat, inputs_flat, targets_flat
            del weight_flat, num_valid_tokens, loss, scaled_loss, ce_loss, penalty
            del penalty_for_log, ce_sum, penalty_sum, sample_weight, attention_mask

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

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    hidden = model_features(
                        X_batch,
                        attention_mask=attention_mask,
                        has_padding=has_padding,
                    )
                    hidden_flat = hidden.reshape(-1, hidden.size(-1))
                    targets_flat = Y_batch.reshape(-1)
                    weight_flat = sample_weight.reshape(-1)
                    num_valid_tokens = weight_flat.sum()
                    loss_sum = chunked_lm_loss(
                        hidden_flat,
                        model_raw.lm_head.weight,
                        targets_flat,
                        weight_flat,
                        chunk_size=LM_LOSS_CHUNK_SIZE,
                    )
                    loss = loss_sum / (num_valid_tokens + 1e-8)
                val_loss += loss.detach()

        val_loss = (val_loss / len(val_ds)).item()
        log_progress(f"{phase_name} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss (CE only): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_raw.state_dict(), model_save_path)
            log_progress(f"Epoch {epoch+1}: Saving model and checkpoint")

        save_checkpoint(
            checkpoint_path, epoch, global_step,
            model_raw, optimizer, scheduler, best_val_loss
        )

    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║                            ĐÁNH GIÁ {phase_name.upper():<4} TRÊN TEST SET                             ║")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    model_raw.load_state_dict(torch.load(model_save_path, map_location=device))
    model_raw.eval()
    test_loss = torch.zeros((), device=device)
    with torch.no_grad():
        for X_batch, Y_batch, sample_weight, attention_mask, has_padding in test_ds:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                hidden = model_features(
                    X_batch,
                    attention_mask=attention_mask,
                    has_padding=has_padding,
                )
                hidden_flat = hidden.reshape(-1, hidden.size(-1))
                targets_flat = Y_batch.reshape(-1)
                weight_flat = sample_weight.reshape(-1)
                num_valid_tokens = weight_flat.sum()
                loss_sum = chunked_lm_loss(
                    hidden_flat,
                    model_raw.lm_head.weight,
                    targets_flat,
                    weight_flat,
                    chunk_size=LM_LOSS_CHUNK_SIZE,
                )
                loss = loss_sum / (num_valid_tokens + 1e-8)
            test_loss += loss.detach()

    test_loss = (test_loss / len(test_ds)).item()
    log_progress(f"{phase_name} Test Loss (CE only): {test_loss:.4f}")
    print(f"╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))

vocab_size = config["vocab_size"]
max_seq_len = config["max_seq_len"]
d_model = config["d_model"]
num_heads = config["num_heads"]
num_kv_heads = config['num_kv_heads']
num_layers = config["num_layers"]
ff_dim = config["ff_dim"]
dropout = config["dropout"]
sft1_epochs = config["sft1_epochs"]
sft2_epochs = config["sft2_epochs"]
batch_size = config["batch_size"]
train_ratio = config["train_ratio"]
val_ratio = config["val_ratio"]
sft1_learning_rate = config["sft1_learning_rate"]
sft1_learning_weight_decay = config["sft1_learning_weight_decay"]
sft2_learning_rate = config["sft2_learning_rate"]
accumulation_steps = config["accumulation_steps"]
sft2_learning_weight_decay = config["sft2_learning_weight_decay"]
freeze = config["freeze"]
num_workers = config["num_workers"]
penalty_engine = (PenaltyEngine()
    .add_rule(WrongTokenMarginPenalty(weight=config["penalty_margin_weight"], detach_max=config["penalty_margin_detach_max"]))
    .add_rule(WrongTokenEntropyPenalty(weight=config["penalty_entropy_weight"], min_entropy=config["penalty_entropy_min_entropy"]))
    .add_rule(FocalOverconfidencePenalty(weight=config["penalty_focal_weight"], gamma=config["penalty_focal_gamma"]))
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

amp_enabled = device.type == "cuda"
amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
scaler = GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
log_progress(f"Sử dụng device: {device}")
if amp_enabled:
    precision_name = "BF16" if amp_dtype == torch.bfloat16 else "FP16 + GradScaler"
    log_progress(f"CUDA mixed precision: {precision_name}")

model_raw = TransformerModel(vocab_size, d_model, num_heads, num_kv_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
# Giữ alias `model` để không thay khung dispatch phase hiện tại.
model = model_raw
if args.limited_max_autotune:
    compile_options = {
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "max_autotune_gemm_search_space": "DEFAULT",
        "epilogue_fusion": True,
        "triton.cudagraphs": False,
    }
    model_features = torch.compile(
        model_raw.forward_features,
        dynamic=True,
        options=compile_options,
    )
    log_progress("torch.compile: limited max-autotune (ATEN,TRITON; DEFAULT; dynamic=True; CUDA Graph off)")
else:
    model_features = torch.compile(model_raw.forward_features, dynamic=True)

phase = args.phase
if phase == "sft1":
    log_progress("Load model từ pretrain...")
    model.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=sft1_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

elif phase == "sft1_resume":
    log_progress("SFT1 resume: khởi tạo model skeleton trước khi load checkpoint...")
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=sft1_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=sft1_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

elif phase == "sft2":
    log_progress("Load model từ sft1...")
    model.load_state_dict(torch.load(sft1_save_path, map_location=device))
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=sft2_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

elif phase == "sft2_resume":
    log_progress("SFT2 resume: khởi tạo model skeleton trước khi load checkpoint...")
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=sft2_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=sft2_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

elif phase == "full":
    # ── SFT1 ──
    log_progress("Load model từ pretrain...")
    model.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    freeze_layers(model, freeze)

    optimizer_sft1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft1_learning_rate, weight_decay=sft1_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft1, device, SFT1_data_ids_file, sub_data=None,
        num_epochs=sft1_epochs,
        model_save_path=sft1_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft1",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=False,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT1 Test Loss: {test_loss:.4f}")

    # ── SFT2 ──
    log_progress("Load model từ sft1...")
    model.load_state_dict(torch.load(sft1_save_path, map_location=device))
    unfreeze_all_layers(model)
    freeze_layers(model, freeze)

    optimizer_sft2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=sft2_learning_rate, weight_decay=sft2_learning_weight_decay,
        fused=amp_enabled,
    )

    test_loss = finetune(
        model, optimizer_sft2, device, SFT2_data_ids_file, SFT1_data_ids_file,
        num_epochs=sft2_epochs,
        model_save_path=sft2_save_path,
        train_ratio=train_ratio, val_ratio=val_ratio,
        batch_size=batch_size, phase_name="sft2",
        num_workers=num_workers,
        penalty_engine=penalty_engine,
        resample_per_epoch=True,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"SFT2 Test Loss: {test_loss:.4f}")

log_progress(f"Model cuối cùng lưu tại: {model_dir / (phase.split('_')[0] + f'_{model_size}.pt')}")
