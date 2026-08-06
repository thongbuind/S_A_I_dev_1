import torch
import time
from pathlib import Path

def get_step_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.4:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.4) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

def freeze_layers(model, freeze):
    for idx in freeze:
        for param in model.blocks[idx].parameters():
            param.requires_grad = False

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

def save_checkpoint(path: Path, epoch: int, global_step: int, model, optimizer, scheduler, best_val_loss: float):
    """Save full training state so training can be resumed exactly."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    """Load full training state. Returns (start_epoch, global_step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    best_val_loss = ckpt["best_val_loss"]
    log_progress(f"Resumed from checkpoint: epoch {ckpt['epoch']+1}, step {global_step}, best_val_loss {best_val_loss:.5f}")
    return start_epoch, global_step, best_val_loss

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"

def estimate_training_flops(
    model: torch.nn.Module,
    num_layers: int,
    max_seq_len: int,
    d_model: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * n_params + 6 * num_layers * max_seq_len * d_model
    total_token_slots = epochs * batches_per_epoch * batch_size * max_seq_len
    total_flops = flops_per_token * total_token_slots
    total_tflops = total_flops / 1e12

    return {
        "n_params": n_params,
        "flops_per_token": flops_per_token,
        "total_batches_per_epoch": batches_per_epoch,
        "total_token_slots": total_token_slots,
        "total_flops": total_flops,
        "total_tflops": total_tflops,
    }

def print_training_flops_summary(flops_info: dict, epochs: int) -> None:
    """In bảng tổng FLOPs cần cho training."""
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                      TINH TONG FLOPs CAN CHO TOAN BO TRAINING                      ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"So tham so model: {flops_info['n_params']:,}")
    print(
        f"Tong so TFLOP can cho toan bo qua trinh training "
        f"({epochs} epoch, {flops_info['total_batches_per_epoch']} batch/epoch): "
        f"{flops_info['total_tflops']:,.2f} TFLOP"
    )
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

class TflopsBenchmarker:
    def __init__(
        self,
        flops_per_token: float,
        total_flops_needed: float,
        total_batches_per_epoch: int,
        epochs: int,
        device: torch.device,
        warmup_steps: int = 100,
        target_steps: int = 500,
    ):
        self.flops_per_token = flops_per_token
        self.total_flops_needed = total_flops_needed
        self.total_batches_per_epoch = total_batches_per_epoch
        self.epochs = epochs
        self.device = device

        self.warmup_steps = min(warmup_steps, total_batches_per_epoch)
        self.target_steps = min(target_steps, total_batches_per_epoch)
        self.reported = self.target_steps <= self.warmup_steps

        self.tokens = 0
        self.start_time = None

    def is_bench_step(self, epoch: int, start_epoch: int, batch_idx: int) -> bool:
        return (
            (not self.reported)
            and (epoch == start_epoch)
            and (self.warmup_steps <= batch_idx < self.target_steps)
        )

    def on_step_begin(self, batch_idx: int, batch_tokens: int) -> None:
        """Gọi khi bắt đầu step nằm trong vùng đo."""
        if batch_idx == self.warmup_steps:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        self.tokens += batch_tokens

    def on_step_end(self, batch_idx: int) -> None:
        """Gọi ở step cuối vùng đo → in kết quả."""
        if batch_idx != self.target_steps - 1:
            return

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self.start_time
        measured_steps = self.target_steps - self.warmup_steps
        flops = self.flops_per_token * self.tokens
        tflops_per_sec = (flops / elapsed) / 1e12 if elapsed > 0 else 0.0

        print()
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        print(
            f"[DO TFLOPS] Do {measured_steps} step training that "
            f"(step {self.warmup_steps + 1} → {self.target_steps}), "
            f"mat {elapsed:.2f}s (da bo qua {self.warmup_steps} step warm-up dau)"
        )

        if tflops_per_sec > 0:
            print(f"[DO TFLOPS] TFLOPS do duoc thuc te cua phan cung: {tflops_per_sec:,.2f} TFLOPS")

            avg_time_per_step = elapsed / measured_steps
            total_steps_all_epochs = self.epochs * self.total_batches_per_epoch

            est_time_actual = avg_time_per_step * total_steps_all_epochs
            print(f"[Cach 1 - Thuc te] Uoc tinh thoi gian train (ngoai suy toc do do duoc): {format_time(est_time_actual)}")

            # Cách 2 – lý thuyết
            est_time_theory = self.total_flops_needed / (tflops_per_sec * 1e12)
            print(f"[Cach 2 - Ly thuyet] Uoc tinh thoi gian train (Tong FLOPs / TFLOPS do duoc): {format_time(est_time_theory)}")
        else:
            print("[CẢNH BÁO] Thời gian đo quá nhỏ để tính TFLOPS đáng tin cậy.")

        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        self.reported = True

class KernelLogger:
    def __init__(self, enabled: bool = False, log_step: int = 500):
        self.enabled = enabled
        self.log_step = log_step
        self.reported = not enabled
        self._prof_ctx = None

    def should_log(self, epoch: int, start_epoch: int, batch_idx: int) -> bool:
        return (
            (not self.reported)
            and (epoch == start_epoch)
            and (batch_idx == self.log_step)
        )

    def start(self):
        self._prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self._prof_ctx.__enter__()

    def stop_and_report(self, batch_idx: int, epoch: int, device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize()

        self._prof_ctx.__exit__(None, None, None)

        key_avgs = self._prof_ctx.key_averages()

        def _self_cuda_us(e):
            if hasattr(e, "self_cuda_time_total"):
                return e.self_cuda_time_total
            return getattr(e, "self_device_time_total", 0)

        def _cuda_total_us(e):
            if hasattr(e, "cuda_time_total"):
                return e.cuda_time_total
            return getattr(e, "device_time_total", 0)

        total_self_cuda = sum(_self_cuda_us(e) for e in key_avgs)
        total_self_cpu = sum(e.self_cpu_time_total for e in key_avgs)

        rows = []
        for e in key_avgs:
            self_cuda = _self_cuda_us(e)
            cuda_total = _cuda_total_us(e)

            cuda_pct = (
                self_cuda / total_self_cuda * 100
                if total_self_cuda > 0 else 0.0
            )

            cpu_pct = (
                e.self_cpu_time_total / total_self_cpu * 100
                if total_self_cpu > 0 else 0.0
            )

            if cuda_pct >= 1.0:
                rows.append((
                    e.key,
                    self_cuda,
                    cuda_total,
                    cuda_pct,
                    cpu_pct,
                    e.count,
                ))

        rows.sort(key=lambda r: r[3], reverse=True)

        print()
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"[LOG KERNEL] Top kernel tại step {batch_idx} (epoch {epoch+1})")
        print(
            f"{'Name':<55}"
            f"{'Self CUDA':>12}"
            f"{'CUDA Total':>12}"
            f"{'CUDA %':>9}"
            f"{'CPU %':>9}"
            f"{'Calls':>9}"
        )

        for name, self_cuda, cuda_total, cuda_pct, cpu_pct, count in rows:
            print(
                f"{name[:55]:<55}"
                f"{self_cuda/1000:>9.2f}ms"
                f"{cuda_total/1000:>9.2f}ms"
                f"{cuda_pct:>8.2f}%"
                f"{cpu_pct:>8.2f}%"
                f"{count:>9}"
            )

        trace_file = f"torch_profile_epoch{epoch+1}_step{batch_idx}.json"
        self._prof_ctx.export_chrome_trace(trace_file)

        print("────────────────────────────────────────────────────────────────────────────────────")
        print(f"Chrome Trace: {trace_file}")
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")

        self.reported = True
        self._prof_ctx = None
