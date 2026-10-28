"""CIFAR-10 (32x32) SDiT training with self-distillation and no feature loss.

Example: .venv/bin/python Cifar10_SDiT_TrainNoFE.py --device mps
The data directory must already contain torchvision's cifar-10-batches-py.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import SDiT_Base, SDiT_Small, SDiT_Tiny
from sdit_losses import lossWithoutFeatureExtraction


class CheckpointedSDiT(nn.Module):
    """학습 중 인코더와 보조 분류기에 활성화 체크포인팅을 적용하는 래퍼."""

    def __init__(self, model, enabled):
        """체크포인팅에 사용할 모델과 활성화 여부의 저장.

        매개변수:
            model: 최종 분류기와 보조 분류기를 가진 SDiT 모델.
            enabled: True이면 학습 시 활성화 체크포인팅 사용.

        반환값:
            없음. 래퍼 객체 초기화.
        """
        super().__init__()
        self.model = model
        self.enabled = enabled

    def forward(self, images):
        """이미지의 최종 분류 결과와 보조 분류 결과 계산.

        매개변수:
            images: (배치 크기, 3, 32, 32) 형태의 이미지 텐서.

        반환값:
            최종 분류 logits를 첫 번째에 두고 보조 분류 logits를
            얕은 인코더 순서로 이어 붙인 튜플. 각 텐서는 (배치 크기, 10) 형태.
        """
        if not self.enabled or not self.training:
            return self.model(images)
        model = self.model
        x = model.patch_embed(images)
        cls = model.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if x.shape[1:] != model.pos_embed.shape[1:]:
            raise ValueError("Input dimensions do not match the model configuration")
        x = model.pos_drop(x + model.pos_embed)
        outputs = []
        for block, head in zip(model.blocks[:-1], model.aux_heads):
            # Capture each block/head as defaults for checkpoint recomputation.
            def block_and_head(tokens, block=block, head=head):
                """인코더 블록과 해당 보조 분류기의 출력 계산.

                매개변수:
                    tokens: 현재 인코더에 입력할 토큰 텐서.
                    block: 이번 체크포인팅 구간의 인코더 블록.
                    head: 해당 블록에 연결된 보조 분류기.

                반환값:
                    다음 인코더에 전달할 토큰과 보조 분류 logits의 튜플.
                """
                tokens = block(tokens)
                return tokens, head(tokens)

            x, logits = checkpoint(block_and_head, x, use_reentrant=False)
            outputs.append(logits)
        x = checkpoint(model.blocks[-1], x, use_reentrant=False)
        final_logits = model.head(model.norm(x)[:, 0])
        return (final_logits, *outputs)


def parse_args():
    """명령행 학습 옵션 파싱 및 허용 범위 검사.

    매개변수:
        없음. 실행 시 전달되는 명령행 인자.

    반환값:
        데이터 경로, 모델, 배치 크기, 장치 등 학습 설정을 담은 Namespace.
        잘못된 인자 입력 시 오류 메시지 출력 및 프로그램 종료.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints_cifar10_nofe"))
    parser.add_argument("--model", choices=("base", "small", "tiny"), default="base")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16, help="Physical batch size")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Microbatches per optimizer update")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10, help="Epoch interval for numbered checkpoints; 0 disables them")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a full training checkpoint")
    args = parser.parse_args()
    if args.patch_size <= 0 or 32 % args.patch_size:
        parser.error("--patch-size must be a positive divisor of 32")
    if min(args.batch_size, args.accumulation_steps, args.eval_batch_size, args.epochs, args.log_interval) <= 0:
        parser.error("Batch sizes, accumulation steps, epochs, and log interval must be positive")
    if not 0 < args.val_size < 50000 or args.workers < 0:
        parser.error("--val-size must be between 1 and 49999 and --workers nonnegative")
    if args.save_every < 0:
        parser.error("--save-every must be nonnegative")
    if not 0 <= args.alpha <= 1 or args.temperature <= 0:
        parser.error("--alpha must be in [0, 1] and --temperature must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("Learning rate must be positive, weight decay nonnegative, and warmup less than epochs")
    return args


def get_device(name):
    """요청한 연산 장치의 선택 및 사용 가능 여부 확인.

    매개변수:
        name: 'auto', 'mps', 'cuda', 'cpu' 중 하나. auto는 MPS,
            CUDA, CPU 순서로 사용 가능한 장치 선택.

    반환값:
        선택한 장치를 나타내는 torch.device. 요청한 GPU를 사용할 수 없는 경우
        RuntimeError 발생.
    """
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process. Check Metal access and permissions.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process.")
    return torch.device(name)


def make_loaders(args, device):
    """기존 CIFAR-10 데이터의 학습·검증·테스트 로더 생성.

    매개변수:
        args: 데이터 경로, 분할 크기, 배치 크기, 시드와 작업자 수를 담은 설정.
        device: 학습 장치. CUDA일 때 데이터 로더의 고정 메모리 사용.

    반환값:
        (학습 로더, 검증 로더, 테스트 로더) 튜플. 학습 데이터 일부를
        검증에 사용하고 테스트 데이터는 별도로 유지하는 구성.
    """
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    try:
        train_data = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=train_transform)
        val_data = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=eval_transform)
        test_data = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=eval_transform)
    except RuntimeError as exc:
        raise RuntimeError(f"CIFAR-10 was not found or failed integrity checks in {args.data_dir}. "
                           "Supply --data-dir containing cifar-10-batches-py.") from exc
    permutation = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_indices = permutation[:args.val_size]
    train_indices = permutation[args.val_size:]
    common = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size,
                              shuffle=True, **common)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=args.eval_batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def evaluate(model, loader, device, amp_enabled):
    """최종 분류기의 교차 엔트로피 손실 및 정확도 평가.

    매개변수:
        model: 평가할 SDiT 래퍼 모델.
        loader: 검증 또는 테스트 데이터를 제공하는 데이터 로더.
        device: 이미지와 정답을 올릴 연산 장치.
        amp_enabled: True이면 평가 시 자동 혼합 정밀도 사용.

    반환값:
        (샘플당 평균 교차 엔트로피 손실, 정확도) 튜플. 정확도 범위 0~1.
    """
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)[0]
            loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count


def main():
    """설정을 읽어 모델 학습, 최적 모델 저장 및 평가 수행.

    매개변수:
        없음. parse_args()를 통한 명령행 설정 읽기.

    반환값:
        없음. 학습 이력, 최적 모델, 테스트 결과의 출력 디렉토리 저장.
    """
    args = parse_args()
    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("format_version") != 1:
            raise ValueError("--resume requires a full training checkpoint, not best.pt")
        # Keep the original schedule and data split; runtime device and worker options remain selectable.
        saved_config = resume["config"]
        for key in ("model", "patch_size", "batch_size", "accumulation_steps", "eval_batch_size",
                    "epochs", "lr", "weight_decay", "warmup_epochs", "alpha", "temperature",
                    "val_size", "seed", "amp", "checkpointing"):
            setattr(args, key, saved_config[key])
        args.data_dir = Path(saved_config["data_dir"])
        args.output_dir = args.resume.resolve().parent
        print(f"Resuming with saved training configuration from {args.resume}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    amp_enabled = args.amp and device.type in ("mps", "cuda")
    if args.amp and not amp_enabled:
        print("AMP disabled on CPU; using float32.")
    train_loader, val_loader, test_loader = make_loaders(args, device)
    factory = {"base": SDiT_Base, "small": SDiT_Small, "tiny": SDiT_Tiny}[args.model]
    model = CheckpointedSDiT(factory(img_size=32, patch_size=args.patch_size, n_classes=10,
                                     p=0.1), args.checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = args.warmup_epochs * updates_per_epoch

    def lr_factor(step):
        """현재 optimizer 업데이트에 적용할 학습률 배율 계산.

        매개변수:
            step: 지금까지 진행한 스케줄러 업데이트 단계.

        반환값:
            초기 학습률에 곱할 실수 배율. 워밍업 후 코사인 방식으로 감소하는 구조.
        """
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print(f"Device: {device}; AMP: {amp_enabled}; checkpointing: {args.checkpointing}; "
          f"physical batch: {args.batch_size}; effective batch: up to {args.batch_size * args.accumulation_steps}")
    print(f"Samples: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    start_epoch = 0
    if resume is not None:
        if resume["device_type"] != device.type:
            raise ValueError("Resume requires the same device type as the saved run, for AMP and RNG state")
        model.model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = resume["epoch"]
        best_accuracy, best_epoch = resume["best_accuracy"], resume["best_epoch"]
        history = resume["history"]
        torch.save({"model": resume["best_model"], "epoch": best_epoch,
                    "val_accuracy": best_accuracy, "config": saved_config}, args.output_dir / "best.pt")
        random.setstate(resume["python_rng"])
        numpy_rng = resume["numpy_rng"]
        np.random.set_state((numpy_rng["name"], np.asarray(numpy_rng["keys"], dtype=np.uint32),
                             numpy_rng["pos"], numpy_rng["has_gauss"], numpy_rng["cached_gaussian"]))
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(resume["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["device_rng"])
        del resume
        print(f"Resuming at epoch {start_epoch + 1}/{args.epochs}; best validation accuracy: {best_accuracy:.4f}")
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = correct = count = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Weight each microbatch by its sample count, including the final partial group.
            group_start = (batch_index // args.accumulation_steps) * args.accumulation_steps * args.batch_size
            group_samples = min(args.batch_size * args.accumulation_steps, len(train_loader.dataset) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
            loss = lossWithoutFeatureExtraction(tuple(x.float() for x in outputs), labels,
                                                alpha=args.alpha, temperature=args.temperature)
            scaler.scale(loss * (labels.size(0) / group_samples)).backward()
            if (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach().item() * labels.size(0)
            correct += (outputs[0].detach().argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
            if (batch_index + 1) % args.log_interval == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{len(train_loader)} "
                      f"loss={loss_sum / count:.4f} accuracy={correct / count:.4f}", flush=True)
        val_loss, val_accuracy = evaluate(model, val_loader, device, amp_enabled)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / count, "train_accuracy": correct / count,
               "val_loss": val_loss, "val_accuracy": val_accuracy, "seconds": time.time() - start}
        history.append(row)
        print(f"Epoch {epoch + 1}: train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}; "
              f"val CE={val_loss:.4f} acc={val_accuracy:.4f}; {row['seconds']:.1f}s", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save({"model": model.model.state_dict(), "epoch": epoch + 1,
                        "val_accuracy": val_accuracy, "config": config}, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        numpy_rng = np.random.get_state()
        best_model = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)["model"]
        checkpoint_data = {
            "format_version": 1, "epoch": epoch + 1, "model": model.model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_accuracy": best_accuracy, "best_epoch": best_epoch,
            "best_model": best_model, "history": history, "config": config, "device_type": device.type,
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": {"name": numpy_rng[0], "keys": numpy_rng[1].tolist(), "pos": numpy_rng[2],
                          "has_gauss": numpy_rng[3], "cached_gaussian": numpy_rng[4]},
            "device_rng": torch.mps.get_rng_state() if device.type == "mps" else
                          torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        # Write to a temporary file first so an interrupted save does not destroy the previous checkpoint.
        temporary_path = args.output_dir / "latest.pt.tmp"
        try:
            torch.save(checkpoint_data, temporary_path)
            temporary_path.replace(args.output_dir / "latest.pt")
        finally:
            temporary_path.unlink(missing_ok=True)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint_data, args.output_dir / f"epoch_{epoch + 1:04d}.pt")
        del checkpoint_data, best_model
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.model.load_state_dict(best["model"])
    test_loss, test_accuracy = evaluate(model, test_loader, device, amp_enabled)
    result = {"best_epoch": best["epoch"], "val_accuracy": best_accuracy,
              "test_loss": test_loss, "test_accuracy": test_accuracy, "config": config}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Best epoch: {best['epoch']}; test CE={test_loss:.4f}, test accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
