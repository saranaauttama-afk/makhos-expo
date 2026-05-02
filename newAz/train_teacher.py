"""
Supervised teacher training for Makhos AZ from minimax-labeled dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from network_az import AZNetwork, DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Train AZ network with teacher-supervised dataset.")
    parser.add_argument("--dataset", required=True, help="Path to teacher dataset .npz")
    parser.add_argument("--baseline-checkpoint", required=True, help="Path to baseline checkpoint (e.g. iter_0079.pt)")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained checkpoint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0, help="Weight on value MSE")
    parser.add_argument("--anchor-weight", type=float, default=0.07, help="Weight on anchor regularization")
    parser.add_argument(
        "--anchor-value-weight",
        type=float,
        default=0.35,
        help="Multiplier for value term inside anchor loss",
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from saved training state if available.")
    parser.add_argument("--state-path", default="", help="Optional explicit path for resume state file.")
    parser.add_argument("--keep-state-on-success", action="store_true", help="Do not delete state file on success.")
    parser.add_argument(
        "--notify-email-to",
        default=os.getenv("NOTIFY_EMAIL_TO", ""),
        help="Receiver email for completion notification (optional).",
    )
    parser.add_argument(
        "--notify-email-from",
        default=os.getenv("NOTIFY_EMAIL_FROM", ""),
        help="Sender email for SMTP login (optional).",
    )
    parser.add_argument(
        "--notify-email-password",
        default=os.getenv("NOTIFY_EMAIL_PASSWORD", ""),
        help="SMTP app password/token for sender email (optional).",
    )
    parser.add_argument(
        "--notify-smtp-host",
        default=os.getenv("NOTIFY_SMTP_HOST", "smtp.gmail.com"),
        help="SMTP host for notification email.",
    )
    parser.add_argument(
        "--notify-smtp-port",
        type=int,
        default=int(os.getenv("NOTIFY_SMTP_PORT", "465")),
        help="SMTP SSL port for notification email.",
    )
    return parser.parse_args()


@dataclass
class EpochMetrics:
    policy_ce: float
    value_loss: float
    anchor_loss: float
    total_loss: float
    policy_acc: float


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher_dataset(path: str):
    data = np.load(path)
    required = {"features", "value_target"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"dataset missing keys: {sorted(missing)}")

    x = data["features"].astype(np.float32)
    if "policy_target" in data.files:
        y_policy_target = data["policy_target"].astype(np.float32)
        if y_policy_target.ndim != 2 or y_policy_target.shape[1] != 1024:
            raise ValueError(f"invalid policy_target shape {y_policy_target.shape}, expected (N, 1024)")
        if "policy_index" in data.files:
            y_policy_idx = data["policy_index"].astype(np.int64)
        else:
            y_policy_idx = np.argmax(y_policy_target, axis=1).astype(np.int64)
    elif "policy_index" in data.files:
        y_policy_idx = data["policy_index"].astype(np.int64)
        y_policy_target = np.zeros((y_policy_idx.shape[0], 1024), dtype=np.float32)
        y_policy_target[np.arange(y_policy_idx.shape[0]), y_policy_idx] = 1.0
    else:
        raise KeyError("dataset missing either policy_target or policy_index")
    y_value = data["value_target"].astype(np.float32)
    if x.ndim != 2 or x.shape[1] != 128:
        raise ValueError(f"invalid features shape {x.shape}, expected (N, 128)")
    if y_policy_idx.shape[0] != x.shape[0] or y_policy_target.shape[0] != x.shape[0] or y_value.shape[0] != x.shape[0]:
        raise ValueError("dataset arrays must have same length")
    row_sums = y_policy_target.sum(axis=1, keepdims=True)
    y_policy_target = y_policy_target / np.maximum(row_sums, 1e-9)
    return x, y_policy_idx, y_policy_target, y_value


def load_meta_for_dataset(path: str) -> dict:
    meta_path = os.path.splitext(os.path.abspath(path))[0] + ".json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_network_from_checkpoint(path: str) -> tuple[AZNetwork, dict]:
    checkpoint = torch.load(path, map_location=DEVICE)
    net = AZNetwork(hidden=checkpoint.get("hidden", 256), n_res=checkpoint.get("n_res", 4))
    net.net.load_state_dict(checkpoint["state_dict"])
    return net, checkpoint


def build_dataloaders(
    features: np.ndarray,
    policy_idx: np.ndarray,
    policy_target: np.ndarray,
    value_target: np.ndarray,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
):
    n = features.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(n * np.clip(val_split, 0.0, 0.5))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("no training samples after split")

    x_train = torch.from_numpy(features[train_idx])
    p_train = torch.from_numpy(policy_idx[train_idx])
    pi_train = torch.from_numpy(policy_target[train_idx])
    v_train = torch.from_numpy(value_target[train_idx])
    train_ds = TensorDataset(x_train, p_train, pi_train, v_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train_idx.size > batch_size,
    )

    val_loader = None
    if n_val > 0:
        x_val = torch.from_numpy(features[val_idx])
        p_val = torch.from_numpy(policy_idx[val_idx])
        pi_val = torch.from_numpy(policy_target[val_idx])
        v_val = torch.from_numpy(value_target[val_idx])
        val_ds = TensorDataset(x_val, p_val, pi_val, v_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, train_idx.size, n_val


def run_epoch(
    *,
    student: AZNetwork,
    teacher: AZNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    value_weight: float,
    anchor_weight: float,
    anchor_value_weight: float,
    grad_clip: float,
) -> EpochMetrics:
    training = optimizer is not None
    if training:
        student.net.train()
    else:
        student.net.eval()

    teacher.net.eval()

    total_n = 0
    sum_policy = 0.0
    sum_value = 0.0
    sum_anchor = 0.0
    sum_total = 0.0
    sum_correct = 0.0

    for x, target_policy_idx, target_policy, target_value in loader:
        x = x.to(DEVICE, non_blocking=True)
        target_policy_idx = target_policy_idx.to(DEVICE, non_blocking=True)
        target_policy = target_policy.to(DEVICE, non_blocking=True)
        target_value = target_value.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits, pred_value = student.net(x)
            log_probs = F.log_softmax(logits, dim=-1)
            policy_ce = -(target_policy * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(pred_value, target_value)

            anchor_loss = torch.zeros((), device=DEVICE)
            if anchor_weight > 0:
                with torch.no_grad():
                    teacher_logits, teacher_value = teacher.net(x)
                    teacher_probs = F.softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(logits, dim=-1)
                policy_anchor = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                value_anchor = F.mse_loss(pred_value, teacher_value)
                anchor_loss = policy_anchor + (anchor_value_weight * value_anchor)

            total_loss = policy_ce + (value_weight * value_loss) + (anchor_weight * anchor_loss)

            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.net.parameters(), grad_clip)
                optimizer.step()

        batch_n = int(x.shape[0])
        total_n += batch_n
        sum_policy += float(policy_ce.detach()) * batch_n
        sum_value += float(value_loss.detach()) * batch_n
        sum_anchor += float(anchor_loss.detach()) * batch_n
        sum_total += float(total_loss.detach()) * batch_n
        preds = torch.argmax(logits.detach(), dim=-1)
        sum_correct += float((preds == target_policy_idx).sum().item())

    if total_n == 0:
        return EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    return EpochMetrics(
        policy_ce=sum_policy / total_n,
        value_loss=sum_value / total_n,
        anchor_loss=sum_anchor / total_n,
        total_loss=sum_total / total_n,
        policy_acc=sum_correct / total_n,
    )


def infer_iter_tag(path: str) -> str:
    match = re.search(r"iter_(\d+)", os.path.basename(path))
    if not match:
        return "unknown"
    return f"{int(match.group(1)):04d}"


def infer_depth_from_meta(meta: dict) -> str:
    depth = meta.get("minimax_depth")
    if isinstance(depth, int):
        return str(depth)
    return "x"


def save_checkpoint(
    *,
    student: AZNetwork,
    output_path: str,
    baseline_path: str,
    dataset_path: str,
    args,
    dataset_meta: dict,
    history: list[dict],
):
    payload = {
        "state_dict": student.net.state_dict(),
        "hidden": student.hidden,
        "n_res": student.n_res,
        "trained_from_baseline": os.path.abspath(baseline_path),
        "teacher_dataset": os.path.abspath(dataset_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "teacher_train_args": vars(args),
        "dataset_meta": dataset_meta,
        "history": history,
    }
    torch.save(payload, output_path)


def maybe_send_notification(args, subject: str, body: str):
    if not args.notify_email_to or not args.notify_email_from or not args.notify_email_password:
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = args.notify_email_from
        msg["To"] = args.notify_email_to
        msg.set_content(body)
        with smtplib.SMTP_SSL(args.notify_smtp_host, args.notify_smtp_port) as smtp:
            smtp.login(args.notify_email_from, args.notify_email_password)
            smtp.send_message(msg)
        print(f"[notify] email sent to {args.notify_email_to}")
    except Exception as exc:
        print(f"[notify] email failed: {exc}")


def save_resume_state(
    *,
    state_path: str,
    epoch: int,
    student: AZNetwork,
    optimizer: torch.optim.Optimizer,
    history: list[dict],
    best_val: float,
    best_state: dict,
):
    payload = {
        "epoch": int(epoch),
        "model_state": student.net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_val": float(best_val),
        "best_state": best_state,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    torch.save(payload, state_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    features, policy_idx, policy_target, value_target = load_teacher_dataset(args.dataset)
    dataset_meta = load_meta_for_dataset(args.dataset)
    student_net, baseline_ckpt = load_network_from_checkpoint(args.baseline_checkpoint)
    teacher_net, _ = load_network_from_checkpoint(args.baseline_checkpoint)
    for param in teacher_net.net.parameters():
        param.requires_grad_(False)

    train_loader, val_loader, n_train, n_val = build_dataloaders(
        features=features,
        policy_idx=policy_idx,
        policy_target=policy_target,
        value_target=value_target,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(student_net.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    iter_tag = infer_iter_tag(args.baseline_checkpoint)
    depth_tag = infer_depth_from_meta(dataset_meta)
    date_tag = datetime.utcnow().strftime("%Y%m%d")
    out_name = f"teacher_iter{iter_tag}_d{depth_tag}_{date_tag}.pt"
    out_path = os.path.join(args.output_dir, out_name)
    log_path = os.path.splitext(out_path)[0] + ".trainlog.json"
    state_path = args.state_path.strip() or (os.path.splitext(out_path)[0] + ".state.pt")

    print(f"[train] device={DEVICE}")
    print(f"[train] dataset={os.path.abspath(args.dataset)}")
    print(f"[train] samples={features.shape[0]} train={n_train} val={n_val}")
    print(f"[train] baseline={os.path.abspath(args.baseline_checkpoint)}")
    print(f"[train] output={os.path.abspath(out_path)}")

    history = []
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in student_net.net.state_dict().items()}
    start_epoch = 1

    if args.resume and os.path.exists(state_path):
        resume_payload = torch.load(state_path, map_location=DEVICE)
        student_net.net.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        best_val = float(resume_payload.get("best_val", best_val))
        best_state = resume_payload.get("best_state", best_state)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        print(f"[resume] loaded state from {os.path.abspath(state_path)} (next epoch={start_epoch})")

    if start_epoch > args.epochs:
        print(f"[resume] training already reached epoch {args.epochs}; skipping train loop")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                student=student_net,
                teacher=teacher_net,
                loader=train_loader,
                optimizer=optimizer,
                value_weight=args.value_weight,
                anchor_weight=args.anchor_weight,
                anchor_value_weight=args.anchor_value_weight,
                grad_clip=args.grad_clip,
            )
            val_metrics = None
            if val_loader is not None:
                val_metrics = run_epoch(
                    student=student_net,
                    teacher=teacher_net,
                    loader=val_loader,
                    optimizer=None,
                    value_weight=args.value_weight,
                    anchor_weight=args.anchor_weight,
                    anchor_value_weight=args.anchor_value_weight,
                    grad_clip=args.grad_clip,
                )

            row = {
                "epoch": epoch,
                "train_policy_ce": train_metrics.policy_ce,
                "train_value_loss": train_metrics.value_loss,
                "train_anchor_loss": train_metrics.anchor_loss,
                "train_total_loss": train_metrics.total_loss,
                "train_policy_acc": train_metrics.policy_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if val_metrics is not None:
                row.update(
                    {
                        "val_policy_ce": val_metrics.policy_ce,
                        "val_value_loss": val_metrics.value_loss,
                        "val_anchor_loss": val_metrics.anchor_loss,
                        "val_total_loss": val_metrics.total_loss,
                        "val_policy_acc": val_metrics.policy_acc,
                    }
                )
                score_for_best = val_metrics.total_loss
            else:
                score_for_best = train_metrics.total_loss
            history.append(row)

            if score_for_best < best_val:
                best_val = score_for_best
                best_state = {k: v.detach().cpu().clone() for k, v in student_net.net.state_dict().items()}

            val_text = ""
            if val_metrics is not None:
                val_text = (
                    f" | val policy_ce={val_metrics.policy_ce:.4f}"
                    f" value={val_metrics.value_loss:.4f}"
                    f" anchor={val_metrics.anchor_loss:.4f}"
                    f" total={val_metrics.total_loss:.4f}"
                    f" acc={val_metrics.policy_acc:.3f}"
                )
            print(
                f"epoch {epoch:02d}/{args.epochs}"
                f" | train policy_ce={train_metrics.policy_ce:.4f}"
                f" value={train_metrics.value_loss:.4f}"
                f" anchor={train_metrics.anchor_loss:.4f}"
                f" total={train_metrics.total_loss:.4f}"
                f" acc={train_metrics.policy_acc:.3f}"
                f"{val_text}"
            )

            if args.save_every_epoch:
                epoch_out = os.path.splitext(out_path)[0] + f".epoch{epoch:02d}.pt"
                save_checkpoint(
                    student=student_net,
                    output_path=epoch_out,
                    baseline_path=args.baseline_checkpoint,
                    dataset_path=args.dataset,
                    args=args,
                    dataset_meta=dataset_meta,
                    history=history,
                )
            save_resume_state(
                state_path=state_path,
                epoch=epoch,
                student=student_net,
                optimizer=optimizer,
                history=history,
                best_val=best_val,
                best_state=best_state,
            )
    except BaseException as exc:
        save_resume_state(
            state_path=state_path,
            epoch=max(start_epoch - 1, 0),
            student=student_net,
            optimizer=optimizer,
            history=history,
            best_val=best_val,
            best_state=best_state,
        )
        maybe_send_notification(
            args,
            subject=f"[Makhos] Teacher train failed ({os.path.basename(out_path)})",
            body=(
                f"Teacher training failed.\n\n"
                f"state_path: {os.path.abspath(state_path)}\n"
                f"dataset: {os.path.abspath(args.dataset)}\n"
                f"baseline: {os.path.abspath(args.baseline_checkpoint)}\n"
                f"error: {exc}\n"
            ),
        )
        raise

    # Restore best model and save final checkpoint.
    student_net.net.load_state_dict(best_state)
    save_checkpoint(
        student=student_net,
        output_path=out_path,
        baseline_path=args.baseline_checkpoint,
        dataset_path=args.dataset,
        args=args,
        dataset_meta=dataset_meta,
        history=history,
    )

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_checkpoint": os.path.abspath(args.baseline_checkpoint),
                "dataset": os.path.abspath(args.dataset),
                "output_checkpoint": os.path.abspath(out_path),
                "train_args": vars(args),
                "dataset_meta": dataset_meta,
                "history": history,
                "best_val_total_loss": best_val,
                "baseline_hidden": baseline_ckpt.get("hidden", student_net.hidden),
                "baseline_n_res": baseline_ckpt.get("n_res", student_net.n_res),
            },
            f,
            indent=2,
        )

    print(f"[done] checkpoint saved: {os.path.abspath(out_path)}")
    print(f"[done] train log saved  : {os.path.abspath(log_path)}")
    if (not args.keep_state_on_success) and os.path.exists(state_path):
        os.remove(state_path)
        print(f"[done] removed resume state: {os.path.abspath(state_path)}")
    maybe_send_notification(
        args,
        subject=f"[Makhos] Teacher train done ({os.path.basename(out_path)})",
        body=(
            f"Teacher training completed.\n\n"
            f"checkpoint: {os.path.abspath(out_path)}\n"
            f"train_log: {os.path.abspath(log_path)}\n"
            f"dataset: {os.path.abspath(args.dataset)}\n"
            f"baseline: {os.path.abspath(args.baseline_checkpoint)}\n"
            f"epochs: {args.epochs}\n"
            f"best_val_total_loss: {best_val:.6f}\n"
        ),
    )


if __name__ == "__main__":
    main()
