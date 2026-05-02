"""
Build teacher-supervised dataset for Makhos AZ.

Dataset targets:
- policy target: minimax root candidates as a soft 1024-move distribution
  (plus policy_index for compatibility/accuracy)
- value target: normalized minimax score in [-1, 1]

Position sources:
- random playout
- minimax-guided playout
- forced-capture positions
- model-vs-minimax disagreement positions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Callable

import numpy as np
import torch

from makhos_engine import (
    Move,
    Position,
    apply_move,
    generate_moves,
    get_features,
    hand_eval,
    initial_position,
    is_draw_by_inactivity,
)
from mcts_az import game_result, move_to_index
from network_az import AZNetwork, DEVICE

_INF = 10_000
MAX_GAME_LEN = 250

SOURCE_RANDOM = 0
SOURCE_MINIMAX_PLAYOUT = 1
SOURCE_FORCED_CAPTURE = 2
SOURCE_DISAGREEMENT = 3

SOURCE_NAMES = {
    SOURCE_RANDOM: "random_playout",
    SOURCE_MINIMAX_PLAYOUT: "minimax_playout",
    SOURCE_FORCED_CAPTURE: "forced_capture",
    SOURCE_DISAGREEMENT: "model_minimax_disagreement",
}


@dataclass
class Sample:
    features: np.ndarray
    policy_index: int
    policy_target: np.ndarray
    value_target: float
    side_to_move: int
    source_id: int
    move_from: int
    move_to: int
    capture_count: int


def parse_args():
    parser = argparse.ArgumentParser(description="Build minimax teacher dataset for Makhos AZ.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--samples", type=int, default=12000, help="Target number of labeled positions.")
    parser.add_argument("--minimax-depth", type=int, default=7, help="Teacher minimax depth for labels.")
    parser.add_argument(
        "--policy-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Use root score distribution ('soft') or legacy one-hot best move ('hard').",
    )
    parser.add_argument("--policy-top-k", type=int, default=4, help="Top root candidates kept in soft policy.")
    parser.add_argument("--policy-margin", type=float, default=80.0, help="Keep candidates within this score gap.")
    parser.add_argument("--policy-temp", type=float, default=35.0, help="Soft policy temperature in score units.")
    parser.add_argument(
        "--label-timeout-s",
        type=float,
        default=2.5,
        help="Per-position minimax label timeout before falling back to a lower depth.",
    )
    parser.add_argument(
        "--fallback-minimax-depth",
        type=int,
        default=5,
        help="Fallback label depth when --minimax-depth times out.",
    )
    parser.add_argument(
        "--playout-minimax-depth",
        type=int,
        default=5,
        help="Minimax depth used while generating minimax-guided playout positions.",
    )
    parser.add_argument("--random-seed", type=int, default=20260425, help="RNG seed.")
    parser.add_argument("--max-playout-plies", type=int, default=40, help="Max plies for source playout samplers.")
    parser.add_argument("--value-scale", type=float, default=400.0, help="Scale for tanh(score/scale) value labels.")
    parser.add_argument(
        "--source-mix",
        default="",
        help="Optional explicit ratios random,minimax_playout,forced_capture,disagreement. Overrides --profile.",
    )
    parser.add_argument(
        "--profile",
        choices=["balanced", "hard"],
        default="balanced",
        help="Dataset source profile. 'hard' emphasizes tactical/disagreement positions.",
    )
    parser.add_argument(
        "--model-checkpoint",
        help="Optional checkpoint used for disagreement source. If omitted, disagreement source is skipped.",
    )
    parser.add_argument(
        "--disagreement-search-attempts",
        type=int,
        default=20,
        help="Inner retries to find disagreement position per request.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint if present.")
    parser.add_argument("--save-every", type=int, default=200, help="Save partial checkpoint every N samples.")
    parser.add_argument(
        "--partial-prefix",
        default="",
        help="Optional partial checkpoint prefix path (without .npz/.json).",
    )
    parser.add_argument(
        "--keep-partial-on-success",
        action="store_true",
        help="Keep .partial files after successful completion.",
    )
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


def load_network(path: str) -> AZNetwork:
    checkpoint = torch.load(path, map_location=DEVICE)
    net = AZNetwork(hidden=checkpoint.get("hidden", 256), n_res=checkpoint.get("n_res", 4))
    net.net.load_state_dict(checkpoint["state_dict"])
    net.net.eval()
    return net


class SearchTimeout(RuntimeError):
    pass


def has_forced_capture(pos: Position) -> bool:
    moves = generate_moves(pos)
    return bool(moves and len(moves[0].captured) > 0)


def tactical_child_depth(depth: int, move: Move, child: Position, ply: int, single: bool = False) -> int:
    d = depth - 1
    # Keep Python teacher finite: quiet endgames can shuffle for many plies, so
    # extension is reserved for actual tactical/capture branches.
    is_capture = len(move.captured) > 0
    if single and is_capture:
        d = min(depth, d + 1)
    total = (
        child.p1_men | child.p1_kings | child.p2_men | child.p2_kings
    )
    if total.bit_count() <= 5 and is_capture:
        d = min(depth, d + 1)
    if ply <= 18 and depth >= 2:
        op_forced = not is_capture and has_forced_capture(child)
        ext = 0
        if is_capture or op_forced:
            ext += 1
        if len(move.captured) >= 2 and depth <= 6:
            ext += 1
        if ext:
            cap = depth + 1 if len(move.captured) >= 2 else depth
            d = min(cap, d + ext)
    return max(0, d)


def order_moves_for_search(moves: list[Move]) -> list[Move]:
    return sorted(
        moves,
        key=lambda m: (len(m.captured), int(m.promote), -m.from_sq, -m.to_sq),
        reverse=True,
    )


def _minimax(
    pos: Position,
    depth: int,
    alpha: int,
    beta: int,
    ply: int = 0,
    deadline: float | None = None,
) -> int:
    if deadline is not None and time.perf_counter() > deadline:
        raise SearchTimeout()
    if is_draw_by_inactivity(pos):
        return 0
    moves = generate_moves(pos)
    if not moves:
        return -_INF
    if depth <= 0:
        return hand_eval(pos)

    best = -_INF
    ordered = order_moves_for_search(moves)
    single = len(ordered) == 1
    for move in ordered:
        child = apply_move(pos, move)
        child_depth = tactical_child_depth(depth, move, child, ply, single)
        score = -_minimax(child, child_depth, -beta, -alpha, ply + 1, deadline)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def minimax_best_move_and_score(pos: Position, depth: int) -> tuple[Move | None, int]:
    candidates = minimax_root_candidates(pos, depth)
    if not candidates:
        return None, -_INF
    return candidates[0][0], candidates[0][1]


def minimax_root_candidates(pos: Position, depth: int) -> list[tuple[Move, int]]:
    return minimax_root_candidates_bounded(pos, depth, None)


def minimax_root_candidates_bounded(
    pos: Position,
    depth: int,
    deadline: float | None,
) -> list[tuple[Move, int]]:
    moves = generate_moves(pos)
    if not moves:
        return []
    best_move = None
    best_score = -_INF
    alpha = -_INF
    beta = _INF
    scored: list[tuple[Move, int]] = []
    ordered = order_moves_for_search(moves)
    single = len(ordered) == 1
    for move in ordered:
        child = apply_move(pos, move)
        child_depth = tactical_child_depth(depth, move, child, 0, single)
        score = -_minimax(child, child_depth, -beta, -alpha, 1, deadline)
        scored.append((move, score))
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def minimax_root_candidates_with_fallback(
    pos: Position,
    depth: int,
    timeout_s: float,
    fallback_depth: int,
) -> tuple[list[tuple[Move, int]], int, bool]:
    depths = [depth]
    fallback_depth = max(1, min(fallback_depth, depth - 1))
    if fallback_depth not in depths:
        depths.append(fallback_depth)
    if 3 not in depths and depth > 3:
        depths.append(3)

    for d in depths:
        deadline = None if timeout_s <= 0 else time.perf_counter() + timeout_s
        try:
            return minimax_root_candidates_bounded(pos, d, deadline), d, d != depth
        except SearchTimeout:
            continue
    return [], 0, True


def policy_target_from_candidates(
    *,
    pos: Position,
    candidates: list[tuple[Move, int]],
    mode: str,
    top_k: int,
    margin: float,
    temp: float,
) -> np.ndarray:
    policy = np.zeros((1024,), dtype=np.float32)
    if not candidates:
        return policy
    best_score = candidates[0][1]
    kept = [
        (move, score)
        for move, score in candidates[: max(1, top_k)]
        if score >= best_score - margin
    ]
    if mode == "hard" or len(kept) <= 1:
        policy[move_to_index(candidates[0][0], pos.side)] = 1.0
        return policy

    scaled = np.array(
        [(score - best_score) / max(temp, 1e-6) for _, score in kept],
        dtype=np.float32,
    )
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    weights /= max(float(np.sum(weights)), 1e-9)
    for (move, _), weight in zip(kept, weights):
        policy[move_to_index(move, pos.side)] = float(weight)
    return policy


def score_to_value(score: int, scale: float) -> float:
    return float(np.clip(math.tanh(score / max(scale, 1e-6)), -1.0, 1.0))


def parse_source_mix(mix_str: str, include_disagreement: bool) -> dict[int, float]:
    parts = [float(x.strip()) for x in mix_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--source-mix must have 4 comma-separated values")
    ids = [SOURCE_RANDOM, SOURCE_MINIMAX_PLAYOUT, SOURCE_FORCED_CAPTURE, SOURCE_DISAGREEMENT]
    raw = dict(zip(ids, parts))
    if not include_disagreement:
        raw[SOURCE_DISAGREEMENT] = 0.0
    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        raise ValueError("invalid source mix: sum must be positive")
    return {k: max(v, 0.0) / total for k, v in raw.items()}


def profile_source_mix(profile: str) -> dict[int, float]:
    if profile == "hard":
        # Heavier tactical pressure and disagreement mining.
        return {
            SOURCE_RANDOM: 0.05,
            SOURCE_MINIMAX_PLAYOUT: 0.20,
            SOURCE_FORCED_CAPTURE: 0.40,
            SOURCE_DISAGREEMENT: 0.35,
        }
    # Default = previous behavior.
    return {
        SOURCE_RANDOM: 0.35,
        SOURCE_MINIMAX_PLAYOUT: 0.25,
        SOURCE_FORCED_CAPTURE: 0.20,
        SOURCE_DISAGREEMENT: 0.20,
    }


def make_source_targets(n_samples: int, mix: dict[int, float]) -> dict[int, int]:
    ordered = [SOURCE_RANDOM, SOURCE_MINIMAX_PLAYOUT, SOURCE_FORCED_CAPTURE, SOURCE_DISAGREEMENT]
    raw = {sid: n_samples * mix.get(sid, 0.0) for sid in ordered}
    floored = {sid: int(math.floor(raw[sid])) for sid in ordered}
    remainder = n_samples - sum(floored.values())
    frac = sorted(((raw[sid] - floored[sid], sid) for sid in ordered), reverse=True)
    for i in range(remainder):
        floored[frac[i % len(frac)][1]] += 1
    return floored


def choose_weighted_source(rng: random.Random, remaining: dict[int, int]) -> int:
    choices = [(sid, cnt) for sid, cnt in remaining.items() if cnt > 0]
    total = sum(cnt for _, cnt in choices)
    roll = rng.uniform(0.0, float(total))
    acc = 0.0
    for sid, cnt in choices:
        acc += cnt
        if roll <= acc:
            return sid
    return choices[-1][0]


def sample_random_playout_position(rng: random.Random, max_plies: int) -> Position | None:
    pos = initial_position()
    n_plies = rng.randint(2, max(2, max_plies))
    for _ in range(min(n_plies, MAX_GAME_LEN)):
        moves = generate_moves(pos)
        if not moves or game_result(pos, moves) is not None:
            break
        pos = apply_move(pos, rng.choice(moves))
    if game_result(pos, generate_moves(pos)) is not None:
        return None
    return pos


def sample_minimax_playout_position(
    rng: random.Random,
    max_plies: int,
    playout_depth: int,
    random_move_prob: float = 0.20,
) -> Position | None:
    pos = initial_position()
    n_plies = rng.randint(2, max(2, max_plies))
    for _ in range(min(n_plies, MAX_GAME_LEN)):
        moves = generate_moves(pos)
        if not moves or game_result(pos, moves) is not None:
            break
        if rng.random() < random_move_prob:
            pos = apply_move(pos, rng.choice(moves))
            continue
        best_move, _ = minimax_best_move_and_score(pos, playout_depth)
        pos = apply_move(pos, best_move or rng.choice(moves))
    if game_result(pos, generate_moves(pos)) is not None:
        return None
    return pos


def sample_forced_capture_position(rng: random.Random, max_plies: int) -> Position | None:
    for _ in range(24):
        pos = initial_position()
        n_plies = rng.randint(2, max(2, max_plies))
        for _ in range(min(n_plies, MAX_GAME_LEN)):
            moves = generate_moves(pos)
            if not moves or game_result(pos, moves) is not None:
                break
            if moves and any(len(m.captured) > 0 for m in moves):
                return pos
            pos = apply_move(pos, rng.choice(moves))
    return None


def model_best_legal_move(pos: Position, net: AZNetwork) -> Move | None:
    moves = generate_moves(pos)
    if not moves:
        return None
    legal_indices = [move_to_index(m, pos.side) for m in moves]
    probs, _ = net.predict(get_features(pos), legal_indices)
    return moves[int(np.argmax(probs))]


def sample_disagreement_position(
    rng: random.Random,
    net: AZNetwork,
    label_depth: int,
    label_timeout_s: float,
    fallback_depth: int,
    max_plies: int,
    attempts: int,
) -> Position | None:
    for _ in range(max(1, attempts)):
        pos = sample_minimax_playout_position(
            rng=rng,
            max_plies=max_plies,
            playout_depth=max(3, label_depth - 2),
            random_move_prob=0.35,
        )
        if pos is None:
            continue
        model_move = model_best_legal_move(pos, net)
        candidates, _, _ = minimax_root_candidates_with_fallback(
            pos,
            label_depth,
            label_timeout_s,
            fallback_depth,
        )
        mm_move = candidates[0][0] if candidates else None
        if model_move is None or mm_move is None:
            continue
        if move_to_index(model_move, pos.side) != move_to_index(mm_move, pos.side):
            return pos
    return None


def should_accept_side(side: int, side_counts: dict[int, int], target_total: int, collected: int) -> bool:
    # Keep side balance close to 50/50 while still allowing final fill.
    if collected >= target_total:
        return False
    p1 = side_counts[1]
    p2 = side_counts[-1]
    if collected < int(target_total * 0.90):
        if side == 1 and p1 > p2 + 1:
            return False
        if side == -1 and p2 > p1 + 1:
            return False
    return True


def infer_baseline_iter(path: str | None) -> int | None:
    if not path:
        return None
    name = os.path.basename(path)
    match = re.search(r"iter_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def resolve_partial_paths(output_path: str, partial_prefix: str) -> tuple[str, str]:
    if partial_prefix:
        base = os.path.abspath(partial_prefix)
    else:
        base = os.path.splitext(os.path.abspath(output_path))[0] + ".partial"
    return base + ".npz", base + ".json"


def serialize_samples(samples: list[Sample]) -> dict[str, np.ndarray]:
    if not samples:
        return {
            "features": np.zeros((0, 128), dtype=np.float32),
            "policy_index": np.zeros((0,), dtype=np.int32),
            "policy_target": np.zeros((0, 1024), dtype=np.float16),
            "value_target": np.zeros((0,), dtype=np.float32),
            "side_to_move": np.zeros((0,), dtype=np.int8),
            "source_id": np.zeros((0,), dtype=np.int8),
            "move_from": np.zeros((0,), dtype=np.int8),
            "move_to": np.zeros((0,), dtype=np.int8),
            "capture_count": np.zeros((0,), dtype=np.int8),
        }
    return {
        "features": np.stack([s.features for s in samples]).astype(np.float32, copy=False),
        "policy_index": np.array([s.policy_index for s in samples], dtype=np.int32),
        "policy_target": np.stack([s.policy_target for s in samples]).astype(np.float16, copy=False),
        "value_target": np.array([s.value_target for s in samples], dtype=np.float32),
        "side_to_move": np.array([s.side_to_move for s in samples], dtype=np.int8),
        "source_id": np.array([s.source_id for s in samples], dtype=np.int8),
        "move_from": np.array([s.move_from for s in samples], dtype=np.int8),
        "move_to": np.array([s.move_to for s in samples], dtype=np.int8),
        "capture_count": np.array([s.capture_count for s in samples], dtype=np.int8),
    }


def restore_samples_from_arrays(arrays: dict[str, np.ndarray]) -> list[Sample]:
    n = int(arrays["features"].shape[0])
    has_policy_target = "policy_target" in (arrays.files if hasattr(arrays, "files") else arrays)
    policy_target = arrays["policy_target"] if has_policy_target else None
    restored: list[Sample] = []
    for i in range(n):
        pidx = int(arrays["policy_index"][i])
        if policy_target is not None:
            ptarget = policy_target[i].astype(np.float32, copy=False)
        else:
            ptarget = np.zeros((1024,), dtype=np.float32)
            ptarget[pidx] = 1.0
        restored.append(
            Sample(
                features=arrays["features"][i].astype(np.float32, copy=False),
                policy_index=pidx,
                policy_target=ptarget,
                value_target=float(arrays["value_target"][i]),
                side_to_move=int(arrays["side_to_move"][i]),
                source_id=int(arrays["source_id"][i]),
                move_from=int(arrays["move_from"][i]),
                move_to=int(arrays["move_to"][i]),
                capture_count=int(arrays["capture_count"][i]),
            )
        )
    return restored


def save_partial_checkpoint(
    *,
    partial_npz: str,
    partial_json: str,
    samples: list[Sample],
    attempts: int,
    source_counts: dict[int, int],
    side_counts: dict[int, int],
    fallback_labels: int,
    remaining: dict[int, int],
    source_targets: dict[int, int],
    args,
    status: str,
    started_at: float,
    error: str | None = None,
):
    arrays = serialize_samples(samples)
    os.makedirs(os.path.dirname(os.path.abspath(partial_npz)), exist_ok=True)
    np.savez_compressed(partial_npz, **arrays)
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.time() - started_at,
        "samples_collected": int(len(samples)),
        "attempts": int(attempts),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "side_counts": {str(k): int(v) for k, v in side_counts.items()},
        "fallback_labels": int(fallback_labels),
        "remaining": {str(k): int(v) for k, v in remaining.items()},
        "source_targets": {str(k): int(v) for k, v in source_targets.items()},
        "output_target": os.path.abspath(args.output),
    }
    if error is not None:
        payload["error"] = error
    with open(partial_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_partial_checkpoint(partial_npz: str, partial_json: str):
    if not (os.path.exists(partial_npz) and os.path.exists(partial_json)):
        return None
    arrays = np.load(partial_npz)
    with open(partial_json, encoding="utf-8") as f:
        meta = json.load(f)
    return arrays, meta


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


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    started = time.time()

    include_disagreement = bool(args.model_checkpoint)
    if args.source_mix:
        source_mix = parse_source_mix(args.source_mix, include_disagreement=include_disagreement)
    else:
        source_mix = profile_source_mix(args.profile)
        if not include_disagreement:
            source_mix[SOURCE_DISAGREEMENT] = 0.0
            total = sum(source_mix.values())
            if total > 0:
                source_mix = {k: v / total for k, v in source_mix.items()}
    source_targets = make_source_targets(args.samples, source_mix)
    remaining = dict(source_targets)
    partial_npz, partial_json = resolve_partial_paths(args.output, args.partial_prefix)

    disagreement_net = load_network(args.model_checkpoint) if args.model_checkpoint else None
    if args.model_checkpoint and disagreement_net is None:
        raise RuntimeError("failed to load --model-checkpoint")

    source_generators: dict[int, Callable[[], Position | None]] = {
        SOURCE_RANDOM: lambda: sample_random_playout_position(rng, args.max_playout_plies),
        SOURCE_MINIMAX_PLAYOUT: lambda: sample_minimax_playout_position(
            rng, args.max_playout_plies, args.playout_minimax_depth
        ),
        SOURCE_FORCED_CAPTURE: lambda: sample_forced_capture_position(rng, args.max_playout_plies),
    }
    if disagreement_net is not None:
        source_generators[SOURCE_DISAGREEMENT] = lambda: sample_disagreement_position(
            rng,
            disagreement_net,
            args.minimax_depth,
            args.label_timeout_s,
            args.fallback_minimax_depth,
            args.max_playout_plies,
            args.disagreement_search_attempts,
        )

    samples: list[Sample] = []
    source_counts = {sid: 0 for sid in SOURCE_NAMES}
    side_counts = {1: 0, -1: 0}
    attempts = 0
    fallback_labels = 0
    max_attempts = max(args.samples * 80, 20_000)

    if args.resume:
        restored = load_partial_checkpoint(partial_npz, partial_json)
        if restored is not None:
            arrays, meta = restored
            samples = restore_samples_from_arrays(arrays)
            source_counts.update({int(k): int(v) for k, v in meta.get("source_counts", {}).items()})
            side_counts.update({int(k): int(v) for k, v in meta.get("side_counts", {}).items()})
            restored_remaining = {int(k): int(v) for k, v in meta.get("remaining", {}).items()}
            for sid in remaining:
                if sid in restored_remaining:
                    remaining[sid] = max(0, restored_remaining[sid])
            attempts = int(meta.get("attempts", 0))
            fallback_labels = int(meta.get("fallback_labels", 0))
            print(
                f"[resume] restored {len(samples)} samples"
                f" attempts={attempts} side(P1/P2)={side_counts.get(1, 0)}/{side_counts.get(-1, 0)}"
            )
        else:
            print("[resume] no partial checkpoint found; starting fresh")

    print(f"[build] target samples={args.samples} depth={args.minimax_depth} device={DEVICE}")
    print(f"[build] profile={args.profile}")
    print(f"[build] source targets: { {SOURCE_NAMES[k]: v for k, v in source_targets.items()} }")

    try:
        while len(samples) < args.samples and attempts < max_attempts and any(v > 0 for v in remaining.values()):
            attempts += 1
            source_id = choose_weighted_source(rng, remaining)
            sampler = source_generators.get(source_id)
            if sampler is None:
                remaining[source_id] = 0
                continue

            pos = sampler()
            if pos is None:
                continue
            if not should_accept_side(pos.side, side_counts, args.samples, len(samples)):
                continue

            moves = generate_moves(pos)
            if not moves or game_result(pos, moves) is not None:
                continue

            candidates, used_depth, used_fallback = minimax_root_candidates_with_fallback(
                pos,
                args.minimax_depth,
                args.label_timeout_s,
                args.fallback_minimax_depth,
            )
            if not candidates:
                continue
            if used_fallback:
                fallback_labels += 1
            mm_move, mm_score = candidates[0]
            policy_target = policy_target_from_candidates(
                pos=pos,
                candidates=candidates,
                mode=args.policy_mode,
                top_k=args.policy_top_k,
                margin=args.policy_margin,
                temp=args.policy_temp,
            )

            sample = Sample(
                features=get_features(pos).astype(np.float32, copy=False),
                policy_index=int(move_to_index(mm_move, pos.side)),
                policy_target=policy_target,
                value_target=score_to_value(mm_score, args.value_scale),
                side_to_move=int(pos.side),
                source_id=int(source_id),
                move_from=int(mm_move.from_sq),
                move_to=int(mm_move.to_sq),
                capture_count=int(len(mm_move.captured)),
            )
            samples.append(sample)
            source_counts[source_id] += 1
            side_counts[pos.side] += 1
            remaining[source_id] = max(0, remaining[source_id] - 1)

            if len(samples) % args.save_every == 0 or len(samples) == args.samples:
                elapsed = time.time() - started
                print(
                    f"  collected {len(samples):5d}/{args.samples} in {elapsed:6.1f}s"
                    f"  side(P1/P2)={side_counts[1]}/{side_counts[-1]}"
                    f"  fallback={fallback_labels}"
                )
                save_partial_checkpoint(
                    partial_npz=partial_npz,
                    partial_json=partial_json,
                    samples=samples,
                    attempts=attempts,
                    source_counts=source_counts,
                    side_counts=side_counts,
                    fallback_labels=fallback_labels,
                    remaining=remaining,
                    source_targets=source_targets,
                    args=args,
                    status="running",
                    started_at=started,
                )
    except BaseException as exc:
        save_partial_checkpoint(
            partial_npz=partial_npz,
            partial_json=partial_json,
            samples=samples,
            attempts=attempts,
            source_counts=source_counts,
            side_counts=side_counts,
            fallback_labels=fallback_labels,
            remaining=remaining,
            source_targets=source_targets,
            args=args,
            status="failed",
            started_at=started,
            error=str(exc),
        )
        maybe_send_notification(
            args,
            subject=f"[Makhos] Teacher dataset failed ({len(samples)}/{args.samples})",
            body=(
                f"Dataset build failed.\n\n"
                f"output target: {os.path.abspath(args.output)}\n"
                f"partial npz: {partial_npz}\n"
                f"partial meta: {partial_json}\n"
                f"samples: {len(samples)} / {args.samples}\n"
                f"attempts: {attempts}\n"
                f"error: {exc}\n"
            ),
        )
        raise

    if len(samples) < args.samples:
        print(
            f"[warn] collected {len(samples)} < requested {args.samples}"
            f" after {attempts} attempts; saving partial dataset."
        )

    arrays = serialize_samples(samples)
    features = arrays["features"]
    policy_index = arrays["policy_index"]
    policy_target = arrays["policy_target"]
    value_target = arrays["value_target"]
    side_to_move = arrays["side_to_move"]
    source_id = arrays["source_id"]
    move_from = arrays["move_from"]
    move_to = arrays["move_to"]
    capture_count = arrays["capture_count"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        policy_index=policy_index,
        policy_target=policy_target,
        value_target=value_target,
        side_to_move=side_to_move,
        source_id=source_id,
        move_from=move_from,
        move_to=move_to,
        capture_count=capture_count,
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_npz": os.path.abspath(args.output),
        "samples_requested": args.samples,
        "samples_collected": int(len(samples)),
        "minimax_depth": int(args.minimax_depth),
        "playout_minimax_depth": int(args.playout_minimax_depth),
        "value_scale": float(args.value_scale),
        "policy_mode": args.policy_mode,
        "policy_top_k": int(args.policy_top_k),
        "policy_margin": float(args.policy_margin),
        "policy_temp": float(args.policy_temp),
        "random_seed": int(args.random_seed),
        "max_playout_plies": int(args.max_playout_plies),
        "model_checkpoint": os.path.abspath(args.model_checkpoint) if args.model_checkpoint else None,
        "model_checkpoint_baseline_iter": infer_baseline_iter(args.model_checkpoint),
        "attempts": int(attempts),
        "fallback_labels": int(fallback_labels),
        "attempts_per_sample": float(attempts / max(len(samples), 1)),
        "source_targets": {SOURCE_NAMES[k]: int(v) for k, v in source_targets.items()},
        "profile": args.profile,
        "source_counts": {SOURCE_NAMES[k]: int(v) for k, v in source_counts.items()},
        "side_counts": {"p1": int(side_counts[1]), "p2": int(side_counts[-1])},
        "array_shapes": {
            "features": list(features.shape),
            "policy_index": list(policy_index.shape),
            "policy_target": list(policy_target.shape),
            "value_target": list(value_target.shape),
        },
    }
    meta_path = os.path.splitext(os.path.abspath(args.output))[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - started
    print(f"[done] dataset saved: {os.path.abspath(args.output)}")
    print(f"[done] metadata saved: {meta_path}")
    print(f"[done] collected={len(samples)} elapsed={elapsed:.1f}s attempts={attempts}")
    print(f"[done] source counts: {metadata['source_counts']}")
    print(f"[done] side counts  : {metadata['side_counts']}")

    save_partial_checkpoint(
        partial_npz=partial_npz,
        partial_json=partial_json,
        samples=samples,
        attempts=attempts,
        source_counts=source_counts,
        side_counts=side_counts,
        fallback_labels=fallback_labels,
        remaining=remaining,
        source_targets=source_targets,
        args=args,
        status="completed",
        started_at=started,
    )
    if not args.keep_partial_on_success:
        for p in (partial_npz, partial_json):
            if os.path.exists(p):
                os.remove(p)

    maybe_send_notification(
        args,
        subject=f"[Makhos] Teacher dataset done ({len(samples)} samples, d{args.minimax_depth})",
        body=(
            f"Dataset build completed.\n\n"
            f"output: {os.path.abspath(args.output)}\n"
            f"meta: {meta_path}\n"
            f"samples: {len(samples)} / requested {args.samples}\n"
            f"depth: {args.minimax_depth}\n"
            f"elapsed_s: {elapsed:.1f}\n"
            f"source_counts: {metadata['source_counts']}\n"
            f"side_counts: {metadata['side_counts']}\n"
        ),
    )


if __name__ == "__main__":
    main()
