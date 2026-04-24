"""
train_az.py — AlphaZero Training Loop for Thai Checkers (Makhos)
=================================================================

Upload to Drive/makhos_az_v5/:
  makhos_engine.py  network_az.py  mcts_az.py  train_az.py

Run cells 1 → 2 → 3 → 4 (main loop).
Checkpoints are saved every EVAL_INTERVAL iterations.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Setup
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, shutil, time, random, json, glob
import numpy as np
import torch
import torch.nn.functional as F

DRIVE_DIR  = '/content/drive/MyDrive/makhos_az_v5'
MODELS_DIR = f'{DRIVE_DIR}/models'
LOG_FILE   = f'{DRIVE_DIR}/training_log.jsonl'
EXTERNAL_EVAL_DIR = f'{DRIVE_DIR}/external_eval'
EXTERNAL_REQUESTS_DIR = f'{EXTERNAL_EVAL_DIR}/requests'
EXTERNAL_RESULTS_DIR = f'{EXTERNAL_EVAL_DIR}/results'
EXTERNAL_DECISIONS_DIR = f'{EXTERNAL_EVAL_DIR}/decisions'
TARGET_STATUS_PATH = f'{EXTERNAL_EVAL_DIR}/target_status.json'
for path in [MODELS_DIR, EXTERNAL_EVAL_DIR, EXTERNAL_REQUESTS_DIR, EXTERNAL_RESULTS_DIR, EXTERNAL_DECISIONS_DIR]:
    os.makedirs(path, exist_ok=True)

# Copy engine + network modules to /content so we can import them
for fname in ['makhos_engine.py', 'network_az.py', 'mcts_az.py']:
    src = f'{DRIVE_DIR}/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{fname}')
        print(f'Copied {fname}')
    else:
        print(f'WARNING: {src} not found — upload it to {DRIVE_DIR}/')

sys.path.insert(0, '/content')

from makhos_engine import (
    initial_position, generate_moves, apply_move, get_features,
    is_draw_by_inactivity, bit_count, hand_eval, Position, Move,
)
from network_az import AZNetwork, N_MOVES, DEVICE
from mcts_az     import mcts, game_result, move_to_index

print(f'\nDevice: {DEVICE}')
print('Setup OK')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Config
# ─────────────────────────────────────────────────────────────────────────────

# Network
HIDDEN    = 256
N_RES     = 6          # v4: deeper network (was 4)

# Self-play
N_SELFPLAY   = 50     # v5: budget-aware (was 100)
N_SIMS       = 400    # v4: 2x sims → higher quality games (was 200)
TEMP_CUTOFF  = 16     # plies before switching to argmax selection
MAX_GAME_LEN = 250    # hard cap per game (safety)

# Training
REPLAY_SIZE  = 200_000  # v4: start moderate, will grow (was 350k)
BATCH_SIZE   = 256
TRAIN_STEPS  = 500      # optimizer steps per iteration
LR           = 8e-5     # rollback-from-79 tune: smaller step to reduce regression
WD           = 1e-4
MIN_BUFFER_TO_TRAIN = 8_192
SELFPLAY_ANCHOR_FRACTION = 0.30
SELFPLAY_POOL_FRACTION = 0.25
SELFPLAY_MINIMAX_FRACTION = 0.20
SELFPLAY_MINIMAX_DEPTHS = [5, 7]
OPPONENT_POOL_MAX = 6
POOL_REFRESH_EVERY = 1
LOSS_MINING_DEPTH = 7
LOSS_MINING_GAMES = 3
LOSS_MINING_POSITIONS_PER_GAME = 6
LOSS_MINING_MAX_SAMPLES = 36
LOSS_MINING_SAMPLE_WEIGHT = 0.50  # conservative weighting to reduce regression from mined samples
OVERRIDE_LR_ON_RESUME = True
STABILITY_REG_WEIGHT = 0.07       # slightly stronger anchor to stabilize policy/value while climbing
STABILITY_VALUE_WEIGHT = 0.25

# Evaluation
EVAL_INTERVAL   = 10    # save checkpoint / quick-eval every N iterations
WIN_THRESHOLD   = 0.55  # quick win-rate needed to replace best net
N_EVAL_SIMS     = 400   # MCTS sims during eval (stronger play than self-play)
MINIMAX_DEPTH   = 3
MINIMAX_DEPTH_5 = 5
MINIMAX_DEPTH_7 = 7
MINIMAX_DEPTH_9 = 9
MINIMAX_DEPTH_11 = 11
TARGET_MINIMAX_DEPTH = MINIMAX_DEPTH_11
TARGET_BEST_MARGIN   = 0.02
OPENING_SUITE_SIZE = 8
OPENING_SUITE_MAX_PLY = 8
OPENING_SUITE_SEED = 20260406
OPENING_SUITE_MM_DEPTH = TARGET_MINIMAX_DEPTH

# Quick in-loop eval for Colab only.
QUICK_EVAL_ENABLED = True
QUICK_EVAL_NET_GAMES = 12
QUICK_EVAL_RANDOM_GAMES = 6
QUICK_EVAL_MINIMAX_PLAN = [
    (MINIMAX_DEPTH, 6),
    (MINIMAX_DEPTH_5, 4),
    (MINIMAX_DEPTH_7, 4),
]
QUICK_EVAL_USE_OPENING_SUITE = False
ENABLE_LOSS_MINING = True
AUTO_LOSS_MINING_FROM_DECISIONS = True
DEFAULT_LOSS_MINING_DEPTH = 7
DEFAULT_LOSS_MINING_GAMES = 3
DEFAULT_LOSS_MINING_POSITIONS_PER_GAME = 6
DEFAULT_LOSS_MINING_MAX_SAMPLES = 36
QUICK_GATE_FOR_EXTERNAL_EVAL = True
QUICK_GATE_MIN_MM3 = 0.75
QUICK_GATE_MIN_MM5 = 0.60

# External eval handshake (local machine reads requests and writes results/decisions).
REQUEST_EXTERNAL_EVAL = True
APPLY_EXTERNAL_DECISIONS = True
EXTERNAL_DECISION_POLL_EVERY = 1
ALLOW_QUICK_PROMOTE_BEST = False
FORCE_IGNORE_OLD_DECISIONS = True
FORCE_DECISION_MTIME_SLACK_S = 2.0

# Run-policy
RUN_BLOCK_ITERS = 20     # stop after each 20-iter block so we can review and adjust
MAX_TOTAL_ITERS = 200

# Force-resume controls (set FORCE_BASELINE_ITER=None to disable).
FORCE_BASELINE_ITER = 79
FORCE_RESET_TRAIN_STATE = True
FORCE_CLEAR_REPLAY_BUFFER = True
FORCE_RESET_LOG_CURSOR = True

# Learning-rate control
LR_PLATEAU_PATIENCE = 2
LR_DECAY_FACTOR     = 0.5
MIN_LR              = 5e-5
MIN_FRONTIER_LR     = 8e-5  # keep LR from collapsing while pushing through mm7/mm9 plateau

# Email notification
EMAIL_FROM     = 'saranaauttama@gmail.com'
EMAIL_TO       = 'saranaauttama@gmail.com'
EMAIL_PASSWORD = 'vktn yrqy apkd gkza'  # Gmail App Password 16 ตัว

print('Config OK')
print(f'  self-play: {N_SELFPLAY} games × {N_SIMS} sims/move')
print(f'  training : {TRAIN_STEPS} steps × batch {BATCH_SIZE}')
print(f'  eval     : every {EVAL_INTERVAL} iterations, quick eval on Colab')
print(f'  external : requests -> {EXTERNAL_REQUESTS_DIR}')
print(f'  target   : full decisions come from local eval vs minimax-{TARGET_MINIMAX_DEPTH}')
print(f'  mining   : {"auto-from-decisions" if AUTO_LOSS_MINING_FROM_DECISIONS else ("enabled" if ENABLE_LOSS_MINING else "disabled")}')
if QUICK_GATE_FOR_EXTERNAL_EVAL:
    print(f'  q-gate   : external eval requires mm3>={QUICK_GATE_MIN_MM3:.0%} and mm5>={QUICK_GATE_MIN_MM5:.0%}')
print(f'  run      : blocks of {RUN_BLOCK_ITERS} iterations (cap {MAX_TOTAL_ITERS})')
print(f'  force    : baseline={FORCE_BASELINE_ITER} reset_state={FORCE_RESET_TRAIN_STATE} clear_buf={FORCE_CLEAR_REPLAY_BUFFER}')
print(f'  guard    : ignore_old_decisions={FORCE_IGNORE_OLD_DECISIONS}')
print(f'  lr floor : global={MIN_LR:.1e}, frontier={MIN_FRONTIER_LR:.1e}')
print(f'  stability: reg={STABILITY_REG_WEIGHT:.2f}, value_w={STABILITY_VALUE_WEIGHT:.2f}, mined_w={LOSS_MINING_SAMPLE_WEIGHT:.2f}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Helper functions
# ─────────────────────────────────────────────────────────────────────────────

# ── Minimax (for evaluation only) ─────────────────────────────────────────────
_INF = 10_000

def _minimax(pos, depth, alpha, beta):
    if is_draw_by_inactivity(pos): return 0
    moves = generate_moves(pos)
    if not moves: return -_INF
    if depth == 0: return hand_eval(pos)
    best = -_INF
    for m in moves:
        s = -_minimax(apply_move(pos, m), depth - 1, -beta, -alpha)
        if s > best: best = s
        if s > alpha: alpha = s
        if alpha >= beta: break
    return best

def _minimax_best_move(pos, depth):
    moves = generate_moves(pos)
    if not moves: return None
    return max(moves, key=lambda m: -_minimax(apply_move(pos, m), depth - 1, -_INF, _INF))

def build_optimizer_and_scheduler(network: AZNetwork, lr: float):
    optimizer = torch.optim.Adam(network.net.parameters(), lr=lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_DECAY_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        threshold=TARGET_BEST_MARGIN,
        threshold_mode='abs',
        min_lr=MIN_LR,
    )
    return optimizer, scheduler


def get_current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


def set_optimizer_lr(optimizer, scheduler, lr: float):
    lr = max(MIN_LR, float(lr))
    for group in optimizer.param_groups:
        group['lr'] = lr
    if hasattr(scheduler, '_last_lr'):
        scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]


def maybe_override_lr(optimizer, scheduler):
    if not OVERRIDE_LR_ON_RESUME:
        return
    set_optimizer_lr(optimizer, scheduler, LR)


def make_actor(kind: str, *, net: AZNetwork | None = None, depth: int | None = None, name: str | None = None):
    return {'kind': kind, 'net': net, 'depth': depth, 'name': name or kind}


def one_hot_policy(pos: Position, move: Move) -> np.ndarray:
    pi_full = np.zeros(N_MOVES, dtype=np.float32)
    pi_full[move_to_index(move, pos.side)] = 1.0
    return pi_full


def choose_actor_move(pos: Position, actor: dict, *, add_noise: bool, temperature: bool, n_sims: int) -> tuple[Move | None, np.ndarray | None]:
    moves = generate_moves(pos)
    if not moves:
        return None, None

    if actor['kind'] == 'net':
        out_moves, visit_probs = mcts(pos, actor['net'], n_sims, add_noise=add_noise)
        if not out_moves:
            return moves[0], one_hot_policy(pos, moves[0])
        pi_full = np.zeros(N_MOVES, dtype=np.float32)
        for m, p in zip(out_moves, visit_probs):
            pi_full[move_to_index(m, pos.side)] = p
        if temperature:
            move = random.choices(out_moves, weights=visit_probs)[0]
        else:
            move = out_moves[int(np.argmax(visit_probs))]
        return move, pi_full

    if actor['kind'] == 'minimax':
        move = _minimax_best_move(pos, actor['depth']) or moves[0]
        return move, one_hot_policy(pos, move)

    move = random.choice(moves)
    return move, one_hot_policy(pos, move)


def assign_outcome_targets(game_buf: list, winner_side: int) -> list:
    training_samples = []
    for features, pi_full, side in game_buf:
        if winner_side == 0:
            z = 0.0
        elif side == winner_side:
            z = 1.0
        else:
            z = -1.0
        training_samples.append((features, pi_full, np.float32(z), np.float32(1.0)))
    return training_samples


def play_training_game(p1_actor: dict, p2_actor: dict, start_pos: Position | None = None) -> list:
    """Play one training game and return replay samples."""
    pos = start_pos or initial_position()
    game_buf = []

    for ply in range(MAX_GAME_LEN):
        moves = generate_moves(pos)
        result = game_result(pos, moves)
        if result is not None:
            break

        actor = p1_actor if pos.side == 1 else p2_actor
        move, pi_full = choose_actor_move(
            pos,
            actor,
            add_noise=(actor['kind'] == 'net'),
            temperature=(actor['kind'] == 'net' and ply < TEMP_CUTOFF),
            n_sims=N_SIMS,
        )
        if move is None or pi_full is None:
            break

        game_buf.append((get_features(pos).copy(), pi_full, pos.side))
        pos = apply_move(pos, move)

    if result is None:
        result = game_result(pos, generate_moves(pos)) or 0.0

    if result == 0.0:
        winner_side = 0
    elif result == 1.0:
        winner_side = pos.side
    else:
        winner_side = -pos.side

    return assign_outcome_targets(game_buf, winner_side)


def build_opening_suite() -> list:
    rng = random.Random(OPENING_SUITE_SEED)
    suite = []
    attempts = 0
    while len(suite) < OPENING_SUITE_SIZE and attempts < OPENING_SUITE_SIZE * 20:
        attempts += 1
        pos = initial_position()
        target_plies = rng.randint(2, OPENING_SUITE_MAX_PLY)
        for _ in range(target_plies):
            moves = generate_moves(pos)
            if not moves or game_result(pos, moves) is not None:
                break
            # Deterministic-but-varied opening sampling.
            moves = sorted(moves, key=lambda m: (len(m.captured), m.from_sq, m.to_sq))
            move = rng.choice(moves[:min(6, len(moves))])
            pos = apply_move(pos, move)
        if game_result(pos, generate_moves(pos)) is None:
            suite.append(pos)
    return suite


def play_eval_game(
    net: AZNetwork,
    *,
    mm_depth: int,
    net_side: int,
    start_pos: Position | None = None,
    collect_net_positions: bool = False,
) -> tuple[float, list]:
    pos = start_pos or initial_position()
    net_positions = []
    result = None
    for _ in range(MAX_GAME_LEN):
        moves = generate_moves(pos)
        result = game_result(pos, moves)
        if result is not None:
            break
        is_net = (pos.side == net_side)
        if is_net:
            out_moves, vp = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
            move = out_moves[int(np.argmax(vp))] if out_moves else moves[0]
            if collect_net_positions:
                net_positions.append(pos)
        else:
            move = _minimax_best_move(pos, mm_depth) or moves[0]
        pos = apply_move(pos, move)

    result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
    if result == 0.0:
        score = 0.5
    elif (result == 1.0 and pos.side == net_side) or (result == -1.0 and pos.side != net_side):
        score = 1.0
    else:
        score = 0.0
    return score, net_positions


def eval_net_on_opening_suite(net: AZNetwork, mm_depth: int, suite: list) -> dict:
    p1_scores, p2_scores = [], []
    for start_pos in suite:
        score_p1, _ = play_eval_game(net, mm_depth=mm_depth, net_side=1, start_pos=start_pos)
        score_p2, _ = play_eval_game(net, mm_depth=mm_depth, net_side=-1, start_pos=start_pos)
        p1_scores.append(score_p1)
        p2_scores.append(score_p2)
    all_scores = p1_scores + p2_scores
    return {
        'overall': float(np.mean(all_scores)) if all_scores else 0.0,
        'p1': float(np.mean(p1_scores)) if p1_scores else 0.0,
        'p2': float(np.mean(p2_scores)) if p2_scores else 0.0,
        'games': len(all_scores),
    }


def mine_losses_vs_minimax(
    net: AZNetwork,
    mm_depth: int,
    n_games: int,
    positions_per_game: int,
    max_samples: int,
) -> list:
    samples = []
    if n_games <= 0 or positions_per_game <= 0 or max_samples <= 0:
        return samples
    for g in range(n_games):
        start_pos = OPENING_SUITE[g % len(OPENING_SUITE)] if OPENING_SUITE else initial_position()
        net_side = 1 if (g % 2 == 0) else -1
        score, net_positions = play_eval_game(
            net,
            mm_depth=mm_depth,
            net_side=net_side,
            start_pos=start_pos,
            collect_net_positions=True,
        )
        if score != 0.0:
            continue
        tail_positions = net_positions[-positions_per_game:]
        for pos in tail_positions:
            move = _minimax_best_move(pos, mm_depth)
            if move is None:
                continue
            samples.append((
                get_features(pos).copy(),
                one_hot_policy(pos, move),
                np.float32(-1.0),
                np.float32(LOSS_MINING_SAMPLE_WEIGHT),
            ))
            if len(samples) >= max_samples:
                return samples
    return samples


def load_recent_pool_snapshots() -> list:
    pool = []
    iter_paths = sorted(glob.glob(f'{MODELS_DIR}/iter_*.pt'))[-max(0, OPPONENT_POOL_MAX - 2):]
    for path in iter_paths:
        try:
            net = AZNetwork(HIDDEN, N_RES)
            net.load(path)
            pool.append({'name': os.path.splitext(os.path.basename(path))[0], 'net': net})
        except Exception as exc:
            print(f'  Pool load failed for {path}: {exc}')
    return pool


def refresh_opponent_pool(pool: list, best_net: AZNetwork, target_net: AZNetwork, curr_net: AZNetwork, iteration: int):
    if iteration % POOL_REFRESH_EVERY != 0:
        return pool
    snapshots = [
        ('best', best_net.copy()),
        ('target', target_net.copy()),
        (f'iter_{iteration:04d}', curr_net.copy()),
    ]
    for name, net in snapshots:
        pool.append({'name': name, 'net': net})
    return pool[-OPPONENT_POOL_MAX:]


def choose_selfplay_matchup(curr_net: AZNetwork, best_net: AZNetwork, target_net: AZNetwork, pool: list):
    roll = random.random()
    if roll < SELFPLAY_MINIMAX_FRACTION:
        mm_depth = random.choice(SELFPLAY_MINIMAX_DEPTHS)
        curr_is_p1 = (random.random() < 0.5)
        rival = make_actor('minimax', depth=mm_depth, name=f'mm{mm_depth}')
        if curr_is_p1:
            return make_actor('net', net=curr_net, name='curr'), rival, f'curr-vs-mm{mm_depth}'
        return rival, make_actor('net', net=curr_net, name='curr'), f'mm{mm_depth}-vs-curr'

    if roll < SELFPLAY_MINIMAX_FRACTION + SELFPLAY_POOL_FRACTION and pool:
        rival_entry = random.choice(pool)
        curr_is_p1 = (random.random() < 0.5)
        rival = make_actor('net', net=rival_entry['net'], name=rival_entry['name'])
        if curr_is_p1:
            return make_actor('net', net=curr_net, name='curr'), rival, f'curr-vs-{rival_entry["name"]}'
        return rival, make_actor('net', net=curr_net, name='curr'), f'{rival_entry["name"]}-vs-curr'

    if roll < SELFPLAY_MINIMAX_FRACTION + SELFPLAY_POOL_FRACTION + SELFPLAY_ANCHOR_FRACTION:
        curr_is_p1 = (random.random() < 0.5)
        rival_net = target_net if random.random() < 0.5 else best_net
        rival_name = 'target' if rival_net is target_net else 'best'
        if curr_is_p1:
            return make_actor('net', net=curr_net, name='curr'), make_actor('net', net=rival_net, name=rival_name), f'curr-vs-{rival_name}'
        return make_actor('net', net=rival_net, name=rival_name), make_actor('net', net=curr_net, name='curr'), f'{rival_name}-vs-curr'

    return make_actor('net', net=curr_net, name='curr'), make_actor('net', net=curr_net, name='curr'), 'curr-vs-curr'

OPENING_SUITE = build_opening_suite()
print(f'Opening suite: {len(OPENING_SUITE)} fixed positions')

# ── Train one iteration ───────────────────────────────────────────────────────
def train_step(
    network: AZNetwork,
    optimizer,
    replay_buffer: list,
    teacher_net: AZNetwork | None = None,
    stability_weight: float = 0.0,
):
    """Sample from replay buffer and do one training step batch."""
    batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
    xs = np.stack([b[0] for b in batch])         # (B, 128)
    pis = np.stack([b[1] for b in batch])        # (B, 1024)
    zs = np.array([b[2] for b in batch], dtype=np.float32)   # (B,)
    ws = np.array([b[3] if len(b) >= 4 else 1.0 for b in batch], dtype=np.float32)

    x = torch.from_numpy(xs).to(DEVICE)
    pi = torch.from_numpy(pis).to(DEVICE)
    z = torch.from_numpy(zs).to(DEVICE)
    w = torch.from_numpy(ws).to(DEVICE)
    w_sum = torch.clamp(w.sum(), min=1e-6)

    network.net.train()
    p_logits, v = network.net(x)

    # Policy loss: cross-entropy with MCTS visit distribution
    log_probs = F.log_softmax(p_logits, dim=-1)
    policy_vec = -(pi * log_probs).sum(dim=-1)
    policy_loss = (policy_vec * w).sum() / w_sum

    # Value loss: MSE
    value_vec = (v - z) ** 2
    value_loss = (value_vec * w).sum() / w_sum

    stability_loss = torch.zeros((), device=DEVICE)
    if teacher_net is not None and stability_weight > 0:
        teacher_net.net.eval()
        with torch.no_grad():
            t_logits, t_v = teacher_net.net(x)
            t_probs = F.softmax(t_logits, dim=-1)
        # KL(student || teacher) on policy head + light value anchor.
        kl_vec = (t_probs * (torch.log(torch.clamp(t_probs, min=1e-8)) - log_probs)).sum(dim=-1)
        stability_policy = kl_vec.mean()
        stability_value = F.mse_loss(v, t_v)
        stability_loss = stability_policy + (STABILITY_VALUE_WEIGHT * stability_value)

    loss = policy_loss + value_loss + (stability_weight * stability_loss)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.net.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss.detach()), float(value_loss.detach()), float(stability_loss.detach())

# ── Net-vs-net evaluation ─────────────────────────────────────────────────────
def eval_net_vs_net(new_net: AZNetwork, best_net: AZNetwork, n_games: int) -> float:
    """Returns win-rate of new_net vs best_net (draws count as 0.5)."""
    wins = 0.0
    for g in range(n_games):
        new_is_p1 = (g % 2 == 0)
        pos = initial_position()
        for _ in range(MAX_GAME_LEN):
            moves  = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None: break
            is_new = (pos.side == 1) == new_is_p1
            net    = new_net if is_new else best_net
            out_moves, visit_probs = mcts(pos, net, n_sims=N_EVAL_SIMS, add_noise=False)
            if not out_moves: break
            pos = apply_move(pos, out_moves[int(np.argmax(visit_probs))])
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        # result from perspective of player at pos when game ended
        if result == 0.0:
            wins += 0.5
        elif (result == 1.0 and (pos.side == 1) == new_is_p1) or \
             (result == -1.0 and (pos.side != 1) == new_is_p1):
            wins += 1.0
    return wins / n_games

# ── Net vs minimax ────────────────────────────────────────────────────────────
def eval_net_vs_minimax(net: AZNetwork, n_games: int, mm_depth: int) -> float:
    """Returns win-rate of net vs minimax (draws count as 0.5)."""
    wins = 0.0
    for g in range(n_games):
        net_is_p1 = (g % 2 == 0)
        pos = initial_position()
        for _ in range(MAX_GAME_LEN):
            moves  = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None: break
            is_net = (pos.side == 1) == net_is_p1
            if is_net:
                out_moves, vp = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
                move = out_moves[int(np.argmax(vp))] if out_moves else moves[0]
            else:
                move = _minimax_best_move(pos, mm_depth) or moves[0]
            pos = apply_move(pos, move)
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        if result == 0.0:
            wins += 0.5
        elif (result == 1.0 and (pos.side == 1) == net_is_p1) or \
             (result == -1.0 and (pos.side != 1) == net_is_p1):
            wins += 1.0
    return wins / n_games

# ── Net vs random ─────────────────────────────────────────────────────────────
def eval_net_vs_random(net: AZNetwork, n_games: int) -> float:
    """Returns win-rate of net vs random mover (draws count as 0.5)."""
    wins = 0.0
    for g in range(n_games):
        net_is_p1 = (g % 2 == 0)
        pos = initial_position()
        result = None
        for _ in range(MAX_GAME_LEN):
            moves  = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None: break
            is_net = (pos.side == 1) == net_is_p1
            if is_net:
                out_moves, vp = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
                move = out_moves[int(np.argmax(vp))] if out_moves else moves[0]
            else:
                move = random.choice(moves)
            pos = apply_move(pos, move)
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        if result == 0.0:
            wins += 0.5
        elif (result == 1.0 and (pos.side == 1) == net_is_p1) or \
             (result == -1.0 and (pos.side != 1) == net_is_p1):
            wins += 1.0
    return wins / n_games


def maybe_override_lr(optimizer, scheduler):
    if not OVERRIDE_LR_ON_RESUME:
        return
    set_optimizer_lr(optimizer, scheduler, LR)

import smtplib, subprocess
from email.mime.text import MIMEText

def get_gpu_info() -> str:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'N/A'

def send_iter_email(it, p_loss, v_loss, elapsed, extra=''):
    subject = f'Makhos AZ — iter {it}/200'
    body = f'iter {it}/200\n\np_loss : {p_loss:.4f}\nv_loss : {v_loss:.4f}\nelapsed: {elapsed:.0f}s'
    if extra:
        body += f'\n\n{extra}'
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.send_message(msg)
        print('  📧 Email sent')
    except Exception as e:
        print(f'  Email failed: {e}')


def checkpoint_path_for_iter(iteration: int) -> str:
    return f'{MODELS_DIR}/iter_{iteration:04d}.pt'


def write_external_eval_request(iteration: int, checkpoint_path: str):
    if not REQUEST_EXTERNAL_EVAL:
        return
    payload = {
        'iter': iteration,
        'checkpoint': checkpoint_path,
        'created_at': int(time.time()),
        'status': 'pending',
        'full_eval_plan': {
            'vs_best_games': 60,
            'vs_random_games': 20,
            'minimax_plan': [[3, 24], [5, 20], [7, 12], [9, 8], [11, 8]],
            'opening_suite_size': OPENING_SUITE_SIZE,
            'opening_suite_max_ply': OPENING_SUITE_MAX_PLY,
            'target_depth': TARGET_MINIMAX_DEPTH,
        },
    }
    request_path = f'{EXTERNAL_REQUESTS_DIR}/iter_{iteration:04d}.request.json'
    with open(request_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'  queued external eval -> {request_path}')


def load_target_status():
    if not os.path.exists(TARGET_STATUS_PATH):
        return None
    try:
        with open(TARGET_STATUS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        print(f'  target status read failed: {exc}')
        return None


def resolve_decision_checkpoint_path(checkpoint: str | None, decision_iter: int) -> str:
    candidates = []
    if checkpoint:
        candidates.append(checkpoint)
        candidates.append(os.path.join(MODELS_DIR, os.path.basename(checkpoint)))
    candidates.append(checkpoint_path_for_iter(decision_iter))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return checkpoint or checkpoint_path_for_iter(decision_iter)


def apply_external_decisions(
    best_net: AZNetwork,
    target_net: AZNetwork,
    target_best_wr: float,
    target_best_suite: float,
    target_best_side_floor: float,
    last_external_decision_iter: int,
    external_decision_min_mtime: float,
    current_loss_mining_enabled: bool,
    current_loss_mining_depth: int,
    current_loss_mining_games: int,
    current_loss_mining_positions_per_game: int,
    current_loss_mining_max_samples: int,
):
    if not APPLY_EXTERNAL_DECISIONS:
        return (
            best_net,
            target_net,
            target_best_wr,
            target_best_suite,
            target_best_side_floor,
            last_external_decision_iter,
            external_decision_min_mtime,
            current_loss_mining_enabled,
            current_loss_mining_depth,
            current_loss_mining_games,
            current_loss_mining_positions_per_game,
            current_loss_mining_max_samples,
            [],
        )

    decision_paths = sorted(glob.glob(f'{EXTERNAL_DECISIONS_DIR}/iter_*.decision.json'))
    applied = []
    for path in decision_paths:
        if external_decision_min_mtime > 0.0:
            try:
                if os.path.getmtime(path) < external_decision_min_mtime:
                    continue
            except OSError:
                continue

        try:
            with open(path, encoding='utf-8') as f:
                decision = json.load(f)
        except Exception as exc:
            print(f'  decision read failed for {path}: {exc}')
            continue

        decision_iter = int(decision.get('iter', -1))
        if decision_iter <= last_external_decision_iter:
            continue

        checkpoint = resolve_decision_checkpoint_path(decision.get('checkpoint'), decision_iter)
        if not os.path.exists(checkpoint):
            print(f'  decision skipped: missing checkpoint {checkpoint}')
            continue

        if decision.get('promote_best'):
            best_net.load(checkpoint)
            best_net.save(BEST_NET_PATH)

        if decision.get('promote_target'):
            target_net.load(checkpoint)
            target_net.save(TARGET_NET_PATH)
            target_best_wr = float(decision.get('wr_vs_minimax11', target_best_wr))
            target_best_suite = float(decision.get('wr_opening_suite', target_best_suite))
            target_best_side_floor = float(decision.get('wr_opening_floor', target_best_side_floor))
            target_status = {
                'iter': decision_iter,
                'checkpoint': checkpoint,
                'wr_vs_minimax11': target_best_wr,
                'wr_opening_suite': target_best_suite,
                'wr_opening_floor': target_best_side_floor,
                'updated_at': int(time.time()),
                'source': 'external_decision',
            }
            with open(TARGET_STATUS_PATH, 'w', encoding='utf-8') as f:
                json.dump(target_status, f, indent=2)

        if AUTO_LOSS_MINING_FROM_DECISIONS:
            current_loss_mining_enabled = bool(decision.get('enable_loss_mining', current_loss_mining_enabled))
            current_loss_mining_depth = int(decision.get('loss_mining_depth', current_loss_mining_depth))
            current_loss_mining_games = int(decision.get('loss_mining_games', current_loss_mining_games))
            current_loss_mining_positions_per_game = int(
                decision.get('loss_mining_positions_per_game', current_loss_mining_positions_per_game)
            )
            current_loss_mining_max_samples = int(
                decision.get('loss_mining_max_samples', current_loss_mining_max_samples)
            )
            gated_depth = decision.get('gated_at_depth')
            # Plateau rescue: if candidate is still gated at mm9, push a stronger mm7-focused mining preset.
            if gated_depth == 9:
                current_loss_mining_enabled = True
                current_loss_mining_depth = 7
                current_loss_mining_games = max(current_loss_mining_games, 3)
                current_loss_mining_positions_per_game = max(current_loss_mining_positions_per_game, 6)
                current_loss_mining_max_samples = max(current_loss_mining_max_samples, 36)

        last_external_decision_iter = decision_iter
        applied.append({
            'iter': decision_iter,
            'promote_best': bool(decision.get('promote_best')),
            'promote_target': bool(decision.get('promote_target')),
            'summary': decision.get('summary', ''),
            'loss_mining_reason': decision.get('loss_mining_reason', ''),
            'loss_mining_enabled': current_loss_mining_enabled,
            'loss_mining_depth': current_loss_mining_depth,
        })

    return (
        best_net,
        target_net,
        target_best_wr,
        target_best_suite,
        target_best_side_floor,
        last_external_decision_iter,
        external_decision_min_mtime,
        current_loss_mining_enabled,
        current_loss_mining_depth,
        current_loss_mining_games,
        current_loss_mining_positions_per_game,
        current_loss_mining_max_samples,
        applied,
    )

print('Helper functions OK')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Main training loop  (this is the cell you keep running)
# ─────────────────────────────────────────────────────────────────────────────

BEST_NET_PATH    = f'{MODELS_DIR}/best.pt'    # best promoted model only
LATEST_NET_PATH  = f'{MODELS_DIR}/latest.pt'  # curr_net saved every iter (for resume)
TRAIN_STATE_PATH = f'{MODELS_DIR}/train_state.pt'   # optimizer + scheduler state
TARGET_NET_PATH  = f'{MODELS_DIR}/target_best.pt'   # best net against minimax-11

# ── Init or resume ────────────────────────────────────────────────────────────
best_net = AZNetwork(HIDDEN, N_RES)
curr_net = AZNetwork(HIDDEN, N_RES)
target_net = AZNetwork(HIDDEN, N_RES)

optimizer, scheduler = build_optimizer_and_scheduler(curr_net, LR)
replay_buffer = []
start_iter    = 0
BUFFER_PATH   = f'{DRIVE_DIR}/replay_buffer.npy'
target_best_wr = -1.0
target_best_suite = -1.0
target_best_side_floor = -1.0
last_external_decision_iter = -1
external_decision_min_mtime = 0.0
current_loss_mining_enabled = ENABLE_LOSS_MINING
current_loss_mining_depth = DEFAULT_LOSS_MINING_DEPTH
current_loss_mining_games = DEFAULT_LOSS_MINING_GAMES
current_loss_mining_positions_per_game = DEFAULT_LOSS_MINING_POSITIONS_PER_GAME
current_loss_mining_max_samples = DEFAULT_LOSS_MINING_MAX_SAMPLES
opponent_pool = load_recent_pool_snapshots()

forced_baseline_iter = FORCE_BASELINE_ITER if isinstance(FORCE_BASELINE_ITER, int) and FORCE_BASELINE_ITER >= 0 else None
if forced_baseline_iter is not None:
    forced_path = checkpoint_path_for_iter(forced_baseline_iter)
    if not os.path.exists(forced_path):
        raise FileNotFoundError(f'Forced baseline checkpoint not found: {forced_path}')

    curr_net.load(forced_path)
    best_net.load(forced_path)
    target_net.load(forced_path)
    start_iter = forced_baseline_iter + 1
    last_external_decision_iter = forced_baseline_iter
    if FORCE_IGNORE_OLD_DECISIONS:
        external_decision_min_mtime = time.time() - FORCE_DECISION_MTIME_SLACK_S

    if FORCE_RESET_TRAIN_STATE:
        optimizer, scheduler = build_optimizer_and_scheduler(curr_net, LR)
        target_best_wr = -1.0
        target_best_suite = -1.0
        target_best_side_floor = -1.0
        current_loss_mining_enabled = ENABLE_LOSS_MINING
        current_loss_mining_depth = DEFAULT_LOSS_MINING_DEPTH
        current_loss_mining_games = DEFAULT_LOSS_MINING_GAMES
        current_loss_mining_positions_per_game = DEFAULT_LOSS_MINING_POSITIONS_PER_GAME
        current_loss_mining_max_samples = DEFAULT_LOSS_MINING_MAX_SAMPLES
    elif os.path.exists(TRAIN_STATE_PATH):
        state = torch.load(TRAIN_STATE_PATH, map_location=DEVICE)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            try:
                scheduler.load_state_dict(state['scheduler'])
            except Exception as exc:
                print(f'  Scheduler state skipped: {exc}')
        target_best_wr = state.get('target_best_wr', -1.0)
        target_best_suite = state.get('target_best_suite', -1.0)
        target_best_side_floor = state.get('target_best_side_floor', -1.0)
        current_loss_mining_enabled = state.get('current_loss_mining_enabled', current_loss_mining_enabled)
        current_loss_mining_depth = state.get('current_loss_mining_depth', current_loss_mining_depth)
        current_loss_mining_games = state.get('current_loss_mining_games', current_loss_mining_games)
        current_loss_mining_positions_per_game = state.get(
            'current_loss_mining_positions_per_game',
            current_loss_mining_positions_per_game,
        )
        current_loss_mining_max_samples = state.get(
            'current_loss_mining_max_samples',
            current_loss_mining_max_samples,
        )
        external_decision_min_mtime = float(
            state.get('external_decision_min_mtime', external_decision_min_mtime)
        )

    maybe_override_lr(optimizer, scheduler)

    if FORCE_CLEAR_REPLAY_BUFFER:
        replay_buffer = []
    elif os.path.exists(BUFFER_PATH):
        replay_buffer = [tuple(x) for x in np.load(BUFFER_PATH, allow_pickle=True)]
        print(f'  Loaded replay buffer: {len(replay_buffer)} samples')

    if FORCE_RESET_LOG_CURSOR:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'iter': forced_baseline_iter}) + '\n')

    print(
        f'✅ Forced baseline resume from iter {forced_baseline_iter}'
        f' -> start {start_iter} (LR={get_current_lr(optimizer):.2e})'
    )
    if FORCE_IGNORE_OLD_DECISIONS:
        print(f'  decision guard active (mtime >= {external_decision_min_mtime:.0f})')
elif os.path.exists(LATEST_NET_PATH):
    curr_net.load(LATEST_NET_PATH)
    best_net.load(BEST_NET_PATH if os.path.exists(BEST_NET_PATH) else LATEST_NET_PATH)
    if os.path.exists(TARGET_NET_PATH):
        target_net.load(TARGET_NET_PATH)
    else:
        target_net = best_net.copy()

    # Restore optimizer + scheduler state
    if os.path.exists(TRAIN_STATE_PATH):
        state = torch.load(TRAIN_STATE_PATH, map_location=DEVICE)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            try:
                scheduler.load_state_dict(state['scheduler'])
            except Exception as exc:
                print(f'  Scheduler state skipped: {exc}')
        target_best_wr = state.get('target_best_wr', -1.0)
        target_best_suite = state.get('target_best_suite', -1.0)
        target_best_side_floor = state.get('target_best_side_floor', -1.0)
        last_external_decision_iter = state.get('last_external_decision_iter', -1)
        external_decision_min_mtime = float(state.get('external_decision_min_mtime', 0.0))
        current_loss_mining_enabled = state.get('current_loss_mining_enabled', current_loss_mining_enabled)
        current_loss_mining_depth = state.get('current_loss_mining_depth', current_loss_mining_depth)
        current_loss_mining_games = state.get('current_loss_mining_games', current_loss_mining_games)
        current_loss_mining_positions_per_game = state.get(
            'current_loss_mining_positions_per_game',
            current_loss_mining_positions_per_game,
        )
        current_loss_mining_max_samples = state.get(
            'current_loss_mining_max_samples',
            current_loss_mining_max_samples,
        )

    target_status = load_target_status()
    if target_status:
        target_best_wr = float(target_status.get('wr_vs_minimax11', target_best_wr))
        target_best_suite = float(target_status.get('wr_opening_suite', target_best_suite))
        target_best_side_floor = float(target_status.get('wr_opening_floor', target_best_side_floor))
        mining_policy = target_status.get('loss_mining_policy') or {}
        current_loss_mining_enabled = bool(mining_policy.get('enable_loss_mining', current_loss_mining_enabled))
        current_loss_mining_depth = int(mining_policy.get('loss_mining_depth', current_loss_mining_depth))
        current_loss_mining_games = int(mining_policy.get('loss_mining_games', current_loss_mining_games))
        current_loss_mining_positions_per_game = int(
            mining_policy.get('loss_mining_positions_per_game', current_loss_mining_positions_per_game)
        )
        current_loss_mining_max_samples = int(
            mining_policy.get('loss_mining_max_samples', current_loss_mining_max_samples)
        )

    maybe_override_lr(optimizer, scheduler)

    # Restore replay buffer
    if os.path.exists(BUFFER_PATH):
        replay_buffer = [tuple(x) for x in np.load(BUFFER_PATH, allow_pickle=True)]
        print(f'  Loaded replay buffer: {len(replay_buffer)} samples')

    # Find start iteration from log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            lines = f.readlines()
        if lines:
            start_iter = json.loads(lines[-1]).get('iter', 0) + 1

    print(f'✅ Resumed from iteration {start_iter}  (LR={get_current_lr(optimizer):.2e})')
else:
    target_net = curr_net.copy()
    print('Starting fresh (random network)')

opponent_pool = refresh_opponent_pool(opponent_pool, best_net, target_net, curr_net, 0)

N_ITER = min(MAX_TOTAL_ITERS, start_iter + RUN_BLOCK_ITERS)

print(f'\nTraining {start_iter} → {N_ITER}  on {DEVICE}')
print('=' * 60)

for it in range(start_iter, N_ITER):
    t0 = time.time()
    print(f'\n{"="*40}\niter {it}/{N_ITER}  buf={len(replay_buffer)}', flush=True)

    if it % EXTERNAL_DECISION_POLL_EVERY == 0:
        (
            best_net,
            target_net,
            target_best_wr,
            target_best_suite,
            target_best_side_floor,
            last_external_decision_iter,
            external_decision_min_mtime,
            current_loss_mining_enabled,
            current_loss_mining_depth,
            current_loss_mining_games,
            current_loss_mining_positions_per_game,
            current_loss_mining_max_samples,
            applied_decisions,
        ) = apply_external_decisions(
            best_net,
            target_net,
            target_best_wr,
            target_best_suite,
            target_best_side_floor,
            last_external_decision_iter,
            external_decision_min_mtime,
            current_loss_mining_enabled,
            current_loss_mining_depth,
            current_loss_mining_games,
            current_loss_mining_positions_per_game,
            current_loss_mining_max_samples,
        )
        for decision in applied_decisions:
            print(
                f'  applied external decision iter {decision["iter"]}'
                f'  best={decision["promote_best"]} target={decision["promote_target"]}'
                f'  mining={decision["loss_mining_enabled"]}@d{decision["loss_mining_depth"]}'
                f'  {decision["loss_mining_reason"]}'
                f'  {decision["summary"]}',
                flush=True,
            )

    # ── Self-play ─────────────────────────────────────────────────────────────
    print('  [self-play]', flush=True)
    new_samples = 0
    matchup_counts = {}
    for g in range(N_SELFPLAY):
        p1_actor, p2_actor, matchup_name = choose_selfplay_matchup(curr_net, best_net, target_net, opponent_pool)
        matchup_counts[matchup_name] = matchup_counts.get(matchup_name, 0) + 1
        samples = play_training_game(p1_actor, p2_actor)
        replay_buffer.extend(samples)
        new_samples += len(samples)
        if (g + 1) % 10 == 0:
            elapsed_sp = time.time() - t0
            top_matchups = ', '.join(f'{k}:{v}' for k, v in sorted(matchup_counts.items())[:3])
            print(
                f'    game {g+1}/{N_SELFPLAY}  new_samples={new_samples}'
                f'  pool={len(opponent_pool)}  {top_matchups}  {elapsed_sp:.0f}s',
                flush=True,
            )

    # Cap replay buffer
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer = replay_buffer[-REPLAY_SIZE:]

    # ── Train ─────────────────────────────────────────────────────────────────
    if len(replay_buffer) < MIN_BUFFER_TO_TRAIN:
        steps = 0
        print(f'  [train]  skipped  buf={len(replay_buffer)} < {MIN_BUFFER_TO_TRAIN}', flush=True)
    else:
        steps = min(TRAIN_STEPS, max(1, len(replay_buffer) // BATCH_SIZE))
        print(f'  [train]  {steps} steps  buf={len(replay_buffer)}', flush=True)
    p_losses, v_losses, s_losses = [], [], []
    for s in range(steps):
        pl, vl, sl = train_step(
            curr_net,
            optimizer,
            replay_buffer,
            teacher_net=target_net,
            stability_weight=STABILITY_REG_WEIGHT,
        )
        p_losses.append(pl)
        v_losses.append(vl)
        s_losses.append(sl)
        if (s + 1) % 100 == 0:
            print(f'    step {s+1}/{steps}  p_loss={pl:.4f}  v_loss={vl:.4f}  s_loss={sl:.4f}', flush=True)

    elapsed = time.time() - t0
    p_loss  = float(np.mean(p_losses)) if p_losses else 0.0
    v_loss  = float(np.mean(v_losses)) if v_losses else 0.0
    s_loss  = float(np.mean(s_losses)) if s_losses else 0.0

    gpu_info = get_gpu_info()
    log = {
        'iter':       it,
        'new_samples': new_samples,
        'pool_size':   len(opponent_pool),
        'buffer_size': len(replay_buffer),
        'p_loss':     round(p_loss, 4),
        'v_loss':     round(v_loss, 4),
        's_loss':     round(s_loss, 4),
        'elapsed_s':  round(elapsed, 1),
        'gpu':        gpu_info,
        'lr':         round(get_current_lr(optimizer), 7),
        'loss_mining_enabled': bool(current_loss_mining_enabled),
        'loss_mining_depth_current': int(current_loss_mining_depth),
        'loss_mining_games_current': int(current_loss_mining_games),
        'loss_mining_max_samples_current': int(current_loss_mining_max_samples),
    }

    # ── Checkpoint every EVAL_INTERVAL iterations ─────────────────────────────
    if (it + 1) % EVAL_INTERVAL == 0:
        print(f'\n── Checkpoint iter {it} ──')

        iter_path = checkpoint_path_for_iter(it)
        curr_net.save(iter_path)
        external_eval_queued = False
        external_eval_status = f'skipped for iter {it:04d}'

        wr_net = 0.0
        wr_rand = 0.0
        mm_results = {}
        suite_metrics = {'overall': 0.0, 'p1': 0.0, 'p2': 0.0, 'games': 0}

        if QUICK_EVAL_ENABLED:
            wr_net = eval_net_vs_net(curr_net, best_net, QUICK_EVAL_NET_GAMES)
            wr_rand = eval_net_vs_random(curr_net, QUICK_EVAL_RANDOM_GAMES)
            print(f'  quick curr vs best   : {wr_net:.1%}  ({QUICK_EVAL_NET_GAMES} games)')
            print(f'  quick curr vs random : {wr_rand:.1%}  ({QUICK_EVAL_RANDOM_GAMES} games)')
            log['quick_wr_vs_best'] = round(wr_net, 3)
            log['quick_wr_vs_random'] = round(wr_rand, 3)

            for depth, n_games in QUICK_EVAL_MINIMAX_PLAN:
                wr = eval_net_vs_minimax(curr_net, n_games, depth)
                mm_results[depth] = wr
                log[f'quick_wr_vs_minimax{depth}'] = round(wr, 3)
                print(f'  quick vs minimax-{depth:<2}: {wr:.1%}  ({n_games} games)')

            if QUICK_EVAL_USE_OPENING_SUITE:
                suite_metrics = eval_net_on_opening_suite(curr_net, MINIMAX_DEPTH_5, OPENING_SUITE)
                log['quick_wr_opening_suite'] = round(suite_metrics['overall'], 3)
                print(
                    f'  quick opening suite : {suite_metrics["overall"]:.1%}'
                    f'  (P1 {suite_metrics["p1"]:.1%} / P2 {suite_metrics["p2"]:.1%}, {suite_metrics["games"]} games)'
                )

            if ALLOW_QUICK_PROMOTE_BEST and wr_net >= WIN_THRESHOLD:
                best_net = curr_net.copy()
                best_net.save(BEST_NET_PATH)
                print(f'  ✅ Updated best net from quick eval  ({wr_net:.1%})')
            elif not ALLOW_QUICK_PROMOTE_BEST:
                print('  ➡  Quick eval will not promote best.pt; waiting for external decision')
            else:
                print(f'  ➡  Keep old best net from quick eval ({wr_net:.1%} < {WIN_THRESHOLD:.0%})')

            quick_frontier = mm_results.get(MINIMAX_DEPTH_5, mm_results.get(MINIMAX_DEPTH, wr_net))
            scheduler.step(quick_frontier)
        else:
            scheduler.step(wr_net)

        should_queue_external_eval = REQUEST_EXTERNAL_EVAL
        if should_queue_external_eval and QUICK_EVAL_ENABLED and QUICK_GATE_FOR_EXTERNAL_EVAL:
            mm3_quick = mm_results.get(MINIMAX_DEPTH)
            mm5_quick = mm_results.get(MINIMAX_DEPTH_5)
            if mm3_quick is not None and mm3_quick < QUICK_GATE_MIN_MM3:
                should_queue_external_eval = False
                print(
                    f'  external eval gate : skip (quick mm3 {mm3_quick:.1%} < {QUICK_GATE_MIN_MM3:.0%})'
                )
            elif mm5_quick is not None and mm5_quick < QUICK_GATE_MIN_MM5:
                should_queue_external_eval = False
                print(
                    f'  external eval gate : skip (quick mm5 {mm5_quick:.1%} < {QUICK_GATE_MIN_MM5:.0%})'
                )

        if should_queue_external_eval:
            write_external_eval_request(it, iter_path)
            external_eval_queued = True
            external_eval_status = f'queued for iter {it:04d}'
        log['external_eval_queued'] = bool(external_eval_queued)

        if current_loss_mining_enabled and current_loss_mining_depth <= 7:
            lr_now = get_current_lr(optimizer)
            if lr_now < MIN_FRONTIER_LR:
                set_optimizer_lr(optimizer, scheduler, MIN_FRONTIER_LR)
                print(
                    f'  LR floor guard     : raised {lr_now:.2e} -> {get_current_lr(optimizer):.2e}'
                    f'  (frontier mining mode)',
                    flush=True,
                )

        if current_loss_mining_enabled:
            mined_samples = mine_losses_vs_minimax(
                curr_net,
                current_loss_mining_depth,
                current_loss_mining_games,
                current_loss_mining_positions_per_game,
                current_loss_mining_max_samples,
            )
            log['loss_mining_depth'] = current_loss_mining_depth
            if mined_samples:
                replay_buffer.extend(mined_samples)
                if len(replay_buffer) > REPLAY_SIZE:
                    replay_buffer = replay_buffer[-REPLAY_SIZE:]
            log['loss_mined'] = len(mined_samples)
            print(
                f'  loss mining        : +{len(mined_samples)} samples'
                f'  (depth={current_loss_mining_depth}, games={current_loss_mining_games},'
                f' tail={current_loss_mining_positions_per_game}, max={current_loss_mining_max_samples})'
            )
        else:
            mined_samples = []
            log['loss_mined'] = 0
            print('  loss mining        : skipped by current policy')

        opponent_pool = refresh_opponent_pool(opponent_pool, best_net, target_net, curr_net, it)

        print(f'  LR now             : {get_current_lr(optimizer):.2e}')
        print(f'  external eval      : {external_eval_status}')
        print(f'  GPU                : {gpu_info}')
        extra = (
            f'quick vs best   : {wr_net:.1%}\n'
            f'quick vs random : {wr_rand:.1%}\n'
            f'quick vs mm3    : {mm_results.get(MINIMAX_DEPTH, 0.0):.1%}\n'
            f'quick vs mm5    : {mm_results.get(MINIMAX_DEPTH_5, 0.0):.1%}\n'
            f'stability loss  : {s_loss:.4f}\n'
            f'loss mined      : {len(mined_samples)}\n'
            f'loss mining     : {"on" if current_loss_mining_enabled else "off"}'
            f' @d{current_loss_mining_depth}'
            f' g{current_loss_mining_games}'
            f' t{current_loss_mining_positions_per_game}'
            f' m{current_loss_mining_max_samples}\n'
            f'LR              : {get_current_lr(optimizer):.2e}\n'
            f'external eval   : {external_eval_status}\n'
            f'\nGPU: {gpu_info}'
        )
        send_iter_email(it, p_loss, v_loss, elapsed, extra)

    else:
        # Quick progress print every iteration
        print(f'iter {it:3d}  samples={new_samples:4d}  buf={len(replay_buffer):6d}'
              f'  p_loss={p_loss:.4f}  v_loss={v_loss:.4f}  s_loss={s_loss:.4f}'
              f'  lr={get_current_lr(optimizer):.2e}'
              f'  {elapsed:.0f}s  GPU: {gpu_info}')
        send_iter_email(it, p_loss, v_loss, elapsed, f'LR: {get_current_lr(optimizer):.2e}\nGPU: {gpu_info}')

    # Append to log
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log) + '\n')

    # Save curr_net + optimizer + scheduler + buffer every iteration (for resume)
    curr_net.save(LATEST_NET_PATH)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'target_best_wr': target_best_wr,
        'target_best_suite': target_best_suite,
        'target_best_side_floor': target_best_side_floor,
        'last_external_decision_iter': last_external_decision_iter,
        'external_decision_min_mtime': external_decision_min_mtime,
        'current_loss_mining_enabled': current_loss_mining_enabled,
        'current_loss_mining_depth': current_loss_mining_depth,
        'current_loss_mining_games': current_loss_mining_games,
        'current_loss_mining_positions_per_game': current_loss_mining_positions_per_game,
        'current_loss_mining_max_samples': current_loss_mining_max_samples,
    }, TRAIN_STATE_PATH)
    np.save(BUFFER_PATH, np.array(replay_buffer, dtype=object))

print('\nTraining complete!')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Show training progress (run anytime)
# ─────────────────────────────────────────────────────────────────────────────
import json

rows = []
with open(LOG_FILE) as f:
    for line in f:
        rows.append(json.loads(line))

print(f'{"iter":>4}  {"p_loss":>7}  {"v_loss":>7}  {"s_loss":>7}  {"q_best":>8}  {"q_mm3":>8}  {"q_mm5":>8}  {"lr":>9}  {"mined":>5}  {"buf":>6}')
print('-' * 96)
for r in rows:
    quick_best = f'{r["quick_wr_vs_best"]:.1%}' if 'quick_wr_vs_best' in r else '      —'
    quick_mm3  = f'{r["quick_wr_vs_minimax3"]:.1%}' if 'quick_wr_vs_minimax3' in r else '      —'
    quick_mm5  = f'{r["quick_wr_vs_minimax5"]:.1%}' if 'quick_wr_vs_minimax5' in r else '      —'
    s_loss     = f'{r["s_loss"]:.4f}' if 's_loss' in r else '      —'
    lr      = f'{r["lr"]:.1e}' if 'lr' in r else '        —'
    mined   = f'{r["loss_mined"]:>5}' if 'loss_mined' in r else '    —'
    print(f'{r["iter"]:>4}  {r["p_loss"]:>7.4f}  {r["v_loss"]:>7.4f}  {s_loss:>7}  '
          f'{quick_best:>8}  {quick_mm3:>8}  {quick_mm5:>8}  {lr:>9}  {mined:>5}  {r["buffer_size"]:>6}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Export model to ONNX  (run anytime, does not affect training)
# ─────────────────────────────────────────────────────────────────────────────
!pip install onnxscript onnx -q

import torch, onnx, os
from network_az import AZNetwork, DEVICE

EXPORT_ITER = 'iter_0059'   # ← เปลี่ยนเป็น iter ที่ต้องการ
PT_PATH   = f'{MODELS_DIR}/{EXPORT_ITER}.pt'
ONNX_PATH = f'{DRIVE_DIR}/{EXPORT_ITER}.onnx'

# Load network using checkpoint metadata
ckpt = torch.load(PT_PATH, map_location=DEVICE)
net = AZNetwork(hidden=ckpt.get('hidden', HIDDEN), n_res=ckpt.get('n_res', N_RES))
net.net.load_state_dict(ckpt['state_dict'])
net.net.eval()

# Export to ONNX
dummy = torch.zeros(1, 128, dtype=torch.float32).to(DEVICE)
torch.onnx.export(
    net.net,
    dummy,
    ONNX_PATH,
    input_names=['features'],
    output_names=['policy_logits', 'value'],
    dynamic_axes={'features': {0: 'batch'}, 'policy_logits': {0: 'batch'}, 'value': {0: 'batch'}},
    opset_version=17,
)

# Consolidate external data into single file (removes .onnx.data)
model = onnx.load(ONNX_PATH)
onnx.save_model(model, ONNX_PATH, save_as_external_data=False)

# Clean up leftover .data file if any
data_file = ONNX_PATH + '.data'
if os.path.exists(data_file):
    os.remove(data_file)
    print(f'Removed {data_file}')

size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f'✅ Single-file ONNX saved: {ONNX_PATH}  ({size_mb:.1f} MB)')
