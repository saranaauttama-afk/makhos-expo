"""
train_az.py — AlphaZero Training Loop for Thai Checkers (Makhos)
=================================================================

Upload to Drive/makhos_az_v3/:
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
import os, sys, shutil, time, random, json
import numpy as np
import torch
import torch.nn.functional as F

DRIVE_DIR  = '/content/drive/MyDrive/makhos_az_v3'
MODELS_DIR = f'{DRIVE_DIR}/models'
LOG_FILE   = f'{DRIVE_DIR}/training_log.jsonl'
os.makedirs(MODELS_DIR, exist_ok=True)

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
N_RES     = 4

# Self-play
N_SELFPLAY   = 100    # games per iteration
N_SIMS       = 200    # MCTS simulations per move (self-play)
TEMP_CUTOFF  = 16     # plies before switching to argmax selection
MAX_GAME_LEN = 250    # hard cap per game (safety)

# Training
REPLAY_SIZE  = 350_000  # max replay buffer size
BATCH_SIZE   = 256
TRAIN_STEPS  = 500      # optimizer steps per iteration
LR           = 3e-4
WD           = 1e-4

# Evaluation
EVAL_INTERVAL   = 10    # eval every N iterations
N_EVAL_GAMES    = 60    # new-net vs best-net (more games = lower variance)
WIN_THRESHOLD   = 0.55  # win-rate needed to replace best net
N_MINIMAX_GAMES = 40    # games vs minimax at each checkpoint
N_EVAL_SIMS     = 400   # MCTS sims during eval (stronger play than self-play)
MINIMAX_DEPTH   = 3
MINIMAX_DEPTH_5 = 5

# Email notification
EMAIL_FROM     = 'saranaauttama@gmail.com'
EMAIL_TO       = 'saranaauttama@gmail.com'
EMAIL_PASSWORD = 'vktn yrqy apkd gkza'  # Gmail App Password 16 ตัว

print('Config OK')
print(f'  self-play: {N_SELFPLAY} games × {N_SIMS} sims/move')
print(f'  training : {TRAIN_STEPS} steps × batch {BATCH_SIZE}')
print(f'  eval     : every {EVAL_INTERVAL} iterations, {N_EVAL_GAMES} games')

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

# ── Self-play one game ────────────────────────────────────────────────────────
def self_play_game(network: AZNetwork) -> list:
    """Play one game via MCTS self-play.
    Returns list of (features[128], pi[1024], side_to_move) tuples.
    """
    pos       = initial_position()
    game_buf  = []   # (features, pi_full, side)

    for ply in range(MAX_GAME_LEN):
        moves  = generate_moves(pos)
        result = game_result(pos, moves)
        if result is not None:
            break

        # Run MCTS
        temperature = (ply < TEMP_CUTOFF)
        out_moves, visit_probs = mcts(pos, network, N_SIMS, add_noise=True)

        if not out_moves:
            break

        # Build full policy vector (sparse)
        pi_full = np.zeros(N_MOVES, dtype=np.float32)
        for m, p in zip(out_moves, visit_probs):
            pi_full[move_to_index(m, pos.side)] = p

        game_buf.append((get_features(pos).copy(), pi_full, pos.side))

        # Select move
        if temperature:
            move = random.choices(out_moves, weights=visit_probs)[0]
        else:
            move = out_moves[int(np.argmax(visit_probs))]

        pos = apply_move(pos, move)

    # Determine game outcome
    # result is from the perspective of the player at `pos` when game ended
    if result is None:
        result = game_result(pos, generate_moves(pos)) or 0.0

    # winner_side: which side won (+1 = P1, -1 = P2, 0 = draw)
    if result == 0.0:
        winner_side = 0
    elif result == 1.0:
        winner_side = pos.side        # current player at end won
    else:
        winner_side = -pos.side       # current player at end lost

    # Assign z (value target) to each recorded position
    training_samples = []
    for features, pi_full, side in game_buf:
        if winner_side == 0:
            z = 0.0
        elif side == winner_side:
            z = 1.0
        else:
            z = -1.0
        training_samples.append((features, pi_full, np.float32(z)))

    return training_samples

# ── Train one iteration ───────────────────────────────────────────────────────
def train_step(network: AZNetwork, optimizer, replay_buffer: list):
    """Sample from replay buffer and do one training step batch."""
    batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
    xs  = np.stack([b[0] for b in batch])        # (B, 128)
    pis = np.stack([b[1] for b in batch])        # (B, 1024)
    zs  = np.array([b[2] for b in batch], dtype=np.float32)  # (B,)

    x   = torch.from_numpy(xs).to(DEVICE)
    pi  = torch.from_numpy(pis).to(DEVICE)
    z   = torch.from_numpy(zs).to(DEVICE)

    network.net.train()
    p_logits, v = network.net(x)

    # Policy loss: cross-entropy with MCTS visit distribution
    # Only penalise positions where at least one legal move was visited
    log_probs   = F.log_softmax(p_logits, dim=-1)
    policy_loss = -(pi * log_probs).sum(dim=-1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(v, z)

    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.net.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss.detach()), float(value_loss.detach())

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

print('Helper functions OK')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Main training loop  (this is the cell you keep running)
# ─────────────────────────────────────────────────────────────────────────────

BEST_NET_PATH    = f'{MODELS_DIR}/best.pt'    # best promoted model only
LATEST_NET_PATH  = f'{MODELS_DIR}/latest.pt'  # curr_net saved every iter (for resume)
TRAIN_STATE_PATH = f'{MODELS_DIR}/train_state.pt'   # optimizer + scheduler state

# ── Init or resume ────────────────────────────────────────────────────────────
best_net = AZNetwork(HIDDEN, N_RES)
curr_net = AZNetwork(HIDDEN, N_RES)

optimizer = torch.optim.Adam(curr_net.net.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
replay_buffer = []
start_iter    = 0
BUFFER_PATH   = f'{DRIVE_DIR}/replay_buffer.npy'

if os.path.exists(LATEST_NET_PATH):
    curr_net.load(LATEST_NET_PATH)
    best_net.load(BEST_NET_PATH if os.path.exists(BEST_NET_PATH) else LATEST_NET_PATH)

    # Restore optimizer + scheduler state
    if os.path.exists(TRAIN_STATE_PATH):
        state = torch.load(TRAIN_STATE_PATH, map_location=DEVICE)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

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

    print(f'✅ Resumed from iteration {start_iter}  (LR={scheduler.get_last_lr()[0]:.2e})')
else:
    print('Starting fresh (random network)')

N_ITER = 200   # total iterations to run — increase if you want more training

print(f'\nTraining {start_iter} → {N_ITER}  on {DEVICE}')
print('=' * 60)

for it in range(start_iter, N_ITER):
    t0 = time.time()
    print(f'\n{"="*40}\niter {it}/{N_ITER}  buf={len(replay_buffer)}', flush=True)

    # ── Self-play ─────────────────────────────────────────────────────────────
    print('  [self-play]', flush=True)
    new_samples = 0
    for g in range(N_SELFPLAY):
        samples = self_play_game(curr_net)
        replay_buffer.extend(samples)
        new_samples += len(samples)
        if (g + 1) % 10 == 0:
            elapsed_sp = time.time() - t0
            print(f'    game {g+1}/{N_SELFPLAY}  new_samples={new_samples}  {elapsed_sp:.0f}s', flush=True)

    # Cap replay buffer
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer = replay_buffer[-REPLAY_SIZE:]

    # ── Train ─────────────────────────────────────────────────────────────────
    steps = min(TRAIN_STEPS, max(1, len(replay_buffer) // BATCH_SIZE))
    print(f'  [train]  {steps} steps  buf={len(replay_buffer)}', flush=True)
    p_losses, v_losses = [], []
    for s in range(steps):
        pl, vl = train_step(curr_net, optimizer, replay_buffer)
        p_losses.append(pl)
        v_losses.append(vl)
        if (s + 1) % 100 == 0:
            print(f'    step {s+1}/{steps}  p_loss={pl:.4f}  v_loss={vl:.4f}', flush=True)
    scheduler.step()

    elapsed = time.time() - t0
    p_loss  = np.mean(p_losses)
    v_loss  = np.mean(v_losses)

    gpu_info = get_gpu_info()
    log = {
        'iter':       it,
        'new_samples': new_samples,
        'buffer_size': len(replay_buffer),
        'p_loss':     round(p_loss, 4),
        'v_loss':     round(v_loss, 4),
        'elapsed_s':  round(elapsed, 1),
        'gpu':        gpu_info,
    }

    # ── Evaluate every EVAL_INTERVAL iterations ───────────────────────────────
    if (it + 1) % EVAL_INTERVAL == 0:
        print(f'\n── Checkpoint iter {it} ──')

        # 1) New net vs best net
        wr_net = eval_net_vs_net(curr_net, best_net, N_EVAL_GAMES)
        print(f'  curr vs best : {wr_net:.1%}  ({N_EVAL_GAMES} games)')

        # 2) Net vs random
        wr_rand = eval_net_vs_random(curr_net, N_MINIMAX_GAMES)
        print(f'  curr vs random   : {wr_rand:.1%}  ({N_MINIMAX_GAMES} games)')

        # 3) Net vs minimax-3
        wr_mm3 = eval_net_vs_minimax(curr_net, N_MINIMAX_GAMES, MINIMAX_DEPTH)
        print(f'  curr vs minimax-3: {wr_mm3:.1%}  ({N_MINIMAX_GAMES} games)')

        # 4) Net vs minimax-5
        wr_mm5 = eval_net_vs_minimax(curr_net, N_MINIMAX_GAMES, MINIMAX_DEPTH_5)
        print(f'  curr vs minimax-5: {wr_mm5:.1%}  ({N_MINIMAX_GAMES} games)')

        log['wr_vs_best']     = round(wr_net, 3)
        log['wr_vs_random']   = round(wr_rand, 3)
        log['wr_vs_minimax3'] = round(wr_mm3, 3)
        log['wr_vs_minimax5'] = round(wr_mm5, 3)

        # 3) Update best if curr is better
        if wr_net >= WIN_THRESHOLD:
            best_net = curr_net.copy()
            best_net.save(BEST_NET_PATH)
            print(f'  ✅ Updated best net  (win rate {wr_net:.1%})')
        else:
            print(f'  ➡  Keep old best net (win rate {wr_net:.1%} < {WIN_THRESHOLD:.0%})')

        # 4) Save iteration checkpoint
        iter_path = f'{MODELS_DIR}/iter_{it:04d}.pt'
        curr_net.save(iter_path)

        # 5) Send checkpoint email
        print(f'  GPU: {gpu_info}')
        extra = (f'vs best net  : {wr_net:.1%}\n'
                 f'vs random    : {wr_rand:.1%}\n'
                 f'vs minimax-3 : {wr_mm3:.1%}\n'
                 f'vs minimax-5 : {wr_mm5:.1%}\n'
                 f'\nGPU: {gpu_info}')
        send_iter_email(it, p_loss, v_loss, elapsed, extra)

    else:
        # Quick progress print every iteration
        print(f'iter {it:3d}  samples={new_samples:4d}  buf={len(replay_buffer):6d}'
              f'  p_loss={p_loss:.4f}  v_loss={v_loss:.4f}  {elapsed:.0f}s  GPU: {gpu_info}')
        send_iter_email(it, p_loss, v_loss, elapsed, f'GPU: {gpu_info}')

    # Append to log
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log) + '\n')

    # Save curr_net + optimizer + scheduler + buffer every iteration (for resume)
    curr_net.save(LATEST_NET_PATH)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
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

print(f'{"iter":>4}  {"p_loss":>7}  {"v_loss":>7}  {"vs_best":>8}  {"vs_rand":>8}  {"vs_mm3":>8}  {"vs_mm5":>8}  {"buf":>6}')
print('-' * 68)
for r in rows:
    vs_best = f'{r["wr_vs_best"]:.1%}'     if 'wr_vs_best'     in r else '      —'
    vs_rand = f'{r["wr_vs_random"]:.1%}'   if 'wr_vs_random'   in r else '      —'
    vs_mm3  = f'{r["wr_vs_minimax3"]:.1%}' if 'wr_vs_minimax3' in r else '      —'
    vs_mm5  = f'{r["wr_vs_minimax5"]:.1%}' if 'wr_vs_minimax5' in r else '      —'
    print(f'{r["iter"]:>4}  {r["p_loss"]:>7.4f}  {r["v_loss"]:>7.4f}  '
          f'{vs_best:>8}  {vs_rand:>8}  {vs_mm3:>8}  {vs_mm5:>8}  {r["buffer_size"]:>6}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Export model to ONNX  (run anytime, does not affect training)
# ─────────────────────────────────────────────────────────────────────────────
!pip install onnxscript onnx -q

import torch, onnx, os
from network_az import AZNetwork, DEVICE

EXPORT_ITER = 'iter_0059'   # ← เปลี่ยนเป็น iter ที่ต้องการ
PT_PATH   = f'{MODELS_DIR}/{EXPORT_ITER}.pt'
ONNX_PATH = f'{DRIVE_DIR}/{EXPORT_ITER}.onnx'

# Load network
net = AZNetwork(hidden=256, n_res=4)
net.load(PT_PATH)
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
