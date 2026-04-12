"""
Local evaluator for the newAz split pipeline.

Purpose:
- read checkpoints produced by Colab training
- run the heavy evaluation locally
- write machine-readable result / decision files back to the shared folder
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
from mcts_az import game_result, mcts, move_to_index
from network_az import AZNetwork, DEVICE, N_MOVES

MAX_GAME_LEN = 250
N_EVAL_SIMS = 400
FULL_NET_GAMES = 60
FULL_RANDOM_GAMES = 20
FULL_MINIMAX_PLAN = [
    (3, 24),
    (5, 20),
    (7, 12),
    (9, 8),
    (11, 8),
]
FULL_EVAL_GATES = {
    7: 0.35,   # do not continue to mm9 unless candidate is at least competitive at mm7
    9: 0.45,   # do not continue to mm11 unless candidate is genuinely close at mm9
}
HEAVY_MM11_EVERY = 20   # only run mm11 on every 20th iteration block checkpoint
OPENING_SUITE_SIZE = 8
OPENING_SUITE_MAX_PLY = 8
OPENING_SUITE_SEED = 20260406
TARGET_MINIMAX_DEPTH = 11
WIN_THRESHOLD = 0.55
TARGET_BEST_MARGIN = 0.02
TS_VERIFY_TRIGGER_MM11 = 0.50
TS_VERIFY_TRIGGER_OPENING_FLOOR = 0.45
PRACTICAL_VERIFY_MM_DEPTH = 7
PRACTICAL_VERIFY_GAMES = 12
PRACTICAL_VERIFY_SIMS = 800
PRACTICAL_VERIFY_TRIGGER_MM5 = 0.75
PRACTICAL_VERIFY_TRIGGER_MM7 = 0.25
VERBOSE_MINIMAX_DEPTHS = {9, 11}

_INF = 10_000


def parse_args():
    parser = argparse.ArgumentParser(description='Run full local evaluation for newAz checkpoints.')
    parser.add_argument('--drive-dir', required=True, help='Shared training folder, e.g. synced Google Drive path.')
    parser.add_argument('--checkpoint', help='Optional direct checkpoint path or iter label (iter_0049).')
    parser.add_argument('--watch', action='store_true', help='Continuously watch request folder for new checkpoints.')
    parser.add_argument('--poll-seconds', type=int, default=30, help='Watch mode poll interval.')
    parser.add_argument('--once', action='store_true', help='Process at most one checkpoint and exit.')
    parser.add_argument('--ts-verify', action='store_true', help='Run final TypeScript battle verify when checkpoint is strong enough.')
    parser.add_argument('--ts-games', type=int, default=12, help='Games for TS final verify.')
    parser.add_argument('--ts-depth', type=int, default=11, help='Minimax depth for TS final verify.')
    parser.add_argument('--ts-sims', type=int, default=1600, help='AZ sims for TS final verify.')
    return parser.parse_args()


def load_network(path: str) -> AZNetwork:
    ckpt = torch.load(path, map_location=DEVICE)
    net = AZNetwork(hidden=ckpt.get('hidden', 256), n_res=ckpt.get('n_res', 4))
    net.net.load_state_dict(ckpt['state_dict'])
    net.net.eval()
    return net


def _minimax(pos: Position, depth: int, alpha: int, beta: int) -> int:
    if is_draw_by_inactivity(pos):
        return 0
    moves = generate_moves(pos)
    if not moves:
        return -_INF
    if depth == 0:
        return hand_eval(pos)
    best = -_INF
    for move in moves:
        score = -_minimax(apply_move(pos, move), depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _minimax_best_move(pos: Position, depth: int) -> Optional[Move]:
    moves = generate_moves(pos)
    if not moves:
        return None
    return max(moves, key=lambda move: -_minimax(apply_move(pos, move), depth - 1, -_INF, _INF))


def play_eval_game(
    net: AZNetwork,
    *,
    mm_depth: int,
    net_side: int,
    start_pos: Position | None = None,
) -> float:
    pos = start_pos or initial_position()
    result = None
    for _ in range(MAX_GAME_LEN):
        moves = generate_moves(pos)
        result = game_result(pos, moves)
        if result is not None:
            break

        if pos.side == net_side:
            out_moves, visit_probs = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
            move = out_moves[int(np.argmax(visit_probs))] if out_moves else moves[0]
        else:
            move = _minimax_best_move(pos, mm_depth) or moves[0]
        pos = apply_move(pos, move)

    result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
    if result == 0.0:
        return 0.5
    if (result == 1.0 and pos.side == net_side) or (result == -1.0 and pos.side != net_side):
        return 1.0
    return 0.0


def eval_net_vs_net(candidate: AZNetwork, best_net: AZNetwork, n_games: int) -> float:
    wins = 0.0
    for game in range(n_games):
        candidate_is_p1 = (game % 2 == 0)
        pos = initial_position()
        result = None
        for _ in range(MAX_GAME_LEN):
            moves = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None:
                break
            is_candidate = (pos.side == 1) == candidate_is_p1
            net = candidate if is_candidate else best_net
            out_moves, visit_probs = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
            if not out_moves:
                break
            pos = apply_move(pos, out_moves[int(np.argmax(visit_probs))])
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        if result == 0.0:
            wins += 0.5
        elif (result == 1.0 and (pos.side == 1) == candidate_is_p1) or (result == -1.0 and (pos.side != 1) == candidate_is_p1):
            wins += 1.0
    return wins / n_games


def eval_net_vs_random(net: AZNetwork, n_games: int) -> float:
    wins = 0.0
    for game in range(n_games):
        net_is_p1 = (game % 2 == 0)
        pos = initial_position()
        result = None
        for _ in range(MAX_GAME_LEN):
            moves = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None:
                break
            if (pos.side == 1) == net_is_p1:
                out_moves, visit_probs = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
                move = out_moves[int(np.argmax(visit_probs))] if out_moves else moves[0]
            else:
                move = random.choice(moves)
            pos = apply_move(pos, move)
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        if result == 0.0:
            wins += 0.5
        elif (result == 1.0 and (pos.side == 1) == net_is_p1) or (result == -1.0 and (pos.side != 1) == net_is_p1):
            wins += 1.0
    return wins / n_games


def eval_net_vs_minimax(net: AZNetwork, n_games: int, mm_depth: int) -> float:
    wins = 0.0
    for game in range(n_games):
        net_is_p1 = (game % 2 == 0)
        pos = initial_position()
        result = None
        for _ in range(MAX_GAME_LEN):
            moves = generate_moves(pos)
            result = game_result(pos, moves)
            if result is not None:
                break
            if (pos.side == 1) == net_is_p1:
                out_moves, visit_probs = mcts(pos, net, N_EVAL_SIMS, add_noise=False)
                move = out_moves[int(np.argmax(visit_probs))] if out_moves else moves[0]
            else:
                move = _minimax_best_move(pos, mm_depth) or moves[0]
            pos = apply_move(pos, move)
        result = result if result is not None else (game_result(pos, generate_moves(pos)) or 0.0)
        game_score = 0.0
        outcome = 'L'
        if result == 0.0:
            game_score = 0.5
            outcome = 'D'
        elif (result == 1.0 and (pos.side == 1) == net_is_p1) or (result == -1.0 and (pos.side != 1) == net_is_p1):
            game_score = 1.0
            outcome = 'W'
        wins += game_score
        if mm_depth in VERBOSE_MINIMAX_DEPTHS:
            side = 'P1' if net_is_p1 else 'P2'
            running = wins / (game + 1)
            print(f'    mm{mm_depth} game {game+1}/{n_games}  net={side}  result={outcome}  running={running:.1%}')
    return wins / n_games


def build_opening_suite() -> list[Position]:
    rng = random.Random(OPENING_SUITE_SEED)
    suite: list[Position] = []
    attempts = 0
    while len(suite) < OPENING_SUITE_SIZE and attempts < OPENING_SUITE_SIZE * 20:
        attempts += 1
        pos = initial_position()
        target_plies = rng.randint(2, OPENING_SUITE_MAX_PLY)
        for _ in range(target_plies):
            moves = generate_moves(pos)
            if not moves or game_result(pos, moves) is not None:
                break
            moves = sorted(moves, key=lambda move: (len(move.captured), move.from_sq, move.to_sq))
            pos = apply_move(pos, rng.choice(moves[:min(6, len(moves))]))
        if game_result(pos, generate_moves(pos)) is None:
            suite.append(pos)
    return suite


def eval_net_on_opening_suite(net: AZNetwork, mm_depth: int, suite: list[Position]) -> dict:
    p1_scores = []
    p2_scores = []
    for start_pos in suite:
        p1_scores.append(play_eval_game(net, mm_depth=mm_depth, net_side=1, start_pos=start_pos))
        p2_scores.append(play_eval_game(net, mm_depth=mm_depth, net_side=-1, start_pos=start_pos))
    all_scores = p1_scores + p2_scores
    return {
        'overall': float(np.mean(all_scores)) if all_scores else 0.0,
        'p1': float(np.mean(p1_scores)) if p1_scores else 0.0,
        'p2': float(np.mean(p2_scores)) if p2_scores else 0.0,
        'games': len(all_scores),
    }


def write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: str, payload: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')


def load_target_status(drive_dir: str) -> Optional[dict]:
    path = os.path.join(drive_dir, 'external_eval', 'target_status.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_target_status(drive_dir: str, payload: dict):
    path = os.path.join(drive_dir, 'external_eval', 'target_status.json')
    write_json(path, payload)


def should_run_mm11(iter_idx: int) -> bool:
    return iter_idx >= 0 and (iter_idx + 1) % HEAVY_MM11_EVERY == 0


def recommend_loss_mining(result: dict) -> dict:
    wr5 = result.get('wr_vs_minimax5')
    wr7 = result.get('wr_vs_minimax7')
    wr9 = result.get('wr_vs_minimax9')
    wr11 = result.get('wr_vs_minimax11')

    if wr5 is None or float(wr5) < 0.20:
        return {
            'enable_loss_mining': False,
            'loss_mining_depth': 5,
            'loss_mining_games': 0,
            'loss_mining_positions_per_game': 0,
            'loss_mining_max_samples': 0,
            'loss_mining_reason': 'mm5 below 20%; keep training on self-play only',
        }

    if wr7 is None or float(wr7) < 0.35:
        return {
            'enable_loss_mining': True,
            'loss_mining_depth': 5,
            'loss_mining_games': 2,
            'loss_mining_positions_per_game': 6,
            'loss_mining_max_samples': 16,
            'loss_mining_reason': 'stronger light mining at mm5 frontier',
        }

    if wr9 is None or float(wr9) < 0.45:
        return {
            'enable_loss_mining': True,
            'loss_mining_depth': 7,
            'loss_mining_games': 2,
            'loss_mining_positions_per_game': 4,
            'loss_mining_max_samples': 16,
            'loss_mining_reason': 'medium mining at mm7 frontier',
        }

    if wr11 is None or float(wr11) < 0.50:
        return {
            'enable_loss_mining': True,
            'loss_mining_depth': 9,
            'loss_mining_games': 2,
            'loss_mining_positions_per_game': 6,
            'loss_mining_max_samples': 24,
            'loss_mining_reason': 'deeper mining near mm11 frontier',
        }

    return {
        'enable_loss_mining': True,
        'loss_mining_depth': 9,
        'loss_mining_games': 1,
        'loss_mining_positions_per_game': 4,
        'loss_mining_max_samples': 8,
        'loss_mining_reason': 'maintenance mining only; avoid overfitting late-stage checkpoints',
    }


def should_run_ts_verify(result: dict, enabled: bool) -> bool:
    if not enabled:
        return False
    mm11 = result.get('wr_vs_minimax11')
    opening_floor = result.get('wr_opening_floor')
    if mm11 is None or opening_floor is None:
        return False
    return float(mm11) >= TS_VERIFY_TRIGGER_MM11 and float(opening_floor) >= TS_VERIFY_TRIGGER_OPENING_FLOOR


def should_run_practical_verify(result: dict) -> bool:
    wr5 = result.get('wr_vs_minimax5')
    wr7 = result.get('wr_vs_minimax7')
    if wr5 is not None and float(wr5) >= PRACTICAL_VERIFY_TRIGGER_MM5:
        return True
    if wr7 is not None and float(wr7) >= PRACTICAL_VERIFY_TRIGGER_MM7:
        return True
    return False


def run_ts_verify(checkpoint_path: str, *, mm_depth: int, n_games: int, az_sims: int) -> dict:
    runner_ts = os.path.join(ROOT_DIR, 'battle_verify.ts')
    out_dir = os.path.join(ROOT_DIR, '..', 'tmp_battle_newaz')
    os.makedirs(out_dir, exist_ok=True)
    compiled_js = os.path.join(out_dir, 'newAz', 'battle_verify.js')

    compile_cmd = [
        'npx', 'tsc',
        '--module', 'commonjs',
        '--moduleResolution', 'node',
        '--target', 'es2019',
        '--outDir', out_dir,
        '--esModuleInterop', 'true',
        '--skipLibCheck', 'true',
        runner_ts,
        os.path.join(ROOT_DIR, '..', 'src', 'coreClaude', 'azFeatures.ts'),
        os.path.join(ROOT_DIR, '..', 'src', 'coreClaude', 'position.ts'),
        os.path.join(ROOT_DIR, '..', 'src', 'coreClaude', 'movegen.ts'),
        os.path.join(ROOT_DIR, '..', 'src', 'coreClaude', 'bitboards.ts'),
        os.path.join(ROOT_DIR, '..', 'src', 'coreClaude', 'eval.ts'),
    ]
    subprocess.run(compile_cmd, cwd=os.path.join(ROOT_DIR, '..'), check=True, capture_output=True, text=True)

    run_cmd = [
        'node',
        compiled_js,
        '--model', checkpoint_path,
        '--games', str(n_games),
        '--depth', str(mm_depth),
        '--sims', str(az_sims),
    ]
    proc = subprocess.run(run_cmd, cwd=os.path.join(ROOT_DIR, '..'), check=True, capture_output=True, text=True)
    output = proc.stdout

    match = re.search(r'Win rate:\s*([0-9.]+)%', output)
    ts_win_rate = float(match.group(1)) / 100.0 if match else None
    return {
        'win_rate': ts_win_rate,
        'games': n_games,
        'depth': mm_depth,
        'sims': az_sims,
        'raw_output': output,
    }


def evaluate_checkpoint(drive_dir: str, checkpoint_path: str) -> dict:
    models_dir = os.path.join(drive_dir, 'models')
    best_path = os.path.join(models_dir, 'best.pt')
    target_path = os.path.join(models_dir, 'target_best.pt')
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    candidate = load_network(checkpoint_path)
    suite = build_opening_suite()

    result = {
        'iter': int(checkpoint_name.split('_')[-1]) if checkpoint_name.startswith('iter_') else -1,
        'checkpoint': checkpoint_path,
        'checkpoint_name': checkpoint_name,
        'evaluated_at': int(time.time()),
        'device': str(DEVICE),
    }

    if os.path.exists(best_path) and os.path.abspath(best_path) != os.path.abspath(checkpoint_path):
        print(f'  vs best ({FULL_NET_GAMES} games)...', flush=True)
        best_net = load_network(best_path)
        result['wr_vs_best'] = round(eval_net_vs_net(candidate, best_net, FULL_NET_GAMES), 3)
        print(f'    wr_vs_best = {result["wr_vs_best"]:.1%}', flush=True)
    else:
        result['wr_vs_best'] = None

    print(f'  vs random ({FULL_RANDOM_GAMES} games)...', flush=True)
    result['wr_vs_random'] = round(eval_net_vs_random(candidate, FULL_RANDOM_GAMES), 3)
    print(f'    wr_vs_random = {result["wr_vs_random"]:.1%}', flush=True)

    mm_results = {}
    gated_at_depth = None
    run_mm11 = should_run_mm11(result['iter'])
    for depth, n_games in FULL_MINIMAX_PLAN:
        if depth == TARGET_MINIMAX_DEPTH and not run_mm11:
            print(f'  skipping mm{depth} for iter {result["iter"]:04d} (heavy eval every {HEAVY_MM11_EVERY} iterations)', flush=True)
            continue
        print(f'  vs mm{depth} ({n_games} games)...', flush=True)
        mm_results[depth] = round(eval_net_vs_minimax(candidate, n_games, depth), 3)
        result[f'wr_vs_minimax{depth}'] = mm_results[depth]
        print(f'    wr_vs_mm{depth} = {mm_results[depth]:.1%}', flush=True)
        gate = FULL_EVAL_GATES.get(depth)
        if gate is not None and mm_results[depth] < gate:
            gated_at_depth = depth
            print(f'  ⏭ gated at mm{depth} (wr {mm_results[depth]:.1%} < {gate:.0%}) — skipping deeper', flush=True)
            break

    result['gated_at_depth'] = gated_at_depth
    result['evaluated_minimax_depths'] = sorted(mm_results.keys())
    result['skipped_minimax_depths'] = [depth for depth, _ in FULL_MINIMAX_PLAN if depth not in mm_results]

    if run_mm11 and 7 in mm_results and mm_results[7] >= FULL_EVAL_GATES[7]:
        print(f'  opening suite...', flush=True)
        suite_metrics = eval_net_on_opening_suite(candidate, TARGET_MINIMAX_DEPTH, suite)
        result['wr_opening_suite'] = round(suite_metrics['overall'], 3)
        result['wr_opening_p1'] = round(suite_metrics['p1'], 3)
        result['wr_opening_p2'] = round(suite_metrics['p2'], 3)
        result['wr_opening_floor'] = round(min(suite_metrics['p1'], suite_metrics['p2']), 3)
    else:
        result['wr_opening_suite'] = None
        result['wr_opening_p1'] = None
        result['wr_opening_p2'] = None
        result['wr_opening_floor'] = None
        if not run_mm11:
            print(f'  opening suite skipped for iter {result["iter"]:04d} (paired with mm11-only heavy eval)', flush=True)

    target_baseline = {
        'wr_vs_minimax11': -1.0,
        'wr_opening_suite': -1.0,
        'wr_opening_floor': -1.0,
        'source': 'none',
    }
    target_status = load_target_status(drive_dir)
    if target_status:
        target_baseline['wr_vs_minimax11'] = float(target_status.get('wr_vs_minimax11', -1.0))
        target_baseline['wr_opening_suite'] = float(target_status.get('wr_opening_suite', -1.0))
        target_baseline['wr_opening_floor'] = float(target_status.get('wr_opening_floor', -1.0))
        target_baseline['source'] = 'target_status'
    elif os.path.exists(target_path) and os.path.abspath(target_path) != os.path.abspath(checkpoint_path):
        target_net = load_network(target_path)
        target_baseline['wr_vs_minimax11'] = round(eval_net_vs_minimax(target_net, 8, TARGET_MINIMAX_DEPTH), 3)
        target_suite = eval_net_on_opening_suite(target_net, TARGET_MINIMAX_DEPTH, suite)
        target_baseline['wr_opening_suite'] = round(target_suite['overall'], 3)
        target_baseline['wr_opening_floor'] = round(min(target_suite['p1'], target_suite['p2']), 3)
        target_baseline['source'] = 'target_best_recomputed'

    result['target_reference'] = target_baseline
    result['run_mm11_this_checkpoint'] = run_mm11
    return result


def make_decision(result: dict) -> dict:
    wr_best = result.get('wr_vs_best')
    wr_mm11 = float(result.get('wr_vs_minimax11', 0.0) or 0.0)
    open_suite = float(result.get('wr_opening_suite', 0.0) or 0.0)
    open_floor = float(result.get('wr_opening_floor', 0.0) or 0.0)
    target_ref = result.get('target_reference', {})
    target_wr = float(target_ref.get('wr_vs_minimax11', -1.0) or -1.0)
    target_suite = float(target_ref.get('wr_opening_suite', -1.0) or -1.0)
    target_floor = float(target_ref.get('wr_opening_floor', -1.0) or -1.0)
    gated_at_depth = result.get('gated_at_depth')
    ts_verify = result.get('ts_verify')
    practical_verify = result.get('practical_verify')
    ts_win_rate = None
    practical_win_rate = None
    if isinstance(ts_verify, dict) and ts_verify.get('win_rate') is not None:
        ts_win_rate = float(ts_verify['win_rate'])
    if isinstance(practical_verify, dict) and practical_verify.get('win_rate') is not None:
        practical_win_rate = float(practical_verify['win_rate'])

    promote_best = wr_best is not None and wr_best >= WIN_THRESHOLD
    promote_target = False
    if result.get('wr_vs_minimax11') is not None and result.get('wr_opening_floor') is not None:
        promote_target = (
            (wr_mm11 >= target_wr + TARGET_BEST_MARGIN and open_floor >= target_floor - 0.05) or
            (
                abs(wr_mm11 - target_wr) <= TARGET_BEST_MARGIN and
                (open_floor > target_floor + 0.02 or open_suite > target_suite + 0.01)
            )
        )
        if target_wr < 0.0:
            promote_target = wr_mm11 >= 0.50 and open_floor >= 0.45
        if ts_win_rate is not None and ts_win_rate < 0.50:
            promote_target = False

    summary = (
        f'best={wr_best if wr_best is not None else "n/a"} '
        f'mm11={wr_mm11:.3f} open={open_suite:.3f} floor={open_floor:.3f} '
        f'practical7={practical_win_rate if practical_win_rate is not None else "n/a"} '
        f'target_ref={target_wr:.3f}/{target_suite:.3f}/{target_floor:.3f} '
        f'gated_at={gated_at_depth} ts={ts_win_rate if ts_win_rate is not None else "n/a"}'
    )

    mining = recommend_loss_mining(result)

    return {
        'iter': result['iter'],
        'checkpoint': result['checkpoint'],
        'promote_best': promote_best,
        'promote_target': promote_target,
        'wr_vs_best': wr_best,
        'wr_vs_minimax11': wr_mm11,
        'wr_opening_suite': open_suite,
        'wr_opening_floor': open_floor,
        'practical_mm7_win_rate': practical_win_rate,
        'ts_verify_win_rate': ts_win_rate,
        'gated_at_depth': gated_at_depth,
        'summary': summary,
        **mining,
    }


def resolve_checkpoint_arg(drive_dir: str, checkpoint_arg: str) -> str:
    if os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if checkpoint_arg.endswith('.pt'):
        return os.path.join(drive_dir, 'models', checkpoint_arg)
    return os.path.join(drive_dir, 'models', f'{checkpoint_arg}.pt')


def process_checkpoint(drive_dir: str, checkpoint_path: str, *, ts_verify: bool, ts_games: int, ts_depth: int, ts_sims: int):
    external_dir = os.path.join(drive_dir, 'external_eval')
    results_dir = os.path.join(external_dir, 'results')
    decisions_dir = os.path.join(external_dir, 'decisions')
    result_log = os.path.join(external_dir, 'external_eval_log.jsonl')

    result = evaluate_checkpoint(drive_dir, checkpoint_path)
    if should_run_practical_verify(result):
        try:
            result['practical_verify'] = run_ts_verify(
                checkpoint_path,
                mm_depth=PRACTICAL_VERIFY_MM_DEPTH,
                n_games=PRACTICAL_VERIFY_GAMES,
                az_sims=PRACTICAL_VERIFY_SIMS,
            )
        except Exception as exc:
            result['practical_verify'] = {
                'error': str(exc),
                'games': PRACTICAL_VERIFY_GAMES,
                'depth': PRACTICAL_VERIFY_MM_DEPTH,
                'sims': PRACTICAL_VERIFY_SIMS,
            }
    if should_run_ts_verify(result, ts_verify):
        try:
            result['ts_verify'] = run_ts_verify(
                checkpoint_path,
                mm_depth=ts_depth,
                n_games=ts_games,
                az_sims=ts_sims,
            )
        except Exception as exc:
            result['ts_verify'] = {
                'error': str(exc),
                'games': ts_games,
                'depth': ts_depth,
                'sims': ts_sims,
            }
    decision = make_decision(result)
    checkpoint_name = result['checkpoint_name']

    result_path = os.path.join(results_dir, f'{checkpoint_name}.result.json')
    decision_path = os.path.join(decisions_dir, f'{checkpoint_name}.decision.json')

    write_json(result_path, result)
    write_json(decision_path, decision)
    append_jsonl(result_log, result)

    if decision['promote_target']:
        write_target_status(drive_dir, {
            'iter': result['iter'],
            'checkpoint': result['checkpoint'],
            'wr_vs_minimax11': decision['wr_vs_minimax11'],
            'wr_opening_suite': decision['wr_opening_suite'],
            'wr_opening_floor': decision['wr_opening_floor'],
            'updated_at': int(time.time()),
            'source': 'eval_local',
            'ts_verify': result.get('ts_verify'),
            'loss_mining_policy': {
                'enable_loss_mining': decision['enable_loss_mining'],
                'loss_mining_depth': decision['loss_mining_depth'],
                'loss_mining_games': decision['loss_mining_games'],
                'loss_mining_positions_per_game': decision['loss_mining_positions_per_game'],
                'loss_mining_max_samples': decision['loss_mining_max_samples'],
                'loss_mining_reason': decision['loss_mining_reason'],
            },
        })

    print(f'[done] {checkpoint_name}')
    print(f'  result   -> {result_path}')
    print(f'  decision -> {decision_path}')
    print(f'  summary  -> {decision["summary"]}')
    if result.get('practical_verify'):
        print(f'  practical verify -> {result["practical_verify"]}')
    if result.get('ts_verify'):
        print(f'  ts verify -> {result["ts_verify"]}')


def watch_requests(drive_dir: str, poll_seconds: int, once: bool, *, ts_verify: bool, ts_games: int, ts_depth: int, ts_sims: int):
    requests_dir = os.path.join(drive_dir, 'external_eval', 'requests')
    results_dir = os.path.join(drive_dir, 'external_eval', 'results')

    while True:
        request_paths = sorted(glob.glob(os.path.join(requests_dir, 'iter_*.request.json')))
        processed_any = False
        for request_path in request_paths:
            with open(request_path, encoding='utf-8') as f:
                request = json.load(f)
            checkpoint_path = request['checkpoint']
            # Translate Colab path to local drive path
            checkpoint_filename = os.path.basename(checkpoint_path)
            checkpoint_path = os.path.join(drive_dir, 'models', checkpoint_filename)
            checkpoint_name = os.path.splitext(checkpoint_filename)[0]
            result_path = os.path.join(results_dir, f'{checkpoint_name}.result.json')
            if os.path.exists(result_path):
                continue
            print(f'[eval] processing request {os.path.basename(request_path)}')
            process_checkpoint(
                drive_dir,
                checkpoint_path,
                ts_verify=ts_verify,
                ts_games=ts_games,
                ts_depth=ts_depth,
                ts_sims=ts_sims,
            )
            processed_any = True
            if once:
                return
        if once and not processed_any:
            return
        time.sleep(poll_seconds)


def main():
    args = parse_args()
    drive_dir = os.path.abspath(args.drive_dir)

    if args.checkpoint:
        process_checkpoint(
            drive_dir,
            resolve_checkpoint_arg(drive_dir, args.checkpoint),
            ts_verify=args.ts_verify,
            ts_games=args.ts_games,
            ts_depth=args.ts_depth,
            ts_sims=args.ts_sims,
        )
        return

    if args.watch or args.once:
        watch_requests(
            drive_dir,
            args.poll_seconds,
            args.once,
            ts_verify=args.ts_verify,
            ts_games=args.ts_games,
            ts_depth=args.ts_depth,
            ts_sims=args.ts_sims,
        )
        return

    raise SystemExit('Use --checkpoint <iter_XXXX.pt> or --watch or --once')


if __name__ == '__main__':
    main()
