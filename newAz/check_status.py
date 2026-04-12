"""
Simple status viewer for the newAz split pipeline.

Usage:
  python newAz/check_status.py --drive-dir "D:\\GoogleDrive\\makhos_az_v5"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description='Show an easy-to-read status summary for the newAz pipeline.')
    parser.add_argument('--drive-dir', required=True, help='Shared training folder, e.g. Google Drive Desktop path.')
    return parser.parse_args()


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_last_jsonl(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    last_line = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    return json.loads(last_line) if last_line else None


def iter_from_name(path: str) -> Optional[int]:
    name = os.path.basename(path)
    try:
        return int(name.split('_')[-1].split('.')[0])
    except Exception:
        return None


def summarize_set(paths: list[str]) -> tuple[list[int], Optional[int]]:
    iters = sorted(i for i in (iter_from_name(path) for path in paths) if i is not None)
    return iters, (iters[-1] if iters else None)


def fmt_iter(value: Optional[int]) -> str:
    return f'iter_{value:04d}' if value is not None else 'none'


def main():
    args = parse_args()
    drive_dir = os.path.abspath(args.drive_dir)

    training_log = os.path.join(drive_dir, 'training_log.jsonl')
    latest_train = load_last_jsonl(training_log)

    requests_dir = os.path.join(drive_dir, 'external_eval', 'requests')
    results_dir = os.path.join(drive_dir, 'external_eval', 'results')
    decisions_dir = os.path.join(drive_dir, 'external_eval', 'decisions')
    target_status_path = os.path.join(drive_dir, 'external_eval', 'target_status.json')

    request_paths = glob.glob(os.path.join(requests_dir, 'iter_*.request.json'))
    result_paths = glob.glob(os.path.join(results_dir, 'iter_*.result.json'))
    decision_paths = glob.glob(os.path.join(decisions_dir, 'iter_*.decision.json'))

    request_iters, latest_request = summarize_set(request_paths)
    result_iters, latest_result = summarize_set(result_paths)
    decision_iters, latest_decision = summarize_set(decision_paths)

    pending_iters = sorted(set(request_iters) - set(result_iters))
    undecided_iters = sorted(set(result_iters) - set(decision_iters))

    latest_decision_payload = None
    if latest_decision is not None:
        latest_decision_payload = load_json(
            os.path.join(decisions_dir, f'iter_{latest_decision:04d}.decision.json')
        )

    target_status = load_json(target_status_path)

    print('=== newAz Status ===')
    print()

    if latest_train:
        print('Training')
        print(f'  latest iter     : {latest_train.get("iter", "n/a")}')
        print(f'  p_loss / v_loss : {latest_train.get("p_loss", "n/a")} / {latest_train.get("v_loss", "n/a")}')
        print(f'  buffer          : {latest_train.get("buffer_size", "n/a")}')
        print(f'  lr              : {latest_train.get("lr", "n/a")}')
        print(f'  elapsed_s       : {latest_train.get("elapsed_s", "n/a")}')
        print()
    else:
        print('Training')
        print('  no training_log.jsonl found yet')
        print()

    print('External Eval Queue')
    print(f'  latest request  : {fmt_iter(latest_request)}')
    print(f'  latest result   : {fmt_iter(latest_result)}')
    print(f'  latest decision : {fmt_iter(latest_decision)}')
    print(f'  pending eval    : {", ".join(fmt_iter(i) for i in pending_iters[:8]) if pending_iters else "none"}')
    print(f'  no decision yet : {", ".join(fmt_iter(i) for i in undecided_iters[:8]) if undecided_iters else "none"}')
    print()

    if latest_decision_payload:
        print('Latest Decision')
        print(f'  iter            : {latest_decision_payload.get("iter", "n/a")}')
        print(f'  promote_best    : {latest_decision_payload.get("promote_best", False)}')
        print(f'  promote_target  : {latest_decision_payload.get("promote_target", False)}')
        print(f'  ts_verify       : {latest_decision_payload.get("ts_verify_win_rate", "n/a")}')
        print(f'  summary         : {latest_decision_payload.get("summary", "")}')
        print()

    print('Target Status')
    if target_status:
        print(f'  iter            : {target_status.get("iter", "n/a")}')
        print(f'  checkpoint      : {os.path.basename(str(target_status.get("checkpoint", "")))}')
        print(f'  mm11            : {target_status.get("wr_vs_minimax11", "n/a")}')
        print(f'  opening_suite   : {target_status.get("wr_opening_suite", "n/a")}')
        print(f'  opening_floor   : {target_status.get("wr_opening_floor", "n/a")}')
        ts_verify = target_status.get('ts_verify')
        if isinstance(ts_verify, dict):
            print(f'  ts_verify       : {ts_verify.get("win_rate", "n/a")}')
    else:
        print('  no target_status.json yet')
    print()

    print('Quick Read')
    if pending_iters:
        print('  local eval is behind Colab; there are pending requests')
    elif request_iters and not result_iters:
        print('  Colab is creating requests, but local evaluator has not produced results yet')
    elif result_iters and not decision_iters:
        print('  local eval produced results, but no decision files were written yet')
    elif latest_decision is not None:
        print('  pipeline handshake is working')
    else:
        print('  waiting for first checkpoint / request')


if __name__ == '__main__':
    main()
