"""
Step 1: Generate training data from AI vs AI self-play games

Workflow:
1. Run this script on Colab to generate training data (10,000 games in batches)
2. Download the training_data.npz file
3. Use that file for training with train.py

Example:
    python gen_data.py --total_games 10000 --batch_size 2000 --time_per_move 500
"""

import json
import numpy as np
import subprocess
import os
import time
import glob
from typing import List, Tuple

def run_game_generator_batch(batch_idx: int, num_games: int, time_per_move: int, output_dir: str = "."):
    """
    Run the TypeScript game generator for one batch

    Args:
        batch_idx: Batch number (for naming)
        num_games: Number of games in this batch
        time_per_move: Time in milliseconds for each AI move
        output_dir: Directory to save the raw game data

    Returns:
        output_file path if successful, None otherwise
    """
    output_file = os.path.join(output_dir, f"games_batch_{batch_idx:04d}.json")

    print(f"\n{'='*60}")
    print(f"Batch {batch_idx}: Generating {num_games} games...")
    print(f"{'='*60}")

    # Find project root (parent of ml directory)
    try:
        script_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(script_dir)
    except NameError:
        # In notebook: go up from current directory
        script_dir = os.getcwd()
        project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'ml' else script_dir

    script_path = os.path.join(project_root, "scripts", "generate_games.ts")

    # Make output_file absolute
    if not os.path.isabs(output_file):
        output_file = os.path.abspath(output_file)

    print(f"  Script: {script_path}")
    print(f"  Output: {output_file}")

    # Run TypeScript using tsx
    # For Colab: use full path to avoid /tools/node conflict
    home = os.path.expanduser("~")
    node_bin = f"{home}/.nvm/versions/node/v20.19.5/bin"
    tsx_cmd = f"{node_bin}/npx tsx {script_path} {num_games} {time_per_move} {output_file}"

    start_time = time.time()
    try:
        # Run from project root directory with modified PATH
        env = os.environ.copy()
        env["PATH"] = f"{node_bin}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["bash", "-c", tsx_cmd],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            env=env
        )
        elapsed = time.time() - start_time
        print(f"✓ Batch {batch_idx} complete in {elapsed/60:.1f} minutes")
        print(f"  Saved to: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running batch {batch_idx}:")
        print(f"  Exit code: {e.returncode}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return None

def generate_in_batches(total_games: int, batch_size: int, time_per_move: int, output_dir: str = "."):
    """
    Generate games in batches with progress tracking

    Args:
        total_games: Total number of games to generate
        batch_size: Games per batch
        time_per_move: Time per move in milliseconds
        output_dir: Directory for output files

    Returns:
        List of batch file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing batches (for resume)
    existing_batches = sorted(glob.glob(os.path.join(output_dir, "games_batch_*.json")))
    if existing_batches:
        print(f"\nFound {len(existing_batches)} existing batch files.")
        resume = input("Resume from checkpoint? (y/n): ").lower().strip() == 'y'
        if resume:
            print("Resuming from existing batches...")
            start_batch = len(existing_batches)
        else:
            print("Starting fresh (existing batches will be kept)...")
            start_batch = 0
    else:
        start_batch = 0

    num_batches = (total_games + batch_size - 1) // batch_size
    batch_files = list(existing_batches) if start_batch > 0 else []

    print(f"\n{'='*60}")
    print(f"GENERATION PLAN")
    print(f"{'='*60}")
    print(f"Total games: {total_games}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"Starting from batch: {start_batch}")
    print(f"Time per move: {time_per_move}ms")
    print(f"{'='*60}\n")

    overall_start = time.time()

    for batch_idx in range(start_batch, num_batches):
        games_in_batch = min(batch_size, total_games - batch_idx * batch_size)

        batch_file = run_game_generator_batch(batch_idx, games_in_batch, time_per_move, output_dir)

        if batch_file:
            batch_files.append(batch_file)

            # Progress summary
            games_done = (batch_idx + 1) * batch_size
            games_done = min(games_done, total_games)
            progress = games_done / total_games * 100
            elapsed = time.time() - overall_start
            avg_time_per_batch = elapsed / (batch_idx - start_batch + 1)
            remaining_batches = num_batches - batch_idx - 1
            eta_seconds = avg_time_per_batch * remaining_batches

            print(f"\n{'─'*60}")
            print(f"PROGRESS: {games_done}/{total_games} games ({progress:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} min")
            if remaining_batches > 0:
                print(f"ETA: {eta_seconds/60:.1f} min ({eta_seconds/3600:.1f} hours)")
            print(f"{'─'*60}\n")
        else:
            print(f"\n✗ Batch {batch_idx} failed. Stopping.")
            break

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Batches completed: {len(batch_files)}/{num_batches}")
    print(f"{'='*60}\n")

    return batch_files

def merge_batch_files(batch_files: List[str]) -> str:
    """
    Merge all batch JSON files into a single file

    Args:
        batch_files: List of batch file paths

    Returns:
        Path to merged file
    """
    print(f"\n{'='*60}")
    print(f"MERGING BATCHES")
    print(f"{'='*60}")

    all_games = []
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            games = json.load(f)
            all_games.extend(games)
            print(f"  Loaded {len(games)} games from {os.path.basename(batch_file)}")

    merged_file = "games_data_merged.json"
    with open(merged_file, 'w') as f:
        json.dump(all_games, f)

    print(f"✓ Merged {len(all_games)} total games into {merged_file}")
    print(f"{'='*60}")
    return merged_file

def position_to_planes(state_array: List[int]) -> np.ndarray:
    """
    Convert position array to neural network input planes

    Input format: [p1Men, p1Kings, p2Men, p2Kings, side, halfmoveClock]
    Output: 6 planes of 32 bits each:
      - plane 0: p1Men
      - plane 1: p1Kings
      - plane 2: p2Men
      - plane 3: p2Kings
      - plane 4: side to move (all 1s for P1, all 0s for P2)
      - plane 5: halfmove clock (normalized 0-1)

    Returns: shape (6, 32) binary array
    """
    p1_men = state_array[0]
    p1_kings = state_array[1]
    p2_men = state_array[2]
    p2_kings = state_array[3]
    side = state_array[4]
    halfmove = state_array[5]

    planes = np.zeros((6, 32), dtype=np.float32)

    # Convert bitboards to binary arrays
    for i in range(32):
        planes[0, i] = (p1_men >> i) & 1
        planes[1, i] = (p1_kings >> i) & 1
        planes[2, i] = (p2_men >> i) & 1
        planes[3, i] = (p2_kings >> i) & 1
        planes[4, i] = 1.0 if side == 1 else 0.0
        planes[5, i] = min(halfmove / 20.0, 1.0)  # normalize by draw threshold

    return planes

def legal_moves_to_mask(legal_moves: List[List[int]]) -> np.ndarray:
    """
    Convert legal moves to binary mask

    Args:
        legal_moves: List of [from, to, num_captured, promote]

    Returns:
        (32, 32) binary mask where mask[from][to] = 1 for legal moves
    """
    mask = np.zeros((32, 32), dtype=np.float32)
    for move in legal_moves:
        from_sq = move[0]
        to_sq = move[1]
        mask[from_sq, to_sq] = 1.0
    return mask

def process_games(games_file: str = "games_data.json") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process raw game data into training dataset

    Returns:
        states: (N, 6, 32) array of board states
        policy_targets: (N, 32, 32) array of policy targets (from search)
        legal_masks: (N, 32, 32) array of legal move masks
        values: (N,) array of game outcomes from current player's perspective
        search_scores: (N,) array of search evaluation scores
        evaluations: (N,) array of static evaluation scores
    """
    print(f"\nLoading games from {games_file}...")

    with open(games_file, 'r') as f:
        games = json.load(f)

    print(f"\n{'='*60}")
    print(f"PROCESSING GAMES")
    print(f"{'='*60}")
    print(f"Total games: {len(games)}")
    print(f"{'='*60}\n")

    states_list = []
    policy_targets_list = []
    legal_masks_list = []
    values_list = []
    search_scores_list = []
    evaluations_list = []

    total_positions = 0
    for game_idx, game in enumerate(games):
        positions = game['positions']
        result = game['result']  # 1 = P1 wins, -1 = P2 wins, 0 = draw

        for pos_data in positions:
            state_array = pos_data['state']
            legal_moves = pos_data['legalMoves']
            policy_target = pos_data['policyTarget']
            search_score = pos_data['searchScore']
            evaluation = pos_data['evaluation']

            # Convert position to input planes
            state = position_to_planes(state_array)

            # Convert policy target to 32x32 array
            policy_array = np.array(policy_target, dtype=np.float32).reshape(32, 32)

            # Create legal moves mask
            legal_mask = legal_moves_to_mask(legal_moves)

            # Game outcome from current player's perspective
            side = state_array[4]
            value = result * side  # Flip result based on whose turn it is

            states_list.append(state)
            policy_targets_list.append(policy_array)
            legal_masks_list.append(legal_mask)
            values_list.append(value)
            search_scores_list.append(search_score * side)  # Flip score too
            evaluations_list.append(evaluation)

            total_positions += 1

        if (game_idx + 1) % 100 == 0:
            print(f"  Processed {game_idx + 1}/{len(games)} games, {total_positions} positions...")

    states = np.array(states_list, dtype=np.float32)
    policy_targets = np.array(policy_targets_list, dtype=np.float32)
    legal_masks = np.array(legal_masks_list, dtype=np.float32)
    values = np.array(values_list, dtype=np.float32)
    search_scores = np.array(search_scores_list, dtype=np.float32)
    evaluations = np.array(evaluations_list, dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"DATASET CREATED")
    print(f"{'='*60}")
    print(f"  States: {states.shape}")
    print(f"  Policy targets: {policy_targets.shape}")
    print(f"  Legal masks: {legal_masks.shape}")
    print(f"  Values: {values.shape}")
    print(f"  Search scores: {search_scores.shape}")
    print(f"  Evaluations: {evaluations.shape}")
    print(f"\nValue distribution:")
    print(f"  P1 wins (+1): {(values == 1).sum():,} ({(values == 1).sum()/len(values)*100:.1f}%)")
    print(f"  Draws (0):    {(values == 0).sum():,} ({(values == 0).sum()/len(values)*100:.1f}%)")
    print(f"  P2 wins (-1): {(values == -1).sum():,} ({(values == -1).sum()/len(values)*100:.1f}%)")
    print(f"\nAverage legal moves per position: {legal_masks.sum(axis=(1,2)).mean():.1f}")
    print(f"{'='*60}")

    return states, policy_targets, legal_masks, values, search_scores, evaluations

def save_dataset(
    states: np.ndarray,
    policy_targets: np.ndarray,
    legal_masks: np.ndarray,
    values: np.ndarray,
    search_scores: np.ndarray,
    evaluations: np.ndarray,
    output_path: str = "training_data.npz"
):
    """Save processed dataset to npz file"""
    np.savez_compressed(
        output_path,
        states=states,
        policy_targets=policy_targets,
        legal_masks=legal_masks,
        values=values,
        search_scores=search_scores,
        evaluations=evaluations
    )

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024

    # Calculate data statistics
    total_size = (
        states.nbytes +
        policy_targets.nbytes +
        legal_masks.nbytes +
        values.nbytes +
        search_scores.nbytes +
        evaluations.nbytes
    )
    uncompressed_mb = total_size / 1024 / 1024
    compression_ratio = total_size / os.path.getsize(output_path)

    print(f"\n{'='*60}")
    print(f"DATASET SAVED")
    print(f"{'='*60}")
    print(f"  File: {output_path}")
    print(f"  Compressed size: {file_size_mb:.2f} MB")
    print(f"  Uncompressed size: {uncompressed_mb:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"{'='*60}")

def main():
    """Main data generation pipeline"""
    import argparse
    import sys

    # Jupyter/Colab compatibility: filter out kernel arguments
    if any('kernel' in arg.lower() for arg in sys.argv):
        sys.argv = [sys.argv[0]]  # Keep only script name

    parser = argparse.ArgumentParser(description="Generate training data from AI vs AI games")
    parser.add_argument("--total_games", type=int, default=5000, help="Total number of games to generate")
    parser.add_argument("--batch_size", type=int, default=1000, help="Games per batch")
    parser.add_argument("--time_per_move", type=int, default=1000, help="Time per move in milliseconds (1000-1200 recommended for quality)")
    parser.add_argument("--output", type=str, default="training_data.npz", help="Output dataset file")
    parser.add_argument("--skip_generation", action="store_true", help="Skip game generation (process existing batches)")
    parser.add_argument("--batch_dir", type=str, default="game_batches", help="Directory for batch files")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MAKHOS DATA GENERATION PIPELINE")
    print(f"{'='*60}\n")

    # Step 1: Generate games in batches
    if not args.skip_generation:
        batch_files = generate_in_batches(args.total_games, args.batch_size, args.time_per_move, args.batch_dir)
        if not batch_files:
            print("\n✗ No games were generated. Exiting.")
            return
    else:
        print("Skipping generation, loading existing batches...")
        batch_files = sorted(glob.glob(os.path.join(args.batch_dir, "games_batch_*.json")))
        if not batch_files:
            print(f"✗ No batch files found in {args.batch_dir}. Exiting.")
            return
        print(f"Found {len(batch_files)} batch files.")

    # Step 2: Merge batches
    merged_file = merge_batch_files(batch_files)

    # Step 3: Process games into training data
    states, policy_targets, legal_masks, values, search_scores, evaluations = process_games(merged_file)

    # Step 4: Save dataset
    save_dataset(states, policy_targets, legal_masks, values, search_scores, evaluations, args.output)

    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")
    print(f"Dataset ready: {args.output}")
    print(f"Use in training with: python train.py --data {args.output}")
    print(f"{'='*60}\n")

def generate_data(total_games=5000, batch_size=1000, time_per_move=1000, output="training_data.npz", skip_generation=False, batch_dir="game_batches"):
    """
    Helper function for Jupyter/Colab - call directly without argparse

    Example:
        generate_data(total_games=1000, batch_size=500, time_per_move=1000)
    """
    print(f"\n{'='*60}")
    print(f"MAKHOS DATA GENERATION PIPELINE")
    print(f"{'='*60}\n")

    # Step 1: Generate games in batches
    if not skip_generation:
        batch_files = generate_in_batches(total_games, batch_size, time_per_move, batch_dir)
        if not batch_files:
            print("\n✗ No games were generated. Exiting.")
            return
    else:
        print("Skipping generation, loading existing batches...")
        batch_files = sorted(glob.glob(os.path.join(batch_dir, "games_batch_*.json")))
        if not batch_files:
            print(f"✗ No batch files found in {batch_dir}. Exiting.")
            return
        print(f"Found {len(batch_files)} batch files.")

    # Step 2: Merge batches
    merged_file = merge_batch_files(batch_files)

    # Step 3: Process games into training data
    states, policy_targets, legal_masks, values, search_scores, evaluations = process_games(merged_file)

    # Step 4: Save dataset
    save_dataset(states, policy_targets, legal_masks, values, search_scores, evaluations, output)

    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")
    print(f"Dataset ready: {output}")
    print(f"Use in training with: python train.py --data {output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()