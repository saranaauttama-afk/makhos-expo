"""Test augmentation logic to find bugs"""
import json
import random

def _flip_features(features):
    """Flip board features for side symmetry augmentation"""
    flipped = features[:]  # Make a copy using slice notation
    # Flip positional features [0-127]: my_men, my_kings, op_men, op_kings
    for layer in range(4):
        for sq in range(32):
            flipped[layer * 32 + sq] = features[layer * 32 + (31 - sq)]
    # Flip mobility features [128-191]
    for layer in range(2):
        for sq in range(32):
            flipped[128 + layer * 32 + sq] = features[128 + layer * 32 + (31 - sq)]
    # Flip threat maps [192-255]
    for layer in range(2):
        for sq in range(32):
            flipped[192 + layer * 32 + sq] = features[192 + layer * 32 + (31 - sq)]
    # Flip hanging flags [256-287]
    for sq in range(32):
        flipped[256 + sq] = features[256 + (31 - sq)]
    # Flip distance to promotion [288-319]
    for sq in range(32):
        flipped[288 + sq] = features[288 + (31 - sq)]
    return flipped

def _flip_move(from_sq, to_sq):
    """Flip move indices for side symmetry"""
    return (31 - from_sq, 31 - to_sq)

# Load one example
data_path = '.tmp/training_data_with_features/chunk_selfplay_0.json'
with open(data_path, 'r') as f:
    data = json.load(f)

# Test on first example
example = data[0]
print("Original example:")
print(f"  bestMove: {example['bestMove']['from']} -> {example['bestMove']['to']}")
print(f"  move_idx: {example['bestMove']['from'] * 32 + example['bestMove']['to']}")
print(f"  value: {example['positionValue']}")

# Print first few features (my_men layer, first 8 squares)
print(f"  features[0:8]: {example['features'][0:8]}")

# Flip
flipped_features = _flip_features(example['features'])
from_sq, to_sq = _flip_move(example['bestMove']['from'], example['bestMove']['to'])
flipped_move_idx = from_sq * 32 + to_sq
flipped_value = -example['positionValue']

print("\nFlipped example:")
print(f"  bestMove: {from_sq} -> {to_sq}")
print(f"  move_idx: {flipped_move_idx}")
print(f"  value: {flipped_value}")
print(f"  features[0:8]: {flipped_features[0:8]}")

# Verify flip is correct
# If original has piece at sq 0, flipped should have piece at sq 31
print("\nVerifying flip correctness:")
print(f"  Original sq 0: {example['features'][0]}")
print(f"  Flipped sq 31: {flipped_features[31]}")
print(f"  Should match: {example['features'][0] == flipped_features[31]}")

print(f"\n  Original sq 31: {example['features'][31]}")
print(f"  Flipped sq 0: {flipped_features[0]}")
print(f"  Should match: {example['features'][31] == flipped_features[0]}")

# Double flip should give original
double_flipped = _flip_features(flipped_features)
from_sq2, to_sq2 = _flip_move(from_sq, to_sq)
print(f"\nDouble flip test:")
print(f"  Original move: {example['bestMove']['from']} -> {example['bestMove']['to']}")
print(f"  Double flipped move: {from_sq2} -> {to_sq2}")
print(f"  Moves match: {example['bestMove']['from'] == from_sq2 and example['bestMove']['to'] == to_sq2}")
print(f"  Features match: {example['features'] == double_flipped}")
