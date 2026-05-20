"""Quick test to verify training augmentation works correctly"""
import sys
import json
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load one example from training data
with open('.tmp/training_data_with_features/chunk_selfplay_0.json', 'r') as f:
    data = json.load(f)
    example = data[0]

print("Original example:")
print(f"  bestMove: {example['bestMove']['from']} -> {example['bestMove']['to']}")
print(f"  value: {example['positionValue']}")
print(f"  features[0:8]: {example['features'][0:8]}")
print(f"  features[64:72] (op_men): {example['features'][64:72]}")

# Simulate augmentation logic from train.py
def _flip_features(features):
    """Flip board features for P1/P2 side symmetry augmentation"""
    flipped = features[:]  # Make a copy

    # Flip positional features [0-127]: my_men, my_kings, op_men, op_kings
    # Swap my pieces <-> opponent pieces
    for layer in range(4):
        for sq in range(32):
            # Swap layers: 0↔2 (my_men↔op_men), 1↔3 (my_kings↔op_kings)
            target_layer = 2 + layer if layer < 2 else layer - 2
            flipped[target_layer * 32 + (31 - sq)] = features[layer * 32 + sq]

    # Flip mobility features [128-191]: my_mobility, op_mobility
    for layer in range(2):
        for sq in range(32):
            # Swap layers: 0↔1 (my_mobility↔op_mobility)
            target_layer = 1 - layer
            flipped[128 + target_layer * 32 + (31 - sq)] = features[128 + layer * 32 + sq]

    # Flip threat maps [192-255]: my_threats, op_threats
    for layer in range(2):
        for sq in range(32):
            # Swap layers: 0↔1 (my_threats↔op_threats)
            target_layer = 1 - layer
            flipped[192 + target_layer * 32 + (31 - sq)] = features[192 + layer * 32 + sq]

    # Flip hanging flags [256-287]: hanging pieces
    for sq in range(32):
        flipped[256 + (31 - sq)] = features[256 + sq]

    # Flip distance to promotion [288-319]: promotion distance
    for sq in range(32):
        # Distance to promotion also inverts (distance from bottom → distance from top)
        # 7 - distance because board height is 8
        original_dist = features[288 + sq]
        flipped[288 + (31 - sq)] = 7 - original_dist if original_dist > 0 else 0

    return flipped

def _flip_move(from_sq, to_sq):
    """Flip move indices for P1/P2 side symmetry"""
    return (31 - from_sq, 31 - to_sq)

# Test augmentation
features = example['features'][:]
flipped_features = _flip_features(features)
from_sq, to_sq = _flip_move(example['bestMove']['from'], example['bestMove']['to'])
flipped_value = -example['positionValue']

print("\nFlipped example:")
print(f"  bestMove: {from_sq} -> {to_sq}")
print(f"  value: {flipped_value}")
print(f"  features[0:8] (now op_men after swap): {flipped_features[0:8]}")
print(f"  features[64:72] (now my_men after swap): {flipped_features[64:72]}")

# Verify swap: original my_men should become flipped op_men
print("\nVerifying layer swap:")
print(f"  Original my_men[0]: {features[0]}")
print(f"  Flipped op_men[31]: {flipped_features[64 + 31]}")
print(f"  Should be same after position flip: {features[0] == flipped_features[64 + 31]}")

print(f"\n  Original op_men[31]: {features[64 + 31]}")
print(f"  Flipped my_men[0]: {flipped_features[0]}")
print(f"  Should be same after position flip: {features[64 + 31] == flipped_features[0]}")

# Double flip should restore original
double_flipped = _flip_features(flipped_features)
from_sq2, to_sq2 = _flip_move(from_sq, to_sq)
print(f"\nDouble flip test:")
print(f"  Original move: {example['bestMove']['from']} -> {example['bestMove']['to']}")
print(f"  Double flipped move: {from_sq2} -> {to_sq2}")
print(f"  Moves match: {example['bestMove']['from'] == from_sq2 and example['bestMove']['to'] == to_sq2}")
print(f"  Features match: {features == double_flipped}")
print(f"  Value match: {example['positionValue'] == -flipped_value}")

print("\n✅ Augmentation test complete!")
