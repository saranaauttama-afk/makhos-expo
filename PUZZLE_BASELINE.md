# Puzzle Suite Baseline Results

**Date**: 2026-05-13
**Branch**: tuneClaude
**Commit**: After opening book implementation

## Summary

Testing AI tactical strength against 14 Thai Checkers puzzles (หมากกล).

### Overall Performance
- **Correct**: 1/14 (7%)
- **Solved**: 14/14 (100%) - AI found moves, but mostly wrong
- **Average Time**: 7.5s per puzzle
- **Average Depth**: 3.2

### By Difficulty
| Difficulty | Correct | Total | Percentage |
|------------|---------|-------|------------|
| Easy       | 1       | 3     | 33%        |
| Medium     | 0       | 6     | 0%         |
| Hard       | 0       | 4     | 0%         |
| Expert     | 0       | 1     | 0%         |

### By Puzzle Type
| Type                    | Puzzles | Correct |
|-------------------------|---------|---------|
| Forced Capture Trap     | 2       | 0       |
| Promotion Race          | 2       | 1       |
| Sacrifice Combination   | 2       | 0       |
| Endgame Technique       | 2       | 0       |
| Escape from Trap        | 2       | 0       |
| King vs Men             | 2       | 0       |
| Tempo Gain              | 2       | 0       |

## Successful Puzzle

✓ **promo-01-race-to-king** (easy)
- Type: Promotion Race
- Expected: 10->6
- AI chose: 10->6 ✓
- Time: 116ms, Depth: 12
- Note: Straightforward promotion - AI handles simple tactical goals well

## Failed Puzzles (13/14)

All other puzzles failed, showing significant tactical weaknesses:

### Forced Capture Traps (0/2)
- **trap-01-bait-sacrifice**: Expected 26->23, chose 26->22
- **trap-02-double-bait**: Expected 27->23, chose 21->16

### Promotion Races (0/1)
- **promo-02-sacrifice-for-promotion**: Expected 14->10, chose 10->7

### King vs Men Endgames (0/2)
- **endgame-01-king-vs-two**: Expected 18->14, chose 18->16
- **endgame-02-king-vs-three**: Expected 5->9, chose 6->10

### Sacrifice Combinations (0/2)
- **sac-01-piece-for-position**: Expected 25->22, chose 22->15
- **sac-02-two-for-promotion**: Expected 14->10, chose 10->1

### Escape from Trap (0/2)
- **escape-01-find-the-exit**: Expected 29->25, chose 25->21
- **escape-02-counter-trap**: Expected 30->26, chose 21->16

### Tempo Gain (0/2)
- **tempo-01-force-response**: Expected 18->14, chose 14->7
- **tempo-02-zugzwang**: Expected 10->6, chose 14->23

### Endgame Techniques (0/2)
- **endgame-03-opposition**: Expected 18->14, chose 18->9
- **endgame-04-triangulation**: Expected 18->22, chose 18->0

## Key Insights

1. **Material Blindness**: AI struggles with sacrifices for positional/tactical advantage
2. **Trap Recognition**: Cannot see bait-and-recapture patterns
3. **Endgame Weakness**: Poor at king vs men techniques (opposition, triangulation)
4. **Tempo Unawareness**: Doesn't understand zugzwang or forced response patterns
5. **Escape Failure**: Cannot find precise defensive moves in squeezed positions

## Comparison to Benchmark Performance

**Benchmark (Expert level)**:
- Solve rate: 100%
- Blunder rate: 0%
- Protected cases: All passing

**Puzzle Suite**:
- Solve rate: 7%
- Most puzzles failed

**Interpretation**:
Current AI excels at **positional play** and **standard tactics** in full games, but severely lacks **advanced tactical pattern recognition** required for puzzles.

## Future Improvement Targets

To improve puzzle performance, consider:

1. **Trap pattern database**: Add specific trap recognition
2. **Sacrifice evaluation**: Better eval of positional compensation
3. **Endgame tablebases**: Expand coverage for king vs men
4. **Deeper search**: Some puzzles require 6-8 ply to see solution
5. **Specialized puzzle mode**: Different search parameters for puzzle-solving

## Value of Puzzle Suite

This baseline establishes a **tactical weakness profile**:
- ✓ Strong in full-game play (expert 100/0)
- ✗ Weak in isolated tactical puzzles (7%)

The puzzle suite serves as:
- **Regression detector**: Any changes should maintain or improve 7% baseline
- **Improvement metric**: Future work can be measured against this baseline
- **Weakness identifier**: Shows exactly what tactical patterns need work

---

**Note**: Puzzles are intentionally challenging - 7% is a realistic baseline for a hand-crafted eval engine. Neural network approaches (like CU_Makhos) likely perform better on these pattern-recognition tasks.
