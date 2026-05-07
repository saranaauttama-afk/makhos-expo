# Engine Phase C Repetition

## Purpose

Phase C adds minimal repetition contempt infrastructure only.

The goal is to isolate a configurable repetition-draw scoring path so contempt can be tested later without mixing it with unrelated search changes.

## Current Status

- Repetition contempt is present as infrastructure only
- Default is `OFF`
- Default behavior should remain unchanged while disabled

## Risks

- Repetition scoring changes can alter move choice even when the rest of search is unchanged
- Small score shifts around repetition nodes can ripple into root decisions
- Repetition contempt is easy to combine accidentally with other loop-avoidance ideas, which makes diagnosis harder

## Why Anti-Loop Ordering Is Intentionally Excluded

Anti-loop ordering is excluded on purpose in this phase because it changes search exploration order rather than only the repetition score path. That would make it harder to tell whether any future benchmark movement comes from contempt scoring itself or from move-order changes.

Phase C is meant to isolate only the draw-score infrastructure.

## How It Is Wired

- File: `src/coreClaude/search/alphabeta.ts`
- Flag: `ENABLE_REPETITION_CONTEMPT`
- Configurable value: `REPETITION_CONTEMPT_CP`
- Isolated helper: `repetitionDrawScore(...)`

When the flag is disabled, `repetitionDrawScore(...)` returns the legacy draw score `0`.

## How To Enable Contempt Experimentally Later

When Phase C experiments are approved later:

1. turn `ENABLE_REPETITION_CONTEMPT` on
2. set `REPETITION_CONTEMPT_CP` to the intended test value
3. run perft first
4. run tactical benchmark comparison against the stable tag baseline

## Guardrail

This phase is infrastructure-only and should not alter benchmark behavior while disabled.
