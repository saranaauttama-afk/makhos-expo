# Phase 3D soundForcedTrap confirmation protocol

This protocol was frozen before tournament results were inspected. The
hypothesis is that disabling only `soundForcedTrap` improves frozen Teacher v1
(`b2e6a35db6a50ea294a10f6b76a90b4e70e0689f`). The baseline explicitly pins
`smallEndgame=false` and `soundForcedTrap=true`; the candidate pins the same
complete extension configuration except `soundForcedTrap=false`. Production
defaults, evaluation, pruning, reductions, null move, ProbCut, IID, move
ordering, and `singleCaptureRecapture` are out of scope.

## Frozen corpus and generator

* Suite: `makhos-phase3d-sound-forced-trap-confirmation-starts-v1`
* Seed: `0x075cd5bf` / `123524543`, the first eight hexadecimal digits of
  canonical branch tip `075cd5bfb69eabac4555eef6d048fa35700efadb`.
* SHA-256: `d68b1b8f33b3f4e7b490e44ff8bd2c4576ba11b14c81a52a987471598cb1ab6d`.
* Generator: xorshift32, lexically sorted legal moves, cyclic opening lengths
  2-17 plies, generator v1. Generate 320 unique legal states, remove complete
  state identities present in every Phase 3A, 3B, superseded Phase 3C, and
  final Phase 3C corpus, then take the first 64. IDs and replay moves are part
  of the fingerprint. No regeneration or alternate seed is permitted.

## Decision protocol

Use the first 32 paired starts, paired colors, 20,000 and 50,000 nodes/move,
100 ms/move, depth cap 64, and maximum 160 played plies. Report candidate
W/D/L, score, Elo and deterministic 20,000-resample pair-bootstrap 95% CI,
main/qnodes, completed depth, elapsed time, NPS, unresolved/errors, and subtype
trigger/added-depth counters. Fixed-node evidence is canonical; equal-time is
production-like descriptive evidence.

Promote the hypothesis only if correctness gates pass, the revealed verified
tactical/holdout/endgame suite has no regression, independent tournament
evidence supports the candidate, and preferably both fixed-work and equal-time
intervals are wholly above zero Elo. Phase 3B's roughly +100 Elo observation is
screening context only. Keep holdout output blinded until configurations,
corpus, and this decision rule are frozen and the strength runs are complete.
This experiment does not change the production default.
