# Phase 3B extension subtype attribution protocol

## Frozen design and controls

Phase 3B keeps the Phase 3A `SearchFeatureFlags` inventory unchanged. `extensions`
remains the backward-compatible master switch. `ExtensionFeatureFlags` is a
separate namespace; every subtype defaults on, the master off disables all of
them, and tournament `extensionOverrides` changes exactly one subtype. Production
defaults are not changed.

Screen every subtype against all-on defaults on the first 32 Phase 3A v1 starts,
5,000 fixed nodes/move, equal depth cap 64, 160 plies, paired colors. Select only
the strongest single subtype. Confirm it on 64 new starts (128 games) under the
same limits. Candidate always means the subtype-disabled engine. Pair-bootstrap
95% score CI wholly above 50% means the enabled subtype is likely harmful; wholly
below means likely beneficial; otherwise the result is inconclusive.

The independent suite is `makhos-extension-confirmation-starts-v1`. Its unsigned
32-bit seed `0x4149c4b1` (1095353521) is mechanically the first eight hex digits
of the Phase 3A merge SHA `4149c4b1989378b2a85746b36190c915fdb16d4d`.
The generator produced candidates once, removed exact complete-state collisions
with Phase 3A v1, retained the first 64, and was frozen before results were run.
Its SHA-256 replay/content fingerprint is
`58475f4a2913f4a6c093d12a54a89e6a30d6160f5921f63964bb83267d5384ab`.
No alternate seed was tried.

## Source audit

All entries below are guarded by the master switch. “Added” is remaining search
plies and is capped as stated. Instrumentation counts a true trigger even when a
prior extension consumes its cap, and separately totals actual added depth.

| Subtype | Exact trigger, addition and cap | Location / fixed-node reachability |
|---|---|---|
| `singleLegalMove` | ordered legal count is 1; +1, capped at nominal `depth` | Root and interior; reachable |
| `smallEndgame` | child has at most 5 pieces; +1, capped at nominal `depth` | Root and interior; reachable and extremely frequent late in games |
| `tacticalCapture` | move captures and nominal depth >=2, ply <=18; +1 under tactical aggregate cap (`depth+2` for multi-capture, otherwise `depth+1`) | Root and interior; reachable |
| `multiCapture` | move captures >=2, nominal depth 2..6, ply <=18; +1 under the same aggregate cap | Root/interior; reachable; also a verification-only +1 capped at `reached+1` |
| `opponentForcedCapture` | quiet move leaves opponent a mandatory capture, depth >=2, ply <=18: +1, plus another +1 at depth <=6, aggregate cap `depth+1` | Root/interior; reachable |
| `singleCaptureRecapture` | code asks for a one-piece capture and `opHasCaptures` at depth <=6 | **Unreachable:** `opHasCaptures` is calculated only when the current move is quiet. Retained as an audited distinct legacy branch, not silently merged or credited |
| `rootLowMobility` | root has <=3 quiet legal moves and <=8 pieces, iteration depth >=4; +4 when <=6 pieces, else +2 | Root only; reachable; cap is nominal depth +4/+2 |
| `soundForcedTrap` | the root quiet move passes exhaustive immediate forced-trap/material-soundness checks | Root only; reachable; +2 capped at nominal depth+2 |

The post-iteration root verification repeats `smallEndgame`, tactical capture,
`soundForcedTrap`, and an explicit `multiCapture` extension. However, the whole
verification block requires `activeNodeLimit === undefined`. Canonical Phase 3A
and 3B fixed-node games therefore **never execute those verification-only
paths**; they are preserved and instrumented for timed/fixed-depth production
search but excluded from fixed-node attribution reasoning. Root move-selection
overrides after verification are not depth extensions and are not subtype flags.

## Commands

```sh
npm run test:phase3b
npm run experiment:extension-ablation -- --stage=screening
npm run experiment:extension-ablation -- --stage=confirmation --subtypes=smallEndgame
```

## Phase 3C promotion confirmation

Phase 3C's original multi-budget run reused the first 32 starts above. It is a
same-corpus robustness replication, not a second independent-corpus result.
The final promotion review freezes `makhos-phase3c-final-confirmation-starts-v1`.
Its seed `0xb59570ad` (3046469805) is mechanically the first eight hexadecimal
digits of immutable, GitHub-resolvable PR #11 commit
`b59570adb3e94dc629fc5a9af262b96b9d030498`. Its 64 retained complete states
are disjoint from Phase 3A, Phase 3B, and the superseded Phase 3C review corpus.
Its content fingerprint is
`fc48f9aee765fd13d30c2577d6c3031bada555214d1d831148a58ae68ea16d0f`.
No alternate seed was tried. Exact metadata is frozen in
`config/phase3c-promotion-v1.json` before final games are inspected.
