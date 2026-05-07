# ENGINE_PHASE_D_OPENING_BOOK_INSPECTION

## Scope

Phase D.0 is inspection only. No engine behavior was changed.

## Files Inspected

- `src/coreClaude/search/openingBook.ts`
- `src/ui/useCodexEngine.ts`
- `src/coreClaude/search/alphabeta.ts`
- `scripts/buildOpeningBook.ts`
- `scripts/matchup.ts`
- `scripts/aiBenchmark.ts`
- `package.json`

## Current Opening Book Flow

1. `scripts/buildOpeningBook.ts` generates book rows offline by searching AI-side opening positions and recording candidate moves.
2. The generated rows are pasted into `src/coreClaude/search/openingBook.ts` as the static `BOOK` array.
3. `openingBook.ts` builds an in-memory `BOOK_MAP` keyed by `hashPosition(pos)`.
4. `lookupOpeningBookCandidates(pos)`:
   - hashes the current position
   - finds matching book rows
   - generates current legal moves
   - keeps only rows whose `from/to` pair still matches a legal move
   - returns candidates sorted by descending weight
5. `lookupOpeningBook(pos)` calls `lookupOpeningBookCandidates(pos)` and picks one candidate with weighted randomness.
6. In the app path, `src/ui/useCodexEngine.ts` does not play a direct book move. It reads `lookupOpeningBookCandidates(pos)`, converts candidate weights into `rootMoveScores`, and passes those hints into `iterativeDeepening(...)`.
7. In `scripts/matchup.ts`, the enhanced side does use `lookupOpeningBook(pos)` directly before falling back to search.

## Existing Opening Book Files

- Primary runtime book module:
  - `src/coreClaude/search/openingBook.ts`
- Offline book generator:
  - `scripts/buildOpeningBook.ts`
- Runtime consumer using book-guided search:
  - `src/ui/useCodexEngine.ts`
- Script consumer using direct book moves:
  - `scripts/matchup.ts`

## Legacy Status

Treat the current opening book implementation as legacy/reference-only for Phase D planning.

- It is useful as a reference for prior data shape and wiring.
- It should not be trusted blindly as the basis for a new production opening book.
- The likely Phase D direction is to replace it later with a fresh implementation after isolation and inspection are complete.

## Entry Count

- `src/coreClaude/search/openingBook.ts` documents:
  - `932 entries covering 7 rounds of play (MAX_PLY=14, TOP_N=3, AI_THINK_MS=1000ms)`

This appears to be the current known book size embedded in the checked-in file.

## Old Entry And Data Format

The current embedded book format in `src/coreClaude/search/openingBook.ts` is:

- `type BookRow = [number, number, number, number?, number?];`

That corresponds to:

- `hash`
- `from`
- `to`
- optional `weight`
- optional `score`

At runtime:

- the static rows are loaded into `BOOK_MAP`
- each map bucket stores `{ from, to, weight, score? }`

At generation time:

- `scripts/buildOpeningBook.ts` prints rows as `[hash, from, to, weight, score]`
- those rows are manually pasted back into `openingBook.ts`

This is a compact legacy format, but it has clear limitations:

- source-embedded data instead of external structured data
- manual regeneration/paste workflow
- no explicit embedded metadata for versioning or generation provenance
- move storage is only `from/to`, with legality revalidated at runtime

## Deterministic Vs Random Behavior

Deterministic pieces:

- `lookupOpeningBookCandidates(pos)` is deterministic for a fixed position and legal move list.
- Candidate ordering is sorted by descending weight.
- Legal move filtering is deterministic.
- `scripts/buildOpeningBook.ts` deterministically filters candidate moves by score margin and rank once a search result exists.

Random pieces:

- `lookupOpeningBook(pos)` uses `weightedPick(...)`, which calls `Math.random()`.
- `src/coreClaude/search/alphabeta.ts` also has `pickDiversifiedRoot(...)`, which uses `Math.random()` when `diversifyRoot` is enabled.

Important distinction:

- The opening book module itself supports both deterministic candidate lookup and random direct move selection.
- The current app engine path is not a pure direct-book path. It is a book-guided search path, and it can also interact with root diversification.

## Where The Old Book Is Wired In

### Engine / Search Adjacency

- `src/coreClaude/search/openingBook.ts`
  - owns the legacy embedded book data
  - performs hash lookup
  - performs legal move filtering
  - performs weighted random direct selection when `lookupOpeningBook(...)` is used

- `src/coreClaude/search/alphabeta.ts`
  - does not import the opening book directly
  - does accept `rootMoveScores` and `diversifyRoot`, which can be fed by book-guided UI logic

### UI Wiring

- `src/ui/useCodexEngine.ts`
  - imports `lookupOpeningBookCandidates`
  - uses book candidates only in the early opening ply window
  - converts candidate weights into `rootMoveScores`
  - enables `diversifyRoot` in that same opening path
  - therefore the old book influences final move choice indirectly through search guidance rather than by forcing a direct move

### Script Wiring

- `scripts/matchup.ts`
  - imports `lookupOpeningBook`
  - directly selects a weighted random book move for the enhanced side when a hit exists

- `scripts/buildOpeningBook.ts`
  - regenerates the legacy book data offline

### Benchmark Wiring

- `scripts/aiBenchmark.ts`
  - does not import `openingBook.ts`
  - tactical benchmark currently bypasses the old book entirely

- `package.json`
  - benchmark scripts route through `scripts/aiBenchmark.ts`
  - no npm script currently wires the opening book into tactical benchmark runs

## Tactical Benchmark Book Usage

The tactical benchmark already bypasses the opening book.

Evidence:

- `scripts/aiBenchmark.ts` does not import `openingBook.ts`.
- It drives move selection through `iterativeDeepening(...)` and `selectStrictLevelMove(...)` directly.
- The benchmark scripts in `package.json` compile and run `scripts/aiBenchmark.ts`, so current `bench:ai*` flows do not rely on the opening book.

## Legal Move Verification

Yes, legal move verification exists.

`lookupOpeningBookCandidates(pos)` does not trust raw book rows blindly. It:

- calls `generateMoves(pos)`
- searches the legal move list for a matching `from/to`
- drops any book entry that is not currently legal

This is an important safety guard because the stored book rows are hash-indexed static data.

## Instrumentation Added In D.1

Lightweight opening book instrumentation is now present in `src/coreClaude/search/openingBook.ts`.

Tracked counters:

- `lookupAttempts`
- `successfulHits`
- `disabledRejects`
- `hashMisses`
- `illegalMoves`
- `benchmarkBypassRejects`

Helper functions:

- `getOpeningBookStats()`
- `resetOpeningBookStats()`

Notes:

- the counters live in the opening book module, not in core search
- the tactical benchmark still does not call into the opening book module
- `benchmarkBypassRejects` exists for future guarded call paths, but should remain `0` while the benchmark bypass stays hard-isolated

## Benchmark Isolation Status

The tactical benchmark remains opening-book-free.

- `scripts/aiBenchmark.ts` still does not import `openingBook.ts`
- benchmark reporting now explicitly records `openingBookBypassed: true`
- console summary now prints that the opening book was bypassed

This keeps tactical benchmark results insulated from legacy book behavior.

## Determinism Status

Direct opening book selection is now deterministic where it is used directly.

- `lookupOpeningBookCandidates(...)` returns deterministically sorted candidates
- `lookupOpeningBook(...)` now selects the top candidate deterministically
- `scripts/matchup.ts` uses that deterministic direct path

The UI path was already candidate-driven rather than random direct selection, but it still remains search-guided rather than a pure standalone book path.

## Risks Found

1. The app path is not a simple on/off opening book switch.
   - `useCodexEngine.ts` uses book candidates as root hints rather than directly forcing a book move.
   - That means changing book behavior can indirectly affect search ordering and final move choice.

2. Direct book selection is random.
   - `lookupOpeningBook(pos)` uses weighted randomness.
   - Any benchmark or experiment that calls this path directly must be treated as non-deterministic unless randomness is explicitly controlled.

3. Opening diversification adds another random layer.
   - `useCodexEngine.ts` sets `diversifyRoot` during opening play when depth is high enough.
   - That means app opening behavior is influenced by both book guidance and a search-layer diversification hook.

4. The offline generator expands only selected opponent responses.
   - `scripts/buildOpeningBook.ts` uses `TOP_N=3` and a specific opponent branching policy.
   - This keeps the book manageable, but it can bias coverage toward likely lines instead of exhaustive lines.

5. The runtime book depends on pasted static data.
   - Regeneration requires manual paste-back into `openingBook.ts`.
   - That creates a risk of stale docs, stale entry counts, or mismatched assumptions after regeneration.

6. There is no dedicated runtime book stats surface in the inspected code.
   - Current inspection did not find an existing book hit-rate or selection audit path.
   - That makes safe experimentation harder without first deciding where passive measurement should live.

7. The current book is wired into two different behavior surfaces.
   - The UI uses book-guided search.
   - The matchup script uses direct random book selection.
   - That split makes it easy to discuss “the opening book” as one thing when it is actually two different runtime behaviors.

8. The current book format and workflow are legacy-oriented.
   - The system is compact and workable as a reference.
   - It is not an ideal clean foundation for a fresh opening book implementation.

9. Deterministic direct selection does not remove all opening-path variability.
   - The UI still feeds book hints into search rather than executing an isolated direct-book policy.
   - Any future measurement work must keep book guidance separate from search behavior.

## How To Disable Or Isolate The Old Book Safely Later

Do not perform these in D.0. These are planning notes only.

1. Safest isolation point for app behavior:
   - gate the `lookupOpeningBookCandidates(pos)` call in `src/ui/useCodexEngine.ts`
   - leave core search logic untouched

2. Safest isolation point for direct script behavior:
   - gate or bypass `lookupOpeningBook(pos)` in `scripts/matchup.ts`
   - keep it out of tactical-quality validation

3. Safest handling for the legacy runtime module:
   - keep `src/coreClaude/search/openingBook.ts` as reference-only until the replacement design is ready
   - avoid mutating the embedded dataset during early Phase D

4. Safest benchmark stance:
   - keep `scripts/aiBenchmark.ts` book-free
   - continue validating tactical search independently from opening book experiments

## Archive, Ignore, Or Replace

Recommended treatment: replace later, keep as reference now.

- `src/coreClaude/search/openingBook.ts`
  - recommendation: `reference now, replace later`
  - reason: valuable for studying current wiring and data shape, but not a clean production baseline to trust blindly

- `scripts/buildOpeningBook.ts`
  - recommendation: `reference now, likely archive or supersede later`
  - reason: useful for understanding previous generation flow, but tied to the old embedded format and assumptions

- `scripts/matchup.ts` direct-book path
  - recommendation: `ignore for tactical validation, inspect separately for book experiments`
  - reason: it uses direct weighted random book selection rather than a deterministic benchmark path

- embedded `BOOK` data
  - recommendation: `do not delete yet`
  - reason: it can serve as a comparison artifact when designing the replacement system

## Recommendation For A Fresh Future Book System

The safest replacement direction is:

1. keep the legacy embedded book as a read-only reference artifact
2. define a new data format with explicit metadata and generation provenance
3. separate direct book policy from search-guidance policy
4. keep tactical benchmark permanently book-free
5. validate any new book in a dedicated opening/book harness before wiring it into app play

## Recommended Safe Phase D Plan

### D.1

Document current policy explicitly before behavior changes:

- keep tactical benchmark book-free
- distinguish direct book move selection from book-guided search
- distinguish deterministic candidate lookup from random weighted pick
- mark the current book implementation as legacy/reference-only

### D.2

Add passive inspection only, outside the search hot path if needed later:

- book hit count
- candidate count by position
- direct-book path usage vs guided-search path usage
- explicit boundary between legacy book and future replacement book

No behavior changes should happen in this step.

### D.3

If experiments are approved later, gate them narrowly:

- legacy book on/off flag at the UI call site
- direct weighted book move selection flag
- deterministic top candidate selection flag
- opening diversification interaction flag
- separate flag namespace for any fresh replacement book

Each switch should be isolated so book experiments do not accidentally mix multiple behaviors at once.

### D.4

Validate book experiments separately from tactical search quality:

- keep `bench:ai*` book-free
- use a dedicated opening/book experiment script or matchup script
- compare deterministic and random book policies in isolation before mixing them into normal app play
- compare legacy and replacement book behavior only after both can be isolated cleanly

## Summary

The current opening book system is already present and functional, but it is used in two different ways:

- direct weighted move selection in `scripts/matchup.ts`
- candidate-guided search in `src/ui/useCodexEngine.ts`

The tactical benchmark already avoids the book, legal move validation exists, and the biggest risk for Phase D is mixing together:

- book randomness
- root diversification
- search guidance

before each piece is isolated behind clear experiment boundaries.

For planning purposes, the existing opening book should be treated as a legacy reference system. The safest direction is to inspect it, isolate it, and then build a fresh opening book implementation rather than reusing the old one blindly.
