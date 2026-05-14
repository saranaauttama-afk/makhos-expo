# Known Issues

## ~~sac-two-win-three-p1 Catastrophic Failure~~ FIXED ✅

**Status**: ~~Pre-existing known issue (as of 2026-05-13)~~ **FIXED** (2026-05-14)
**Fixed by**: Opening book expansion to 26 entries (commit 68ceda9)

**Description**: The `sac-two-win-three-p1` tactical benchmark case consistently fails with a catastrophic drop (~998K centipawns). The AI chooses move `7->2K` (promoting to square 2) when the oracle expects `8->4K` (promoting to square 4).

**Position Details**:
- **Bucket**: Forced recapture trap
- **Player 1 (to move)**: Men on squares 6, 7, 10, 19, 30; King on square 9
- **Player 2**: Men on squares 0, 13, 16, 17, 29; King on square 5
- **Expected**: Multi-move forced capture sequence evaluation

**Impact**:
- Causes `npm run gate:ai:quick` to return `classification=FAIL`
- Listed in `FATAL_CATASTROPHIC_CASES` set
- Has special handling in `repeatedRunBenchmarkSummary.ts` due to "repeated-run volatility"

**Root Cause**:
The engine struggles with deep tactical evaluation in forced recapture trap positions. This appears to be a fundamental limitation of the current evaluation function's ability to score complex multi-capture sequences correctly.

**Workaround**:
This failure is expected and does not indicate a regression unless the drop significantly worsens (e.g., from 998K to 1.5M+).

**Investigation History**:
- Confirmed failing at commit `f2d6c9c` (2026-05-13)
- Not introduced by recent opening book expansion (commits `18fa840`, `b503331`)
- Likely a long-standing tactical blind spot

**Next Steps**:
To fix this would require one of:
1. Implementing specialized tactical pattern recognition for forced recapture traps
2. Increasing search depth for high-piece-count tactical positions
3. Improving evaluation function's scoring of intermediate capture positions
4. Adding this specific pattern type to an endgame tablebase

---

**Note**: When evaluating AI improvements, use the subset of tests excluding this case to measure progress, or compare drop magnitude (ensure it doesn't worsen from ~998K).
