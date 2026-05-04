# Benchmark Runbook

Use this file when running AI benchmarks from PowerShell.

## Current Overnight Run Notes

If a full benchmark is already running with:

```powershell
npm run bench:ai:full *> full-benchmark.log
```

the first terminal will be quiet. Watch progress from a second terminal:

```powershell
cd "d:\My App\makhos-v2"
Get-Content .\full-benchmark.log -Tail 120 -Wait
```

Do not start a fresh run unless you intentionally want to reset the checkpoint.
If the process stops, resume with:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:full *> full-benchmark-resume.log
```

Current checkpoint file:

```text
.tmp\benchmarks\ai-benchmark-full-checkpoint.json
```

## Quick Benchmark

Fast smoke test:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai
```

Fast smoke test from scratch:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:fresh
```

Latest quick JSON report:

```text
.tmp\benchmarks\ai-benchmark-quick-latest.json
```

## Full Benchmark Overnight

Run full benchmark and save all output to a log file:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:full *> full-benchmark.log
```

The terminal will look quiet because all output is redirected into
`full-benchmark.log`.

## Teacher / Deep Oracle Benchmark

Use this when we want the slower, deeper tactical oracle to judge candidate
moves more strictly. It does tactical cases only, with no head-to-head matrix.

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:teacher *> teacher-benchmark.log
```

Watch progress from another terminal:

```powershell
cd "d:\My App\makhos-v2"
Get-Content .\teacher-benchmark.log -Tail 120 -Wait
```

Resume after interruption:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:teacher *> teacher-benchmark-resume.log
```

Start teacher benchmark from scratch:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:teacher:fresh *> teacher-benchmark-fresh.log
```

Teacher output files:

```text
.tmp\benchmarks\ai-benchmark-teacher-latest.json
.tmp\benchmarks\ai-benchmark-teacher-checkpoint.json
```

## Watch Progress In Another Terminal

Open a second PowerShell window:

```powershell
cd "d:\My App\makhos-v2"
Get-Content .\full-benchmark.log -Tail 80 -Wait
```

Stop watching with `Ctrl+C`. This does not stop the benchmark running in the
first terminal.

## Check If The Log Is Updating

```powershell
cd "d:\My App\makhos-v2"
Get-Item .\full-benchmark.log | Select-Object Length, LastWriteTime
```

## Resume After Terminal Closes Or Machine Sleeps

Run the same command again:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:full *> full-benchmark-resume.log
```

The benchmark resumes from:

```text
.tmp\benchmarks\ai-benchmark-full-checkpoint.json
```

## Start Full Benchmark From Scratch

This clears the full benchmark checkpoint:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:full:fresh *> full-benchmark-fresh.log
```

## See Progress On Screen And Save Log

Use this instead of redirecting everything to a file:

```powershell
cd "d:\My App\makhos-v2"
cmd /c "npm run bench:ai:full 2>&1" | Tee-Object -FilePath full-benchmark.log
```

## Output Files

Latest full report:

```text
.tmp\benchmarks\ai-benchmark-full-latest.json
```

Timestamped full reports:

```text
.tmp\benchmarks\ai-benchmark-full-*.json
```

Resume checkpoint:

```text
.tmp\benchmarks\ai-benchmark-full-checkpoint.json
```

## Useful Verification Commands

```powershell
cd "d:\My App\makhos-v2"
npm run test:rules
npm run test:tactical
npx tsc --noEmit
```

## Analyze Latest Full Report

After a full benchmark finishes:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:analyze
```

Analyze a specific report:

```powershell
cd "d:\My App\makhos-v2"
npm run bench:ai:analyze -- .tmp\benchmarks\ai-benchmark-full-latest.json
```

## Expected Progress Lines

During tactical benchmark:

```text
[tactical 12/39] oracle all-kings-2v1-corner-win
  [45/156] easy all-kings-2v1-corner-win
    chose 7->2, oracle 7->2, drop=0, 503ms
```

During head-to-head:

```text
[h2h 13/72] normal vs hard game 1/6
    outcome=p2, score normal vs hard = 0.0
```

## If Nothing Appears

If the terminal is quiet but you used `*> full-benchmark.log`, that is normal.
Watch the log from another terminal:

```powershell
cd "d:\My App\makhos-v2"
Get-Content .\full-benchmark.log -Tail 80 -Wait
```

If `full-benchmark.log` is still empty after 10-15 minutes, check whether the
process is still running:

```powershell
Get-Process node -ErrorAction SilentlyContinue
```
