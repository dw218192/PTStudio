---
name: triage-ci
description: Wait for CI pipeline to finish, then triage and fix any failures. Loops until CI is green or the turn limit is reached. Use this after pushing code, opening a PR, or whenever the user says "check CI", "wait for CI", "triage CI", "fix CI", or asks about build/test failures on the current branch.
argument-hint: "[max turns] [PR number] -- defaults to 3 turns, current branch's PR"
---

Wait for the CI pipeline to complete, then diagnose and fix failures.
Repeats the triage-fix-push-wait cycle until CI is green or the turn
limit is reached.

## Arguments

Parse the argument string for:
- A small integer (1-10) -> max turns (default: 3)
- A larger integer or `#N` -> PR number (default: current branch's PR)

Examples: `/triage-ci` (3 turns, auto PR), `/triage-ci 5` (5 turns),
`/triage-ci 29` (PR #29, 3 turns), `/triage-ci 5 29` (5 turns, PR #29).

## Loop

```
for turn in 1..max_turns:
    1. Wait for CI (background -- user can work while waiting)
    2. Check results -- if green, report success and stop
    3. Triage failures
    4. Apply fixes, build and test locally
    5. Push and go to next turn
```

If the turn limit is reached with CI still failing, report what's left
and stop. Don't loop forever.

## Steps (per turn)

### 1. Wait for CI

Find the PR:
```bash
gh pr view --json number -q .number
```

Get the latest run:
```bash
gh run list --branch <branch> --limit 1
```

If the run is already completed, skip to step 2.

If still in progress, use `run_in_background: true` on the Bash tool to
watch without blocking the conversation:

```bash
gh run watch <run-id> --exit-status
```

This lets the user continue working. When the background task completes,
a notification arrives -- pick up from step 2 at that point. Tell the
user: "CI run <id> is in progress. I'm watching in the background --
you'll be notified when it finishes. Feel free to keep working."

### 2. Check results

```bash
gh pr checks <pr-number>
```

If all checks pass, report success and stop the loop.

### 3. Triage each failure

For each failed job:
```bash
gh run view <run-id> --job <job-id> --log-failed
```

Classify as:
- **COMPILE** -- build error. Read the error, find the file/line, fix it.
- **TEST** -- test failure. Check if it's a real regression or stale test.
- **INFRA** -- CI infrastructure (missing artifacts, timeouts, network).
  Check if caused by a code change or transient.
- **FLAKY** -- passes locally, fails in CI with no code cause.

Present the triage:
```
Turn 1/3 -- CI run <id> -- 2 checks failed

  1. [COMPILE] Build (windows-x64, Release)
     error at file.cpp:42 -- description
     Fix: ...

  2. [INFRA] Build (emscripten, Release)
     Downstream of #1
```

### 4. Apply fixes

For each actionable failure:
1. Read the failing file
2. Apply the fix
3. Build locally: `./repo build`
4. Test locally: `./repo test`

### 5. Push

```bash
git add <changed files>
git commit -m "Fix CI: <brief description>"
git push
```

Then loop back to step 1 for the next turn.

## Notes

- Cascade failures are common: one build failure causes downstream jobs
  to fail (e.g. missing artifacts). Identify the root cause first.
- `max-parallel: 1` in the CI matrix means jobs run sequentially --
  if the first matrix entry fails, later entries may fail for dependent
  reasons.
- Always build and test locally before pushing a fix to avoid churn.
- INFRA failures that are purely transient (network blip, runner OOM)
  can be retried without code changes: `gh run rerun <run-id> --failed`.
- Do NOT use `sleep` for polling -- the hook blocks it. Use
  `run_in_background: true` on `gh run watch` and wait for the
  notification instead.
