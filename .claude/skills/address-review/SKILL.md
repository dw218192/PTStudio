---
name: address-review
description: Fetch CodeRabbit (or other bot) review comments from the current PR and address each one with code changes.
argument-hint: "[PR number] — defaults to the PR for the current branch"
---

Address review comments on a GitHub PR. Fetches comments, applies fixes, commits.

## Steps

### 1. Identify the PR

If an argument is provided, use it as the PR number. Otherwise, find the PR
for the current branch:

```bash
gh pr view --json number,headRefName -q .number
```

### 2. Fetch review comments

```bash
gh api repos/{owner}/{repo}/pulls/{number}/comments --paginate
```

Parse each comment for:
- `path`: file path
- `line` / `original_line`: line number
- `body`: the comment text (may contain a diff suggestion block)
- `in_reply_to_id`: skip replies (only process top-level comments)

Also fetch the PR review body (walkthrough / summary) but only act on
**inline file comments**, not the summary.

### 3. Triage comments

**Default: fix everything.** Trivial fixes (add an attribute, remove an
unused import, rename a file) are still valid — they improve the codebase.
Do not skip a comment because it's "just a nit." The whole point of this
skill is to handle the tedious stuff.

For each comment, classify as one of:
- **FIX**: the comment requests a concrete change. This is the default.
  Includes: add attribute, rename, remove dead code, fix a race condition,
  add a test case, quote a path, catch an exception, improve an error
  message — no matter how small.
- **REJECT**: the suggestion is wrong or conflicts with project conventions
  (CLAUDE.md). You must state *why* it's wrong — "not important" is not a
  valid reason. Examples: suggesting a pattern the codebase explicitly
  avoids, proposing a change that breaks ABI, misunderstanding the code.
- **STALE**: the file/line no longer exists in the current code (already
  fixed or code was deleted).

Present the triage to the user:
```
PR #27 — 10 comments found
  1. [FIX]    renderWorld.h:346 — add [[nodiscard]]
  2. [FIX]    worker.h:114 — remove default-constructibility requirement
  3. [REJECT] editorApplication.cpp:500 — suggests X but CLAUDE.md says Y
  4. [STALE]  oldFile.cpp:30 — file no longer exists
  ...
Proceed with 8 fixes? (y/n)
```

Wait for user confirmation before making changes.

### 4. Apply fixes

For each actionable comment:
1. Read the file at the referenced path
2. Locate the relevant code (line number is a hint, not exact — find by context)
3. Apply the fix using the Edit tool
4. If the comment contains a diff suggestion (```suggestion block), apply it
   directly

### 5. Build & verify

After all fixes:
```bash
./repo build
./repo test
```

If build or test fails, diagnose and fix before proceeding.

### 6. Commit & push

Stage all changed files and commit:
```
git add <changed files>
git commit -m "Address review comments on PR #<number>"
git push
```

### 7. Reply to comments (optional)

If the user asks, reply to each addressed comment on GitHub:
```bash
gh api repos/{owner}/{repo}/pulls/{number}/comments/{comment_id}/replies \
  -f body="Addressed in <commit-sha>"
```

## Notes

- Never blindly apply suggestions without reading the surrounding code —
  the suggestion may be based on stale context
- If a suggestion conflicts with project conventions (CLAUDE.md), skip it
  and explain why
- Group related fixes into a single commit, not one commit per comment
- Run `./repo launch editor --capture-and-quit` if any GPU/rendering code
  was changed
