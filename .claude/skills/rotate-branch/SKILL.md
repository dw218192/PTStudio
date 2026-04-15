---
name: rotate-branch
description: After a PR is merged, prune the old dev branch and start a fresh one off develop.
argument-hint: "[branch name] -- defaults to dev/rendering-next"
---

Rotate a dev branch after its PR has been merged into develop.

## Steps

### 1. Ensure clean working tree

```bash
git status --porcelain
```

If dirty, stop and ask the user to commit or stash.

### 2. Clean up stale worktrees

Remove all agent worktrees and their branches left over from ticket work.

```bash
./repo python .claude/skills/rotate-branch/scripts/cleanup_worktrees.py
```

### 3. Switch to develop and pull

```bash
git checkout develop
git pull --ff-only origin develop
```

### 4. Delete the old dev branch (local + remote)

```bash
git branch -D dev/rendering-next
git push origin --delete dev/rendering-next
```

Use the argument if provided, otherwise default to `dev/rendering-next`.

### 5. Create and push the new branch

```bash
git checkout -b dev/rendering-next
git push -u origin dev/rendering-next
```

### 6. Print summary

```
Rotated dev/rendering-next
  Base: develop @ <short-hash> (<commit message>)
  Worktrees cleaned: <count>
  Old branch pruned (local + remote)
  New branch pushed to origin
```
