---
name: contribute-framework
description: Publish pending changes in tools/framework/ to the repokit upstream repo. Handles branch switch, testing, versioning, push, and submodule pin.
argument-hint: one-line summary of the change (used in changelog)
---

Publish pending `tools/framework/` changes upstream and pin the new version.
Abort if the submodule has no uncommitted changes.

## Steps

### 1. Switch to `main`

```bash
cd tools/framework
git stash
git checkout main
git pull --ff-only origin main
git stash pop   # conflict -> stop, ask user
```

### 2. Bootstrap test driver & run tests

`test_driver/` exists on `main` but not on the release tag. It needs a
junction so the framework code under test resolves correctly.

```bash
# Windows junction (idempotent)
cmd.exe /c "mklink /J test_driver\tools\framework . 2>nul || echo junction exists"
bash test_driver/tools/framework/bootstrap.sh test_driver
cd test_driver && ./repo test
```

**Fail -> fix the issue and re-run tests. Do NOT bump version or proceed until
tests pass.** Loop fix -> test until green.

### 3. Clean up test driver

Remove the junction and `test_driver/` artifacts **before** bumping or
switching away from `main`. The release tag doesn't contain `test_driver/`,
so these would linger as untracked files.

```bash
cd tools/framework
cmd.exe /c "rmdir test_driver\tools\framework"   # remove junction first
rm -rf test_driver
```

### 4. Bump version & changelog

- Patch-increment `version` in `pyproject.toml` (e.g. `0.7.26` -> `0.7.27`)
- Prepend to `CHANGELOG.md`:

```markdown
## <new-version>

- **<Area>**: <user's argument>
```

### 5. Commit & push

```bash
cd tools/framework
git add -A
git commit -m "v<new-version>: <summary>"
git push origin main
```

### 6. Wait for CI tag

CI auto-tags `v<version>` on green. Poll `git fetch origin --tags && git tag -l "v<new-version>"` every ~30s, up to 3 min. Timeout -> print manual finish instructions and stop.

### 7. Pin in parent repo

```bash
cd tools/framework && git checkout v<new-version>
cd ../.. && git add tools/framework && git commit -m "Pin repokit submodule to v<new-version>"
```

Print: old version -> new version, changelog entry, pin commit hash.
