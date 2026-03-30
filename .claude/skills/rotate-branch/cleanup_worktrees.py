"""Clean up stale agent worktrees and their branches."""

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]  # .claude/skills/rotate-branch -> repo root
    worktrees_dir = root / "_agent" / "worktrees"

    # Prune any worktrees git still tracks but whose directories are gone
    subprocess.run(["git", "worktree", "prune"], cwd=root, check=True)

    if not worktrees_dir.is_dir():
        print("No worktrees to clean")
        return

    entries = [d for d in worktrees_dir.iterdir() if d.is_dir()]
    if not entries:
        print("No worktrees to clean")
        return

    cleaned = 0
    for wt in sorted(entries):
        name = wt.name
        branch = f"worktree-{name}"

        # Try git worktree remove first (handles registered worktrees)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=root, capture_output=True,
        )

        # Delete the directory if it still exists (unregistered leftovers)
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)

        # Delete the branch if it exists
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=root, capture_output=True,
        )

        cleaned += 1

    # Remove the worktrees dir itself if empty
    if worktrees_dir.exists() and not any(worktrees_dir.iterdir()):
        worktrees_dir.rmdir()

    print(f"Cleaned {cleaned} worktree(s)")


if __name__ == "__main__":
    main()
