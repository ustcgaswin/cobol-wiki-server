import argparse
import os
import stat
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_DIR_NAME = "mlartifacts"


def _make_writable(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # best effort


def safe_delete_directory(target: Path, dry_run: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Delete a directory tree safely.
    Returns (success, error_message). error_message is None on success.
    """
    if not target.exists():
        return False, f"Directory not found: {target}"

    if not target.is_dir():
        return False, f"Path is not a directory: {target}"

    # Safety: ensure we're not at filesystem root or project root accidentally
    if len(target.parts) < 2 or target == target.anchor:
        return False, f"Refusing to delete unsafe path: {target}"

    if dry_run:
        return True, None

    try:
        shutil.rmtree(target, onerror=_make_writable)
        return True, None
    except PermissionError as e:
        return False, f"PermissionError deleting {target}: {e}"
    except OSError as e:
        return False, f"OSError deleting {target}: {e}"
    except Exception as e:
        return False, f"Unexpected error deleting {target}: {e}"


def resolve_target(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path).expanduser().resolve()
    # Default: sibling folder to scripts directory (project root assumed one level up)
    return Path(__file__).resolve().parents[1] / DEFAULT_DIR_NAME


def parse_args():
    p = argparse.ArgumentParser(description="Delete the mlartifacts directory.")
    p.add_argument("--path", "-p", help="Explicit path to mlartifacts directory.")
    p.add_argument("--dry-run", "-n", action="store_true", help="Show what would be deleted without deleting.")
    p.add_argument("--force", "-f", action="store_true", help="Exit 0 even if directory not found.")
    return p.parse_args()


def main():
    args = parse_args()
    target = resolve_target(args.path)

    print(f"[delete_mlartifacts] Target: {target}")
    if args.dry_run:
        if target.exists():
            print(f"[delete_mlartifacts] DRY-RUN: Would delete directory (size may be large): {target}")
            for i, p in enumerate(target.rglob("*")):
                if i >= 10:
                    print("  ... (truncated list)")
                    break
                print(f"  {p}")
            sys.exit(0)
        else:
            msg = f"[delete_mlartifacts] DRY-RUN: Directory does not exist: {target}"
            print(msg)
            sys.exit(0 if args.force else 1)

    success, error = safe_delete_directory(target, dry_run=False)
    if success:
        print(f"[delete_mlartifacts] Deleted: {target}")
        sys.exit(0)
    else:
        print(f"[delete_mlartifacts] {error}")
        if "not found" in (error or "") and args.force:
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()