import argparse
import os
import stat
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# Common log folder names; first existing will be chosen if --path not provided
CANDIDATE_DIR_NAMES = ["logs", "log"]


def _make_writable(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # best effort


def _partial_delete_on_permission_error(target: Path) -> Tuple[bool, Optional[str]]:
    """
    Attempt best-effort deletion of contents when some files are locked.
    Returns (success, message). success True if at least partial cleanup performed.
    """
    locked: List[Path] = []
    # Walk bottom-up so we can remove empty dirs afterwards
    for root, dirs, files in os.walk(target, topdown=False):
        root_path = Path(root)
        # Files
        for fname in files:
            fpath = root_path / fname
            try:
                os.chmod(fpath, stat.S_IWRITE)
                fpath.unlink()
            except Exception:
                locked.append(fpath)
        # Dirs
        for dname in dirs:
            dpath = root_path / dname
            try:
                dpath.rmdir()
            except Exception:
                # Could be not empty due to locked children; ignore
                pass

    # Try to remove the top directory if empty
    try:
        target.rmdir()
    except Exception:
        pass

    if not target.exists():
        return True, "Deleted directory after removing unlocked files."
    if locked:
        sample = ", ".join(str(p) for p in locked[:5])
        more = " ..." if len(locked) > 5 else ""
        return True, f"Partial delete: kept locked files: {sample}{more}"
    # If we get here, nothing was locked (unexpected), but removal failed.
    return False, f"Could not fully delete {target} for unknown reasons."


def safe_delete_directory(target: Path, dry_run: bool = False) -> Tuple[bool, Optional[str]]:
    if not target.exists():
        return False, f"Directory not found: {target}"
    if not target.is_dir():
        return False, f"Path is not a directory: {target}"
    if len(target.parts) < 2 or target == target.anchor:
        return False, f"Refusing to delete unsafe path: {target}"

    if dry_run:
        return True, None

    try:
        shutil.rmtree(target, onerror=_make_writable)
        return True, None
    except PermissionError:
        # Fallback: delete what we can
        success, msg = _partial_delete_on_permission_error(target)
        return success, msg
    except OSError as e:
        return False, f"OSError deleting {target}: {e}"
    except Exception as e:
        return False, f"Unexpected error deleting {target}: {e}"


def resolve_target(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path).expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    for name in CANDIDATE_DIR_NAMES:
        cand = root / name
        if cand.exists():
            return cand
    return root / CANDIDATE_DIR_NAMES[0]


def parse_args():
    p = argparse.ArgumentParser(description="Delete the log folder.")
    p.add_argument("--path", "-p", help="Explicit path to log directory.")
    p.add_argument("--dry-run", "-n", action="store_true", help="Show what would be deleted without deleting.")
    p.add_argument("--force", "-f", action="store_true", help="Exit 0 even if directory not found.")
    return p.parse_args()


def main():
    args = parse_args()
    target = resolve_target(args.path)
    print(f"[delete_log_folder] Target: {target}")

    if args.dry_run:
        if target.exists():
            print(f"[delete_log_folder] DRY-RUN: Would delete directory: {target}")
            for i, p in enumerate(target.rglob("*")):
                if i >= 10:
                    print("  ... (truncated list)")
                    break
                print(f"  {p}")
            sys.exit(0)
        else:
            print(f"[delete_log_folder] DRY-RUN: Directory does not exist: {target}")
            sys.exit(0 if args.force else 1)

    success, message = safe_delete_directory(target, dry_run=False)
    if success:
        if target.exists():
            # Partial cleanup
            print(f"[delete_log_folder] {message}")
            sys.exit(0)
        else:
            print(f"[delete_log_folder] Deleted: {target}")
            if message:
                print(f"[delete_log_folder] {message}")
            sys.exit(0)
    else:
        print(f"[delete_log_folder] {message}")
        if message and "not found" in message and args.force:
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()