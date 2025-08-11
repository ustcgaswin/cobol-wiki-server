import os
import shutil
import subprocess
import uuid
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional
import stat
from app.utils.logger import logger

import requests

_executor = ThreadPoolExecutor(max_workers=5)

def rmtree_onerror(func, path, exc_info):
    """
    Error handler for shutil.rmtree that handles read-only files, common in .git directories.
    """
    # exc_info[1] is the exception instance
    if isinstance(exc_info[1], PermissionError):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            logger.error(f"Failed to fix permissions for {path}: {e}")
            raise
    else:
        raise exc_info[1]



def clone_github_repo_async(
    github_url: str, token: Optional[str], project_id: str, target_path: str
) -> Future:
    """
    Asynchronously clones a GitHub repo.

    token is optional (None or "" ⇒ public clone).
    project_id must be a valid UUID string.
    target_path is the destination directory for the clone.
    Returns a Future whose .result() is a dict:
      • success: bool
      • path: str           # on success
      • error: str          # on failure
      • code: str           # machine-readable error code
      • details: str?       # optional extra info
    """
    return _executor.submit(
        _clone_github_repo, github_url, token, project_id, target_path
    )


def _clone_github_repo(
    github_url: str, token: Optional[str], project_id: str, target_path: str
) -> Dict:
    # 1. Parse & validate URL
    try:
        parsed = urlparse(github_url)
    except Exception:
        return {"success": False, "error": "Malformed URL", "code": "URL_PARSE_ERROR"}

    if parsed.scheme not in ("http", "https") or "github.com" not in parsed.netloc:
        return {"success": False, "error": "Invalid GitHub URL", "code": "INVALID_URL"}

    parts = parsed.path.lstrip("/").rstrip("/").split("/")
    if len(parts) < 2:
        return {
            "success": False,
            "error": "URL does not point to a repository",
            "code": "INVALID_URL",
        }
    owner, repo = parts[0], parts[1]
    repo_name = repo[:-4] if repo.endswith(".git") else repo

    # 2. Validate project_id as UUID
    try:
        uuid.UUID(project_id)
    except ValueError:
        return {
            "success": False,
            "error": "project_id must be a valid UUID",
            "code": "INVALID_PROJECT_ID",
        }

    # 3. Prepare target directory
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        # Clean up directory if it exists to ensure a fresh clone
        shutil.rmtree(target_path, ignore_errors=True)

    # 4. GitHub API check (unauth if no token)
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        r = requests.get(api_url, headers=headers, timeout=10, verify=False)
    except requests.RequestException as e:
        return {"success": False, "error": f"Network error: {e}", "code": "NETWORK_ERROR"}

    if r.status_code == 404:
        return {"success": False, "error": "Repository not found", "code": "REPO_NOT_FOUND"}
    if r.status_code in (401, 403):
        return {"success": False, "error": "Access denied. Check your token and permissions.", "code": "ACCESS_DENIED"}
    if r.status_code != 200:
        return {
            "success": False,
            "error": f"GitHub API error: {r.status_code}",
            "code": "API_ERROR",
        }

    # 5. Clone (with or without token)
    if token:
        clone_url = f"https://{token}@github.com/{owner}/{repo_name}.git"
    else:
        clone_url = f"https://github.com/{owner}/{repo_name}.git"

    try:
        subprocess.run(
            ["git", "clone", clone_url, target_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(target_path, ignore_errors=True)
        return {
            "success": False,
            "error": "Git clone failed",
            "code": "CLONE_FAILED",
            "details": e.stderr.strip(),
        }
    except Exception as e:
        shutil.rmtree(target_path, ignore_errors=True)
        return {"success": False, "error": f"Unexpected error during clone: {e}", "code": "UNKNOWN_ERROR"}

    return {"success": True, "path": target_path}