"""What the cloud trainings share (GitHub's free machines: 4 processor cores, no graphics card, 6 hours per run).

- the saved state between runs lives in the repo's release "state" (files attached to it, up to 2 GB each)
- the live progress goes to the branch "status" (the progress website reads it)
- Auric v2's private material and checkpoints are ENCRYPTED with the secret AURIC_KEY (the public repo only holds
  scrambled bytes; the key is only on Saad's PC and in the repo's secrets)
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "saadvvsaadvv344-star/auric-training")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = "https://api.github.com"
WORK = Path(os.environ.get("AURIC_WORK", "work"))
WORK.mkdir(parents=True, exist_ok=True)
STARTED = time.time()
BUDGET = float(os.environ.get("AURIC_MINUTES", "335")) * 60          # (a GitHub run may last 6 h; keep a margin)


def time_left():
    return BUDGET - (time.time() - STARTED)


def _req(method, url, data=None, headers=None, raw=False, timeout=600):
    h = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json", "User-Agent": "auric-training"}
    h.update(headers or {})
    body = data if isinstance(data, (bytes, type(None))) else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = r.read()
    return out if raw else (json.loads(out) if out else None)


# ---------------------------------------------------------------- state between runs (release "state")
def _release(tag="state"):
    try:
        return _req("GET", f"{API}/repos/{REPO}/releases/tags/{tag}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        return _req("POST", f"{API}/repos/{REPO}/releases", {"tag_name": tag, "name": "Training state (between runs)",
                                                               "body": "Saved automatically by the trainings."})


def download(name, dest, tag="state"):
    """A file attached to the release -> dest (False if it doesn't exist yet)."""
    for a in _release(tag).get("assets", []):
        if a["name"] == name:
            data = _req("GET", a["url"], headers={"Accept": "application/octet-stream"}, raw=True, timeout=1800)
            Path(dest).write_bytes(data)
            return True
    return False


def upload(path, name=None, tag="state"):
    """Attaches (replaces) a file on the release."""
    name = name or Path(path).name
    rel = _release(tag)
    for a in rel.get("assets", []):
        if a["name"] == name:
            _req("DELETE", a["url"])
    url = rel["upload_url"].split("{")[0] + f"?name={name}"
    _req("POST", url, Path(path).read_bytes(), headers={"Content-Type": "application/octet-stream"}, timeout=1800)


# ---------------------------------------------------------------- live progress (branch "status")
def _ensure_status_branch():
    try:
        _req("GET", f"{API}/repos/{REPO}/git/ref/heads/status")
    except urllib.error.HTTPError:
        main = _req("GET", f"{API}/repos/{REPO}/git/ref/heads/main")
        _req("POST", f"{API}/repos/{REPO}/git/refs", {"ref": "refs/heads/status", "sha": main["object"]["sha"]})


def publish_status(files):
    """{path in the status branch: bytes or json-able} -> committed (the website shows it within ~1 minute)."""
    _ensure_status_branch()
    for path, content in files.items():
        data = content if isinstance(content, bytes) else json.dumps(content, indent=1).encode()
        sha = None
        try:
            sha = _req("GET", f"{API}/repos/{REPO}/contents/{path}?ref=status")["sha"]
        except urllib.error.HTTPError:
            pass
        body = {"message": f"progress: {path}", "content": base64.b64encode(data).decode(), "branch": "status"}
        if sha:
            body["sha"] = sha
        for attempt in range(3):
            try:
                _req("PUT", f"{API}/repos/{REPO}/contents/{path}", body)
                break
            except urllib.error.HTTPError:
                time.sleep(3)


def read_status(path):
    try:
        out = _req("GET", f"{API}/repos/{REPO}/contents/{path}?ref=status")
        return json.loads(base64.b64decode(out["content"]))
    except Exception:
        return {}


# ---------------------------------------------------------------- privacy (Auric v2)
def _fernet():
    from cryptography.fernet import Fernet
    key = os.environ.get("AURIC_KEY", "")
    if not key:
        raise SystemExit("AURIC_KEY is missing (repo secret): Auric v2's private material can't be read")
    return Fernet(key.encode())


def encrypt_file(src, dst):
    Path(dst).write_bytes(_fernet().encrypt(Path(src).read_bytes()))


def decrypt_file(src, dst):
    Path(dst).write_bytes(_fernet().decrypt(Path(src).read_bytes()))


def history_add(job, entry, keep=60):
    """One line per cloud run (shown on the website)."""
    h = read_status(f"{job}_history.json") or []
    h = (h if isinstance(h, list) else []) + [entry]
    publish_status({f"{job}_history.json": h[-keep:]})
