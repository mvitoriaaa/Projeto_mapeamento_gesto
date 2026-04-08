import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_FILES = {
    Path("scripts/check_repo_safety.py"),
    Path(".github/workflows/repo-safety.yml"),
}

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA|PRIVATE KEY)-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]

BANNED_TERMS = [
    "middle_finger",
    "easter_egg",
]


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        path = Path(line)
        if path not in SKIP_FILES:
            yield path


def is_text_file(path):
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def scan_file(path):
    content = (ROOT / path).read_text(encoding="utf-8")
    findings = []

    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            findings.append(f"{path}: possible secret -> {match.group(0)[:80]}")

    lowered = content.lower()
    for term in BANNED_TERMS:
        if term in lowered:
            findings.append(f"{path}: banned term -> {term}")

    return findings


def main():
    findings = []
    for relative_path in tracked_files():
        full_path = ROOT / relative_path
        if not full_path.is_file() or not is_text_file(full_path):
            continue
        findings.extend(scan_file(relative_path))

    if findings:
        print("Repository safety check failed:")
        for finding in findings:
            print(f" - {finding}")
        sys.exit(1)

    print("Repository safety check passed.")


if __name__ == "__main__":
    main()
