#!/usr/bin/env python3
"""
Repository prompt dumper

Primary target ecosystems:
- Python
- TypeScript / JavaScript
- Java
- Go

Design:
- Binary denylist, text allow-by-default
- Language-aware directory exclusions
- Safe handling of secrets, env files, and large files
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

# Known binary extensions (hard denylist)
BINARY_EXTS = {
    # images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".ico", ".svg",
    # audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
    # video
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    # archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # compiled / bytecode
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".class",
    # package formats
    ".jar", ".war",
    # fonts
    ".ttf", ".otf", ".woff", ".woff2",
    # office
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    # data / ML
    ".h5", ".hdf5", ".parquet", ".feather", ".npy", ".npz",
    ".pb", ".onnx", ".pt", ".pth",
    # disk images
    ".iso", ".dmg",
}

# Extensions that are binary but explicitly allowed
ALWAYS_ALLOW_EXTS = {
    ".pdf",
}

# Always-include important repo metadata (extensionless or critical)
ALWAYS_INCLUDE_NAMES = {
    # general
    "README",
    "README.md",
    "README.txt",
    "LICENSE",
    "LICENSE.txt",
    "Makefile",
    "Dockerfile",

    # JS / TS
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "jsconfig.json",
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "tailwind.config.js",
    "tailwind.config.ts",
    "postcss.config.js",

    # Python
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",

    # Java
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",

    # Go
    "go.mod",
    "go.sum",
}

# Explicit filenames to exclude
EXCLUDE_NAMES = {
    "project_dump.py",
}

# Environment / secrets (exclude ALL variants)
ENV_FILE_PREFIXES = (
    ".env",
)

# Directories to skip entirely (language-aware)
SKIP_DIRS = {
    # VCS / IDE
    ".git",
    ".idea",
    ".vscode",

    # Python
    "__pycache__",
    ".venv",
    "venv",

    # JS / TS
    "node_modules",
    ".next",
    "dist",
    "out",
    "coverage",

    # Java
    "target",
    "build",

    # Go
    "bin",
    "pkg",

    # Misc
    ".cache",
}

# Safety limits
MAX_BYTES_PER_FILE = 200_000

# Text summarization policy
TXT_LINE_LIMIT = 50
TXT_HEAD_LINES = 10
TXT_TAIL_LINES = 10

# ---------------------------------------------------------------------
# TREE BUILDER
# ---------------------------------------------------------------------

def build_tree(root_dir: Path) -> str:
    """Return a tree-style view of the directory."""
    lines: List[str] = []

    def walk(dir_path: Path, prefix: str = ""):
        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return

        total = len(entries)
        for i, entry in enumerate(entries):
            connector = "└── " if i == total - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir() and entry.name not in SKIP_DIRS:
                child_prefix = (
                    f"{prefix}    " if i == total - 1 else f"{prefix}│   "
                )
                walk(entry, child_prefix)

    lines.append(root_dir.name)
    walk(root_dir)
    return "\n".join(lines)

# ---------------------------------------------------------------------
# FILE FILTERING
# ---------------------------------------------------------------------

def is_env_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name == p or name.startswith(p + ".") for p in ENV_FILE_PREFIXES)

def should_include_file(path: Path) -> bool:
    if path.is_dir():
        return False

    name = path.name
    suffix = path.suffix.lower()

    # Explicit excludes
    if name in EXCLUDE_NAMES:
        return False

    # Secrets / env
    if is_env_file(path):
        return False

    # OS junk
    if name == ".DS_Store":
        return False

    # Exclude the running script itself
    try:
        if Path(__file__).resolve().name == name:
            return False
    except NameError:
        pass

    # Always include important metadata
    if name in ALWAYS_INCLUDE_NAMES:
        return True

    # Explicitly allowed binary-like formats
    if suffix in ALWAYS_ALLOW_EXTS:
        return True

    # Include everything that is not a known binary
    if suffix and suffix not in BINARY_EXTS:
        return True

    return False

# ---------------------------------------------------------------------
# FILE READING
# ---------------------------------------------------------------------

def _read_file_bytes(path: Path):
    try:
        raw = path.read_bytes()
    except Exception as e:
        return f"<<ERROR READING FILE: {e}>>", False

    truncated = False
    if len(raw) > MAX_BYTES_PER_FILE:
        raw = raw[:MAX_BYTES_PER_FILE]
        truncated = True

    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="replace")
            break
        except Exception:
            continue
    else:
        text = "<<UNDECODABLE CONTENT>>"

    if truncated:
        text += f"\n\n<<TRUNCATED at {MAX_BYTES_PER_FILE} bytes>>"

    return text, truncated

def summarize_txt(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= TXT_LINE_LIMIT:
        return text

    return "\n".join(
        [
            f"<<SHOWING FIRST {TXT_HEAD_LINES} AND LAST {TXT_TAIL_LINES} LINES "
            f"(total {len(lines)} lines)>>",
            "",
            "<<BEGIN FIRST>>",
            *lines[:TXT_HEAD_LINES],
            "<<END FIRST>>",
            "",
            "<<BEGIN LAST>>",
            *lines[-TXT_TAIL_LINES:],
            "<<END LAST>>",
        ]
    )

def extract_pdf_text(path: Path) -> str:
    try:
        import pypdf
    except Exception:
        return (
            "<<PDF CONTENT NOT EXTRACTED: pypdf not installed>>\n"
            "Run: pip install pypdf"
        )

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        return f"<<PDF READ ERROR: {e}>>"

    chunks = [
        f"<<PDF EXTRACTED TEXT: {path.name} ({len(reader.pages)} pages)>>",
        "",
    ]

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = f"<<ERROR extracting page {i+1}: {e}>>"

        chunks.append(f"----- [PAGE {i+1}] -----")
        chunks.append(text)

    return "\n".join(chunks)

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)

    text, _ = _read_file_bytes(path)

    if path.suffix.lower() == ".txt":
        return summarize_txt(text)

    return text

# ---------------------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------------------

def collect_files(root_dir: Path) -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            path = Path(dirpath) / fname
            if should_include_file(path):
                rel = path.relative_to(root_dir)
                collected.append((str(rel), read_file(path)))

    collected.sort(key=lambda x: x[0].lower())
    return collected

# ---------------------------------------------------------------------
# PROMPT BUILDING
# ---------------------------------------------------------------------

def build_prompt(
    root_dir: Path,
    tree_text: str,
    files: List[Tuple[str, str]],
) -> str:
    parts: List[str] = []

    parts.append("This is my repository.\n")
    parts.append(f"Repository root: {root_dir}\n")

    parts.append("### DIRECTORY TREE ###")
    parts.append("```text")
    parts.append(tree_text)
    parts.append("```")

    parts.append("\n### FILES ###")

    for rel_path, content in files:
        parts.append(f"\n[BEGIN FILE: {rel_path}]")
        parts.append("```text")
        parts.append(content)
        parts.append("```")
        parts.append(f"[END FILE: {rel_path}]\n")

    return "\n".join(parts)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 dump_repo_prompt.py <repo_root> <output_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    tree = build_tree(root_dir)
    files = collect_files(root_dir)
    prompt = build_prompt(root_dir, tree, files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt, encoding="utf-8")

    print(f"Prompt written to {out_path}")

if __name__ == "__main__":
    main()
