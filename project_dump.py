#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# ---------------- settings ----------------

# Extensions we actually care about
ALLOWED_EXTS = {
    ".go",
    ".py",
    ".proto",
    ".txt",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".conf",
    ".ini",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sum",
    ".mod",
    ".lock",
    ".makefile",
    ".mk",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".java",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".sql",
    ".csv",
    ".pdf",
}

# Also include these specific filenames even if extension is weird / missing
ALWAYS_INCLUDE_NAMES = {
    "Makefile",
    "Dockerfile",
    "LICENSE",
    "LICENSE.txt",
    "README",
    "README.txt",
    "README.md",
    "go.mod",
    "go.sum",
    "buf.yaml",
    "buf.gen.yaml",
    "buf.lock",
}

# Directories to skip (not shown in traversal for file contents, but still appear in tree)
SKIP_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "vendor",
    "node_modules",
    ".venv",
    "venv",
}

# For safety, don't read insanely huge binary-ish files for non-PDF
MAX_BYTES_PER_FILE = 200_000

# .txt policy
TXT_LINE_LIMIT = 50
TXT_HEAD_LINES = 10
TXT_TAIL_LINES = 10


# ---------------- helpers ----------------

def build_tree(root_dir: Path) -> str:
    """
    Return a 'tree' style view of the directory, including skipped dirs for context.
    """
    lines = []

    def walk(dir_path: Path, prefix: str = ""):
        # sort dirs first then files
        entries = sorted(
            dir_path.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )

        total = len(entries)
        for i, entry in enumerate(entries):
            connector = "└── " if i == total - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():
                child_prefix = f"{prefix}    " if i == total - 1 else f"{prefix}│   "
                walk(entry, child_prefix)

    lines.append(root_dir.name)
    walk(root_dir)
    return "\n".join(lines)


def should_include_file(path: Path) -> bool:
    """
    Decide if this file should be dumped.
    Rules:
    - if it's in ALWAYS_INCLUDE_NAMES: yes
    - else if its extension is in ALLOWED_EXTS: yes
    - else no
    """
    if path.is_dir():
        return False

    name = path.name
    suffix = path.suffix.lower()

    if name in ALWAYS_INCLUDE_NAMES:
        return True

    if suffix in ALLOWED_EXTS:
        return True

    # Handle some common no-suffix cases like "Makefile" or "Dockerfile"
    # (Dockerfile has no suffix but isn't always in ALWAYS_INCLUDE_NAMES by default)
    if name == "Dockerfile":
        return True

    # Ignore obvious macOS junk
    if name == ".DS_Store":
        return False

    return False


def _read_file_bytes_and_maybe_truncate(path: Path):
    """
    Internal helper for non-PDF files.
    Reads raw bytes and truncates if larger than MAX_BYTES_PER_FILE.
    Returns (decoded_text, was_truncated, raw_len_bytes).
    """
    try:
        raw = path.read_bytes()
    except Exception as e:
        return (f"<<ERROR READING FILE: {e}>>", False, 0)

    raw_len = len(raw)

    truncated = False
    if raw_len > MAX_BYTES_PER_FILE:
        raw = raw[:MAX_BYTES_PER_FILE]
        truncated = True

    # Try to decode as text
    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="replace")
            break
        except Exception:
            continue

    if text is None:
        text = "<<UNDECODABLE CONTENT>>"

    if truncated:
        text += (
            "\n\n<<TRUNCATED: file was larger than "
            f"{MAX_BYTES_PER_FILE} bytes>>\n"
        )

    return (text, truncated, raw_len)


def summarize_txt_if_large(full_text: str) -> str:
    """
    .txt rule:
    - If <= 50 lines total: include whole file
    - If > 50 lines: include first 10 and last 10 lines,
      with an annotation
    """
    lines = full_text.splitlines()
    total_lines = len(lines)

    if total_lines <= TXT_LINE_LIMIT:
        return full_text

    head_lines = lines[:TXT_HEAD_LINES]
    tail_lines = lines[-TXT_TAIL_LINES:]

    preview = []
    preview.append(
        f"<<SHOWING FIRST {TXT_HEAD_LINES} AND LAST {TXT_TAIL_LINES} LINES OF LARGE TEXT FILE "
        f"(total {total_lines} lines)>>"
    )
    preview.append("")
    preview.append(f"<<BEGIN FIRST {TXT_HEAD_LINES} LINES>>")
    preview.extend(head_lines)
    preview.append(f"<<END FIRST {TXT_HEAD_LINES} LINES>>")
    preview.append("")
    preview.append(f"<<BEGIN LAST {TXT_TAIL_LINES} LINES>>")
    preview.extend(tail_lines)
    preview.append(f"<<END LAST {TXT_TAIL_LINES} LINES>>")

    return "\n".join(preview)


def extract_pdf_text(path: Path) -> str:
    """
    Extract ALL text from a PDF using pypdf.
    No summarization.
    If pypdf is not installed or extraction fails, return a fallback note.
    """
    try:
        import pypdf
    except Exception:
        # pypdf not available, warn instead of crashing
        try:
            size_bytes = path.stat().st_size
        except Exception:
            size_bytes = "UNKNOWN"
        return (
            "<<PDF CONTENT NOT EXTRACTED: 'pypdf' not installed>>\n"
            f"File name: {path.name}\n"
            f"Approx size: {size_bytes} bytes\n"
            "Run: pip install pypdf\n"
        )

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        return (
            f"<<PDF READ ERROR: {e}>>\n"
            f"File name: {path.name}\n"
        )

    # dump every page in full, mark page boundaries
    out_chunks = [f"<<PDF EXTRACTED TEXT: {path.name} ({len(reader.pages)} pages)>>", ""]
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            page_text = f"<<ERROR extracting page {i}: {e}>>"
        out_chunks.append(f"----- [PAGE {i+1}] -----")
        out_chunks.append(page_text)

    return "\n".join(out_chunks)


def read_file(path: Path) -> str:
    """
    Returns the text content we embed in the final prompt per file.
    Behavior:
    - .pdf  -> extract full text (no summarization)
    - .txt  -> keep whole file if <= 50 lines, otherwise first/last 10 lines
    - other -> try to decode as text (utf-8/latin-1), with byte cap
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(path)

    text, _trunc, _raw_len = _read_file_bytes_and_maybe_truncate(path)

    if suffix == ".txt":
        return summarize_txt_if_large(text)

    return text


def collect_files(root_dir: Path):
    """
    Walk the directory tree and collect (relative_path, file_content)
    for all include-worthy files.
    We skip SKIP_DIRS while descending.
    """
    collected = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # prune noisy dirs from recursion
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            p = Path(dirpath) / fname
            if should_include_file(p):
                rel_path = p.relative_to(root_dir)
                file_text = read_file(p)
                collected.append((str(rel_path), file_text))

    # deterministic sort
    collected.sort(key=lambda x: x[0].lower())
    return collected


def build_prompt(root_dir: Path, tree_text: str, files_data: list[tuple[str, str]]) -> str:
    """
    Final prompt text:
    - Directory tree
    - Then each file in sorted order
    """
    parts = []

    parts.append("This is my repository. Below is the directory tree, followed by the contents of key files.\n")
    parts.append(f"Repository root: {root_dir}\n")

    parts.append("### DIRECTORY TREE ###")
    parts.append("```text")
    parts.append(tree_text)
    parts.append("```")

    parts.append("### FILES ###")

    for rel_path, content in files_data:
        parts.append(f"\n[BEGIN FILE: {rel_path}]\n```text")
        parts.append(content)
        parts.append("```")
        parts.append(f"[END FILE: {rel_path}]\n")

    return "\n".join(parts)


def main():
    # arg1 = repo root, arg2 = output file
    if len(sys.argv) != 3:
        print("Usage: python3 dump_repo_prompt.py /path/to/repo_root /path/to/output_file.txt", file=sys.stderr)
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    if not (root_dir.exists() and root_dir.is_dir()):
        print(f"Error: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # build tree
    tree_text = build_tree(root_dir)

    # collect and read files
    files_data = collect_files(root_dir)

    # build full prompt
    prompt_text = build_prompt(root_dir, tree_text, files_data)

    # make sure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    print(f"Prompt written to {out_path}")


if __name__ == "__main__":
    main()
