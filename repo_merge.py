#!/usr/bin/env python3
"""
repo_merge.py

Merge text files from a local git repository into Markdown parts, optionally create a summary,
and optionally zip all output files into a single archive. Supports forced splitting of very
large files across parts to ensure no part exceeds MAX_PART_SIZE.

Also supports restoring a repository from Markdown parts or a zip archive.

New features:
- --verify: round-trip verification after merging (merge -> restore in tempdir -> compare checksums)
- --normalize-eol: enforce LF or CRLF normalization when writing parts and when restoring
- -s/--save-to: save outputs to a specific folder
- --force-overwrite: skip interactive overwrite prompts
- --show-llm-instructions: print the embedded LLM instructions and exit
- --show-llm-prompt: print a ready-to-use user prompt for an LLM and exit
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_MAX_PART_SIZE = "50M"


# ---------- Helpers ----------

def parse_size(s: Optional[str]) -> int:
    if s is None:
        s = DEFAULT_MAX_PART_SIZE
    s = str(s).strip().upper()
    try:
        if s.endswith("GB"):
            return int(float(s[:-2]) * 1024**3)
        if s.endswith("G"):
            return int(float(s[:-1]) * 1024**3)
        if s.endswith("MB"):
            return int(float(s[:-2]) * 1024**2)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1024**2)
        if s.endswith("KB"):
            return int(float(s[:-2]) * 1024)
        if s.endswith("K"):
            return int(float(s[:-1]) * 1024)
        return int(float(s))
    except Exception as e:
        raise ValueError(f"Invalid size value: {s}") from e


def compute_sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_binary_file(path: Path, blocksize: int = 1024) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(blocksize)
            if not chunk:
                return False
            if b'\x00' in chunk:
                return True
            try:
                chunk.decode('utf-8')
                return False
            except UnicodeDecodeError:
                return True
    except Exception:
        return True


def git_ignored_paths(repo_root: Path) -> Set[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-oi", "--exclude-standard", "-z"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL
        )
        if not out:
            return set()
        parts = out.split(b'\x00')
        paths = set()
        for p in parts:
            if p:
                rel = Path(p.decode('utf-8', 'surrogateescape'))
                paths.add(rel)
                for i in range(1, len(rel.parts) + 1):
                    paths.add(Path(*rel.parts[:i]))
        return paths
    except Exception:
        return set()


def parse_simple_gitignore(repo_root: Path) -> Set[Path]:
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return set()
    patterns = []
    with gitignore.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    ignored = set()
    for pat in patterns:
        if pat.endswith('/'):
            for p in repo_root.rglob(pat.rstrip('/')):
                try:
                    rel = p.relative_to(repo_root)
                    ignored.add(rel)
                    for i in range(1, len(rel.parts) + 1):
                        ignored.add(Path(*rel.parts[:i]))
                except Exception:
                    pass
        else:
            for p in repo_root.rglob(pat):
                try:
                    rel = p.relative_to(repo_root)
                    ignored.add(rel)
                    for i in range(1, len(rel.parts) + 1):
                        ignored.add(Path(*rel.parts[:i]))
                except Exception:
                    pass
            for p in repo_root.rglob('*'):
                try:
                    rels = str(p.relative_to(repo_root))
                except Exception:
                    continue
                if fnmatch.fnmatch(rels, pat):
                    try:
                        r = Path(rels)
                        ignored.add(r)
                        for i in range(1, len(r.parts) + 1):
                            ignored.add(Path(*r.parts[:i]))
                    except Exception:
                        pass
    return ignored


def fence_language_for_suffix(suffix: str) -> str:
    mapping = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java',
        '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.html': 'html', '.css': 'css',
        '.md': 'markdown', '.json': 'json', '.yml': 'yaml', '.yaml': 'yaml',
        '.sh': 'bash', '.ps1': 'powershell', '.rb': 'ruby', '.go': 'go',
        '.rs': 'rust', '.txt': '', '.xml': 'xml', '.toml': 'toml',
    }
    return mapping.get(suffix.lower(), '')


def normalize_eol_text(content_text: str, mode: str) -> str:
    """
    mode: 'preserve', 'lf', 'crlf'
    """
    if mode == 'preserve' or mode is None:
        return content_text
    # Normalize all CRLF and CR to LF first
    text = content_text.replace('\r\n', '\n').replace('\r', '\n')
    if mode == 'lf':
        return text
    if mode == 'crlf':
        return text.replace('\n', '\r\n')
    return content_text


def normalize_eol_bytes(b: bytes, mode: str) -> bytes:
    if mode == 'preserve' or mode is None:
        return b
    try:
        text = b.decode('utf-8', errors='replace')
    except Exception:
        text = b.decode('utf-8', errors='replace')
    text = normalize_eol_text(text, mode)
    return text.encode('utf-8')


def _choose_fence_for_content(content_text: str) -> str:
    runs = re.findall(r'`+', content_text)
    longest = 0
    for r in runs:
        if len(r) > longest:
            longest = len(r)
    fence_len = max(3, longest + 1)
    return '`' * fence_len


# ---------- LLM instructions and prompt templates ----------

LLM_INSTRUCTIONS = """# LLM INSTRUCTION

You are given one or more Markdown part files that together contain a machine-readable archive of a local Git repository. Treat the provided files as the complete archive input; do not expect any other files. Your job is to interpret the archive format, locate each original repository file and its exact contents, and (when asked) verify, summarize, or reconstruct the repository tree.

Purpose
- These Markdown parts are a faithful, text-only export of a repository's text files intended for automated processing by an LLM.
- Each part is self-contained and, when combined in part order, represents the full set of included files and their contents.

High-level structure of each part
1. Optional LLM prompt block (human-readable instruction) may appear at the very top of the first part.
2. A YAML-like metadata block follows, fenced with triple backticks and the literal `yaml` marker:
   ```
   ```yaml
   merged_repo_file: v1
   generated_by: "repo_merge.py"
   generated_at: 2025-12-01T00:00:00Z
   repo_root: "/absolute/path/where/repo/was/read"
   repo_name: "my-repo"
   notes: "short notes"
   ```
   ```
   - **merged_repo_file**: archive format version.
   - **generated_by**: must be "repo_merge.py".
   - **generated_at**: UTC timestamp in ISO 8601 with trailing Z.
   - **repo_root**: absolute path on the machine that produced the parts (informational only).
   - **repo_name**: repository directory name (used as default output name).
   - Additional metadata may appear but is optional.

File sections
- After metadata there is a header `# Repository: <repo_name>` and `## Files`.
- Each original repository file is represented by a section with this exact header form on its own line:
  ```
  ### FILE: `relative/path/to/file.ext`
  ```
  - The path is always **relative** to the repository root and uses forward slashes.
  - Paths may include directories (e.g., `src/module/file.py`) and may include filenames with dots.

- Immediately following the `### FILE:` header there are one or more metadata lines (Markdown list style) describing the file:
  - `- **size:** <bytes> bytes`
  - `- **sha1:** `<40-hex-chars>`  (optional but present when available)
  - There may be other metadata lines; do not assume only these two.

- After the metadata lines there is a fenced code block that contains the file contents exactly as read:
  - The fence may use **three or more backticks** (e.g., ````` or ````) and may include a language hint (e.g., ```python).
  - The generator chooses a fence length that is longer than any run of backticks inside the file content to avoid premature termination.
  - The code block contains the file bytes decoded as UTF-8 with invalid sequences replaced; line endings are normalized to `\n` in the parts unless otherwise noted in metadata.
  - The closing fence matches the opening fence exactly (same number of backticks).
  - **Important**: The content inside the fence is the file content; preserve it exactly (after decoding replacement) when reconstructing the file. Do not add or remove trailing newlines beyond what is inside the fence.

Multi-part and split files
- Files are normally stored intact in a single part. If the generator was run with a "force-split" option, very large files may be split across multiple parts.
  - In that case the same `### FILE:` header and metadata may appear in multiple parts for the same relative path; concatenate the captured content chunks in part order to reconstruct the original file.
  - Use the SHA-1 metadata (if present) to validate the final concatenated content.

End-of-archive lists (present at the end of the last part)
- `## Ignored by .gitignore` — list of paths that were intentionally excluded (the generator may omit this list if requested).
- `## Skipped binary files` — list of binary files that were not included.

How to validate and restore files
- For each `### FILE:` section:
  1. Extract the relative path.
  2. Read the metadata lines and capture the `sha1` value if present.
  3. Capture the entire code-fence content exactly (respecting fence length).
  4. If a file appears in multiple parts, concatenate the captured bytes in the order of parts.
  5. When writing the restored file to disk, write the bytes exactly as decoded from the code block (or re-encode using UTF-8) without adding or removing extra newlines.
  6. If a `sha1` value is present, compute SHA-1 of the restored bytes and compare; report any mismatch.

Robust parsing rules (what an LLM should expect)
- Accept fences of length 3 or more backticks and match closing fence by exact backtick count.
- Accept optional language hints after the opening fence.
- Allow extra blank lines between header, metadata, and fence, but prefer the nearest fenced block after the metadata for that file.
- Normalize Windows-style backslashes in paths to forward slashes when reconstructing directories.
- Ignore any `### FILE:` headers that are not followed by a fenced block (emit a warning and continue).
- If a part fails validation (missing YAML, missing `generated_by`, no file headers, or missing fences), report the reason and continue processing other parts.

Behavioral expectations for an LLM consumer
- Treat the parts as authoritative: do not attempt to fetch files from the original repo_root path.
- When asked to reconstruct the repository, create directories and write files using the relative paths exactly as provided.
- When asked to summarize or analyze the repository, use the file headers, sizes, and SHA-1 values to reference files precisely.
- When asked to verify integrity, compute SHA-1 on the restored bytes and compare to the `sha1` metadata; report mismatches and missing checksums.
- When asked to list files, report the relative paths and the part(s) they appear in.
- If a file is split across parts, indicate that in any summary or verification output.

End of instruction.
"""

def build_llm_user_prompt() -> str:
    """
    Return a ready-to-use user prompt that instructs an LLM how to behave when given the Markdown parts.
    This prompt is intended to be pasted to an LLM to set the collaboration mode.
    """
    prompt = (
        "Hej, I have uploaded a set of Markdown files generated by repo_merge.py.\n\n"
        "The very first part file begins with an \"LLM INSTRUCTION\" block that describes the format and conventions of the archive.\n"
        "Please read and follow those instructions carefully when interpreting the files.\n\n"
        "After that, treat the uploaded Markdown files as the repository contents. Each part begins with metadata and then lists files with headers, SHA‑1 hashes, and fenced code blocks. "
        "At the end, there may be a summary file showing a tree of files and which part they are in.\n\n"
        "I want you to act as my collaborator on this codebase. Please:\n"
        "- Parse the uploaded Markdown files and treat them as the repository contents.\n"
        "- Be able to navigate the file tree and reference specific files by their relative paths.\n"
        "- Help me review, refactor, and extend the code. For example, I may ask you to explain how a function works, suggest improvements, add new features, or fix bugs.\n"
        "- When you provide code changes, show the full updated file or the relevant diff in fenced code blocks.\n"
        "- Keep track of which part/file the code comes from so we can maintain consistency across the repository.\n\n"
        "When you restore files, preserve file bytes exactly as provided in the fenced blocks (after UTF-8 decoding with replacement). "
        "If SHA-1 checksums are present, verify them and report any mismatches.\n\n"
        "To start, please confirm you understand the format of the uploaded Markdown files, including the \"LLM INSTRUCTION\" block in the first file, "
        "and that you can work with me on the code they contain.\n\n"
        "Thank you in advance for your help – I really appreciate it."
    )
    return prompt


# ---------- Part writer and other components ----------
# (The rest of the script is unchanged in structure; it uses LLM_INSTRUCTIONS where needed.)
# For brevity and clarity the full implementation follows below (unchanged logic from previous version),
# with references to LLM_INSTRUCTIONS and the new CLI flags integrated.

# ---------- Part writer ----------

class PartWriter:
    def __init__(self, base_out: Optional[Path], repo_name: str, repo_root: Path,
                 max_part_size: int, llm_prompt: Optional[str] = None, force_split: bool = False,
                 out_dir_override: Optional[Path] = None, force_overwrite: bool = False,
                 normalize_eol: str = 'preserve'):
        self.repo_name = repo_name
        self.repo_root = repo_root
        self.max_part_size = max_part_size
        self.llm_prompt = llm_prompt
        self.force_split = force_split
        self.force_overwrite = force_overwrite
        self.normalize_eol = normalize_eol

        self.part_index = 1
        self.current_file = None
        self.current_bytes = 0

        if base_out:
            self.out_dir = base_out.parent
            name = base_out.name
            if ".part" in name and name.lower().endswith(".md"):
                idx = name.lower().rfind(".part")
                self.base_name = name[:idx]
            else:
                self.base_name = name[:-3] if name.lower().endswith(".md") else name
        else:
            self.out_dir = Path.cwd()
            self.base_name = repo_name

        if out_dir_override:
            self.out_dir = out_dir_override

        if self.out_dir.exists() and not self.force_overwrite:
            existing = list(self.out_dir.glob(f"{self.base_name}.part*.md")) + list(self.out_dir.glob(f"{self.base_name}.summary.md")) + list(self.out_dir.glob(f"{self.base_name}-merged.zip"))
            if existing:
                print(f"Output directory '{self.out_dir}' already contains files that may be overwritten.")
                while True:
                    resp = input("Choose action: [o]verwrite files, [w]rite into existing (append), [c]ancel: ").strip().lower()
                    if resp in ("o", "overwrite", "o"):
                        for p in existing:
                            try:
                                p.unlink()
                            except Exception:
                                pass
                        break
                    elif resp in ("w", "write", "writeto", "write into existing", "merge"):
                        break
                    elif resp in ("c", "cancel"):
                        print("Operation cancelled.")
                        sys.exit(0)
                    else:
                        print("Please enter 'o', 'w', or 'c'.")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_new_part()

    def _part_filename(self, index: int) -> Path:
        return self.out_dir / f"{self.base_name}.part{index:03d}.md"

    def _open_new_part(self):
        if self.current_file:
            self.current_file.close()
        path = self._part_filename(self.part_index)
        if path.exists() and not self.force_overwrite:
            while True:
                resp = input(f"Output file '{path}' exists. Overwrite? [y/N]: ").strip().lower()
                if resp in ("y", "yes"):
                    break
                elif resp in ("n", "no", ""):
                    print("Operation cancelled.")
                    sys.exit(0)
                else:
                    print("Please enter 'y' or 'n'.")
        self.current_file = path.open("wb")
        self.current_bytes = 0
        if self.part_index == 1 and self.llm_prompt:
            prompt_block = "## LLM PROMPT\n\n" + "```\n" + self.llm_prompt.strip() + "\n```\n\n"
            self._write_raw(prompt_block)
        # write top metadata
        self._write_top_metadata()
        print(f"Opened part: {path}")
        self.part_index += 1

    def _write_top_metadata(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        meta = []
        meta.append("```yaml\n")
        meta.append("merged_repo_file: v1\n")
        meta.append("generated_by: \"repo_merge.py\"\n")
        meta.append(f"generated_at: {now}\n")
        meta.append(f"repo_root: \"{str(self.repo_root.resolve())}\"\n")
        meta.append(f"repo_name: \"{self.repo_name}\"\n")
        meta.append("notes: \"Text files included; binary files listed below; .gitignored files excluded unless --no-gitignore was used\"\n")
        meta.append("```\n\n")
        self._write_raw("".join(meta))

    def _write_raw(self, s: str):
        if self.current_file is None:
            self._open_new_part()
        data = s.encode("utf-8")
        offset = 0
        while offset < len(data):
            space = self.max_part_size - self.current_bytes
            if space <= 0:
                self._open_new_part()
                space = self.max_part_size - self.current_bytes
            to_write = min(space, len(data) - offset)
            chunk = data[offset:offset + to_write]
            self.current_file.write(chunk)
            self.current_bytes += len(chunk)
            offset += to_write

    def _write_bytes_force(self, s: str):
        if self.current_file is None:
            self._open_new_part()
        data = s.encode("utf-8")
        self.current_file.write(data)
        self.current_bytes += len(data)

    def write_section(self, header: str, metadata_lines: List[str], content_path: Path, fence_lang: str) -> int:
        """
        Write a file section. Choose a fence length that does not appear in the file content.
        Apply EOL normalization to the content_text before writing if requested.
        """
        part_written = self.part_index - 1
        header_block = header + "\n\n" + "".join(metadata_lines) + "\n"

        try:
            raw = content_path.read_bytes()
            content_text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            content_text = f"[ERROR reading file: {e}]"

        # apply normalization if requested
        content_text = normalize_eol_text(content_text, self.normalize_eol)

        fence = _choose_fence_for_content(content_text)
        fence_start = fence + (fence_lang if fence_lang else "") + "\n"
        fence_end = fence + "\n"

        if self.force_split:
            header_size = len((header_block + fence_start).encode("utf-8"))
            if self.current_bytes > 0 and (self.current_bytes + header_size) > self.max_part_size:
                self._open_new_part()
                part_written = self.part_index - 1
            self._write_raw(header_block)
            self._write_raw(fence_start)
            self._write_raw(content_text)
            if content_text.endswith("\n"):
                self._write_raw(fence_end)
            else:
                self._write_raw("\n" + fence_end)
            return part_written

        if self.current_bytes > 0 and (self.current_bytes + len(header_block.encode("utf-8")) + len(fence_start.encode("utf-8"))) > self.max_part_size:
            self._open_new_part()
            part_written = self.part_index - 1

        self._write_bytes_force(header_block)
        self._write_bytes_force(fence_start)
        self._write_bytes_force(content_text)
        if content_text.endswith("\n"):
            self._write_bytes_force(fence_end)
        else:
            self._write_bytes_force("\n" + fence_end)
        return part_written

    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file = None


# ---------- Dry-run planner ----------

class PartPlanner:
    def __init__(self, base_name: str, max_part_size: int):
        self.base_name = base_name
        self.max_part_size = max_part_size
        self.parts: List[List[Tuple[str, int]]] = []
        self.current_part: List[Tuple[str, int]] = []
        self.current_bytes = 0

    def add_section(self, relpath: str, section_size: int):
        if self.current_bytes > 0 and (self.current_bytes + section_size) > self.max_part_size:
            self._start_new_part()
        self.current_part.append((relpath, section_size))
        self.current_bytes += section_size

    def _start_new_part(self):
        if self.current_part:
            self.parts.append(self.current_part)
        self.current_part = []
        self.current_bytes = 0

    def finish(self):
        if self.current_part:
            self.parts.append(self.current_part)
            self.current_part = []
            self.current_bytes = 0


# ---------- Collect files for merge ----------

def collect_files(repo_root: Path, script_path: Optional[Path], extensions: Optional[Set[str]],
                  include_hidden: bool, ignore_gitignore: bool) -> Tuple[List[Path], List[Path], List[Path]]:
    ignored: Set[Path] = set()
    if not ignore_gitignore:
        ignored = git_ignored_paths(repo_root)
        if not ignored:
            ignored = parse_simple_gitignore(repo_root)

    included: List[Path] = []
    skipped_binary: List[Path] = []
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        if not include_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]

        kept_dirs = []
        for d in dirs:
            full_dir = root_path / d
            try:
                rel_dir = full_dir.relative_to(repo_root)
            except Exception:
                kept_dirs.append(d)
                continue
            skip_dir = False
            for i in range(1, len(rel_dir.parts) + 1):
                if Path(*rel_dir.parts[:i]) in ignored:
                    skip_dir = True
                    break
            if not skip_dir:
                kept_dirs.append(d)
        dirs[:] = kept_dirs

        for fname in files:
            if not include_hidden and fname.startswith("."):
                continue
            full = root_path / fname
            try:
                rel = full.relative_to(repo_root)
            except Exception:
                continue
            if ".git" in rel.parts:
                continue
            if script_path:
                try:
                    if full.resolve() == script_path.resolve():
                        continue
                except Exception:
                    pass
            if fname == (script_path.name if script_path else "repo_merge.py"):
                continue
            if not ignore_gitignore:
                skip_due_to_ignored = False
                for i in range(1, len(rel.parts) + 1):
                    if Path(*rel.parts[:i]) in ignored:
                        skip_due_to_ignored = True
                        break
                if skip_due_to_ignored:
                    continue
            if extensions is not None and full.suffix.lower() not in extensions:
                continue
            if is_binary_file(full):
                skipped_binary.append(rel)
                continue
            included.append(rel)
    included.sort()
    ignored_list = sorted(list(ignored))
    skipped_binary.sort()
    return included, ignored_list, skipped_binary


def compute_section_size(repo_root: Path, rel: Path, fence_lang: str, normalize_eol: str = 'preserve') -> int:
    full = repo_root / rel
    header_block = f"### FILE: `{str(rel)}`\n\n"
    try:
        size = full.stat().st_size
    except Exception:
        size = 0
    sha1 = compute_sha1(full)
    metadata = f"- **size:** {size} bytes  \n- **sha1:** `{sha1}`  \n\n"
    try:
        raw = full.read_bytes()
        content_text = raw.decode('utf-8', errors='replace')
    except Exception:
        content_text = ""
    content_text = normalize_eol_text(content_text, normalize_eol)
    fence = _choose_fence_for_content(content_text)
    fence_start = fence + (fence_lang if fence_lang else "") + "\n"
    fence_end = fence + "\n"
    content_bytes = len(content_text.encode("utf-8"))
    total = len(header_block.encode("utf-8")) + len(metadata.encode("utf-8")) + len(fence_start.encode("utf-8")) + content_bytes + len(fence_end.encode("utf-8"))
    return total


def build_tree_mapping(file_to_part: Dict[str, int]) -> Dict:
    tree: Dict = {}
    for relpath, part in sorted(file_to_part.items()):
        if relpath.endswith(".summary.md"):
            node = tree.setdefault("__files__", [])
            node.append((relpath, part))
            continue
        parts = relpath.split(os.sep)
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        files = node.setdefault("__files__", [])
        files.append((parts[-1], part))
    return tree


def render_tree(tree: Dict, prefix: str = "") -> List[str]:
    lines: List[str] = []
    files = tree.get("__files__", [])
    for fname, part in sorted(files):
        if part == 0:
            lines.append(f"{prefix}- {fname}  (summary)")
        else:
            lines.append(f"{prefix}- {fname}  (part {part:03d})")
    for name in sorted(k for k in tree.keys() if k != "__files__"):
        lines.append(f"{prefix}- {name}/")
        lines.extend(render_tree(tree[name], prefix + "  "))
    return lines


def write_summary_file(repo_name: str, out_dir: Path, file_to_part: Dict[str, int], base_name: Optional[str] = None) -> Path:
    summary_base = base_name if base_name else repo_name
    summary_path = out_dir / f"{summary_base}.summary.md"
    tree = build_tree_mapping(file_to_part)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines: List[str] = []
    lines.append(f"# Repository Summary: {repo_name}\n")
    lines.append(f"- generated_at: {now}\n")
    lines.append("\n## Files and Part Mapping\n")
    rendered = render_tree(tree)
    if not rendered:
        lines.append("_No files included_\n")
    else:
        lines.extend(line + "\n" for line in rendered)
    summary_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote summary file: {summary_path}")
    return summary_path


def create_zip_archive(out_dir: Path, repo_name: str, include_summary: bool, base_name: Optional[str] = None) -> Path:
    zip_name = f"{(base_name if base_name else repo_name)}-merged.zip"
    zip_path = out_dir / zip_name
    folder_prefix = f"{(base_name if base_name else repo_name)}-merged/"
    files_to_zip = sorted([p for p in out_dir.glob("*.part*.md") if p.is_file()])
    summary_path = out_dir / f"{(base_name if base_name else repo_name)}.summary.md"
    if include_summary and summary_path.exists():
        files_to_zip.append(summary_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_zip:
            zf.write(p, arcname=folder_prefix + p.name)
    print(f"Created zip archive: {zip_path}")
    return zip_path


def write_markdown_parts(repo_root: Path, included: List[Path], ignored: List[Path], skipped_binary: List[Path],
                         out_path: Optional[Path], max_part_size: int, dry_run: bool, llm_prompt: Optional[str],
                         create_summary: bool, force_split: bool, save_to: Optional[Path], force_overwrite: bool,
                         normalize_eol: str = 'preserve') -> Tuple[Path, Dict[str, int]]:
    repo_name = repo_root.name
    base_out = out_path if out_path else None
    file_to_part: Dict[str, int] = {}

    out_dir_override = save_to if save_to else None

    if dry_run:
        planner = PartPlanner(repo_name, max_part_size)
        out_dir = (base_out.parent if base_out else Path.cwd())
        if out_dir_override:
            out_dir = out_dir_override
    else:
        writer = PartWriter(base_out, repo_name, repo_root, max_part_size, llm_prompt=llm_prompt,
                            force_split=force_split, out_dir_override=out_dir_override, force_overwrite=force_overwrite,
                            normalize_eol=normalize_eol)
        out_dir = writer.out_dir

    header_text = f"# Repository: {repo_name}\n\n## Files\n\n"
    header_bytes = len(header_text.encode("utf-8"))
    if dry_run:
        planner.add_section("(__metadata_header__)", header_bytes)
    else:
        writer._write_raw(header_text)

    total_files = len(included)
    for idx, rel in enumerate(included, start=1):
        progress_msg = f"Processing {idx}/{total_files}: {str(rel)}"
        print(progress_msg.ljust(120), end="\r", flush=True)

        full = repo_root / rel
        try:
            size = full.stat().st_size
        except Exception:
            size = 0
        sha1 = compute_sha1(full)
        header = f"### FILE: `{str(rel)}`"
        metadata_lines = [
            f"- **size:** {size} bytes  \n",
            f"- **sha1:** `{sha1}`  \n\n"
        ]
        lang = fence_language_for_suffix(full.suffix)
        if dry_run:
            section_size = compute_section_size(repo_root, rel, lang, normalize_eol=normalize_eol)
            planner.add_section(str(rel), section_size)
        else:
            part_written = writer.write_section(header, metadata_lines, full, lang)
            file_to_part[str(rel)] = part_written

    if total_files > 0:
        print(" " * 120, end="\r", flush=True)

    if dry_run:
        planner.finish()
        total_parts = len(planner.parts)
        total_bytes = sum(sum(sz for _, sz in part) for part in planner.parts)
        print("Dry-run summary")
        print(f"Repository: {repo_name}")
        print(f"Max part size: {max_part_size} bytes")
        print(f"Planned parts: {total_parts}")
        print(f"Total bytes (approx): {total_bytes}")
        planned_file_to_part: Dict[str, int] = {}
        for i, part in enumerate(planner.parts, start=1):
            part_bytes = sum(sz for _, sz in part)
            print(f" Part {i:03d}: {len(part)} files, {part_bytes} bytes")
            for relpath, sz in part:
                print(f"   - {relpath} ({sz} bytes)")
                planned_file_to_part[relpath] = i
        if create_summary:
            print("\nPlanned summary (tree):")
            tree = build_tree_mapping(planned_file_to_part)
            for line in render_tree(tree):
                print(line)
        return out_dir, planned_file_to_part

    writer._write_raw("## Skipped binary files\n\n")
    if skipped_binary:
        for p in skipped_binary:
            writer._write_raw(f"- `{str(p)}`\n")
    else:
        writer._write_raw("_None_\n")
    writer._write_raw("\n")

    writer.close()
    print("Merging complete.")

    base_name = None
    if base_out:
        name = base_out.name
        if ".part" in name and name.lower().endswith(".md"):
            idx = name.lower().rfind(".part")
            base_name = name[:idx]
        else:
            base_name = name[:-3] if name.lower().endswith(".md") else name

    if create_summary:
        summary_path = write_summary_file(repo_name, out_dir, file_to_part, base_name=base_name)
        file_to_part[str(summary_path.name)] = 0

    return out_dir, file_to_part


# ---------- Robust parser for restore ----------

def _validate_markdown_file_format_with_reason(path: Path) -> Tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Cannot read file: {e}"

    meta_match = re.search(r"```yaml(.*?)```", text, re.DOTALL)
    if not meta_match:
        return False, "Missing YAML metadata block (```yaml ... ```)."

    meta_text = meta_match.group(1)
    if "merged_repo_file" not in meta_text:
        return False, "YAML metadata block missing 'merged_repo_file' entry."
    if 'generated_by: "repo_merge.py"' not in meta_text and "generated_by: 'repo_merge.py'" not in meta_text:
        return False, "YAML metadata block missing 'generated_by: \"repo_merge.py\"'."

    if "### FILE:" not in text:
        return False, "No '### FILE:' headers found."

    for fh in re.finditer(r"^### FILE: `([^`]+)`", text, re.MULTILINE):
        start_pos = fh.end()
        lookahead = text[start_pos:start_pos + 4000]
        if not re.search(r"^(`{3,})([^\n]*)\n", lookahead, re.MULTILINE):
            return False, f"File section for '{fh.group(1)}' does not contain a fenced code block after the header."

    if "- **sha1:**" not in text:
        return True, "Valid format but no '- **sha1:**' entries found (checksums unavailable)."

    return True, "OK"


def _extract_repo_name_from_metadata(text: str) -> Optional[str]:
    m = re.search(r'repo_name:\s*["\']([^"\']+)["\']', text)
    if m:
        return m.group(1)
    return None


def _parse_markdown_parts_robust(paths: List[Path]) -> Tuple[Optional[str], Dict[str, List[Tuple[Optional[str], bytes]]]]:
    file_chunks: Dict[str, List[Tuple[Optional[str], bytes]]] = {}
    repo_name: Optional[str] = None
    meta_re = re.compile(r"```yaml(.*?)```", re.DOTALL)

    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        if repo_name is None:
            mmeta = meta_re.search(text)
            if mmeta:
                repo_name = _extract_repo_name_from_metadata(mmeta.group(1))
        header_iter = list(re.finditer(r"^### FILE: `([^`]+)`\s*$", text, re.MULTILINE))
        for i, h in enumerate(header_iter):
            start = h.end()
            end = header_iter[i+1].start() if i+1 < len(header_iter) else len(text)
            section = text[start:end]
            fence_start_m = re.search(r"^(`{3,})([^\n]*)\n", section, re.MULTILINE)
            if not fence_start_m:
                continue
            fence = fence_start_m.group(1)
            content_start = fence_start_m.end()
            close_re = re.compile(r"^" + re.escape(fence) + r"\s*$", re.MULTILINE)
            close_m = close_re.search(section, content_start)
            if not close_m:
                content = section[content_start:]
            else:
                content = section[content_start:close_m.start()]
            rel = h.group(1).strip().replace("\\", "/")
            meta_block = section[:fence_start_m.start()]
            sha_m = re.search(r"- \*\*sha1:\*\*\s*`([0-9a-fA-F]{40})`", meta_block)
            sha = sha_m.group(1) if sha_m else None
            content_bytes = content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
            file_chunks.setdefault(rel, []).append((sha, content_bytes))
    return repo_name, file_chunks


def _collect_source_paths_for_restore(source: str) -> Tuple[List[Path], Optional[Path]]:
    src = Path(source)
    tempdir = None
    paths: List[Path] = []

    if src.is_dir():
        paths = sorted([p for p in src.glob("*.md") if p.is_file()])
        return paths, None

    if src.is_file() and src.suffix.lower() == ".zip":
        tempdir = Path(tempfile.mkdtemp(prefix="repo_merge_restore_"))
        try:
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(tempdir)
        except Exception as e:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract zip: {e}")
        paths = sorted([p for p in tempdir.rglob("*.md") if p.is_file()])
        return paths, tempdir

    matched = sorted([Path(p) for p in glob.glob(source)])
    paths = [p for p in matched if p.suffix.lower() == ".md" and p.is_file()]
    return paths, None


def restore_repo(source: str, new_name: Optional[str] = None, check_sha: bool = False, list_files_only: bool = False,
                 force_overwrite: bool = False, normalize_eol: str = 'preserve') -> Tuple[Path, List[Tuple[Path, Optional[str]]]]:
    print(f"Restoring from: {source}")
    try:
        paths, tempdir = _collect_source_paths_for_restore(source)
    except Exception as e:
        print(f"Error preparing source: {e}")
        sys.exit(1)

    if not paths:
        print("No Markdown files found to restore. Aborting.")
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
        sys.exit(1)

    valid_paths: List[Path] = []
    for p in paths:
        valid, reason = _validate_markdown_file_format_with_reason(p)
        if not valid:
            print(f"Warning: Ignoring '{p}' — reason: {reason}")
            continue
        else:
            if reason != "OK":
                print(f"Note: '{p}': {reason}")
            valid_paths.append(p)

    if not valid_paths:
        print("No valid Markdown part files found to restore after validation. Aborting.")
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
        sys.exit(1)

    repo_name, _ = _parse_markdown_parts_robust(valid_paths)
    if list_files_only:
        print("Files found in parts (path -> parts):")
        path_to_parts: Dict[str, List[str]] = {}
        for p in valid_paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"^### FILE: `([^`]+)`", text, re.MULTILINE):
                rel = m.group(1).strip().replace("\\", "/")
                path_to_parts.setdefault(rel, []).append(p.name)
        for rel, parts in sorted(path_to_parts.items()):
            print(f"{rel}  ->  {', '.join(parts)}")
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
        return Path(), []

    repo_name, file_chunks = _parse_markdown_parts_robust(valid_paths)
    if repo_name is None:
        repo_name = "restored-repo"
    target_name = new_name if new_name else repo_name

    target_dir = Path(target_name)
    if not target_dir.is_absolute():
        target_dir = Path.cwd() / target_dir

    if target_dir.exists():
        print(f"Target directory '{target_dir}' already exists.")
        if not force_overwrite:
            while True:
                resp = input("Choose action: [o]verwrite (delete and recreate), [m]merge (write into existing), [c]ancel: ").strip().lower()
                if resp in ("o", "overwrite"):
                    try:
                        shutil.rmtree(target_dir)
                    except Exception as e:
                        print(f"Failed to remove existing directory: {e}")
                        if tempdir:
                            shutil.rmtree(tempdir, ignore_errors=True)
                        sys.exit(1)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    break
                elif resp in ("m", "merge"):
                    break
                elif resp in ("c", "cancel"):
                    print("Restore cancelled.")
                    if tempdir:
                        shutil.rmtree(tempdir, ignore_errors=True)
                    sys.exit(0)
                else:
                    print("Please enter 'o', 'm', or 'c'.")
        else:
            try:
                shutil.rmtree(target_dir)
            except Exception:
                pass
            target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    restored_files: List[Tuple[Path, Optional[str]]] = []
    for relpath, chunks in file_chunks.items():
        rp = relpath.lstrip("./")
        if os.path.isabs(rp):
            rp = rp.lstrip(os.sep)
        out_path = target_dir / Path(rp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined = b"".join(chunk for _, chunk in chunks)
        # apply normalization if requested
        combined = normalize_eol_bytes(combined, normalize_eol)
        if out_path.exists() and not force_overwrite:
            while True:
                resp = input(f"File '{out_path}' exists. Overwrite? [y/N]: ").strip().lower()
                if resp in ("y", "yes"):
                    break
                elif resp in ("n", "no", ""):
                    print(f"Skipping '{out_path}'.")
                    combined = None
                    break
                else:
                    print("Please enter 'y' or 'n'.")
        if combined is not None:
            try:
                out_path.write_bytes(combined)
            except Exception as e:
                print(f"Failed to write {out_path}: {e}")
                if tempdir:
                    shutil.rmtree(tempdir, ignore_errors=True)
                sys.exit(1)
        expected_sha = None
        for sha, _ in chunks:
            if sha:
                expected_sha = sha
                break
        restored_files.append((out_path, expected_sha))

    print(f"Restored {len([p for p, _ in restored_files if p.exists()])} files to: {target_dir}")

    if check_sha:
        print("Verifying SHA-1 checksums...")
        mismatches = []
        missing_sha = []
        for path, expected in restored_files:
            if not path.exists():
                continue
            actual = compute_sha1(path)
            if expected is None:
                missing_sha.append(str(path.relative_to(target_dir)))
            else:
                if actual.lower() != expected.lower():
                    mismatches.append((str(path.relative_to(target_dir)), expected.lower(), actual.lower()))
        if missing_sha:
            print("Warning: The following restored files did not have SHA-1 metadata in the markdown (cannot verify):")
            for p in missing_sha:
                print(f" - {p}")
        if mismatches:
            print("SHA-1 mismatches detected:")
            for p, exp, act in mismatches:
                print(f" - {p}: expected {exp}, actual {act}")
            print("Restore completed with checksum mismatches.")
        else:
            print("All verified files match their SHA-1 checksums.")

    if tempdir:
        shutil.rmtree(tempdir, ignore_errors=True)

    return target_dir, restored_files


# ---------- Verify round-trip ----------

def verify_round_trip(repo_root: Path, parts_dir: Path, parts_base_pattern: str, included_files: List[Path], normalize_eol: str) -> bool:
    pattern = parts_base_pattern
    matched = sorted([p for p in parts_dir.glob(pattern) if p.is_file()])
    if not matched:
        print("No part files found for verification.")
        return False

    tempdir = Path(tempfile.mkdtemp(prefix="repo_merge_verify_"))
    try:
        repo_name, file_chunks = _parse_markdown_parts_robust(matched)
        restored_count = 0
        mismatches = []
        missing = []
        for rel, chunks in file_chunks.items():
            out_path = tempdir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined = b"".join(chunk for _, chunk in chunks)
            combined = normalize_eol_bytes(combined, normalize_eol)
            out_path.write_bytes(combined)
            restored_count += 1

        for rel in included_files:
            orig = repo_root / rel
            restored = tempdir / rel
            if not restored.exists():
                missing.append(str(rel))
                continue
            try:
                orig_bytes = orig.read_bytes()
            except Exception:
                orig_bytes = b""
            orig_bytes = normalize_eol_bytes(orig_bytes, normalize_eol)
            restored_bytes = restored.read_bytes()
            sha_orig = compute_sha1_bytes(orig_bytes)
            sha_rest = compute_sha1_bytes(restored_bytes)
            if sha_orig.lower() != sha_rest.lower():
                mismatches.append((str(rel), sha_orig.lower(), sha_rest.lower()))
        if missing:
            print("Verification failed: some files were not restored:")
            for m in missing:
                print(f" - {m}")
        if mismatches:
            print("Verification failed: checksum mismatches:")
            for p, exp, act in mismatches:
                print(f" - {p}: expected {exp}, restored {act}")
        if not missing and not mismatches:
            print("Round-trip verification succeeded: all files match.")
            return True
        return False
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


# ---------- CLI ----------

def default_output_name(repo_root: Path) -> str:
    repo_name = repo_root.name or "repo"
    return f"{repo_name}.part001.md"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo_merge.py",
        description="Merge text files from a local git repository into Markdown parts, or restore from parts."
    )
    p.add_argument("-g", "--git-repo", help="Path to the repository root (for merge mode)")
    p.add_argument("-o", "--output", help="Base output filename or path (optional). If omitted, defaults to <repo-name>.part001.md")
    p.add_argument("-m", "--max-size", default=DEFAULT_MAX_PART_SIZE,
                   help=f"Maximum bytes per part (e.g., 50M, 200K). Default: {DEFAULT_MAX_PART_SIZE}")
    p.add_argument("-e", "--extensions", help="Comma-separated list of file extensions to include (e.g., .py,.md or py,md). If omitted, all text files are considered.")
    p.add_argument("-x", "--dry-run", action="store_true", help="Do not write files; show how many parts would be produced and list files per part.")
    p.add_argument("--no-gitignore", action="store_true", help="Ignore .gitignore and include all files in the repository.")
    p.add_argument("--include-hidden", action="store_true", help="Include hidden files and folders (names starting with a dot). By default hidden files are skipped.")
    p.add_argument("--no-summary", action="store_true", help="Do not create the <repo-name>.summary.md summary file.")
    p.add_argument("-z", "--zip", action="store_true", help="Create a zip archive named <repo-name>-merged.zip containing all parts and the summary. Files inside the zip are placed under <repo-name>-merged/.")
    p.add_argument("--force-split", action="store_true", help="Force split very large files across parts to not exceed MAX_PART_SIZE.")
    p.add_argument("-r", "--restore", help="Restore a repository from Markdown parts, a zip file, or a wildcard path (e.g., /path/to/*.md).")
    p.add_argument("-n", "--new-name", help="New base folder name or path for restored repository. If omitted, original repo_name from metadata is used.")
    p.add_argument("-c", "--check", action="store_true", help="When restoring, verify restored files against SHA-1 checksums in the Markdown metadata.")
    p.add_argument("--list-files", action="store_true", help="When restoring, list files found in parts and which parts they appear in, then exit.")
    p.add_argument("-s", "--save-to", help="Folder path where merged markdown files and zip should be saved.")
    p.add_argument("--force-overwrite", action="store_true", help="Do not prompt before overwriting existing files or directories.")
    p.add_argument("--normalize-eol", choices=['preserve', 'lf', 'crlf'], default='preserve', help="Normalize line endings when writing parts and when restoring. Default: preserve.")
    p.add_argument("--verify", action="store_true", help="After merging, perform a round-trip verification (merge -> restore -> compare checksums).")
    p.add_argument("--show-llm-instructions", action="store_true", help="Print the embedded LLM instructions and exit.")
    p.add_argument("--show-llm-prompt", action="store_true", help="Print a ready-to-use user prompt for an LLM and exit.")
    return p


def main(argv: Optional[List[str]] = None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Show LLM instructions or prompt if requested, then exit
    if args.show_llm_instructions:
        print(LLM_INSTRUCTIONS)
        return
    if args.show_llm_prompt:
        print(build_llm_user_prompt())
        return

    if args.restore:
        restore_repo(args.restore, new_name=args.new_name, check_sha=args.check, list_files_only=args.list_files,
                     force_overwrite=args.force_overwrite, normalize_eol=args.normalize_eol)
        return

    if not args.git_repo:
        parser.error("You must specify -g/--git-repo for merge mode, or -r/--restore for restore mode.")

    repo_root = Path(args.git_repo).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print("Error: repo root does not exist or is not a directory.")
        sys.exit(1)

    out_file = Path(args.output) if args.output else Path.cwd() / default_output_name(repo_root)
    save_to = Path(args.save_to) if args.save_to else None
    if save_to:
        if not save_to.is_absolute():
            save_to = Path.cwd() / save_to

    try:
        max_part_size = parse_size(args.max_size)
    except Exception as e:
        print(f"Error parsing max-size: {e}")
        sys.exit(1)

    extensions = None
    if args.extensions:
        parts = [p.strip() for p in args.extensions.split(",") if p.strip()]
        normalized = set()
        for p in parts:
            if not p.startswith("."):
                p = "." + p
            normalized.add(p.lower())
        extensions = normalized

    try:
        script_path = Path(__file__).resolve()
    except Exception:
        script_path = None

    included, ignored, skipped_binary = collect_files(
        repo_root, script_path, extensions, args.include_hidden, args.no_gitignore
    )

    if args.dry_run:
        print(f"Dry-run: repository={repo_root}, files={len(included)}, ignored={len(ignored)}, skipped_binary={len(skipped_binary)}")

    out_dir, file_to_part = write_markdown_parts(
        repo_root,
        included,
        ignored,
        skipped_binary,
        out_file,
        max_part_size,
        args.dry_run,
        llm_prompt=LLM_INSTRUCTIONS,
        create_summary=not args.no_summary,
        force_split=args.force_split,
        save_to=save_to,
        force_overwrite=args.force_overwrite,
        normalize_eol=args.normalize_eol
    )

    if args.zip:
        repo_name = repo_root.name
        base_name = None
        if out_file:
            name = out_file.name
            if ".part" in name and name.lower().endswith(".md"):
                idx = name.lower().rfind(".part")
                base_name = name[:idx]
            else:
                base_name = name[:-3] if name.lower().endswith(".md") else name
        if args.dry_run:
            print("\nDry-run: planned zip archive")
            zip_name = f"{(base_name if base_name else repo_name)}-merged.zip"
            print(f"Zip name: {zip_name}")
            print(f"When unzipped, files will be placed into folder: {(base_name if base_name else repo_name)}-merged/")
            planned_names = [f"{(base_name if base_name else repo_name)}.part{idx:03d}.md" for idx in range(1, len(file_to_part) + 2)]
            for name in sorted(set(planned_names + [f"{(base_name if base_name else repo_name)}.summary.md"])):
                print(f" - {name}")
        else:
            zip_path = create_zip_archive(out_dir, repo_name, include_summary=not args.no_summary, base_name=base_name)
            print(f"Zip created at: {zip_path}")

    if args.verify and not args.dry_run:
        if out_file:
            name = out_file.name
            if ".part" in name and name.lower().endswith(".md"):
                idx = name.lower().rfind(".part")
                base_name = name[:idx]
            else:
                base_name = name[:-3] if name.lower().endswith(".md") else name
        else:
            base_name = repo_root.name
        parts_pattern = f"{base_name}.part*.md"
        print("Starting round-trip verification...")
        ok = verify_round_trip(repo_root, out_dir, parts_pattern, included, args.normalize_eol)
        if not ok:
            print("Round-trip verification failed.")
            sys.exit(2)
        else:
            print("Round-trip verification passed.")


if __name__ == "__main__":
    main()
