#!/usr/bin/env python3
"""
repo_merge.py

Merge text files from a local git repository into Markdown parts, optionally create a summary,
and optionally zip all output files into a single archive. Supports forced splitting of very
large files across parts to ensure no part exceeds MAX_PART_SIZE.

Features:
- Never splits a file across parts by default (intact sections). With --force-split, large files can span parts.
- LLM prompt at the top of the first part to explain purpose and format.
- Excludes .gitignore from the ignored listing.
- Progress indicator while processing files.
- Summary file (<repo-name>.summary.md) showing a tree and the part for each file (can be disabled).
- Optional ZIP archive (<repo-name>-merged.zip) placing outputs under <repo-name>-merged/.
- Options for extension filtering, hidden files, ignoring .gitignore, and dry-run.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

# ---------- Configuration ----------
DEFAULT_MAX_PART_SIZE = "50M"  # default 50 megabytes

# ---------- Helpers ----------

def parse_size(s: Optional[str]) -> int:
    """Parse human-friendly size strings into bytes. Examples: '50M', '200K', '1024'."""
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
    """Return set of paths (relative to repo_root) ignored by git, using `git` if available."""
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
                paths.add(Path(p.decode('utf-8', 'surrogateescape')))
        return paths
    except Exception:
        return set()

def parse_simple_gitignore(repo_root: Path) -> Set[Path]:
    """
    Very simple .gitignore fallback: read top-level .gitignore and match patterns using glob.
    This is a fallback and does not implement full gitignore semantics.
    """
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
                    ignored.add(p.relative_to(repo_root))
                except Exception:
                    pass
        else:
            for p in repo_root.rglob(pat):
                try:
                    ignored.add(p.relative_to(repo_root))
                except Exception:
                    pass
            for p in repo_root.rglob('*'):
                rel = str(p.relative_to(repo_root))
                if fnmatch.fnmatch(rel, pat):
                    try:
                        ignored.add(p.relative_to(repo_root))
                    except Exception:
                        pass
    return ignored

def fence_language_for_suffix(suffix: str) -> str:
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.html': 'html',
        '.css': 'css',
        '.md': 'markdown',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.sh': 'bash',
        '.ps1': 'powershell',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.txt': '',
        '.xml': 'xml',
        '.toml': 'toml',
    }
    return mapping.get(suffix.lower(), '')

def normalize_extensions(ext_arg: Optional[str]) -> Optional[Set[str]]:
    """Return a set of normalized suffixes (with leading dot), or None if no filter."""
    if not ext_arg:
        return None
    parts = [p.strip() for p in ext_arg.split(",") if p.strip()]
    normalized = set()
    for p in parts:
        if not p.startswith("."):
            p = "." + p
        normalized.add(p.lower())
    return normalized if normalized else None

# ---------- LLM prompt text ----------
LLM_PROMPT = """# LLM INSTRUCTION

You are given one or more Markdown part files that together contain the contents of a local Git repository.
Each part is a Markdown document with a YAML-like metadata block at the top and multiple file sections.

Purpose
- Provide a machine-readable archive of the repository's text files so an LLM can analyze, summarize, or answer questions about the repository.

Format
- Top of each part: a fenced YAML-like block with metadata:
  - merged_repo_file: version identifier
  - generated_by: "repo_merge.py"
  - generated_at: UTC timestamp (ISO 8601, trailing Z)
  - repo_root: absolute path where the repo was read (may be different on the machine that generated the parts)
  - repo_name: repository directory name
  - notes: short notes

- After metadata: a header "# Repository: <repo_name>" and "## Files".

- Each file included appears as a section:
  - "### FILE: `relative/path/to/file`"
  - Metadata lines:
    - "- **size:** <bytes>"
    - "- **sha1:** `<sha1-hex>`"
  - A fenced code block with an appropriate language hint (e.g., ```python, ```json, or ``` for plain text) containing the file contents exactly as read (UTF-8, invalid sequences replaced).
  - File sections are never split across parts by default. If this archive was generated with --force-split, very large files may span multiple parts (each chunk remains inside a fenced block).

- At the end of the last part:
  - "## Ignored by .gitignore" (list of paths; note: ".gitignore" itself is excluded from this list)
  - "## Skipped binary files" (list of binary file paths that were not included)

Notes for the LLM
- Use the FILE headers and SHA-1 values to identify and reference files.
- If multiple parts exist, treat them as sequential parts of the same archive.
- Files listed under "Skipped binary files" are not present in the parts.
- Files listed under "Ignored by .gitignore" were intentionally excluded (unless the generator was run with --no-gitignore).

End of instruction.
"""

# ---------- Part writer ----------

class PartWriter:
    """
    Handles writing parts, splitting logic, and ensuring part size boundaries.
    - Default: do not split a single file section across parts (intact sections).
    - With force_split=True: stream content and split across parts wherever needed to not exceed max_part_size.
    """
    def __init__(self, base_out: Optional[Path], repo_name: str, repo_root: Path, max_part_size: int, llm_prompt: Optional[str] = None, force_split: bool = False):
        self.repo_name = repo_name
        self.repo_root = repo_root
        self.max_part_size = max_part_size
        self.llm_prompt = llm_prompt
        self.force_split = force_split

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

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_new_part()

    def _part_filename(self, index: int) -> Path:
        return self.out_dir / f"{self.base_name}.part{index:03d}.md"

    def _open_new_part(self):
        if self.current_file:
            self.current_file.close()
        path = self._part_filename(self.part_index)
        self.current_file = path.open("wb")
        self.current_bytes = 0
        # LLM prompt goes before metadata in the first part
        if self.part_index == 1 and self.llm_prompt:
            prompt_block = "## LLM PROMPT\n\n" + "```\n" + self.llm_prompt.strip() + "\n```\n\n"
            self._write_raw(prompt_block)
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
        """Write string, opening new parts as needed to stay within max_part_size."""
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
        """Write without splitting (may exceed max_part_size)."""
        if self.current_file is None:
            self._open_new_part()
        data = s.encode("utf-8")
        self.current_file.write(data)
        self.current_bytes += len(data)

    def write_section(self, header: str, metadata_lines: List[str], content_path: Path, fence_lang: str) -> int:
        """
        Write a file section. Returns the 1-based part index where the header began.
        - Default: place entire section in one part; if too big, that part may exceed max_part_size.
        - With force_split: stream content and split across parts as needed to never exceed max_part_size.
        """
        part_written = self.part_index - 1  # current part number

        header_block = header + "\n\n" + "".join(metadata_lines) + "\n"
        fence_start = "```" + (fence_lang if fence_lang else "") + "\n"
        fence_end = "\n```\n"

        if self.force_split:
            # Ensure header and fence_start fit: if not, open new part
            header_size = len((header_block + fence_start).encode("utf-8"))
            if self.current_bytes > 0 and (self.current_bytes + header_size) > self.max_part_size:
                self._open_new_part()
                part_written = self.part_index - 1
            # Write header and start fence
            self._write_raw(header_block)
            self._write_raw(fence_start)
            # Stream content, splitting across parts as needed
            try:
                with content_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.endswith("\n"):
                            line += "\n"
                        self._write_raw(line)
            except Exception as e:
                self._write_raw(f"[ERROR reading file: {e}\\n]")
            # Close fence and add spacing (may trigger part split if at boundary)
            self._write_raw(fence_end)
            self._write_raw("\n")
            return part_written

        # Default behavior: intact section in one part
        # Compute section size (excluding file content to avoid double-reading); we will still stream content
        header_bytes = len(header_block.encode("utf-8"))
        fence_start_bytes = len(fence_start.encode("utf-8"))
        fence_end_bytes = len(fence_end.encode("utf-8"))
        # If placing header+start fence would overflow current part, open new part
        if self.current_bytes > 0 and (self.current_bytes + header_bytes + fence_start_bytes) > self.max_part_size:
            self._open_new_part()
            part_written = self.part_index - 1

        # Write header and start fence without splitting (force)
        self._write_bytes_force(header_block)
        self._write_bytes_force(fence_start)
        # Stream file content (force, no splitting)
        try:
            with content_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.endswith("\n"):
                        line += "\n"
                    self._write_bytes_force(line)
        except Exception as e:
            self._write_bytes_force(f"[ERROR reading file: {e}\\n]")
        # Close fence and spacing
        self._write_bytes_force(fence_end)
        self._write_bytes_force("\n")
        return part_written

    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file = None

# ---------- Dry-run planner ----------

class PartPlanner:
    """
    Simulate packing file sections into parts without writing files.
    Uses intact-section rules (no force-split simulation).
    """
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

# ---------- Main logic ----------

def collect_files(repo_root: Path, script_path: Optional[Path], extensions: Optional[Set[str]], include_hidden: bool, ignore_gitignore: bool) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Returns (included_text_files, ignored_paths, skipped_binary_files). Paths are relative to repo_root.
    """
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
                if rel in ignored:
                    continue
                skip_due_to_parent = False
                for i in range(1, len(rel.parts)+1):
                    if Path(*rel.parts[:i]) in ignored:
                        skip_due_to_parent = True
                        break
                if skip_due_to_parent:
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

def compute_section_size(repo_root: Path, rel: Path, fence_lang: str) -> int:
    """
    Compute the approximate byte size of the section (header + metadata + fences + content).
    Used for dry-run planning (intact sections only).
    """
    full = repo_root / rel
    header_block = f"### FILE: `{str(rel)}`\n\n"
    try:
        size = full.stat().st_size
    except Exception:
        size = 0
    sha1 = compute_sha1(full)
    metadata = f"- **size:** {size} bytes  \n- **sha1:** `{sha1}`  \n\n"
    fence_start = "```" + (fence_lang if fence_lang else "") + "\n"
    fence_end = "\n```\n"
    content_bytes = 0
    try:
        with full.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.endswith("\n"):
                    line = line + "\n"
                content_bytes += len(line.encode("utf-8"))
    except Exception:
        content_bytes = 0
    total = len(header_block.encode("utf-8")) + len(metadata.encode("utf-8")) + len(fence_start.encode("utf-8")) + content_bytes + len(fence_end.encode("utf-8")) + len("\n".encode("utf-8"))
    return total

def build_tree_mapping(file_to_part: Dict[str, int]) -> Dict:
    """
    Build a nested dict representing the directory tree.
    Leaves store '__files__': [(filename, part_index)].
    """
    tree: Dict = {}
    for relpath, part in sorted(file_to_part.items()):
        if relpath.endswith(".summary.md"):
            # place summary at root
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

def write_summary_file(repo_name: str, out_dir: Path, file_to_part: Dict[str, int]) -> Path:
    """
    Write <repo-name>.summary.md in out_dir listing files and which part they are in.
    """
    summary_path = out_dir / f"{repo_name}.summary.md"
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

def create_zip_archive(out_dir: Path, repo_name: str, include_summary: bool) -> Path:
    """
    Create <repo-name>-merged.zip in out_dir containing parts (and summary if present),
    with files under folder <repo-name>-merged/.
    """
    zip_name = f"{repo_name}-merged.zip"
    zip_path = out_dir / zip_name
    folder_prefix = f"{repo_name}-merged/"
    files_to_zip = sorted([p for p in out_dir.glob("*.part*.md") if p.is_file()])
    summary_path = out_dir / f"{repo_name}.summary.md"
    if include_summary and summary_path.exists():
        files_to_zip.append(summary_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_zip:
            zf.write(p, arcname=folder_prefix + p.name)
    print(f"Created zip archive: {zip_path}")
    return zip_path

def write_markdown_parts(repo_root: Path, included: List[Path], ignored: List[Path], skipped_binary: List[Path], out_path: Optional[Path], max_part_size: int, dry_run: bool, llm_prompt: Optional[str], create_summary: bool, force_split: bool) -> Tuple[Path, Dict[str, int]]:
    repo_name = repo_root.name
    base_out = out_path if out_path else None
    file_to_part: Dict[str, int] = {}

    if dry_run:
        planner = PartPlanner(repo_name, max_part_size)
        out_dir = base_out.parent if base_out else Path.cwd()
    else:
        writer = PartWriter(base_out, repo_name, repo_root, max_part_size, llm_prompt=llm_prompt, force_split=force_split)
        out_dir = writer.out_dir

    # Initial header
    header_text = f"# Repository: {repo_name}\n\n## Files\n\n"
    header_bytes = len(header_text.encode("utf-8"))
    if dry_run:
        planner.add_section("(__metadata_header__)", header_bytes)
    else:
        # Header goes into the first part
        writer._write_raw(header_text)

    total_files = len(included)
    for idx, rel in enumerate(included, start=1):
        # Progress indicator
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
            section_size = compute_section_size(repo_root, rel, lang)
            planner.add_section(str(rel), section_size)
        else:
            part_written = writer.write_section(header, metadata_lines, full, lang)
            # Map file to the part where its header was written
            file_to_part[str(rel)] = part_written

    # Clear progress line
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
        # Return planned dir and mapping
        return out_dir, planned_file_to_part

    # Append ignored and skipped lists at the end of the last part
    writer._write_raw("## Ignored by .gitignore\n\n")
    filtered_ignored = [p for p in ignored if p.name != ".gitignore"]
    if filtered_ignored:
        for p in filtered_ignored:
            writer._write_raw(f"- `{str(p)}`\n")
    else:
        writer._write_raw("_None_\n")
    writer._write_raw("\n")

    writer._write_raw("## Skipped binary files\n\n")
    if skipped_binary:
        for p in skipped_binary:
            writer._write_raw(f"- `{str(p)}`\n")
    else:
        writer._write_raw("_None_\n")
    writer._write_raw("\n")

    writer.close()
    print("Merging complete.")

    # Write summary file if requested
    if create_summary:
        summary_path = write_summary_file(repo_name, out_dir, file_to_part)
        file_to_part[str(summary_path.name)] = 0

    return out_dir, file_to_part

# ---------- CLI ----------

def default_output_name(repo_root: Path) -> str:
    repo_name = repo_root.name or "repo"
    return f"{repo_name}.part001.md"

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo_merge.py",
        description="Merge text files from a local git repository into Markdown parts."
    )
    p.add_argument("repo", help="Path to the repository root")
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
    return p

def main(argv: Optional[List[str]] = None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print("Error: repo root does not exist or is not a directory.")
        sys.exit(1)

    out_file = Path(args.output) if args.output else Path.cwd() / default_output_name(repo_root)

    try:
        max_part_size = parse_size(args.max_size)
    except Exception as e:
        print(f"Error parsing max-size: {e}")
        sys.exit(1)

    extensions = normalize_extensions(args.extensions)

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
        llm_prompt=LLM_PROMPT,
        create_summary=not args.no_summary,
        force_split=args.force_split
    )

    # Create zip if requested
    if args.zip:
        repo_name = repo_root.name
        if args.dry_run:
            print("\nDry-run: planned zip archive")
            zip_name = f"{repo_name}-merged.zip"
            print(f"Zip name: {zip_name}")
            print(f"When unzipped, files will be placed into folder: {repo_name}-merged/")
            print("Files to include:")
            # Show planned parts and summary names
            planned_names = [f"{repo_name}.part{idx:03d}.md" for idx in range(1, len(file_to_part) + 2)]  # approximate
            for name in sorted(set(planned_names + [f"{repo_name}.summary.md"])):
                print(f" - {name}")
        else:
            zip_path = create_zip_archive(out_dir, repo_name, include_summary=not args.no_summary)
            print(f"Zip created at: {zip_path}")

if __name__ == "__main__":
    main()
