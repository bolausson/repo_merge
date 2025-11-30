# repo_merge

A command-line tool that walks a local Git repository and merges its text files into one or more Markdown part files suitable for ingestion by an LLM or for archival. The tool preserves file contents, includes metadata (size and SHA‑1), respects `.gitignore` by default, and can produce a tree-style summary and a ZIP archive of the outputs. It also supports forced splitting of very large files to keep parts under a configured size.

## Features
- Merge text files into Markdown parts named `<repo-name>.partNNN.md`.
- Never split a file across parts by default; each file section is kept intact.
- Optional `--force-split` to split very large files across parts so no part exceeds `MAX_PART_SIZE`.
- Configurable max part size with human-friendly values (e.g., `50M`, `200K`).
- Respects `.gitignore` by default; optional `--no-gitignore` to include everything.
- Skip binary files and list them in the output.
- LLM prompt included at the top of the first part to explain format and purpose.
- Summary file `<repo-name>.summary.md` with a tree showing which part contains each file (can be disabled).
- Optional ZIP archive `<repo-name>-merged.zip` containing all parts and the summary; when unzipped, files go into a folder named `<repo-name>-merged`.
- Filters and options: include only certain extensions, include hidden files, dry-run mode, progress indicator.

## Installation
1. Ensure you have Python 3.8+ installed.
2. Save the script as `repo_merge.py` and make it executable if desired:
   ```bash
   chmod +x repo_merge.py
   ```
3. Run the script from a shell. No external Python packages are required.

## Usage
Basic invocation:
```bash
python repo_merge.py /path/to/repo
```

### Common options
- `-o, --output` — Base output filename or path. If omitted, defaults to `<repo-name>.part001.md` in the current directory.
- `-m, --max-size` — Maximum bytes per part (e.g., `50M`, `200K`). Default: `50M`.
- `-e, --extensions` — Comma-separated list of file extensions to include (e.g., `.py,.md` or `py,md`).
- `-x, --dry-run` — Do not write files; show how many parts would be produced and list files per part.
- `--no-gitignore` — Ignore `.gitignore` and include all files.
- `--include-hidden` — Include hidden files and folders (names starting with `.`).
- `--no-summary` — Do not create the `<repo-name>.summary.md` summary file.
- `-z, --zip` — Create a zip archive named `<repo-name>-merged.zip` containing all parts and the summary. Files inside the zip are placed under `<repo-name>-merged/`.
- `--force-split` — Force split very large input files across parts so no part exceeds `MAX_PART_SIZE`.

## Examples
- Merge a repository with default settings:
  ```bash
  python repo_merge.py /home/user/my-repo
  ```
  Output: `my-repo.part001.md`, `my-repo.part002.md`, ...

- Merge and set max part size to 100 MB:
  ```bash
  python repo_merge.py /home/user/my-repo -m 100M
  ```

- Merge only `.py` and `.md` files and include hidden files:
  ```bash
  python repo_merge.py /home/user/my-repo -e .py,.md --include-hidden
  ```

- Dry-run to see how many parts would be produced:
  ```bash
  python repo_merge.py /home/user/my-repo -x
  ```

- Produce parts, summary, and a ZIP archive:
  ```bash
  python repo_merge.py /home/user/my-repo -z
  ```
  Result: `my-repo.part001.md`, `my-repo.part002.md`, `my-repo.summary.md`, and `my-repo-merged.zip` (contains `my-repo-merged/` folder with the files).

- Force split very large files:
  ```bash
  python repo_merge.py /home/user/my-repo --force-split -m 50M
  ```
  Large files will be split across parts to keep each part ≤ 50 MB.

## Output format and conventions
- Top of each part: a fenced YAML-like metadata block with `merged_repo_file`, `generated_by`, `generated_at`, `repo_root`, `repo_name`, and `notes`.
- LLM prompt: the first part begins with a clear LLM instruction block describing the format and purpose.
- File sections: each file appears as:
  - `### FILE: \`relative/path/to/file\``
  - Metadata lines: `- **size:** <bytes>` and `- **sha1:** \`<sha1-hex>\``
  - A fenced code block with a language hint (when available) containing the file contents.
- Summary file: `<repo-name>.summary.md` lists files in a tree-like structure and shows the part number where each file is located.
- ZIP archive: `<repo-name>-merged.zip` contains all parts and the summary under the folder `<repo-name>-merged/` so unzipping places files into that folder.

## Notes and tips
- Large single files: by default, if a file section exceeds `--max-size`, it is written whole into its own part (which may exceed the limit). Use `--force-split` to ensure no part exceeds the limit.
- `.gitignore`: by default the script respects `.gitignore`. Use `--no-gitignore` to include everything.
- Binary files: binary files are detected and skipped; they are listed under `## Skipped binary files` in the last part.
- Script exclusion: the script excludes itself from the merged output.
- Avoid overwriting: the script will overwrite files with the same names in the output directory. If you need safe behavior, run in a clean directory or implement a wrapper to move/rename existing files first.
- Dry-run: use `-x` to preview parts and the planned summary without writing files.

## Troubleshooting
- If you see timezone or datetime warnings, ensure you are running Python 3.8+.
- If `git` is not available, the script falls back to a simple `.gitignore` parser; results may differ from `git` behavior for complex ignore rules.
- If the script fails to read a file due to encoding issues, it reads with `errors="replace"` so the file content is still included with replacement characters.
