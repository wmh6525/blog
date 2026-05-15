#!/usr/bin/env python3
"""
Fix LaTeX math expressions in markdown posts.

Two fixes applied inside math blocks only:
  1. "\\_"   -> "_"     (defensive markdown escape no longer needed with Passthrough)
  2. "\\\\\\\\" -> "\\\\"  (4 backslashes were used to survive markdown)
  3. raw "<" / ">" -> "\\lt" / "\\gt" (Hugo's HTML parser was eating x_{<t})

Safety guards:
  - Only process $$...$$ and $...$ that LOOK LIKE math (contain LaTeX commands).
    Avoids false positives on currency markers like $0, $$, $$$, $$$$ used in tables.
  - Skip code fences (``` ... ```).
  - For inline $...$, also skip if the content contains | (markdown table cell).
"""

import re
from pathlib import Path

POSTS_DIR = Path(__file__).parent.parent / "content" / "posts"

# DISPLAY math: $$ ... $$
DISPLAY_RE = re.compile(r"\$\$([\s\S]*?)\$\$")
# INLINE math: $ ... $ (single line, no $)
INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")

# Heuristic: only treat as math if it contains a LaTeX command or structural marker.
# This avoids matching currency markers ($0, $$, $$$ in tables).
MATH_LIKE_RE = re.compile(
    r"\\[a-zA-Z]+"          # \frac, \sum, \mathcal, etc.
    r"|\\[\\\[\]{}|]"        # \\, \[, \], \{, \}, \|
    r"|[_^]\{"               # subscript/superscript with braces: x_{i}, x^{i}
    r"|[_^][a-zA-Z0-9]"      # subscript/superscript single char: x_i, x^i
    r"|\\\\"                 # \\ for newlines
)


def looks_like_math(content: str) -> bool:
    """Return True if the captured text looks like real LaTeX math."""
    if not content.strip():
        return False
    # Markdown table rows (with |) inside a "$$...$$" capture indicate
    # we accidentally matched currency-marker $$ across table cells.
    if "|" in content and "\n" in content:
        return False
    return bool(MATH_LIKE_RE.search(content))


def fix_math_content(math: str) -> str:
    """Apply LaTeX cleanup to math content."""
    fixed = math.replace("\\_", "_")
    fixed = fixed.replace("\\\\\\\\", "\\\\")

    # Replace raw <, > with KaTeX-safe \lt, \gt to prevent Hugo's HTML parser
    # from interpreting them as tags. Don't touch:
    #   - escaped \< (rare in LaTeX, defensive)
    #   - <=, <! (HTML-comment / comparison)
    #   - => (arrow), >= (comparison)
    fixed = re.sub(r"(?<!\\)<(?![=!/])", r"\\lt ", fixed)
    fixed = re.sub(r"(?<!\\)(?<!=)>", r"\\gt ", fixed)
    return fixed


def split_code_blocks(text: str):
    """Yield (segment, is_code) pairs, splitting on triple-backtick code fences."""
    parts = re.split(r"(```[\s\S]*?```)", text)
    for part in parts:
        yield part, part.startswith("```")


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    new_segments = []
    changed = False

    for segment, is_code in split_code_blocks(text):
        if is_code:
            new_segments.append(segment)
            continue

        def repl_display(m):
            nonlocal changed
            inner = m.group(1)
            if not looks_like_math(inner):
                # leave currency / table-cell markers alone
                return m.group(0)
            new_inner = fix_math_content(inner)
            if new_inner != inner:
                changed = True
            return f"$${new_inner}$$"

        def repl_inline(m):
            nonlocal changed
            inner = m.group(1)
            if not looks_like_math(inner):
                return m.group(0)
            new_inner = fix_math_content(inner)
            if new_inner != inner:
                changed = True
            return f"${new_inner}$"

        segment = DISPLAY_RE.sub(repl_display, segment)
        segment = INLINE_RE.sub(repl_inline, segment)
        new_segments.append(segment)

    if changed:
        path.write_text("".join(new_segments), encoding="utf-8")

    return changed


def main():
    files = sorted(POSTS_DIR.glob("*.md"))
    fixed_count = 0
    for f in files:
        if fix_file(f):
            print(f"  fixed: {f.name}")
            fixed_count += 1
    print(f"\nDone. Modified {fixed_count} / {len(files)} files.")


if __name__ == "__main__":
    main()
