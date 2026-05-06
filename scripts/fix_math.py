#!/usr/bin/env python3
"""
Fix LaTeX math expressions in markdown posts.

With Hugo Passthrough enabled, markdown no longer mangles math content,
so the defensive escapes (\\_ and \\\\\\\\) used to avoid markdown italic/escape
must be reverted to their proper LaTeX form (_ and \\\\).

This script:
  1. Finds $...$ and $$...$$ blocks (skipping code fences)
  2. Replaces inside math:
     - "\\_" -> "_"
     - "\\\\\\\\" -> "\\\\"
  3. Saves the file in place
"""

import re
import sys
from pathlib import Path

POSTS_DIR = Path(__file__).parent.parent / "content" / "posts"

# Match $$ ... $$ (display) and $ ... $ (inline)
# We deliberately avoid matching inside ``` code fences by splitting first
DISPLAY_RE = re.compile(r"\$\$([\s\S]*?)\$\$")
INLINE_RE = re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)")


def fix_math_content(math: str) -> str:
    # Only modify inside the math content
    # \_ -> _   (defensive markdown escape no longer needed)
    fixed = math.replace("\\_", "_")
    # \\\\ -> \\  (4 backslashes were used to survive markdown; now Passthrough leaves them alone)
    fixed = fixed.replace("\\\\\\\\", "\\\\")
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
            new_inner = fix_math_content(inner)
            if new_inner != inner:
                changed = True
            return f"$${new_inner}$$"

        def repl_inline(m):
            nonlocal changed
            inner = m.group(1)
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
