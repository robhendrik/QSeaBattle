# QSeaBattle Markdown Style Guide (Pandoc-Safe)

This document defines **strict authoring rules** for all `.md` files in this repository.

**Goal:** Every Markdown file must render correctly and consistently in:
- MkDocs (HTML)
- Pandoc → LaTeX → PDF

No post-processing fixes should be required.

---

## 1. General principle

> Markdown files are **source code**, not layout documents.

- Do not rely on visual wrapping
- Do not rely on renderer heuristics
- Be explicit and unambiguous

If something is ambiguous, Pandoc will break it.

---

## 2. Math rules (MANDATORY)

### Allowed
- Inline math:
  ```
  $n^2$, $k = \log_2(n^2)$, $(B, m)$
  ```
- Display math:
  ```
  $$
  k = \log_2(n^2)
  $$
  ```

### Forbidden
- `\(...\)` or `\[...\]`
- Unicode math symbols: `n`, `^2`, `B`, `ℝ`, `∑`, etc.
- Math outside `$...$`

### Rule
**All math MUST be LaTeX math inside `$...$` or `$$...$$`.**

---

## 3. Horizontal rules / separators

### Forbidden
```
---
```

Reason: Interpreted as YAML front matter or parsing boundary by Pandoc.

### Allowed
- Blank lines
- Headings
- Optional: `***`

---

## 4. Headings

### Correct
```
## Section title
```

### Forbidden
```
    ## Indented heading
```

### Rules
- Headings must start at column 1
- Always leave a blank line after headings

---

## 5. Lists

### Correct
```
- One bullet = one line
- No hard line breaks
- Full sentence on one line
```

### Forbidden
```
- Communication
limited
to
comms_size
```

### Rules
- One bullet per physical line
- No manual line wrapping
- Blank line before and after lists

---

## 6. Paragraphs

### Correct
```
This is a full paragraph written on a single line.
```

### Forbidden
```
This paragraph
is broken
across lines.
```

### Rule
**Never insert line breaks inside sentences.**

---

## 7. Code blocks

### Allowed
```
```python
def foo():
    pass
```
```

### Rules
- Code blocks are verbatim
- Do not place renderable math inside code blocks
- Do not use code blocks for layout

---

## 8. Inline code vs math

Use:
- `` `identifier` `` for variable names
- `$n^2$` for mathematical meaning

Never mix Unicode math with text.

---

## 9. Shapes and dimensions

### Preferred
```
- `field`: $(B, n^2)$
- `gun`: $(B, n^2)$ one-hot
- `comm`: $(B, m)$
```

### Forbidden
```
(B, n^2)
(B, n^2)
(B, n^2)
```

---

## 10. Encoding, mojibake, and control characters (CRITICAL)

### Required
- Files MUST be saved as UTF-8 (no BOM preferred)
- Plain ASCII for prose
- LaTeX math for mathematical symbols

### Forbidden (MUST NOT appear anywhere)
- Unicode superscripts/subscripts (e.g. `^2`)
- Mathematical alphabets (`B`, `n`)
- Smart quotes
- **C0/C1 control characters** (e.g. U+009D)
- **Mojibake sequences** caused by encoding errors, including but not limited to:
  - ``
  - `â”`
  - `-`
  - `--`

These sequences indicate UTF-8 text decoded as Latin-1/Windows-1252 and
will cause broken rendering or missing glyph warnings in PDF output.

### Normative requirement
> **Markdown files MUST NOT contain mojibake or control characters.  
> Any appearance of such characters is a style violation and must be fixed immediately.**

### Detection (recommended)
```
# Detect mojibake and control characters
Select-String docs\**\*.md '|â”|-|--|[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
```

### Remediation (safe cleanup)
```
# Remove all C0/C1 control characters except tab/newline
Get-ChildItem docs -Recurse -Filter *.md |
  ForEach-Object {
    $p = $_.FullName
    $s = Get-Content -LiteralPath $p -Raw
    $t = [regex]::Replace($s, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "")
    if ($t -ne $s) {
      Set-Content -LiteralPath $p -Value $t -Encoding UTF8
    }
  }
```

---

## 11. Repository trees and diagrams

### Forbidden
- Box-drawing characters:
  - `├`, `└`, `|`, `-`
- Corrupted variants:
  - `|`, `-`, ```

### Required
Use ASCII-only trees:

```
QSeaBattle/
|-- docs/
|-- mkdocs.yml
|-- src/
|-- tests/
`-- notebooks/
```

---

## 12. YAML / front matter

### Forbidden
```
---
title: Something
---
```

### Rule
**No YAML front matter is allowed in spec files.**

---

## 13. Tables

### Allowed
- Simple tables
- Single-line cells

Avoid math-heavy tables. Prefer bullet lists.

---

## 14. Validation rules (recommended)

Run these periodically from repo root:

```
# Forbidden separators
Select-String docs\**\*.md '^\s*---\s*$'

# Forbidden math delimiters
Select-String docs\**\*.md '\\(|\\)'

# Unicode math characters
Select-String docs\**\*.md '[\u00B2\u2070-\u209F\u1D400-\u1D7FF]'

# Mojibake / encoding garbage
Select-String docs\**\*.md '|â”|-|--'

# Suspicious hard-wrapped prose
Select-String docs\**\*.md '^\w+$'
```

---

## 15. Golden rule

> If it looks nicely wrapped in your editor, it is probably wrong.

Let the renderer handle layout.

---

## 16. Enforcement

Violations of this style guide are **spec bugs** and must be fixed before:
- PDF export
- version tagging
- public release

This document is normative.