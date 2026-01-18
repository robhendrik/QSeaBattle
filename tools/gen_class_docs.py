#!/usr/bin/env python3
"""
Generate MkDocs class pages from a folder of .py modules using a fixed prompt.

Usage:
  export OPENAI_API_KEY="..."
  python gen_class_docs.py \
      --src Q_Sea_Battle/players \
      --package Q_Sea_Battle.players \
      --out docs/players \
      --model gpt-5.2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import ast
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI


SYSTEM_PROMPT_CLASS = r"""
You are a technical writer producing specification-grade Markdown for MkDocs (Material). Output MUST be VALID MARKDOWN ONLY.

Hard rules (reproducibility + Pandoc-safe):
- Output ONLY the final Markdown. No preamble, no analysis, no commentary.
- No HTML. No YAML front matter. Do NOT use lines containing only '---'.
- Do not hard-wrap prose: each paragraph is a single physical line.
- Lists: one bullet = one physical line. No manual line breaks inside bullets.
- Headings start at column 1. Leave a blank line after each heading.
- Use fenced code blocks only for code examples.
- Use admonitions only in MkDocs style, exactly:
  !!! note "Title"
      text...
  !!! warning "Title"
      text...
  !!! example "Title"
      ```python
      ...
      ```
- Never put code in tables.
- Math must be LaTeX inside $...$ or $$...$$ only (no unicode math symbols).
- Always describe every argument and return value with: type + constraints + shape (if applicable).
  - Arrays: "np.ndarray, dtype int {0,1}, shape (n2,)".
  - Tensors: "tf.Tensor, dtype float32, shape (B, n2)".
- Use symbols consistently: field_size, comms_size, n2, m (define them once when needed).

Document structure MUST be EXACTLY in this order for classes:
1) H1 title: "# {ClassName}"
2) Role one-liner as a quote line: "> Role: ..."
3) Location line: "Location: `{python.import.path.ClassName}`"
4) (Optional) Derived constraints section (only if derivable)
5) "## Constructor"
   - A 3-column Markdown table with headers: Parameter | Type | Description
   - Then: Preconditions, Postconditions, Errors, Example
6) "## Public Methods" (methods in source order)
7) "## Data & State"
8) "## Planned (design-spec)"
9) "## Deviations"
10) "## Notes for Contributors"
11) "## Related"
12) "## Changelog"

Extraction rules (do NOT invent APIs):
- Only document classes/functions that exist in the provided module text.
- If a behavior/argument/return is not in code, do not add it as fact.
- If "design notes" are provided, items present there but missing in code go under “Planned (design-spec)”.
- If code and design notes disagree, describe the mismatch under “Deviations” without resolving.
- Prefer "Unknown" / "Not specified" over guessing.
""".strip()

SYSTEM_PROMPT_MODULE = r"""
You are a technical writer producing specification-grade Markdown for MkDocs (Material). Output MUST be VALID MARKDOWN ONLY.

Hard rules (reproducibility + Pandoc-safe):
- Output ONLY the final Markdown. No preamble, no analysis, no commentary.
- No HTML. No YAML front matter. Do NOT use lines containing only '---'.
- Do not hard-wrap prose: each paragraph is a single physical line.
- Lists: one bullet = one physical line.
- Headings start at column 1. Leave a blank line after each heading.
- Use fenced code blocks only for code examples.
- Use admonitions only in MkDocs style.

Extraction rules (do NOT invent APIs):
- Only document functions/classes/constants that exist in the provided module text.
- Prefer "Unknown" / "Not specified" over guessing.

Module page structure MUST be EXACTLY:
1) "# {ModuleName}"
2) "> Role: ..."
3) "Location: `{python.import.path}`"
4) "## Overview"
5) "## Public API"
   - "### Functions" (each function in source order; Signature/Purpose/Arguments/Returns/Errors/Example)
   - "### Constants" (if any)
   - "### Types" (if any, e.g. aliases, TypedDict)
6) "## Dependencies"
7) "## Planned (design-spec)"
8) "## Deviations"
9) "## Notes for Contributors"
10) "## Related"
11) "## Changelog"
""".strip()

SYSTEM_PROMPT_FAMILY_AB = r"""
You are a technical writer producing specification-grade Markdown for MkDocs (Material). Output MUST be VALID MARKDOWN ONLY.

Hard rules:
- Output ONLY the final Markdown. No preamble, no analysis, no commentary.
- No HTML. No YAML front matter. Do NOT use lines containing only '---'.
- Do not hard-wrap prose: each paragraph is a single physical line.
- Lists: one bullet = one physical line.
- Headings start at column 1. Leave a blank line after each heading.
- Use fenced code blocks only for code examples.
- Never put code in tables.
- Use MkDocs admonitions only (!!! note / warning / example / tip).

Do NOT invent APIs:
- Only document classes/methods/attributes that exist in the provided module text.
- If a type/shape/behavior cannot be proven, write "Not specified".
- If design notes are provided, items present there but missing in code go under "Planned (design-spec)".
- If code and design notes disagree, describe both under "Deviations" without resolving.

Module page structure MUST be EXACTLY:
1) "# {ModuleName}"
2) "> Role: ..."
3) "Location: `{python.import.path}`"
4) "## Overview"
5) "## Class Groups"
   - Group A/B pairs by base name: e.g., {Base}A and {Base}B.
   - If multiple pairs exist, list groups in deterministic alphabetical order by base name.
6) "## Classes"
   - For each group:
     - "### {Base} (A/B pair)"
     - "#### {ClassNameA}"
       - Constructor table + Preconditions/Postconditions/Errors + Example
       - Public Methods (source order)
       - Data & State
     - "#### {ClassNameB}"
       - same structure
   - If any class has no A/B pair, document it under "### Other Classes" (alphabetical).
7) "## Planned (design-spec)"
8) "## Deviations"
9) "## Notes for Contributors"
10) "## Related"
11) "## Changelog"
""".strip()

USER_PROMPT_CLASS = r"""
Generate a single MkDocs class specification page in Markdown, following the mandated structure and rules.

ASSUMPTION (MUST ENFORCE)
- The module contains exactly ONE top-level public class (class name does not start with "_").
- If this assumption is false, still output Markdown, but:
  - Use H1: "# INVALID MODULE SHAPE"
  - Add a "## Errors" section listing:
    - The number of public classes found
    - Their names
  - Do not attempt to document any class.

INPUTS
1) Python module text (verbatim):
<<<PY_MODULE
{py_module_text}
PY_MODULE>>>

2) (Optional) Design notes / intended behavior (verbatim):
<<<DESIGN_NOTES
{design_notes}
DESIGN_NOTES>>>

OUTPUT REQUIREMENTS
- Output Markdown ONLY.
- The H1 MUST be exactly the class name discovered in the module.
- "Location:" MUST use the module path I provide below, plus the class name.
- Module import path (verbatim, do not modify):
{module_import_path}
""".strip()

USER_PROMPT_MODULE = r"""
Generate a single MkDocs MODULE documentation page in Markdown, following the mandated module structure and rules.

INPUTS
1) Python module text (verbatim):
<<<PY_MODULE
{py_module_text}
PY_MODULE>>>

2) (Optional) Design notes / intended behavior (verbatim):
<<<DESIGN_NOTES
{design_notes}
DESIGN_NOTES>>>

OUTPUT REQUIREMENTS
- Output Markdown ONLY.
- Module import path (verbatim, do not modify):
{module_import_path}
""".strip()

USER_PROMPT_FAMILY_AB = r"""
Generate a single MkDocs MODULE page documenting ALL public classes in the module, following the mandated structure.

INPUTS
1) Python module text (verbatim):
<<<PY_MODULE
{py_module_text}
PY_MODULE>>>

2) (Optional) Design notes / intended behavior (verbatim):
<<<DESIGN_NOTES
{design_notes}
DESIGN_NOTES>>>

OUTPUT REQUIREMENTS
- Output Markdown ONLY.
- Module import path (verbatim, do not modify):
{module_import_path}
""".strip()

SYSTEM_PROMPT_MIXED = SYSTEM_PROMPT_MODULE
USER_PROMPT_MIXED = USER_PROMPT_MODULE

@dataclass
class ModuleShape:
    public_classes: list[str]
    public_functions: list[str]
    has_constants: bool

def parse_module_shape(py_text: str) -> ModuleShape:
    t = ast.parse(py_text)

    public_classes = []
    public_functions = []
    has_constants = False

    for node in t.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public_functions.append(node.name)
        elif isinstance(node, ast.Assign):
            # "constants" heuristic: module-level assignment to ALL_CAPS names
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    has_constants = True

    return ModuleShape(
        public_classes=public_classes,
        public_functions=public_functions,
        has_constants=has_constants,
    )

def is_utilities_by_name(module_import_path: str, py_path: Path) -> bool:
    s = f"{module_import_path}::{py_path.as_posix()}".lower()
    return (
        ".utilities" in s
        or "/utilities/" in s
        or s.endswith("_utilities.py")
        or "utilities" in py_path.stem.lower()
    )

def looks_like_ab_family(class_names: list[str]) -> bool:
    # Detect presence of suffix A/B pairs (FooA + FooB), possibly multiple bases.
    if len(class_names) < 2:
        return False

    bases = {}
    for name in class_names:
        if name.endswith("A") and len(name) > 1:
            bases.setdefault(name[:-1], set()).add("A")
        elif name.endswith("B") and len(name) > 1:
            bases.setdefault(name[:-1], set()).add("B")

    # If at least one base has both A and B, treat as family.
    return any(v == {"A", "B"} for v in bases.values())

def classify_module(module_import_path: str, py_path: Path, py_text: str) -> str:
    shape = parse_module_shape(py_text)

    # 1) Utilities by name wins
    if is_utilities_by_name(module_import_path, py_path):
        return "utilities"

    # 2) If no public classes but functions/constants exist -> utilities-like
    if len(shape.public_classes) == 0 and (len(shape.public_functions) > 0 or shape.has_constants):
        return "utilities"

    # 3) Single-class module
    if len(shape.public_classes) == 1:
        return "single_class"

    # 4) Multi-class A/B family module
    if looks_like_ab_family(shape.public_classes):
        return "family_ab"

    # 5) Fallback
    return "mixed_other"

def iter_py_files(src_dir: Path) -> list[Path]:
    files = []
    for p in src_dir.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        files.append(p)
    return sorted(files)

def is_utilities_module(module_import_path: str, py_path: Path) -> bool:
    # Most robust: catch both folder names and filename conventions
    s = f"{module_import_path}::{py_path.as_posix()}".lower()
    return (
        ".utilities" in s
        or "/utilities/" in s
        or s.endswith("_utilities.py")
        or "utilities" in py_path.stem.lower()
    )

def rel_module_path(py_file: Path, src_dir: Path, package_prefix: str) -> str:
    rel = py_file.relative_to(src_dir).with_suffix("")
    # convert path segments to module segments
    mod = ".".join(rel.parts)
    return f"{package_prefix}.{mod}" if mod else package_prefix

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src/Q_Sea_Battle", type=Path, help="Folder containing .py modules")
    ap.add_argument("--package", default="Q_Sea_Battle", help="Python import prefix, e.g. Q_Sea_Battle.players")
    ap.add_argument("--out", default="docs", type=Path, help="Docs output folder")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--design-notes", default="", help="Optional design notes text (applied to all modules)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    client = OpenAI()

    args.out.mkdir(parents=True, exist_ok=True)

    py_files = iter_py_files(args.src)
    if not py_files:
        raise SystemExit(f"No .py files found under {args.src}")

    for py_path in py_files:
        module_import_path = rel_module_path(py_path, args.src, args.package)
        py_text = py_path.read_text(encoding="utf-8")

        kind = classify_module(module_import_path, py_path, py_text)

        if kind == "single_class":
            instructions = SYSTEM_PROMPT_CLASS
            user_prompt = USER_PROMPT_CLASS.format(
                py_module_text=py_text,
                design_notes=args.design_notes or "",
                module_import_path=module_import_path,
            )

        elif kind == "utilities":
            instructions = SYSTEM_PROMPT_MODULE
            user_prompt = USER_PROMPT_MODULE.format(
                py_module_text=py_text,
                design_notes=args.design_notes or "",
                module_import_path=module_import_path,
            )

        elif kind == "family_ab":
            instructions = SYSTEM_PROMPT_FAMILY_AB
            user_prompt = USER_PROMPT_FAMILY_AB.format(
                py_module_text=py_text,
                design_notes=args.design_notes or "",
                module_import_path=module_import_path,
            )

        else:
            instructions = SYSTEM_PROMPT_MIXED
            user_prompt = USER_PROMPT_MIXED.format(
                py_module_text=py_text,
                design_notes=args.design_notes or "",
                module_import_path=module_import_path,
            )

        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / py_path.with_suffix(".md").name

        if out_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {out_path}")
            continue

        resp = client.responses.create(
            model=args.model,
            instructions=instructions,
            input=user_prompt,
        )
        md = resp.output_text

        if not md.lstrip().startswith("# "):
            raise RuntimeError(f"Model output for {py_path} did not start with an H1")

        out_path.write_text(md, encoding="utf-8")
        print(f"WROTE: {out_path}")



if __name__ == "__main__":
    main()
