"""Compile the current article and supplement in an external build directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

if __package__:
    from .paths import OUTPUT_DIR, PAPER_DIR
else:
    from paths import OUTPUT_DIR, PAPER_DIR


def prepare_build(template_dir: Path | None = None) -> Path:
    """Copy approved assets and separately installed Springer template dependencies.

    ``template_dir`` (or ``FACTORLASSO_LATEX_TEMPLATE_DIR``) points to the
    unpacked official Springer Nature template. Publisher files remain external
    to the repository. If omitted, the TeX distribution must supply them.
    """
    destination = OUTPUT_DIR / "latex-build"
    destination.mkdir(parents=True, exist_ok=True)
    for line in (PAPER_DIR / ".gitignore").read_text(encoding="utf-8").splitlines():
        if not line.startswith("!/paper/") or line.endswith("/"):
            continue
        relative = Path(line[len("!/paper/"):])
        # Reading PDFs are outputs, never inputs to the new compilation.
        if relative.name in {"article.pdf", "supplement.pdf"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER_DIR / "paper" / relative, target)
    configured = template_dir or os.environ.get("FACTORLASSO_LATEX_TEMPLATE_DIR")
    if configured:
        template_root = Path(configured).expanduser().resolve()
        for name in ("sn-jnl.cls", "sn-basic.bst"):
            choices = (template_root / name, template_root / "bst" / name)
            source = next((path for path in choices if path.is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing {name} in template directory {template_root}")
            shutil.copy2(source, destination / name)
    return destination


def compile_documents(destination: Path) -> dict:
    """Build both documents until mutual references are stable, then audit logs."""
    stems = [stem for stem in ("article", "supplement")
             if (destination / f"{stem}.tex").is_file()]
    if not stems:
        raise FileNotFoundError(f"No article.tex or supplement.tex in {destination}")
    # Remove only this helper's known generated files. Stale auxiliary files
    # must not conceal a missing label or bibliography in revised sources.
    for stem in stems:
        for suffix in (".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".pdf"):
            (destination / f"{stem}{suffix}").unlink(missing_ok=True)

    def latex(stem):
        command = ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", f"{stem}.tex"]
        subprocess.run(command, cwd=destination, check=True)

    for stem in stems:
        latex(stem)
    for stem in stems:
        subprocess.run(["bibtex", stem], cwd=destination, check=True)

    def reference_state():
        return {
            f"{stem}{suffix}": (destination / f"{stem}{suffix}").read_bytes()
            for stem in stems for suffix in (".aux", ".out", ".toc")
            if (destination / f"{stem}{suffix}").is_file()
        }

    state = reference_state()
    passes = 1
    for _ in range(8):
        for stem in stems:
            latex(stem)
        passes += 1
        updated = reference_state()
        if updated == state:
            break
        state = updated
    else:
        raise RuntimeError("Article/supplement references did not stabilize after nine passes")

    failures = {}
    warnings = {}
    for stem in stems:
        log = (destination / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        bad = [line.strip() for line in log.splitlines() if re.search(
            r"undefined|multiply defined|Label\(s\) may have changed|Rerun to get|"
            r"Please rerun", line, re.IGNORECASE
        )]
        if bad:
            failures[stem] = bad
        warnings[stem] = [line.strip() for line in log.splitlines()
                          if "Warning" in line or "Overfull" in line]
    receipt = {
        "latex_passes_per_document": passes,
        "reference_failures": failures,
        "warnings": warnings,
        "pdf_sha256": {
            stem: hashlib.sha256((destination / f"{stem}.pdf").read_bytes()).hexdigest()
            for stem in stems
        },
    }
    (destination / "build_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Unresolved or unstable document references: {failures}")
    return receipt


def _brace_group(text: str, start: int) -> tuple[str, int]:
    """Read one balanced TeX brace group without interpreting its contents."""
    if text[start] != "{":
        raise ValueError("Expected a TeX brace group")
    depth = 1
    i = start + 1
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("Unbalanced TeX brace group")


def _aux_labels(path: Path) -> dict[str, list[str]]:
    """Read standard LaTeX label fields from a successfully built document."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(r"\newlabel{"):
            continue
        name, at = _brace_group(line, len(r"\newlabel"))
        value, _ = _brace_group(line, at)
        fields = []
        at = 0
        while at < len(value):
            field, at = _brace_group(value, at)
            fields.append(field)
        if name in labels:
            raise ValueError(f"Duplicate auxiliary label {name} in {path}")
        labels[name] = fields
    return labels


def export_flat_sources(build: Path) -> Path:
    """Create journal source files with flat figure paths and frozen external labels.

    The export removes the circular ``xr`` dependency: each source contains only
    the other document's labels that it actually cites, linked to ``main.pdf``
    or ``ESM_1.pdf``. It is regenerated from a stable build, never hand edited.
    The manifest lists the exact files to put in the source upload ZIP.
    """
    destination = OUTPUT_DIR / "submission-source"
    destination.mkdir(parents=True, exist_ok=True)
    stems = ("article", "supplement")
    labels = {stem: _aux_labels(build / f"{stem}.aux") for stem in stems}
    records = {}
    external_counts = {}

    def copy_file(source, name):
        target = destination / name
        if source.is_file():
            shutil.copy2(source, target)
        else:
            raise FileNotFoundError(source)
        records[name] = hashlib.sha256(target.read_bytes()).hexdigest()

    for name in ("sn-jnl.cls", "sn-basic.bst", "refs.bib"):
        copy_file(build / name, name)
    for stem in stems:
        source = build / f"{stem}.tex"
        text = source.read_text(encoding="utf-8")
        other = "supplement" if stem == "article" else "article"
        external_pdf = "ESM_1.pdf" if stem == "article" else "main.pdf"
        referenced = set(re.findall(r"\\(?:ref|eqref|pageref|autoref)\*?\{([^}]+)\}", text))
        missing = referenced - labels[stem].keys() - labels[other].keys()
        if missing:
            raise ValueError(f"Unknown references in {stem}: {sorted(missing)}")
        external = sorted(referenced - labels[stem].keys())
        definitions = ["% Cross-document labels frozen from the verified paired build."]
        for name in external:
            fields = (labels[other][name] + [""] * 4)[:4] + [external_pdf]
            definitions.append(
                r"\expandafter\gdef\csname r@" + name + r"\endcsname{"
                + "".join("{" + value + "}" for value in fields) + "}"
            )
        external_counts[stem] = len(external)
        text = re.sub(r"^\\usepackage\{xr\}\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\\externaldocument[^\n]*\n", "", text, flags=re.MULTILINE)
        text = text.replace(r"\begin{document}", "\n".join(definitions) + "\n"
                            + r"\begin{document}", 1)
        if r"\externaldocument" in text or re.search(r"\\(?:input|include)\{", text):
            raise ValueError("Submission sources must have inline text and no xr dependencies")
        figure_names = {}

        def figure(match):
            name = match.group(2)
            if name not in figure_names:
                figure_source = (build / name).resolve()
                if build.resolve() not in figure_source.parents:
                    raise ValueError(f"Figure path escapes build directory: {name}")
                prefix = "Fig" if stem == "article" else "FigS"
                flat_name = prefix + str(len(figure_names) + 1) + figure_source.suffix
                figure_names[name] = flat_name
                copy_file(figure_source, flat_name)
            return match.group(1) + "{" + figure_names[name] + "}"

        text = re.sub(r"(\\includegraphics(?:\[[^]]*\])?)\{([^}]+)\}", figure, text)
        target = destination / source.name
        target.write_text(text, encoding="utf-8")
        records[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "purpose": "Flat editable source upload; publisher dependencies remain unmodified",
        "external_pdf_names": {"article": "main.pdf", "supplement": "ESM_1.pdf"},
        "frozen_external_labels": external_counts,
        "source_sha256": {
            stem: hashlib.sha256((build / f"{stem}.tex").read_bytes()).hexdigest()
            for stem in stems
        },
        "files_sha256": records,
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination


def main() -> None:
    """Run LaTeX without writing build products into the checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", type=Path,
                        help="External unpacked Springer Nature template directory")
    parser.add_argument("--flat-source", action="store_true",
                        help="Also export and compile flat journal submission sources")
    args = parser.parse_args()
    destination = prepare_build(args.template_dir)
    compile_documents(destination)
    if args.flat_source:
        flat = export_flat_sources(destination)
        compile_documents(flat)
        shutil.copy2(flat / "article.pdf", flat / "main.pdf")
        shutil.copy2(flat / "supplement.pdf", flat / "ESM_1.pdf")
        print(flat / "source_manifest.json")
    for name in ("article.pdf", "supplement.pdf"):
        if (destination / name).is_file():
            print(destination / name)


if __name__ == "__main__":
    main()
