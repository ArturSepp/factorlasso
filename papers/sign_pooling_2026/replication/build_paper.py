"""Compile the approved current manuscript in an external build directory."""

from pathlib import Path
import shutil
import subprocess

if __package__:
    from .paths import OUTPUT_DIR, PAPER_DIR
else:
    from paths import OUTPUT_DIR, PAPER_DIR


def prepare_build() -> Path:
    """Copy exactly approved manuscript assets and available local CAS templates."""
    destination = OUTPUT_DIR / "latex-build"
    destination.mkdir(parents=True, exist_ok=True)
    for line in (PAPER_DIR / ".gitignore").read_text(encoding="utf-8").splitlines():
        if not line.startswith("!/paper/") or line.endswith("/"):
            continue
        relative = Path(line[len("!/paper/"):])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER_DIR / "paper" / relative, target)
    for name in ("cas-sc.cls", "cas-common.sty", "cas-model2-names.bst"):
        source = PAPER_DIR / "paper" / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return destination


def main() -> None:
    """Run LaTeX without writing build products into the checkout."""
    destination = prepare_build()
    commands = (
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "article.tex"],
        ["bibtex", "article"],
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "article.tex"],
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "article.tex"],
    )
    for command in commands:
        subprocess.run(command, cwd=destination, check=True)
    print(destination / "article.pdf")


if __name__ == "__main__":
    main()
