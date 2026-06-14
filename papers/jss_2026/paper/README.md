# JSS 2026 paper — source

LaTeX source for the JSS submission. Compile with `pdfLaTeX`:

```bash
pdflatex article
bibtex article
pdflatex article
pdflatex article
```

or equivalently:

```bash
texi2pdf article.tex
```

(or hit the "Compile PDF" button in your LaTeX editor of choice).

## Files

- `article.tex` — main manuscript
- `refs.bib` — bibliography in JSS-conforming BibTeX format
- `jss.cls` — JSS document class (do not edit)
- `jss.bst` — JSS BibTeX style (do not edit)
- `jsslogo.jpg` — logo embedded by the class

## Style guidelines enforced

Every change to `article.tex` should preserve these JSS conventions:

- Title in title case; section headings and figure/table captions in sentence case
- `\proglang{}` for programming languages, `\pkg{}` for packages, `\code{}` for functions and arguments
- Bibliography titles in title case; software cited via `\pkg{...}` / `\proglang{...}` markup inside the BibTeX entry; DOIs included
- `\cite{...}` / `\citet{...}` / `\citep{...}` everywhere; never hard-coded
- Figures and tables labelled with `\label{fig:...}` or `\label{tab:...}`, referenced as `Figure~\ref{...}` / `Table~\ref{...}`
- Equations with `\label{eq:...}`, referenced as `Equation~\ref{...}`
- `\top` for transpose (write `X^\top`, not `X^T` or `X^{T}`)
- `$p$~value`, `$t$~statistic`, etc., with a tilde and no hyphen
- `e.g.,` and `i.e.,` always followed by a comma
- "Section x.y", never "Subsection x.y"
- Abbreviations all-caps, no periods, introduced with lowercase expansion (e.g., "capital market assumption (CMA)")
- Code chunks use `Code` or `CodeInput`/`CodeOutput` environments; no inline comments in code, comments go in the surrounding LaTeX text

## Status

Section 1 (Introduction) drafted. Sections 2–7 carry placeholders and a one-line content statement each; drafting proceeds top-down.

Bibliography seeded with the methodology genealogy (Tibshirani 1996 through Wang–Leng 2008), univariate-guided regression lineage (Chatterjee et al. 2025, Richland et al. 2025), Python ecosystem references (Pedregosa et al. 2011, Diamond–Boyd 2016 for CVXPY, Fu–Narasimhan–Boyd 2020 for CVXR), and the related sparse-regression software (`sparsegl`, `oem`, `adelie`, `gglasso`).

Additional references will be added as later sections are drafted.
