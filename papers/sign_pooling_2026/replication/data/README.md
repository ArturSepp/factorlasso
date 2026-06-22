# Data

Yeast expression-QTL cross of *Saccharomyces cerevisiae* from Brem & Kruglyak
(2005), "The landscape of genetic complexity across 5,700 gene expression traits
in yeast," PNAS 102(5):1572-1577 (doi:10.1073/pnas.0408709102). The data are
public and are redistributed with the `trigger` Bioconductor package.

- `yeast_full.npz` : `Y` (112 x 6216 gene expression), `X` (112 x 3244 marker
  genotypes), and `genes` (gene labels). Loaded with `numpy.load`.
- `yeast.rda` : the `trigger::yeast` object, used here only for
  `marker.pos` (chromosome and position of the 3244 markers). Loaded with the
  `rdata` package.

`eqtl_pipeline.load_data()` reads both files from this directory.
