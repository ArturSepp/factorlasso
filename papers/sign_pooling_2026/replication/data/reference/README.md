# Frozen reference results

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [factorlasso](https://github.com/ArturSepp/factorlasso).
Citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

These 16 CSV/NPZ files were moved without numerical changes from the former
`replication/results/` directory. They preserve the existing simulation and eQTL
reference results; this migration does not certify a new full reproduction.
Their SHA-256 digests, and those of the yeast inputs, are in `../sha256.json`.
Git's normal text line-ending conversion is accounted for by the resource check.

`exhibits.py` reads these inputs by default. `exhibits.py --from-run` reads only
new run results in the external output directory. Keep provenance and compare
new results before deciding to replace any frozen input. Never mix missing new
run files with these cached results silently.
