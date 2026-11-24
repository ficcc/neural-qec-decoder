# Paper draft

`paper.tex` is the draft; `paper.pdf` is the compiled version. Every table and headline number in it is generated from the stored results, so the text cannot drift from the data:

```bash
python -m scripts.make_paper_tables        # writes docs/paper/tables/*.tex and docs/paper/macros.tex
cd docs/paper && pdflatex paper.tex && pdflatex paper.tex
```

Figures are read from `../../figures/`. The reference list should be checked against the publishers' records before submission. For a plain-language version of the project see `../ELI5.md`.
