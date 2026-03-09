# Building the LaTeX report

This folder contains a minimal semester-project report: `report.tex`, `references.bib` (example), and `images/`.

## Required software (macOS)

Recommended: install the full MacTeX distribution which includes all common packages and binaries.

- MacTeX (full): https://tug.org/mactex/  (recommended, ~4GB)
  or via Homebrew:

```bash
brew install --cask mactex
```

- Lightweight alternative: BasicTeX (smaller) via Homebrew:

```bash
brew install --cask basictex
```

If you install BasicTeX, you will likely need to install missing packages with `tlmgr` (see below).

Make sure TeX binaries are on your PATH (MacTeX usually does this automatically):

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

Verify installation:

```bash
which pdflatex
pdflatex --version
which latexmk
latexmk --version
which bibtex
bibtex --version
```

## Recommended build command

From this folder run:

```bash
latexmk -pdf report.tex
```

`latexmk` will run `pdflatex`/`bibtex` as needed. If you prefer manual steps:

```bash
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

## If packages are missing (BasicTeX)

Install `tlmgr` packages as needed, for example:

```bash
sudo tlmgr update --self
sudo tlmgr install latexmk geometry graphicx amsmath amsfonts amssymb caption subcaption setspace hyperref natbib cm-super
```

Adjust the package list according to errors reported by the build.

## Files in this folder

- `report.tex` — main LaTeX source (simple article-style semester report)
- `references.bib` — example bibliography (edit or replace with your `.bib`)
- `images/` — document images (example: `DDI_LM.png`)
- `report.pdf` — generated output after building

Notes:
- `pdflatex` accepts PNG/PDF/JPG images. If you need to use EPS images, `epstopdf` (included in MacTeX) will be used.
- If the bibliography is empty, add `\cite{...}` entries in the text or populate `references.bib`.

## Quick troubleshooting

- Missing `.sty` errors: install the package via `tlmgr` or install full MacTeX.
- `latexmk` reports missing `.bbl`: run `bibtex report` or ensure `\bibliography{references}` is present and `references.bib` exists.

If you want, I can:
- Populate `references.bib` with your real entries,
- Add example `\cite{example2022}` to `report.tex`, or
- Produce a longer dummy report to reach a target page count.
