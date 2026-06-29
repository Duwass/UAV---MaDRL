# Build Notes

## Recommended LaTeX Engine

Use XeLaTeX for Vietnamese text:

```powershell
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Working directory:

```powershell
docs/report_latex/
```

## Required Packages

The preview uses:

- `iftex`
- `fontspec` for XeLaTeX/LuaLaTeX
- `babel` with Vietnamese support
- `geometry`
- `amsmath`, `amssymb`, `amsfonts`
- `graphicx`
- `booktabs`, `array`, `multirow`
- `algorithm`, `algpseudocode`
- `natbib`
- `hyperref`
- `xcolor`
- `enumitem`
- `caption`
- `float`

## Compilation Attempt

Chapter I--II compilation was previously attempted with MiKTeX XeLaTeX and succeeded.

The current final-draft build was compiled with:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Result:

- `main.pdf` was generated successfully.
- Final output length: 27 pages.
- BibTeX completed successfully.
- Final XeLaTeX pass completed with exit code 0.
- The generated preview PDFs are:
  - `main.pdf`
  - `build/report_preview_current.pdf`
- The generated final-draft PDF is:
  - `build/uav_madrl_final_draft.pdf`

Observed warnings after the final build:

- `babel` reported that Vietnamese hyphenation patterns were not preloaded and fallback hyphenation patterns were used. This does not block compilation, but the final report environment should configure Vietnamese hyphenation if strict typography is required.
- No undefined citations, undefined references, overfull hbox, or underfull hbox warnings remained in the final `main.log`.

## Notes

- Bibliography entries with `TODO:` notes must be verified before final submission.
- `main.tex` now includes Chapters I--V.
- Chapter V now includes draft conclusion and future-work sections. Remaining work is official metadata, citation verification, final template alignment, and human review.
