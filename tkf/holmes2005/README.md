# Holmes (2005) integral equations: algebraic solution

Proof that the closed-form per-zone expectations of paper 1 (eq:score-bds and the
zone forms I_z, Z_z) solve the integral equations (8)-(17) of the Holmes (2005)
Bioinformatics supplement, with (17) corrected (both "t +" terms deleted), and that
the solution is unique. Also relates them to paper 1's eq:volterra. Paper 1 (MBE revision) summarises this in Appendix C.

- `proof.md` - the note (source); `proof.tex` / `proof.pdf` - pandoc build (12 pages)
- `verify_all.py` / `verify_all_output.txt` - sympy checks of every identity used (172 checks, all pass)
- `numeric_check.py` / `numeric_check_output.txt` - numerical checks only (port of the DART
  first-order scheme; converges to the closed forms at first order)
- `dart_tkfcoeff_note.md` - what DART's `src/tkf/tkfcoeff.cc` (github.com/ihh/dart) computes, and its CVS history back to 2003

Build:

    pandoc proof.md -s -o proof.tex -V geometry:margin=2.2cm -V fontsize=10pt -H pandoc_header.tex
    pdflatex proof.tex

Rerun checks (CPU is enough):

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python verify_all.py > verify_all_output.txt
    python numeric_check.py > numeric_check_output.txt
