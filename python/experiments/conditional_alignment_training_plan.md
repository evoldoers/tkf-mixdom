# Conditional (maximum-discrimination) training of TKF-family models on reference alignments — plan

Status: approved and queued (2026-11-08), to start when MBE paper 1 is done; no compute before that. Lives outside
the paper.

## 0. Decisions (2026-11-08)

- Numerator: path-conditioned (the alignment's own pair-HMM path; gap orderings as given). The order-summed
  numerator (section 4b) is deferred.
- Models: TKF92, MixFrag F=2, MixDom-d2f1 (the released configuration), each started from its maximum-likelihood
  fit of paper 1 (TKF92 and MixFrag: the Table-2 counts-EM fits; d2f1: `pfam/svibw_d2f1_nobali_1620.npz`).
- Run order when the time comes: (1) gradient check of C against the two E-steps on 1,000 pairs; (2) TKF92 and
  MixFrag on a 20,000-pair subset of the BAliBASE-free Pfam split, L-BFGS with per-pair 2D VJPs for the
  denominator, alpha in {0, 0.1, 1}; (3) d2f1 the same
  way on the subset, then minibatch Adam on the full BAliBASE-free split if the subset fit transfers;
  (4) evaluation per section 4f. Structural training references (section 4e) are a second phase.


## 1. The objective

For a training set of sequence pairs with reference alignments, {(x_n, y_n, a_n)}, fit theta to maximize the
conditional log-likelihood of the alignments given the sequences,

    C(theta) = sum_n log P(a_n | x_n, y_n; theta)
             = sum_n [ log P(a_n, x_n, y_n; theta) - log P(x_n, y_n; theta) ],

instead of the joint log-likelihood sum_n log P(a_n, x_n, y_n; theta) that counts-based EM and svi-BW maximize now.

The two terms are objects we already compute:

- numerator, log P(a, x, y): the path-conditioned likelihood (1D forward along the alignment; for MixDom
  `tkfmixdom.jax.train.vjp.mixdom_constrained_log_prob`, for TKF92/MixFrag the summarized-count likelihood of
  supplement B.6 / `mixfrag_cherry_em.em_fit`'s objective), or its order-summed variant (2D forward constrained to
  the reference's matched pairs, the scorer of `experiments/ordsum_val/score_matchaligned_val.py` with PATHCOND=0);
- denominator, log P(x, y): the unconstrained 2D forward (`vjp.tkf91_log_prob`, `tkf92_log_prob`, `mixdom_log_prob`),
  each with a custom VJP.

Which numerator. The benchmarks score matched residue pairs; the order of insertions and deletions inside a gap is
unobservable and the references do not fix it. So the natural target is a = the set of matched pairs and the
numerator should be the matches-fixed, orderings-summed likelihood. The path-conditioned numerator is the cheap
first version (closed-form counts for TKF92/MixFrag, existing 1D code for MixDom) and is what the paper's
path-conditioned scoring already uses; the order-summed version needs a gradient for the constrained 2D forward
(section 4b).

For MixDom the latent domain types, fragtypes and site classes are summed in both terms, so P(a | x, y) is the
posterior of the indel path alone, which is what the aligner uses.

## 2. Why EM does not apply, and what does

EM maximizes a lower bound on log P(a, x, y) built from the complete-data log-likelihood. C(theta) subtracts a second
marginal log-likelihood, log P(x, y), which enters with a negative sign, so the same bound is not a bound on C: the
objective is a difference of two log-partition functions (a difference of convex functions in the natural
parameters), and plain EM/Baum–Welch is not monotone for it. Three routes exist.

(a) Gradient methods with the gradient we already have. By Fisher's identity the gradient of each term is an
expected sufficient statistic:

    d/dtheta log P(a, x, y) = E[ d/dtheta log P(complete) | a, x, y ]      (E-step given the alignment: 1D)
    d/dtheta log P(x, y)    = E[ d/dtheta log P(complete) | x, y ]         (E-step given the sequences: 2D)

so dC/dtheta = (statistics expected under the alignment) - (statistics expected under the model's own posterior over
alignments), pushed through d eta / d theta exactly as the closed-form M-step and the score identity do now. This
is the classical MMIE gradient (Bahl et al. 1986) with "competing word sequences" replaced by competing alignments.
Both expectations are already produced by our two E-steps, and the custom VJPs return their difference directly as
grad(constrained_log_prob - log_prob). L-BFGS on a fixed subset (the pattern of `mixdom_lbfgs_refine.py`, 20,000
pairs) or Adam on minibatches then applies without new derivations. In paper 1 the EM-vs-gradient comparison showed
L-BFGS driven by these statistics converging in 5–16 passes on the joint objective; nothing in that changes here
except the objective and the cost per pass (section 3).

Count tensors are of no help for the denominator. The summarized counts (transition and gap-length counts per
branch-length bin) are sufficient for the numerator, the joint likelihood of the given alignments, and nothing
else: P(x, y) sums over every alignment of each pair and depends on the sequences themselves, so each evaluation
of C or its gradient needs one 2D forward pass (with VJP) per pair. The three models are therefore treated alike:
per-pair 2D VJP for the denominator, constrained 1D (or counts) gradient for the numerator; the counts route is a
convenience for the numerator, not a speed-up for the fit.

(b) Growth transformations / extended Baum–Welch (Gopalakrishnan et al. 1991; Povey & Woodland 2002). For
multinomial HMM parameters the rational objective P(a,x,y)/P(x,y) admits an EM-like monotone update, counts(num) -
counts(den) + D*theta_old, with D a damping constant. Our parameters are the TKF rates behind the transition matrix,
not free multinomials, so EBW would apply only to a free-transition surrogate (useful as a diagnostic: it is the
"CRF-like" model of section 5). Not the primary route.

(c) CEM / CCCP (Jebara & Pentland 1998): bound the concave part (the numerator) by its EM Q-function and linearize
the convex part (-log P(x,y)) at the current theta. Each step maximizes Q_a(theta) - <d/dtheta log P(x,y)|theta_t,
theta>, is monotone in C, and in the natural parameters is a closed-form M-step with the denominator's expected
statistics subtracted. In our rate parameterization the inner step is a small nonlinear solve (the same shape as the
TKF92 M-step's cubic/line search). Worth implementing only if (a) turns out to need many passes; it gives the
monotone guarantee EM users expect.

## 3. Cost

The numerator is cheap: for TKF92/MixFrag it is a count tensor (no per-pair DP); for MixDom a 1D pass per pair.
The denominator is the unaligned-training cost and dominates everything: one 2D forward (and its VJP) per pair per
evaluation, O(L^2) memory and time per pair, with no summarization possible. From paper 1: 1,000 Pfam pairs, one 2D gradient pass = 11 s (TKF92), 23 s (MixFrag) on
one GPU. So:

- 20,000-pair subset, L-BFGS, 50 passes: TKF92 ~3 h, MixFrag ~6 h, MixDom roughly 3x that (the 2D MixDom forward is
  the expensive kernel; its VJP costs about the same as an E-step).
- Full Pfam train split (1.13 M cherries): minibatch Adam only; the per-step cost is the denominator's 2D forward on
  the minibatch, i.e. the svi-BW E-step cost.

tau per pair: use the per-pair branch length as the aligners do (ML tau from the unaligned pair, re-estimated at
each outer iteration, or a fixed tau grid summed in both numerator and denominator). Must be the same treatment in
both terms or the objective is not a conditional probability.

## 4. Implementation steps (in order; no runs until paper 1 is submitted)

a. `tkfmixdom/jax/train/conditional.py`: per-pair `cond_log_prob = constrained_log_prob - log_prob` for TKF92,
   MixFrag and MixDom, batched like `likelihood.py`; gradient check that grad(cond) equals the difference of the two
   E-step statistic vectors through d eta/d theta (the three-way check of `compare_optimizers.py`, with the
   constrained E-step as the third leg).
b. Order-summed numerator: a masked 2D forward (paths restricted to the reference's match pairs) with gradient.
   Options: autodiff through the masked forward (memory as for the unconstrained one), or a masked E-step custom VJP
   (the mask is applied to the same forward–backward; counts follow). Validate against the PATHCOND=0 scorer.
c. Trainer: L-BFGS on a fixed subset (reuse `mixdom_lbfgs_refine.py`); Adam minibatch variant; optional CEM step
   (section 2c) if L-BFGS needs > ~50 passes.
d. Regularization: keep the Gamma priors on rates and the CTMC prior, and train a hybrid objective
   J(theta) = C(theta) + alpha * sum_n log P(a_n, x_n, y_n; theta), alpha in {0, 0.1, 1}
   (equivalently the joint term weighted (1 + alpha) against the marginal term weighted 1). Pure conditional
   training (alpha = 0) sees only what P(a | x, y) can see, so it can drift to parameters that align well but
   describe the sequences badly and will overfit a few hundred reference families; the joint term anchors the fit
   to the generative model at a chosen strength, and sweeping alpha traces alignment accuracy against held-out
   likelihood. Precedents: I-smoothing in MMIE (Povey & Woodland 2002), which adds maximum-likelihood statistics to
   the discriminative ones; generative–discriminative hybrids / multi-conditional learning (Bouchard & Triggs 2004;
   Lasserre, Bishop & Minka 2006; McCallum, Pal, Druck & Wang 2006).
e. Training references, with their leak status against the three benchmarks:
   - Pfam seed alignments (large; circular with profile-HMM alignment conventions; the BAliBASE-free split exists);
   - structural pairwise alignments: SCOPe/ASTRAL pairs superposed with TM-align, HOMSTRAD, SABmark's own
     superfamily sets split by family (then SABmark cannot be a test benchmark);
   - never BAliBASE: it stays the held-out test for the headline comparison.
f. Evaluation: alignment accuracy under the paper's matched progressive decoding and under MUSCLE's full decoding,
   on all three benchmarks; held-out joint and order-summed likelihood to measure what conditional training gives
   up; the parameter drift (gap-opening rates, fragment extensions) against the ML fits and against ProbCons's
   divergence-free parameters. Prediction to test: conditional refitting of the indel parameters to references
   helps where ML refitting (paper 1, Decoding section) hurt.
g. Simulation sanity check: data simulated from TKF92 at known parameters; conditional ML is consistent for the
   conditional model, so it should recover the generating parameters (up to parameters that P(a | x, y) cannot
   identify, e.g. anything that scales the joint without changing the posterior over paths).

## 5. Relationship to prior work (what has been done before)

- Conditional ML / MMIE for HMMs. Maximizing P(label sequence | observations) rather than the joint is the
  conditional-maximum-likelihood or maximum-mutual-information training of speech recognition: Nádas (1983) set
  up the comparison of unconditional vs conditional ML; Bahl, Brown, de Souza & Mercer (1986) gave the MMIE gradient
  (expected counts under the correct transcription minus expected counts under all transcriptions), which is
  our dC/dtheta; Gopalakrishnan, Kanevsky, Nádas & Nahamoo (1991) proved the growth-transformation inequality that
  makes the extended Baum–Welch update monotone for rational objectives; Povey & Woodland (2002) is the mature
  form. Jebara & Pentland (1998) is the bound-maximization (CEM) version.
- Maximum discrimination HMMs (Eddy, Mitchison & Durbin 1995). Related but a different ratio: they maximize
  sum_n log P(x_n | M) / P(x_n | null), the discrimination of family members from a null model, so that
  over-represented subfamilies do not dominate (it replaced sequence weighting). The denominator there is a
  different model; ours is the same model marginalized over alignments. Same family of ideas, hence the name the
  user remembered; not the same objective.
- Conditionally trained alignment models. CRFs (Lafferty, McCallum & Pereira 2001) are exactly conditional
  training of a path model; McCallum, Bellare & Pereira (2005) built a CRF for string edit distance trained on
  aligned string pairs; CONTRAlign (Do, Gross & Batzoglou 2006) is the direct precedent for protein alignment:
  a pair-CRF whose parameters are fitted by maximizing the conditional log-likelihood of reference alignments
  (structural references) by gradient methods, with regularization and held-out validation; CONTRAfold did the same
  for RNA structure. Yu, Joachims, Elber & Pillardy (2008) trained alignment models by structural SVM (max-margin
  rather than conditional likelihood). What none of these keep is the evolutionary parameterization: their
  parameters are free potentials with no time dependence. The proposal here is CONTRAlign's objective on TKF92,
  MixFrag and MixDom, so that the fitted object is still an evolutionary model with branch lengths, rates and
  fragment lengths, and the generative and discriminative fits of the same model can be compared.
- Motivation from paper 1 / the literature on ML parameters vs accurate alignments: Holmes & Durbin (1998);
  Frith, Hamada & Horton (2010), who state that log-likelihood scores are not necessarily optimal for
  residue-level alignment accuracy. An alternative discriminative objective is expected accuracy itself,
  sum over reference pairs of P(i ~ j | x, y) (the quantity MEA decoding maximizes), differentiable through the
  same 2D forward–backward; keep as a second objective to try after conditional likelihood.

## References

- Bahl LR, Brown PF, de Souza PV, Mercer RL (1986). Maximum mutual information estimation of hidden Markov model
  parameters for speech recognition. Proc. ICASSP 1986, 49–52.
- Nádas A (1983). A decision theoretic formulation of a training problem in speech recognition and a comparison of
  training by unconditional versus conditional maximum likelihood. IEEE Trans. Acoust. Speech Signal Process.
  31:814–817.
- Gopalakrishnan PS, Kanevsky D, Nádas A, Nahamoo D (1991). An inequality for rational functions with applications
  to some statistical estimation problems. IEEE Trans. Inf. Theory 37:107–113.
- Povey D, Woodland PC (2002). Minimum phone error and I-smoothing for improved discriminative training.
  Proc. ICASSP 2002, I-105–108.
- Jebara T, Pentland A (1998). Maximum conditional likelihood via bound maximization and the CEM algorithm.
  Advances in Neural Information Processing Systems 11.
- Eddy SR, Mitchison G, Durbin R (1995). Maximum discrimination hidden Markov models of sequence consensus.
  J. Comput. Biol. 2:9–23. doi:10.1089/cmb.1995.2.9.
- Lafferty J, McCallum A, Pereira F (2001). Conditional random fields: probabilistic models for segmenting and
  labeling sequence data. Proc. ICML 2001, 282–289.
- McCallum A, Bellare K, Pereira F (2005). A conditional random field for discriminatively-trained finite-state
  string edit distance. Proc. UAI 2005. (details from memory; verify before citing)
- Do CB, Gross SS, Batzoglou S (2006). CONTRAlign: discriminative training for protein sequence alignment.
  Proc. RECOMB 2006, LNBI 3909:160–174.
- Yu C-N, Joachims T, Elber R, Pillardy J (2008). Support vector training of protein alignment models.
  J. Comput. Biol. 15:867–880. (details from memory; verify before citing)
- Bouchard G, Triggs B (2004). The trade-off between generative and discriminative classifiers. Proc. COMPSTAT 2004.
- Lasserre JA, Bishop CM, Minka TP (2006). Principled hybrids of generative and discriminative models. Proc. CVPR 2006.
- McCallum A, Pal C, Druck G, Wang X (2006). Multi-conditional learning: generative/discriminative training for
  clustering and classification. Proc. AAAI 2006.
- Holmes I, Durbin R (1998). Dynamic programming alignment accuracy. J. Comput. Biol. 5:493–504.
- Frith MC, Hamada M, Horton P (2010). Parameters for accurate genome alignment. BMC Bioinformatics 11:80.
