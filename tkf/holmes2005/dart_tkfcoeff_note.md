DART ~/dart/src/tkf/tkfcoeff.cc (unchanged since the 2009 first commit apart from compile warnings), TKF_coeff::initialise:
  Z01[n] += b * (lambdaD*Z0[n-m-1] + lambdaI*(Z0[n-m-1] + Z01[n-m-1]))     # = printed (16), no t
  Z11[n] += c * (lambdaD*Z1[n-m-1] + lambdaI*(Z1[n-m-1] + Z01[n-m-1]))     # printed (17) has "t +" in both brackets; code has none
  I0,I1,I01,I11, Z0, Z1, v, a, b, c match (8)-(15) and (13) as printed (first-order Euler: rates at index n-m, I/Z at n-m-1, alpha[m] for alpha(t)).
  accumulate_counts: D = I + n0 + n01 - n01 - n11 = I + N_o - N_**; Z = sum N_z Z_z - T (immortal link's site time removed); M-step lambda = I/(Z+T), mu = D/Z.

Earlier history (SourceForge CVS snapshot, https://sourceforge.net/code-snapshots/cvs/d/da/dart.zip):
before 2009 the file was dart/src/handel/tkfcoeff.cc (revisions 1.1, 2003-11-10, to 1.5, 2005-07-16;
moved to src/tkf/ on 2009-01-27). Revision 1.1 (2003-11-10) already has
  Z11[n] += c * (tmp_lambdaD * tmp_Z1 + tmp_lambdaI * (tmp_Z1 + tmp_Z01));
  deltaZ -= T;  /* subtract site-time contribution of immortal link */
and no later revision changes either line. The implementation used the corrected forms from its first
recorded revision, before the 2005 paper appeared.
