# Algorithm 1 validation summary

Pooled: 900 tests, 14 below p=0.01, 43 below 0.05, min p=0.0001; KS vs uniform p=0.366.
H1 paired z-scores: 120 statistics, max |z|=2.84, mean=-0.01, sd=1.00.

## H1 successive-conditional test (whole branches, 4 ancestral links)
| regime | λ | μ | t | n | distinct alignments | min p (12 tests) | max paired z |
|---|---|---|---|---|---|---|---|
| equal-long | 0.5 | 0.5 | 2.0 | 100000 | 16301 | 0.233 | 0.85 |
| equal | 0.9 | 0.9 | 1.0 | 100000 | 14932 | 0.147 | 2.45 |
| fast | 1.5 | 2.0 | 0.5 | 100000 | 8988 | 0.130 | 1.54 |
| kappa=2 | 2.0 | 1.0 | 0.8 | 100000 | 25066 | 0.103 | 2.62 |
| kappa>1 | 1.2 | 0.6 | 1.5 | 100000 | 60208 | 0.038 | 1.22 |
| kappa<1 | 0.7 | 1.1 | 1.0 | 100000 | 6988 | 0.172 | 2.55 |
| long | 0.3 | 0.4 | 3.0 | 100000 | 10013 | 0.030 | 1.87 |
| many-anc | 1.0 | 1.5 | 1.0 | 100000 | 50225 | 0.381 | 0.89 |
| short | 0.7 | 1.1 | 0.3 | 100000 | 24222 | 0.009 | 2.84 |
| very-long | 0.2 | 0.3 | 5.0 | 100000 | 933 | 0.128 | 1.97 |

## H2 exact conditional by rejection (whole alignments, 3 ancestral links)
| regime | alignment | accepted | Algorithm 1 draws | min p (12 tests) | tests below 0.01 |
|---|---|---|---|---|---|
| kappa<1 | IDDD | 8759 | 20000 | 0.094 | 0 |
| kappa<1 | IIDDD | 5000 | 20000 | 0.150 | 0 |
| kappa<1 | IDDM | 5000 | 20000 | 0.201 | 0 |
| kappa<1 | MIDD | 5000 | 20000 | 0.103 | 0 |
| kappa<1 | DMID | 5000 | 20000 | 0.001 | 1 |
| kappa<1 | IDMD | 5000 | 20000 | 0.150 | 0 |
| kappa>1 | IDDD | 5000 | 20000 | 0.046 | 0 |
| kappa>1 | IIDDD | 5000 | 20000 | 0.144 | 0 |
| kappa>1 | IIIDDD | 5000 | 20000 | 0.132 | 0 |
| kappa>1 | IIIIDDD | 5000 | 20000 | 0.028 | 0 |
| kappa>1 | IIIIIDDD | 5000 | 20000 | 0.010 | 1 |
| kappa>1 | DMID | 5000 | 20000 | 0.021 | 0 |
| equal | IDDD | 5313 | 20000 | 0.119 | 0 |
| equal | IIDDD | 5000 | 20000 | 0.077 | 0 |
| equal | IDDM | 5000 | 20000 | 0.130 | 0 |
| equal | DMID | 5000 | 20000 | 0.249 | 0 |
| equal | MIDD | 5000 | 20000 | 0.034 | 0 |
| equal | IDMD | 5000 | 20000 | 0.031 | 0 |
| long | IDDD | 9703 | 20000 | 0.332 | 0 |
| long | IIDDD | 5000 | 20000 | 0.006 | 1 |
| long | IMDD | 5000 | 20000 | 0.115 | 0 |
| long | IDDM | 5000 | 20000 | 0.140 | 0 |
| long | IDMD | 5000 | 20000 | 0.056 | 0 |
| long | MIDD | 5000 | 20000 | 0.068 | 0 |
| fast | IDDD | 7189 | 20000 | 0.017 | 0 |
| fast | IMDD | 5000 | 20000 | 0.000 | 2 |
| fast | IDMD | 5000 | 20000 | 0.024 | 0 |
| fast | MIDD | 5000 | 20000 | 0.059 | 0 |
| fast | DMID | 5000 | 20000 | 0.142 | 0 |
| fast | DDMI | 5000 | 20000 | 0.104 | 0 |
| kappa<1 | MIIDD | 5000 | 20000 | 0.009 | 3 |
| kappa<1 | IMIDD | 5000 | 20000 | 0.080 | 0 |
| kappa<1 | IIDMD | 5000 | 20000 | 0.014 | 0 |
| kappa<1 | DMIID | 5000 | 20000 | 0.036 | 0 |
| kappa<1 | DDMII | 5000 | 20000 | 0.009 | 1 |
| kappa<1 | IIMDD | 5000 | 20000 | 0.045 | 0 |
| kappa>1 | DDMII | 5000 | 20000 | 0.076 | 0 |
| kappa>1 | IMIDD | 5000 | 20000 | 0.040 | 0 |
| kappa>1 | IIMDD | 5000 | 20000 | 0.059 | 0 |
| kappa>1 | IIDDM | 5000 | 20000 | 0.216 | 0 |
| kappa>1 | DMIID | 5000 | 20000 | 0.079 | 0 |
| kappa>1 | MIIDD | 5000 | 20000 | 0.174 | 0 |
| equal | DMIID | 5000 | 20000 | 0.067 | 0 |
| equal | MIIDD | 5000 | 20000 | 0.212 | 0 |
| equal | IIDMD | 5000 | 20000 | 0.005 | 2 |
| equal | DDMII | 5000 | 20000 | 0.086 | 0 |
| equal | IDDMI | 5000 | 20000 | 0.024 | 0 |
| equal | IMIDD | 5000 | 20000 | 0.011 | 0 |
| long | IMIDD | 5000 | 20000 | 0.261 | 0 |
| long | IDMID | 5000 | 20000 | 0.061 | 0 |
| long | MIIDD | 5000 | 20000 | 0.096 | 0 |
| long | DMIID | 5000 | 20000 | 0.088 | 0 |
| long | IIDMD | 5000 | 20000 | 0.126 | 0 |
| long | IIDDM | 5000 | 20000 | 0.014 | 0 |
| fast | IMIDD | 5000 | 20000 | 0.065 | 0 |
| fast | IDMID | 5000 | 20000 | 0.061 | 0 |
| fast | DMIID | 5000 | 20000 | 0.084 | 0 |
| fast | IIMDD | 5000 | 20000 | 0.255 | 0 |
| fast | IIDDM | 5000 | 20000 | 0.166 | 0 |
| fast | IIDMD | 5000 | 20000 | 0.060 | 0 |

## H3 exact ghost-count pmf per block (bivariate PGF)
| regime | block | chi² p | bins | E[ghosts] exact | sampled |
|---|---|---|---|---|---|
| kappa<1 | immortal/alive root, n=0 | 0.964 | 6 | 0.2614 | 0.2609 ± 0.0017 |
| kappa<1 | immortal/alive root, n=1 | 0.005 | 7 | 0.3949 | 0.3923 ± 0.0022 |
| kappa<1 | immortal/alive root, n=2 | 0.429 | 8 | 0.5283 | 0.5278 ± 0.0025 |
| kappa<1 | immortal/alive root, n=3 | 0.572 | 8 | 0.6617 | 0.6591 ± 0.0028 |
| kappa<1 | immortal/alive root, n=4 | 0.036 | 9 | 0.7952 | 0.7993 ± 0.0031 |
| kappa<1 | immortal/alive root, n=5 | 0.550 | 9 | 0.9286 | 0.9260 ± 0.0033 |
| kappa<1 | deleted root, n=0 | 0.966 | 6 | 0.1334 | 0.1327 ± 0.0013 |
| kappa<1 | deleted root, n=1 | 0.461 | 7 | 0.4666 | 0.4653 ± 0.0023 |
| kappa<1 | deleted root, n=2 | 0.747 | 8 | 0.6001 | 0.6038 ± 0.0027 |
| kappa<1 | deleted root, n=3 | 0.878 | 8 | 0.7335 | 0.7370 ± 0.0030 |
| kappa<1 | deleted root, n=4 | 0.272 | 9 | 0.8670 | 0.8643 ± 0.0032 |
| kappa<1 | deleted root, n=5 | 0.760 | 9 | 1.0004 | 0.9999 ± 0.0035 |
| kappa>1 | immortal/alive root, n=0 | 0.882 | 8 | 0.4840 | 0.4831 ± 0.0025 |
| kappa>1 | immortal/alive root, n=1 | 0.420 | 9 | 0.7016 | 0.6989 ± 0.0030 |
| kappa>1 | immortal/alive root, n=2 | 0.103 | 10 | 0.9192 | 0.9175 ± 0.0035 |
| kappa>1 | immortal/alive root, n=3 | 0.001 | 11 | 1.1368 | 1.1452 ± 0.0039 |
| kappa>1 | immortal/alive root, n=4 | 0.996 | 12 | 1.3544 | 1.3538 ± 0.0043 |
| kappa>1 | immortal/alive root, n=5 | 0.092 | 12 | 1.5720 | 1.5739 ± 0.0047 |
| kappa>1 | deleted root, n=0 | 0.119 | 7 | 0.2176 | 0.2182 ± 0.0018 |
| kappa>1 | deleted root, n=1 | 0.502 | 10 | 0.8584 | 0.8568 ± 0.0033 |
| kappa>1 | deleted root, n=2 | 0.299 | 10 | 1.0760 | 1.0715 ± 0.0037 |
| kappa>1 | deleted root, n=3 | 0.218 | 11 | 1.2936 | 1.2951 ± 0.0042 |
| kappa>1 | deleted root, n=4 | 0.100 | 12 | 1.5112 | 1.5122 ± 0.0045 |
| kappa>1 | deleted root, n=5 | 0.404 | 13 | 1.7288 | 1.7279 ± 0.0048 |
| equal | immortal/alive root, n=0 | 0.307 | 6 | 0.2771 | 0.2773 ± 0.0018 |
| equal | immortal/alive root, n=1 | 0.317 | 7 | 0.4192 | 0.4175 ± 0.0022 |
| equal | immortal/alive root, n=2 | 0.189 | 8 | 0.5613 | 0.5556 ± 0.0026 |
| equal | immortal/alive root, n=3 | 0.580 | 8 | 0.7034 | 0.7018 ± 0.0029 |
| equal | immortal/alive root, n=4 | 0.668 | 9 | 0.8455 | 0.8439 ± 0.0032 |
| equal | immortal/alive root, n=5 | 0.967 | 9 | 0.9876 | 0.9879 ± 0.0035 |
| equal | deleted root, n=0 | 0.460 | 6 | 0.1421 | 0.1439 ± 0.0013 |
| equal | deleted root, n=1 | 0.893 | 7 | 0.4951 | 0.4948 ± 0.0024 |
| equal | deleted root, n=2 | 0.684 | 8 | 0.6372 | 0.6388 ± 0.0028 |
| equal | deleted root, n=3 | 0.435 | 9 | 0.7793 | 0.7799 ± 0.0031 |
| equal | deleted root, n=4 | 0.099 | 9 | 0.9214 | 0.9274 ± 0.0033 |
| equal | deleted root, n=5 | 0.348 | 10 | 1.0635 | 1.0678 ± 0.0036 |
| fast | immortal/alive root, n=0 | 0.328 | 6 | 0.2576 | 0.2608 ± 0.0017 |
| fast | immortal/alive root, n=1 | 0.352 | 7 | 0.3902 | 0.3937 ± 0.0021 |
| fast | immortal/alive root, n=2 | 0.084 | 8 | 0.5229 | 0.5246 ± 0.0025 |
| fast | immortal/alive root, n=3 | 0.669 | 8 | 0.6556 | 0.6555 ± 0.0028 |
| fast | immortal/alive root, n=4 | 0.973 | 9 | 0.7883 | 0.7879 ± 0.0031 |
| fast | immortal/alive root, n=5 | 0.916 | 9 | 0.9210 | 0.9165 ± 0.0033 |
| fast | deleted root, n=0 | 0.488 | 6 | 0.1327 | 0.1335 ± 0.0013 |
| fast | deleted root, n=1 | 0.168 | 7 | 0.4601 | 0.4605 ± 0.0023 |
| fast | deleted root, n=2 | 0.874 | 8 | 0.5928 | 0.5927 ± 0.0026 |
| fast | deleted root, n=3 | 0.020 | 8 | 0.7255 | 0.7189 ± 0.0029 |
| fast | deleted root, n=4 | 0.158 | 9 | 0.8582 | 0.8608 ± 0.0032 |
| fast | deleted root, n=5 | 0.284 | 9 | 0.9909 | 0.9941 ± 0.0035 |
| long | immortal/alive root, n=0 | 0.575 | 7 | 0.3542 | 0.3539 ± 0.0020 |
| long | immortal/alive root, n=1 | 0.388 | 8 | 0.5286 | 0.5344 ± 0.0026 |
| long | immortal/alive root, n=2 | 0.770 | 9 | 0.7031 | 0.7051 ± 0.0030 |
| long | immortal/alive root, n=3 | 0.995 | 9 | 0.8775 | 0.8774 ± 0.0033 |
| long | immortal/alive root, n=4 | 0.356 | 10 | 1.0520 | 1.0481 ± 0.0036 |
| long | immortal/alive root, n=5 | 0.356 | 11 | 1.2264 | 1.2320 ± 0.0040 |
| long | deleted root, n=0 | 0.320 | 6 | 0.1744 | 0.1734 ± 0.0015 |
| long | deleted root, n=1 | 0.898 | 8 | 0.6314 | 0.6293 ± 0.0028 |
| long | deleted root, n=2 | 0.814 | 9 | 0.8058 | 0.8087 ± 0.0032 |
| long | deleted root, n=3 | 0.429 | 10 | 0.9803 | 0.9824 ± 0.0035 |
| long | deleted root, n=4 | 0.990 | 10 | 1.1547 | 1.1559 ± 0.0038 |
| long | deleted root, n=5 | 0.279 | 11 | 1.3292 | 1.3210 ± 0.0041 |
