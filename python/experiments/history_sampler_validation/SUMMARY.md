# Algorithm 1 validation summary

Pooled: 308 tests, 1 below p=0.01, 9 below 0.05, min p=0.0053; KS vs uniform p=0.161.
H1 paired z-scores: 60 statistics, max |z|=2.68, mean=+0.33, sd=0.94.

## H1 successive-conditional test (whole branches, 4 ancestral links)
| regime | λ | μ | t | n | distinct alignments | min p (12 tests) | max paired z |
|---|---|---|---|---|---|---|---|
| kappa<1 | 0.7 | 1.1 | 1.0 | 20000 | 2961 | 0.067 | 2.09 |
| kappa>1 | 1.2 | 0.6 | 1.5 | 20000 | 15561 | 0.042 | 1.44 |
| equal | 0.9 | 0.9 | 1.0 | 20000 | 5698 | 0.046 | 1.57 |
| long | 0.3 | 0.4 | 3.0 | 20000 | 3962 | 0.105 | 2.68 |
| fast | 1.5 | 2.0 | 0.5 | 20000 | 3749 | 0.135 | 1.31 |
| long (rerun, grid 2049) | 0.3 | 0.4 | 3.0 | 60000 | 7588 | 0.159 | 1.60 |
| long (rerun, grid 2049) | 0.3 | 0.4 | 3.0 | 60000 | 7611 | 0.016 | 2.24 |

## H2 exact conditional by rejection (whole alignments, 3 ancestral links)
| regime | alignment | accepted | Algorithm 1 draws | min p (12 tests) | tests below 0.01 |
|---|---|---|---|---|---|
| kappa<1 | IDDD | 8655 | 20000 | 0.278 | 0 |
| kappa<1 | MIDD | 5000 | 20000 | 0.207 | 0 |
| kappa<1 | IIDDD | 5000 | 20000 | 0.086 | 0 |
| kappa>1 | IDDD | 5000 | 20000 | 0.508 | 0 |
| kappa>1 | IIDDD | 5000 | 20000 | 0.218 | 0 |
| kappa>1 | IIIDDD | 5000 | 20000 | 0.138 | 0 |
| equal | IDDD | 5302 | 20000 | 0.161 | 0 |
| equal | IIDDD | 5000 | 20000 | 0.406 | 0 |
| equal | DMID | 5000 | 20000 | 0.078 | 0 |
| kappa<1 | IIDMD | 5000 | 20000 | 0.026 | 0 |
| kappa<1 | IIMDD | 5000 | 20000 | 0.028 | 0 |
| kappa<1 | IDMID | 5000 | 20000 | 0.192 | 0 |
| kappa>1 | IDMID | 5000 | 20000 | 0.015 | 0 |
| kappa>1 | DDMII | 5000 | 20000 | 0.105 | 0 |
| kappa>1 | IIMDD | 5000 | 20000 | 0.005 | 1 |
| equal | IIDDM | 5000 | 20000 | 0.154 | 0 |
| equal | IDDMI | 5000 | 20000 | 0.024 | 0 |
| equal | IDMID | 5000 | 20000 | 0.054 | 0 |

## H3 exact ghost-count pmf per block (bivariate PGF)
| regime | block | chi² p | bins | E[ghosts] exact | sampled |
|---|---|---|---|---|---|
| kappa<1 | immortal/alive root, n=0 | 0.878 | 5 | 0.2614 | 0.2616 ± 0.0039 |
| kappa<1 | immortal/alive root, n=1 | 0.785 | 6 | 0.3949 | 0.3970 ± 0.0048 |
| kappa<1 | immortal/alive root, n=2 | 0.700 | 7 | 0.5283 | 0.5292 ± 0.0056 |
| kappa<1 | immortal/alive root, n=3 | 0.603 | 7 | 0.6617 | 0.6641 ± 0.0063 |
| kappa<1 | deleted root, n=0 | 0.152 | 5 | 0.1334 | 0.1271 ± 0.0028 |
| kappa<1 | deleted root, n=1 | 0.834 | 6 | 0.4666 | 0.4682 ± 0.0052 |
| kappa<1 | deleted root, n=2 | 0.778 | 7 | 0.6001 | 0.5992 ± 0.0059 |
| kappa<1 | deleted root, n=3 | 0.331 | 7 | 0.7335 | 0.7379 ± 0.0066 |
| kappa>1 | immortal/alive root, n=0 | 0.192 | 7 | 0.4840 | 0.4838 ± 0.0055 |
| kappa>1 | immortal/alive root, n=1 | 0.761 | 8 | 0.7016 | 0.6915 ± 0.0067 |
| kappa>1 | immortal/alive root, n=2 | 0.518 | 9 | 0.9192 | 0.9237 ± 0.0079 |
| kappa>1 | immortal/alive root, n=3 | 0.309 | 9 | 1.1368 | 1.1428 ± 0.0087 |
| kappa>1 | deleted root, n=0 | 0.363 | 6 | 0.2176 | 0.2184 ± 0.0040 |
| kappa>1 | deleted root, n=1 | 0.877 | 8 | 0.8584 | 0.8546 ± 0.0074 |
| kappa>1 | deleted root, n=2 | 0.013 | 9 | 1.0760 | 1.0535 ± 0.0082 |
| kappa>1 | deleted root, n=3 | 0.889 | 10 | 1.2936 | 1.2884 ± 0.0093 |
| equal | immortal/alive root, n=0 | 0.753 | 5 | 0.2771 | 0.2817 ± 0.0040 |
| equal | immortal/alive root, n=1 | 0.444 | 6 | 0.4192 | 0.4132 ± 0.0049 |
| equal | immortal/alive root, n=2 | 0.351 | 7 | 0.5613 | 0.5557 ± 0.0058 |
| equal | immortal/alive root, n=3 | 0.221 | 7 | 0.7034 | 0.6932 ± 0.0065 |
| equal | deleted root, n=0 | 0.644 | 5 | 0.1421 | 0.1426 ± 0.0030 |
| equal | deleted root, n=1 | 0.357 | 6 | 0.4951 | 0.4938 ± 0.0053 |
| equal | deleted root, n=2 | 0.847 | 7 | 0.6372 | 0.6302 ± 0.0061 |
| equal | deleted root, n=3 | 0.202 | 7 | 0.7793 | 0.7788 ± 0.0068 |
| fast | immortal/alive root, n=0 | 0.120 | 5 | 0.2576 | 0.2498 ± 0.0038 |
| fast | immortal/alive root, n=1 | 0.323 | 6 | 0.3902 | 0.3804 ± 0.0047 |
| fast | immortal/alive root, n=2 | 0.158 | 7 | 0.5229 | 0.5227 ± 0.0056 |
| fast | immortal/alive root, n=3 | 0.352 | 7 | 0.6556 | 0.6488 ± 0.0062 |
| fast | deleted root, n=0 | 0.273 | 5 | 0.1327 | 0.1267 ± 0.0028 |
| fast | deleted root, n=1 | 0.506 | 6 | 0.4601 | 0.4601 ± 0.0051 |
| fast | deleted root, n=2 | 0.538 | 7 | 0.5928 | 0.5864 ± 0.0059 |
| fast | deleted root, n=3 | 0.967 | 7 | 0.7255 | 0.7249 ± 0.0065 |
