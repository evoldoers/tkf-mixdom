### simulated (TKF92) corpus, TKF92  (N=1000 pairs; LL/pair at init -39.8325, best -25.7623; one pass: E-step 6.0s, gradient 0.3s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 30 | 41 | 53 | 245 | 79 | -25.7623 | 4.2e-07 | λ=0.0575 μ=0.0595 r=0.754 |
| L-BFGS | 6 | 7 | 8 | 2 | 18 | -25.7623 | 1.1e-14 | λ=0.0575 μ=0.0595 r=0.754 |
| Adam (lr 0.1) | 55 | 59 | 89 | 20 | 468 | -25.7623 | 0.0e+00 | λ=0.0575 μ=0.0595 r=0.754 |
| Adam (lr 0.01) | 734 | 1034 | 1278 | 335 | 2249 | -25.7623 | 7.8e-11 | λ=0.0575 μ=0.0595 r=0.754 |

### simulated (TKF92) corpus, MixFrag (F=2)  (N=1000 pairs; LL/pair at init -35.9928, best -25.7623; one pass: E-step 5.6s, gradient 0.3s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 80 | 212 | 430 | 1189 | 680 | -25.7624 | 1.2e-05 | λ=0.0575 μ=0.0595 r=(0.748,0.756) w=(0.257,0.743) |
| L-BFGS | 10 | 14 | 17 | 5 | 31 | -25.7623 | 0.0e+00 | λ=0.0575 μ=0.0595 r=(0.754,0.754) w=(0.007,0.993) |
| Adam (lr 0.1) | 63 | 114 | 337 | 40 | 1077 | -25.7623 | 5.1e-11 | λ=0.0575 μ=0.0595 r=(0.754,0.754) w=(0.060,0.940) |
| Adam (lr 0.01) | 786 | 1595 | 2369 | 546 | 3000 | -25.7624 | 8.5e-06 | λ=0.0575 μ=0.0595 r=(0.746,0.755) w=(0.099,0.901) |

### simulated (MixFrag) corpus, MixFrag (F=2)  (N=1000 pairs; LL/pair at init -36.4986, best -23.9767; one pass: E-step 5.4s, gradient 0.5s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 186 | 465 | 724 | 2496 | 968 | -23.9767 | 1.1e-05 | λ=0.0327 μ=0.0334 r=(0.403,0.845) w=(0.681,0.319) |
| L-BFGS | 12 | 14 | 15 | 8 | 40 | -23.9767 | 5.7e-14 | λ=0.0327 μ=0.0334 r=(0.406,0.846) w=(0.684,0.316) |
| Adam (lr 0.1) | 127 | 184 | 231 | 105 | 655 | -23.9767 | 0.0e+00 | λ=0.0327 μ=0.0334 r=(0.406,0.846) w=(0.684,0.316) |
| Adam (lr 0.01) | 977 | 1269 | 1525 | 717 | 2691 | -23.9767 | 4.2e-09 | λ=0.0327 μ=0.0334 r=(0.406,0.846) w=(0.684,0.316) |

### Pfam corpus, TKF92  (N=1000 pairs; LL/pair at init -33.4412, best -21.1468; one pass: E-step 8.8s, gradient 0.6s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 40 | 61 | 82 | 538 | 125 | -21.1468 | 7.9e-07 | λ=0.0366 μ=0.0375 r=0.668 |
| L-BFGS | 6 | 8 | 9 | 5 | 31 | -21.1468 | 8.2e-14 | λ=0.0366 μ=0.0376 r=0.668 |
| Adam (lr 0.1) | 76 | 99 | 118 | 69 | 493 | -21.1468 | 0.0e+00 | λ=0.0366 μ=0.0376 r=0.668 |
| Adam (lr 0.01) | 911 | 1202 | 1443 | 832 | 2414 | -21.1468 | 8.3e-11 | λ=0.0366 μ=0.0376 r=0.668 |
