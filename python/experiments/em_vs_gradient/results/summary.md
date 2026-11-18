### simulated (TKF91) corpus, TKF91  (N=1000 pairs; LL/pair at init -159.5693, best -155.0512; one pass: E-step 3.2s, gradient 1.8s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 4 | 5 | 7 | 16 | 12 | -155.0512 | 2.6e-08 | λ=0.0563 μ=0.0581 |
| L-BFGS | 5 | 6 | 6 | 11 | 42 | -155.0512 | 0.0e+00 | λ=0.0563 μ=0.0581 |
| Adam (lr 0.1) | 27 | 72 | 81 | 131 | 322 | -155.0512 | 2.8e-14 | λ=0.0563 μ=0.0581 |
| Adam (lr 0.01) | 469 | 627 | 765 | 1144 | 1000 | -155.0512 | 9.5e-07 | λ=0.0563 μ=0.0581 |

### simulated (TKF92) corpus, TKF91  (N=1000 pairs; LL/pair at init -468.5565, best -450.1396; one pass: E-step 18.9s, gradient 6.0s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 5 | 7 | 9 | 134 | 15 | -450.1396 | 3.4e-08 | λ=0.0697 μ=0.0704 |
| L-BFGS | 7 | 7 | 8 | 43 | 25 | -450.1396 | 2.3e-13 | λ=0.0697 μ=0.0704 |
| Adam (lr 0.1) | 73 | 100 | 126 | 613 | 387 | -450.1396 | 0.0e+00 | λ=0.0697 μ=0.0704 |
| Adam (lr 0.01) | – | – | – | – | 1000 | -450.1569 | 1.7e-02 | λ=0.0696 μ=0.0705 |

### simulated (TKF92) corpus, TKF92  (N=1000 pairs; LL/pair at init -449.6457, best -440.0722; one pass: E-step 19.0s, gradient 6.0s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 37 | 52 | 68 | 1061 | 102 | -440.0722 | 6.1e-07 | λ=0.0560 μ=0.0584 r=0.744 |
| L-BFGS | 6 | 7 | 9 | 47 | 41 | -440.0722 | 1.7e-13 | λ=0.0560 μ=0.0585 r=0.744 |
| Adam (lr 0.1) | 52 | 57 | 88 | 377 | 386 | -440.0722 | 0.0e+00 | λ=0.0560 μ=0.0585 r=0.744 |
| Adam (lr 0.01) | 568 | 888 | – | 5708 | 1000 | -440.0726 | 3.7e-04 | λ=0.0566 μ=0.0592 r=0.747 |

### simulated (MixFrag) corpus, MixFrag (F=2)  (N=1000 pairs; LL/pair at init -453.2220, best -446.4294; one pass: E-step 13.1s, gradient 6.6s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 85 | – | – | – | 300 | -446.4310 | 1.6e-03 | λ=0.0328 μ=0.0339 r=(0.304,0.828) w=(0.612,0.388) |
| L-BFGS | 8 | 15 | 16 | 106 | 52 | -446.4294 | 0.0e+00 | λ=0.0326 μ=0.0337 r=(0.354,0.842) w=(0.670,0.330) |
| Adam (lr 0.1) | 108 | 168 | 211 | 1262 | 507 | -446.4294 | 0.0e+00 | λ=0.0326 μ=0.0337 r=(0.354,0.842) w=(0.670,0.330) |
| Adam (lr 0.01) | 756 | – | – | – | 1000 | -446.4314 | 2.0e-03 | λ=0.0332 μ=0.0344 r=(0.309,0.830) w=(0.609,0.391) |

### Pfam corpus, TKF91  (N=1000 pairs; LL/pair at init -594.4158, best -571.8200; one pass: E-step 27.9s, gradient 11.0s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 4 | 6 | 8 | 166 | 14 | -571.8200 | 2.1e-08 | λ=0.0392 μ=0.0395 |
| L-BFGS | 7 | 8 | 9 | 84 | 14 | -571.8200 | 4.5e-13 | λ=0.0392 μ=0.0395 |
| Adam (lr 0.1) | 100 | 141 | 176 | 1479 | 467 | -571.8200 | 0.0e+00 | λ=0.0392 μ=0.0395 |
| Adam (lr 0.01) | – | – | – | – | 1000 | -571.8710 | 5.1e-02 | λ=0.0392 μ=0.0396 |

### Pfam corpus, TKF92  (N=1000 pairs; LL/pair at init -578.7814, best -566.8030; one pass: E-step 27.7s, gradient 11.2s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | 71 | 115 | 159 | 3398 | 234 | -566.8030 | 1.7e-06 | λ=0.0446 μ=0.0458 r=0.679 |
| L-BFGS | 6 | 7 | 9 | 83 | 26 | -566.8030 | 3.4e-13 | λ=0.0446 μ=0.0458 r=0.680 |
| Adam (lr 0.1) | 91 | 132 | 164 | 1554 | 434 | -566.8030 | 0.0e+00 | λ=0.0446 μ=0.0458 r=0.680 |
| Adam (lr 0.01) | – | – | – | – | 1000 | -566.8173 | 1.4e-02 | λ=0.0492 μ=0.0508 r=0.714 |

### Pfam corpus, MixFrag (F=2)  (N=1000 pairs; LL/pair at init -576.0536, best -566.6005; one pass: E-step 40.7s, gradient 22.8s)

| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | passes run | best LL/pair | gap to best | parameters at best |
|---|---|---|---|---|---|---|---|---|
| closed-form EM | – | – | – | – | 300 | -566.6289 | 2.8e-02 | λ=0.0490 μ=0.0503 r=(0.305,0.856) w=(0.672,0.328) |
| L-BFGS | 13 | 15 | 17 | 306 | 38 | -566.6005 | 0.0e+00 | λ=0.0510 μ=0.0524 r=(0.457,0.920) w=(0.868,0.132) |
| Adam (lr 0.1) | 231 | 307 | 370 | 4241 | 720 | -566.6005 | 1.1e-13 | λ=0.0510 μ=0.0524 r=(0.457,0.920) w=(0.868,0.132) |
| Adam (lr 0.01) | – | – | – | – | 1000 | -566.6298 | 2.9e-02 | λ=0.0515 μ=0.0531 r=(0.327,0.865) w=(0.675,0.325) |
