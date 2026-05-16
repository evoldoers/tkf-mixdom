# Unified test-spec statistics

Per-family aggregates over the four canonical reconstruction benchmark specs. Each cell formatted as `mean (median, [min–max])`. Tree-derived stats use the spec-pruned tree (held_out ∪ remaining; xhard additionally drops K nearest neighbours). MSA-derived stats use the same leaf set; `col_entropy` averages per-column amino-acid entropy in bits (gaps excluded from the distribution before entropy); `gap_fraction` reports gaps as a share of all (leaf, column) cells in the kept rows.

| Metric | short | long | hard | xhard |
|---|---|---|---|---|
| Families (target) | 200 | 200 | 141 | 200 |
| Families (kept) | 200 | 200 | 141 | 200 |
| n_leaves | 21.2 (med 18.0, [5.0–49.0]) | 20.7 (med 18.0, [5.0–49.0]) | 29.8 (med 30.0, [8.0–49.0]) | 75.2 (med 96.0, [5.0–98.0]) |
| true_len (residues) | 68.0 (med 68.0, [14.0–100.0]) | 112.1 (med 106.5, [61.0–191.0]) | 94.8 (med 97.0, [14.0–143.0]) | 129.2 (med 119.0, [24.0–309.0]) |
| n_cols | 74.6 (med 77.5, [19.0–100.0]) | 130.2 (med 127.0, [80.0–199.0]) | 139.2 (med 143.0, [49.0–200.0]) | 249.3 (med 232.5, [50.0–500.0]) |
| total tree length | 11.30 (med 8.26, [1.37–40.03]) | 12.93 (med 9.91, [1.47–56.61]) | 24.60 (med 23.43, [5.44–61.01]) | 61.16 (med 60.25, [8.14–154.56]) |
| target→nearest dist | 2.40 (med 2.32, [0.59–6.32]) | 2.69 (med 2.62, [0.71–6.65]) | 2.34 (med 2.28, [1.51–5.72]) | 3.14 (med 2.99, [2.09–8.31]) |
| col entropy (bits) | 1.44 (med 1.41, [0.51–2.55]) | 1.51 (med 1.51, [0.59–2.61]) | 1.90 (med 1.88, [1.17–2.61]) | 1.99 (med 1.96, [0.69–2.72]) |
| gap fraction | 0.078 (med 0.051, [0.000–0.463]) | 0.134 (med 0.101, [0.000–0.533]) | 0.277 (med 0.266, [0.101–0.526]) | 0.441 (med 0.421, [0.212–0.734]) |
| all-gap col fraction | 0.006 (med 0.000, [0.000–0.130]) | 0.013 (med 0.000, [0.000–0.252]) | 0.009 (med 0.000, [0.000–0.116]) | 0.060 (med 0.020, [0.000–0.385]) |
| families with ≥1 all-gap col | 37 / 200 | 60 / 200 | 44 / 141 | 138 / 200 |

Notes:
- short / long use `mean_dist` (mean tree distance from the held-out target to all other leaves) for the `target→nearest dist` row, since neither builder records a true nearest-neighbour distance. hard uses `min_nn_dist` over `remaining`; xhard uses `min_nn_dist_post_drop` (distance to nearest leaf AFTER dropping the K closest as the spec specifies).
- xhard subsamples large families to leaf_cap=100; `n_leaves` for xhard reflects (held + remaining) — i.e. usually 98, occasionally less when the source tree was already small.
- `all-gap col fraction` is the share of columns in the kept submatrix that have no amino acid in any kept leaf (i.e. an "empty column"). A side-effect of subsampling: when xhard drops the K nearest neighbours and caps at 100 leaves, columns whose residues lived only in the dropped/subsampled rows become empty. These columns are trivially predicted as absent and don't exercise the model — interpret xhard headline F1 with this in mind. short / hard never show this; long has 2 of 200; xhard has it in over half of families.

## Representative `short` family

`PF16967` — held_out=`D4ZMB8_SHEVD/284-351`, K (remaining leaves) = 8, n_cols = 69, true_len = 68.
mean target→other = 2.52.

```
  D4ZMB8_SHEVD/284-351  SERRLYFSSPINGRYEILRDNKVIVGDKVKIGQNHIVYSRLPNGIYNVKIKLYG.ADGVVSDLSMKIYN
  A0KED8_AERHH/287-354  STTPVQVFMPANGEVRVYREDRLISLQNLAIGNQNIDTSAFPSGVYNVTVEVYV.DGRLTSTTTQRVTK
  Q6LFV8_PHOPR/302-369  SLLPVEIIMSRAGRVEIYKDSDLIDTQYINAGIARLNTSSFPQGNYLVDVRIYD.GDTFVRSETKQVIK
  I2BD28_SHIBC/289-356  DATPLVIQTNRNARVDIYRGSQLLGSQYFAPGINNINTSTFPPGSYPLELRVFE.NGVLQRTEQQPFTK
  A1SZ51_PSYIN/260-327  RGSKIILFLSQSSQVEIWREDHLLNTASYEAGNQEIDTSSLPNGTYLINLKIRG.TSGVVREEQQLFIK
  D9QNY1_BRESC/310-376  AATPIDVVLPRASRVEIYRNGALVSAAQYSGGLQLIDTSRLPGGSYPIRIVVRD.ASGVTLDEVRTFT.
  D4ZAD9_SHEVD/272-340  NNRRLFYILPNKGRIEVYRDGHLIHSQNVDAGQQSIAFRDLAYGSYTATIVVISAGREILREQQQIFNN
  A0KH17_AERHH/266-333  SATPVYVTPSRPGMVEIYRDGQLINSQQVIAGLQALDTRVLPAGIYEVELRILE.DGQVTERRRETIYK
```

## Representative `long` family

`PF23360` — held_out=`A0A821WZS0_9NEOP/374-482`, K (remaining leaves) = 36, n_cols = 141, true_len = 106.
mean target→other = 3.18.

```
  A0A821WZS0_9NEOP/374-482  PLLDIQYELTGATKNGWIEARITSAVPLDMLFIYCNNKLVI..QTD.TAAVLSLCPPQ.D.........RES........
  A0A0N4VFK4_ENTVE/283-386  PTFSVQDHFVLDKQNGCYVLSLELLIPVDYVLLQSDVYVELDDVDK.GSAVVSQT..............SGN........
  A0A8B8I2B9_VANTA/372-483  PLLDIEHELSGATHNGWQEAKITSAVPLDMLFVYCDSQLEIQ..TD.NAAVLSTCAPQEY.........NRT........
  A0A2A6BBH3_PRIPA/363-471  PTFHIHESFELNKVTGYYILSISLVLPIEFIVVMSQVDIKLVDVER.NNAVVGYTKPKEN.........EEF........
  A0A8S1GSK4_9PELO/395-503  PYFQVYDKFEFSPQLGVYNLTIELVIPIDFLLIQSKMPIRLVEVEK.NASVVCEIRQSEM.........NPW........
  A0A8S1ESM8_9PELO/430-538  PNFQIHDKFEFSPDLGLYNLTIELVIPIDFVLIQSHLPIRLVEVEK.NASVVCQIRKNEL.........NPW........
  A0A0B2UU12_TOXCA/386-494  PRFAIQDHFTLDKILACYTLSIELIIPIDYILLQSDVGVELLDVEK.NSAVVSITTPDEK.........SGN........
  A0A183J7J9_9BILA/147-255  PLFPIHDRFLLNKSDATHQLIIELPVPIECIVLQSEVAIDLLDVKE.NSAIVSFSPCDPQ.........YSN........
```

## Representative `hard` family

`PF17246` — held_out=`B6JXW1_SCHJY/3-99`, K (remaining leaves) = 26, n_cols = 159, true_len = 97.
Family median pairwise tree distance = 3.54.
min target→nn = 3.13.
held-out gap_frac = 0.39.

```
  B6JXW1_SCHJY/3-99        IEAVLNDVTTELKPL............................TSTLPFNEKQLENIIVTELVNITESFSWGISRAVLLT
  A0A067JVK7_JATCU/16-146  FLAFIDYARSILSPDEEEGDVGCDTNGLVRE............TSGPGW......SWIASRILKTCIAYSSGVTAAILFS
  A0A0D2S9S4_GOSRA/30-155  FLAFIEYAWSVISPEEDEDPSGNEEG.................YNGAGW......SWIASRILKTCISYSSGVTAAILLS
  A0A0D3ACZ2_BRAOL/22-156  FLAFVDYARAVISPEQDEIEEEEVRKKNPSETTAE........ASGPGW......GWIASRVLKTCTAYSSGVTAAILLS
  B9SCD1_RICCO/33-158      FIAFVDYARSVLSPVEEEEEGEENIG.................NGGPGW......SWIASRILKTCIAYSSGVTPAILLS
  V4S1R0_CITCL/26-159      FLGLIEYARSVLWPGEEEEGRDESGQDPNNTGSE.........SRGPGW......SWIASRILKTCIAYSSGVTVAILLS
  G7JMU8_MEDTR/26-150      FLNFVDQARSELLSLEDDSNRGDSD..................TSGYGW......SWIVSRILKTCIAYSSGVTPAILLS
  V7C8B0_PHAVU/36-160      FLKFVEHARSELLSLEGDANRDDEG..................SAGLGW......SWIVSRILKTCIAYSSGVTPAILLS
```

## Representative `xhard` family

`PF08522` — held_out=`C7M5X8_CAPOD/56-174`, K (remaining leaves) = 87, n_cols = 256, true_len = 119.
Family median pairwise tree distance = 3.49.
min target→nn (post-drop) = 3.76.
held-out gap_frac = 0.54.
dropped neighbours = ['I4A224_ORNRL/53-163', 'E4T1K6_PALPW/27-166'].

```
  C7M5X8_CAPOD/56-174   RLN...Y...IYLR..............P.TV...ADLPMLSF....GNVTINRN..YT........KEVE.VRL...LD
  I4A3E3_ORNRL/164-282  KSD...L...ILLQPKVVTPV..VSFLEN.SA....TLQIADN.......SAKQN..VT........LSLPFKSL.....
  Q8A890_BACTN/34-161   FPETGGIG..LSMG.....ILQSDNYAME.NP....QINMDHA.....SLSDQFH............ISLT.EPA...SQ
  G0L0T3_ZOBGA/31-144   YTD.......VYFP......................KPELQRS.....IVSGEGL.SIK........VGVY.LGG...LR
  G0L8D4_ZOBGA/36-161   FESTA.....AYFA...........NQYP.VR....TVILDPG.......SDTFE..IN........VGAT.YGG...KY
  H1YGB8_9SPHI/27-161   YPNFTYST..VYFA...........SQYP.ER....TVELGEDLFIDNTLDNQHK..VS........VKAT.LGG.VYEN
  L0G1M2_ECHVK/30-158   YQT.......VYFA...........YQFP.VR....TITFGED.IFDTSLDNEGK..FK........LMVT.TGG.VYSS
  F0S4T8_PSESL/28-157   FEHQS.....VYFA...........YQGP.VR....TITLGED.VFDTSLDNEHK..CE........IIAT.MGG...VY
```
