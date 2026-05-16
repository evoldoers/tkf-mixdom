#!/bin/bash
# Run ProteinGym pair HMM scoring in batches to avoid OOM from JIT cache
set -e

PARAMS="params/best/bw_d3f2_fullseed_15iter.npz"
OUTDIR="experiments/proteingym_pair_results"
mkdir -p "$OUTDIR"

# All short assays (<=100 aa), sorted by length
ASSAYS=(
    TCRG1_MOUSE_Tsuboyama_2023_1E0L_indels
    PIN1_HUMAN_Tsuboyama_2023_1I6C_indels
    YNZC_BACSU_Tsuboyama_2023_2JVD_indels
    SQSTM_MOUSE_Tsuboyama_2023_2RRU_indels
    VG08_BPP22_Tsuboyama_2023_2GP8_indels
    MAFG_MOUSE_Tsuboyama_2023_1K1V_indels
    THO1_YEAST_Tsuboyama_2023_2WQG_indels
    OTU7A_HUMAN_Tsuboyama_2023_2L2D_indels
    SAV1_MOUSE_Tsuboyama_2023_2YSB_indels
    ODP2_GEOSE_Tsuboyama_2023_1W4G_indels
    RD23A_HUMAN_Tsuboyama_2023_1IFY_indels
    SDA_BACSU_Tsuboyama_2023_1PV0_indels
    AMFR_HUMAN_Tsuboyama_2023_4G3O_indels
    SR43C_ARATH_Tsuboyama_2023_2N88_indels
    CBX4_HUMAN_Tsuboyama_2023_2K28_indels
    BCHB_CHLTE_Tsuboyama_2023_2KRU_indels
    CUE1_YEAST_Tsuboyama_2023_2MYX_indels
    PITX2_HUMAN_Tsuboyama_2023_2L7M_indels
    POLG_PESV_Tsuboyama_2023_2MXD_indels
    RAD_ANTMA_Tsuboyama_2023_2CJJ_indels
    DN7A_SACS2_Tsuboyama_2023_1JIC_indels
    HCP_LAMBD_Tsuboyama_2023_2L6Q_indels
    NUSG_MYCTU_Tsuboyama_2023_2MI6_indels
    SPG2_STRSG_Tsuboyama_2023_5UBS_indels
    VRPI_BPT7_Tsuboyama_2023_2WNM_indels
    RCD1_ARATH_Tsuboyama_2023_5OAO_indels
    SOX30_HUMAN_Tsuboyama_2023_7JJK_indels
    UBR5_HUMAN_Tsuboyama_2023_1I2T_indels
    TNKS2_HUMAN_Tsuboyama_2023_5JRT_indels
    SPTN1_CHICK_Tsuboyama_2023_1TUD_indels
    MYO3_YEAST_Tsuboyama_2023_2BTT_indels
    NKX31_HUMAN_Tsuboyama_2023_2L9R_indels
    RPC1_BP434_Tsuboyama_2023_1R69_indels
    PR40A_HUMAN_Tsuboyama_2023_1UZC_indels
    RS15_GEOSE_Tsuboyama_2023_1A32_indels
    BBC1_YEAST_Tsuboyama_2023_1TG0_indels
    DNJA1_HUMAN_Tsuboyama_2023_2LO1_indels
    OBSCN_HUMAN_Tsuboyama_2023_1V1C_indels
    VILI_CHICK_Tsuboyama_2023_1YU5_indels
    DOCK1_MOUSE_Tsuboyama_2023_2M0Y_indels
    EPHB2_HUMAN_Tsuboyama_2023_1F0M_indels
    MBD11_ARATH_Tsuboyama_2023_6ACV_indels
    SRBS1_HUMAN_Tsuboyama_2023_2O2W_indels
    PSAE_PICP2_Tsuboyama_2023_1PSE_indels
    ARGR_ECOLI_Tsuboyama_2023_1AOY_indels
    NUSA_ECOLI_Tsuboyama_2023_1WCL_indels
    UBE4B_HUMAN_Tsuboyama_2023_3L1X_indels
    ILF3_HUMAN_Tsuboyama_2023_2L33_indels
    PKN1_HUMAN_Tsuboyama_2023_1URF_indels
    CATR_CHLRE_Tsuboyama_2023_2AMI_indels
    CBPA2_HUMAN_Tsuboyama_2023_1O6X_indels
    CSN4_MOUSE_Tsuboyama_2023_1UFM_indels
    FECA_ECOLI_Tsuboyama_2023_2D1U_indels
    HECD1_HUMAN_Tsuboyama_2023_3DKM_indels
)

BATCH_SIZE=6
TOTAL=${#ASSAYS[@]}
echo "Running $TOTAL assays in batches of $BATCH_SIZE"

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    batch=()
    for ((j=i; j<i+BATCH_SIZE && j<TOTAL; j++)); do
        batch+=("${ASSAYS[$j]}")
    done
    batch_str=$(IFS=,; echo "${batch[*]}")
    batch_num=$((i/BATCH_SIZE + 1))
    total_batches=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))

    outfile="$OUTDIR/batch_${batch_num}.csv"

    # Skip if already done
    if [ -f "$outfile" ]; then
        echo "Batch $batch_num/$total_batches: already done, skipping"
        continue
    fi

    echo "Batch $batch_num/$total_batches: ${batch[*]}"
    JAX_PLATFORMS=cpu uv run python experiments/score_proteingym_indel.py \
        --params "$PARAMS" \
        --pair --tau 0.1 \
        --assays "$batch_str" \
        --out "$outfile" 2>&1

    echo "Batch $batch_num done, results in $outfile"
    echo ""
done

# Combine all batch results
echo "Combining results..."
head -1 "$OUTDIR/batch_1.csv" > experiments/proteingym_pair_bw_d3f2.csv
for f in "$OUTDIR"/batch_*.csv; do
    tail -n +2 "$f" >> experiments/proteingym_pair_bw_d3f2.csv
done

echo "Combined results in experiments/proteingym_pair_bw_d3f2.csv"

# Compute summary stats
uv run python -c "
import csv, numpy as np
from scipy import stats as sp_stats
with open('experiments/proteingym_pair_bw_d3f2.csv') as f:
    rows = list(csv.DictReader(f))
rhos = [float(r['rho']) for r in rows]
print(f'Total assays: {len(rhos)}')
print(f'Mean Spearman rho: {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}')
print(f'Median Spearman rho: {np.median(rhos):.4f}')
print(f'Min: {min(rhos):.4f}, Max: {max(rhos):.4f}')
print(f'Baseline HMMER: 0.389')
"
