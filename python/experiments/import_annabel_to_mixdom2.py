"""Import Annabel's trained MixDom params into our MixDom2 checkpoint format.

Loads from /home/shared/mixdom_params/<model_dir>, converts via the exact
flat-class import (option 1: n_classes = D*F*S*R), and saves a checkpoint that
train_pfam.py can load with --eval-only.

Also emits a filtered val-split JSON that excludes families in Annabel's
training set, so the val LL can be computed on a non-overlap subset for a
clean comparison.

Usage:

    cd python && JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python \
        experiments/import_annabel_to_mixdom2.py \
        --params-dir /home/shared/mixdom_params/GTR_3dom_3frag_3site \
        --out-checkpoint pfam/annabel_gtr3_imported.npz \
        --train-set-tsv /home/shared/mixdom_params/training_set_pfams.tsv \
        --pfam-split /home/yam/bio-datasets/data/pfam/seed/splits/v1.json \
        --filtered-split-out /tmp/v1_val_minus_annabel_gtr_train.json

After this, run val LL evals (the script prints both invocations):

    # Full v1 val (2430 fams, ~52% in Annabel's training)
    uv run python train_pfam.py --eval-only \
        --checkpoint pfam/annabel_gtr3_imported.npz \
        --split val \
        --split-file /home/yam/bio-datasets/data/pfam/seed/splits/v1.json \
        --msa-dir pfam/

    # Non-overlap subset (~1167 fams)
    uv run python train_pfam.py --eval-only \
        --checkpoint pfam/annabel_gtr3_imported.npz \
        --split val \
        --split-file /tmp/v1_val_minus_annabel_gtr_train.json \
        --msa-dir pfam/
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params-dir', required=True,
                        help='Annabel params directory '
                             '(e.g. /home/shared/mixdom_params/GTR_3dom_3frag_3site)')
    parser.add_argument('--out-checkpoint', required=True,
                        help='Output .npz checkpoint compatible with train_pfam.py')
    parser.add_argument('--t-rep', type=float, default=0.4,
                        help='Representative t for chi construction (default 0.4 — '
                             'typical Pfam median; matches our postfix training runs)')
    parser.add_argument('--train-set-tsv', default='/home/shared/mixdom_params/training_set_pfams.tsv',
                        help='Annabel training-set TSV (with F81_pfams, GTR_pfams cols)')
    parser.add_argument('--train-set-col', default='GTR_pfams',
                        choices=['GTR_pfams', 'F81_pfams'],
                        help='Which Annabel training column to filter on')
    parser.add_argument('--pfam-split', default='/home/yam/bio-datasets/data/pfam/seed/splits/v1.json',
                        help='Our Pfam split JSON (with train/val/test family lists)')
    parser.add_argument('--filtered-split-out', default=None,
                        help='If set, write a copy of the split JSON with the '
                             'val list filtered to exclude Annabel training families')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip the per-(d,f) emission verification step')
    args = parser.parse_args()

    # Make tkfmixdom importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Force float64 for the verification
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    from tkfmixdom.jax.models.annabel_to_mixdom2 import (
        annabel_to_mixdom2_params, verify_import_matches_annabel,
    )

    print(f"Importing {args.params_dir}")
    params = annabel_to_mixdom2_params(args.params_dir)

    if not args.no_verify:
        print("Verifying per-(d,f) joint match emission matches Annabel exactly...")
        res = verify_import_matches_annabel(args.params_dir, t=0.5)
        print(f"  shape={res['shape']}, n_classes={res['n_classes']}")
        print(f"  max_abs_diff={res['max_abs_diff']:.3e}, "
              f"max_rel_diff={res['max_rel_diff']:.3e}")
        if not res['ok']:
            print(f"WARNING: import mismatch — abort save")
            sys.exit(2)
        print("  OK")

    # Save in train_pfam.py checkpoint format
    n_dom = int(params['dom_weights'].shape[0])
    n_frag = int(params['frag_weights'].shape[1])
    config = {
        'n_dom': n_dom,
        'n_frag': n_frag,
        'source': f'imported_from_annabel:{os.path.basename(args.params_dir.rstrip("/"))}',
    }

    save = {
        'main_ins': np.float64(params['main_ins']),
        'main_del': np.float64(params['main_del']),
        'dom_ins': np.array(params['dom_ins']),
        'dom_del': np.array(params['dom_del']),
        'dom_weights': np.array(params['dom_weights']),
        'frag_weights': np.array(params['frag_weights']),
        'ext_rates': np.array(params['ext_rates']),
        'dom_Qs': np.array(params['dom_Qs']),
        'dom_pis': np.array(params['dom_pis']),
        'dom_S_exch': np.array(params['dom_S_exch']),
        'n_classes_frag': np.int32(params['n_classes']),
        'classdist': np.array(params['classdist']),
        'class_pis': np.array(params['class_pis']),
        'class_S_exch': np.array(params['class_S_exch']),
        'em_iter': np.int32(0),
        'step_file_idx': np.int32(0),
        'log_probs': np.array([]),
        '_config': np.array(json.dumps(config)),
        't_rep': np.float64(args.t_rep),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_checkpoint)) or '.',
                exist_ok=True)
    np.savez_compressed(args.out_checkpoint, **save)
    print(f"\nSaved checkpoint: {args.out_checkpoint}")
    print(f"  n_dom={n_dom}, n_frag={n_frag}, n_classes={int(params['n_classes'])}")
    print(f"  t_rep={args.t_rep}")

    # Build filtered val split if requested
    if args.filtered_split_out:
        with open(args.pfam_split) as f:
            split_data = json.load(f)
        if 'val' not in split_data:
            print(f"WARNING: 'val' key not found in {args.pfam_split}")
            sys.exit(3)

        annabel_train = set()
        with open(args.train_set_tsv) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row.get(args.train_set_col) == '1':
                    annabel_train.add(row['pfam'])

        val_full = list(split_data['val'])
        val_filtered = [f for f in val_full if f not in annabel_train]
        overlap = len(val_full) - len(val_filtered)
        print(f"\nVal-split filter ({args.train_set_col}):")
        print(f"  original val:   {len(val_full)}")
        print(f"  Annabel train:  {len(annabel_train)}")
        print(f"  overlap:        {overlap} ({100.0 * overlap / len(val_full):.1f}%)")
        print(f"  filtered val:   {len(val_filtered)}")

        out_split = dict(split_data)
        out_split['val'] = val_filtered
        out_split['_filtered_excludes_annabel_train'] = {
            'source_tsv': args.train_set_tsv,
            'col': args.train_set_col,
            'n_excluded': overlap,
        }
        with open(args.filtered_split_out, 'w') as f:
            json.dump(out_split, f, indent=2)
        print(f"  saved filtered split: {args.filtered_split_out}")

    # Emit the eval invocations
    print("\n" + "=" * 60)
    print("Next: run val-LL evals")
    print("=" * 60)
    print(f"\n# Full v1 val (with Annabel training overlap):")
    print(f"cd python && uv run python train_pfam.py --eval-only \\")
    print(f"    --checkpoint {args.out_checkpoint} \\")
    print(f"    --split val \\")
    print(f"    --split-file {args.pfam_split} \\")
    print(f"    --msa-dir pfam/")
    if args.filtered_split_out:
        print(f"\n# Non-overlap subset (clean comparison):")
        print(f"cd python && uv run python train_pfam.py --eval-only \\")
        print(f"    --checkpoint {args.out_checkpoint} \\")
        print(f"    --split val \\")
        print(f"    --split-file {args.filtered_split_out} \\")
        print(f"    --msa-dir pfam/")


if __name__ == '__main__':
    main()
