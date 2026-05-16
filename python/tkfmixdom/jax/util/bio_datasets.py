"""Integration with ~/bio-datasets for reproducible data fetching.

If ~/bio-datasets exists (github.com:${REPO_OWNER}/bio-datasets), data is fetched
there and symlinked into the project. If not, data is fetched locally.

Usage in scripts:

    from tkfmixdom.jax.util.bio_datasets import resolve_data_dir

    # Returns a path with data in it. If bio-datasets exists, the data
    # lives there and a symlink is created at the local path.
    data_dir = resolve_data_dir("pfam", local_fallback="pfam/")
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Standard location for bio-datasets repo.
# Override with BIO_DATASETS_HOME env var or --bio-datasets CLI flag.
BIO_DATASETS_HOME = Path(os.environ.get("BIO_DATASETS_HOME",
                                         Path.home() / "bio-datasets"))


def set_bio_datasets_home(path: str):
    """Override the bio-datasets repo location (from CLI or env)."""
    global BIO_DATASETS_HOME
    BIO_DATASETS_HOME = Path(path)


def add_bio_datasets_arg(parser):
    """Add --bio-datasets argument to an argparse parser.

    Call apply_bio_datasets_arg(args) after parse_args() to apply.
    """
    parser.add_argument(
        '--bio-datasets', type=str, default=None, metavar='DIR',
        help='Path to bio-datasets repo (default: $BIO_DATASETS_HOME or ~/bio-datasets)')


def apply_bio_datasets_arg(args):
    """Apply --bio-datasets CLI override if provided."""
    val = getattr(args, 'bio_datasets', None)
    if val:
        set_bio_datasets_home(val)


def bio_datasets_available() -> bool:
    """Check if ~/bio-datasets repo exists."""
    return (BIO_DATASETS_HOME / "fetch").is_dir()


def bio_datasets_data_dir(dataset: str) -> Path:
    """Return ~/bio-datasets/data/<dataset>, creating if needed."""
    d = BIO_DATASETS_HOME / "data" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_data_dir(dataset: str, local_fallback: str = None) -> Path:
    """Resolve the best directory for a dataset.

    Strategy:
    1. If ~/bio-datasets exists, use ~/bio-datasets/data/<dataset>/
       and create a symlink at local_fallback pointing there.
    2. Otherwise, use local_fallback directly.

    Args:
        dataset: dataset name (e.g. "pfam", "treefam", "balibase")
        local_fallback: local directory path to use/symlink (e.g. "pfam/")

    Returns:
        Path to the data directory (may be the bio-datasets path or local)
    """
    if bio_datasets_available():
        data_dir = bio_datasets_data_dir(dataset)
        log.info("Using bio-datasets: %s → %s", dataset, data_dir)

        # Create symlink at local_fallback if requested and not already there
        if local_fallback:
            local = Path(local_fallback)
            if not local.exists():
                local.symlink_to(data_dir)
                log.info("Created symlink: %s → %s", local, data_dir)
            elif local.is_symlink():
                # Already a symlink — check it points to the right place
                target = local.resolve()
                if target != data_dir.resolve():
                    log.warning("Symlink %s points to %s, expected %s",
                                local, target, data_dir)
        return data_dir
    else:
        if local_fallback:
            local = Path(local_fallback)
            local.mkdir(parents=True, exist_ok=True)
            return local
        return Path(dataset)


def ensure_symlinks(data_dir: Path, local_dir: Path, pattern: str = "*.sto"):
    """Create per-file symlinks from local_dir to data_dir for matching files.

    Useful when local_dir contains project-specific files (checkpoints, logs)
    alongside dataset symlinks.

    Skips files that already exist in local_dir (as files or symlinks).
    Never deletes existing files or symlinks.
    """
    import glob
    local_dir.mkdir(parents=True, exist_ok=True)
    n_created = 0
    for src in sorted(data_dir.glob(pattern)):
        dest = local_dir / src.name
        if not dest.exists():
            dest.symlink_to(src)
            n_created += 1
    if n_created:
        log.info("Created %d symlinks: %s/%s → %s/", n_created, local_dir, pattern, data_dir)
    return n_created
