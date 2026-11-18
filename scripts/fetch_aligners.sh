#!/bin/bash
# Fetch MUSCLE 5 + MAFFT static binaries to ~/.local/bin/. Idempotent.
# Used by tests/eval_balibase_full.py for sanity-comparing TKF-DP-corrected
# FSA against widely-used aligners under a uniform SP/TC scoring path.
set -e
mkdir -p ~/.local/bin ~/.local/share/mafft

if ! command -v muscle > /dev/null; then
    echo "Fetching MUSCLE 5.3 ..."
    curl -sL -o ~/.local/bin/muscle \
        "https://github.com/rcedgar/muscle/releases/download/v5.3/muscle-linux-x86.v5.3"
    chmod +x ~/.local/bin/muscle
fi
muscle --version | head -1

if ! command -v mafft > /dev/null || ! mafft --version 2>&1 | grep -q v7; then
    echo "Fetching MAFFT 7.526 ..."
    cd /tmp
    curl -sL -o /tmp/mafft.tgz \
        "https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz"
    tar -xzf /tmp/mafft.tgz -C /tmp/
    mv /tmp/mafft-linux64/mafftdir ~/.local/share/mafft/mafftdir
    mv /tmp/mafft-linux64/mafft.bat ~/.local/share/mafft/mafft
    cat > ~/.local/bin/mafft << 'WRAP'
#!/bin/bash
exec "$HOME/.local/share/mafft/mafft" "$@"
WRAP
    chmod +x ~/.local/bin/mafft
fi
mafft --version 2>&1 | head -1
