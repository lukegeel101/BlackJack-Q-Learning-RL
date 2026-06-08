#!/usr/bin/env bash
# Run the 3-way agent comparison across multiple seeds and summarize.
#
# Use this after train_supervised.py produces dqn_agent.pt.
#
# Each seed runs simulate.py for $HANDS hands per agent. Edges are
# averaged across seeds. With 4 seeds × 200k hands, total = 800k hands
# per agent (matches the previous Agent 3 evaluation methodology).

set -euo pipefail

HANDS="${HANDS:-200000}"
SEEDS="${SEEDS:-42 43 44 45}"
MODEL="${MODEL:-dqn_agent.pt}"
MARGIN="${MARGIN:-0.05}"

OUTDIR="$(mktemp -d /tmp/blackjack-eval-XXXXXX)"
echo "Running eval: hands=$HANDS  seeds=$SEEDS  model=$MODEL  margin=$MARGIN"
echo "Per-seed logs: $OUTDIR/"
echo

for s in $SEEDS; do
    log="$OUTDIR/seed_${s}.log"
    echo "  seed=$s ..."
    python simulate.py \
        --hands "$HANDS" \
        --seed "$s" \
        --model "$MODEL" \
        --no-plot \
        > "$log" 2>&1
    # Pull just the edge lines for each agent.
    grep -E "(Agent|Flat|Counting|DQN).*edge|edge \(per" "$log" | head -3
    echo
done

echo "=== aggregate ==="
python - <<PY
import glob, re, statistics
edges = {'Flat-bet Basic Strategy': [], 'Counting Basic Strategy': [], 'DQN Card Counting': []}
for f in sorted(glob.glob("$OUTDIR/seed_*.log")):
    name = None
    with open(f) as fh:
        for line in fh:
            m = re.match(r"^\s*(\S.+?)\s*$", line)
            stripped = line.strip()
            if stripped in edges:
                name = stripped
            elif name and 'edge (per $1)' in line:
                m = re.search(r"edge \(per \$1\)\s*:\s*([+-]?\d+\.\d+)%", line)
                if m:
                    edges[name].append(float(m.group(1)))
                    name = None

for agent, vals in edges.items():
    if not vals:
        continue
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    n = len(vals)
    print(f"  {agent:35s}  edge = {mean:+.3f}% +/- {stdev:.3f}%  (n={n})")
PY
