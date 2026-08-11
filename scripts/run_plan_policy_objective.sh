#!/usr/bin/env bash
# Run the policy-alignment-only planner + 3D EE optimization-history viz.
# Mirrors the verified command; extra args are passed through to the script, e.g.
#   ./run_plan_policy_objective.sh --episode 3 --policy-objective-weight 0.1
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python plan_policy_objective_and_visualize.py \
    --repo-id Sorozco0612/pusht_200_20260706_130548 \
    --episode 0 \
    --n-actions 16 \
    --max-trajectories-per-iter 15 \
    --out-html /tmp/planned_opt_history_3d.html \
    "$@"
