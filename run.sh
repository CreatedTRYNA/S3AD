#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run.sh CONFIG [train|test] [key=value ...]" >&2
    exit 2
fi
config=$1
shift
mode=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
python3 -m torch.distributed.run \
    --nproc_per_node="${nproc_per_node:-1}" \
    --nnodes="${nnodes:-1}" \
    --node_rank="${node_rank:-0}" \
    --master_addr="${master_addr:-127.0.0.1}" \
    --master_port="${master_port:-29500}" \
    run.py -c "$config" -m "$mode" --logger_rank 0 "$@"
