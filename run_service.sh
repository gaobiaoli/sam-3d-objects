#!/usr/bin/env bash
set -eo pipefail

source /home/bgao491/miniconda3/etc/profile.d/conda.sh
conda activate sam3d
set -u
cd /home/bgao491/MySAM3d

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SAM3D_PIPELINE_CONFIG=/home/bgao491/Dataset/sam-3d-objects/checkpoints/pipeline.yaml
export SAM3D_OUTPUT_DIR=/home/bgao491/MySAM3d/outputs/service/results
export SAM_MODEL_TYPE=vit_h
export SAM_CHECKPOINT_PATH=/home/bgao491/Dataset/segment-anything/sam_vit_h_4b8939.pth
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

mkdir -p "$SAM3D_OUTPUT_DIR"
exec python -m uvicorn server:app --host 0.0.0.0 --port "${SAM3D_PORT:-6006}" --workers 1
