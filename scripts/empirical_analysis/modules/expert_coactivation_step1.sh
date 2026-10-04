#!/bin/bash
# Local version - no GCP download, use existing local actives

mkdir -p $SSD_MOUNT/actives/$MODEL_NAME
echo "Using local actives at: $SSD_MOUNT/actives/$MODEL_NAME"

# GCP download disabled for local-only operation:
# gcloud storage cp --recursive $GCP_MOUNT/actives/$TRAIN_JOB_NAME/$TRAIN_JOB_ID/* $SSD_MOUNT/actives > /dev/null 2>&1
