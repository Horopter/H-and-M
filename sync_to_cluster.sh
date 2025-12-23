#!/bin/bash
# Sync entire project to cluster, deleting remote folder first

REMOTE="santoshd@greatlakes.arc-ts.umich.edu"
REMOTE_DIR="/scratch/si670f25_class_root/si670f25_class/santoshd/Kaggle1"
LOCAL_DIR="/Users/santoshdesai/Documents/Desai_Projects/Kaggle_1"

echo "Deleting remote folder..."
ssh $REMOTE "rm -rf $REMOTE_DIR"

echo "Syncing project..."
rsync -avzP --delete \
  --exclude='.*' \
  --exclude='*venv*/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='checkpoints/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='mlruns/' \
  --exclude='.pip-cache/' \
  --exclude='.git/' \
  $LOCAL_DIR/ \
  $REMOTE:$REMOTE_DIR/

echo "Done!"

