#!/bin/bash
# Sync only source code to cluster, preserving all remote data and checkpoints

REMOTE="santoshd@greatlakes.arc-ts.umich.edu"
REMOTE_DIR="/scratch/si670f25_class_root/si670f25_class/santoshd/Kaggle1"
LOCAL_DIR="/Users/santoshdesai/Documents/Desai_Projects/Kaggle_1"

echo "Syncing source code only (preserving remote data, checkpoints, outputs, logs)..."
echo "This will NOT delete or overwrite:"
echo "  - outputs/ (including arrow_data, checkpoints, plots)"
echo "  - logs/"
echo "  - data/"
echo "  - mlruns/"
echo "  - Any other data files"
echo ""

# Sync only source code files, preserving all data on remote
rsync -avzP \
  --include='lib/' \
  --include='lib/**' \
  --include='src/' \
  --include='src/**' \
  --include='scripts/' \
  --include='scripts/**' \
  --include='tests/' \
  --include='tests/**' \
  --include='requirements*.txt' \
  --include='README.md' \
  --include='sync_to_cluster.sh' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*venv*/' \
  --exclude='.git/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='data/' \
  --exclude='checkpoints/' \
  --exclude='mlruns/' \
  --exclude='.pip-cache/' \
  --exclude='*.ipynb_checkpoints' \
  --exclude='notebooks/' \
  --exclude='submissions/' \
  --exclude='*' \
  $LOCAL_DIR/ \
  $REMOTE:$REMOTE_DIR/

echo ""
echo "Cleaning up __pycache__ directories and .pyc files on remote..."
# Delete all __pycache__ directories and .pyc/.pyo files in one SSH command
ssh $REMOTE "cd $REMOTE_DIR && find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; find . -type f -name '*.pyc' -delete 2>/dev/null; find . -type f -name '*.pyo' -delete 2>/dev/null; echo 'Cleanup complete'"

echo ""
echo "Sync complete!"
echo "Remote data, checkpoints, and outputs are preserved."
echo "__pycache__ directories and .pyc/.pyo files cleaned up on remote."

