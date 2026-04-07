"""One-time setup script for PostgresSaver checkpoint tables.

Run once during deployment to create the 4 tables PostgresSaver needs:
  checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations

Usage:
    python scripts/setup_checkpoints.py
"""

from quantstack.checkpointing import setup_checkpoint_tables

if __name__ == "__main__":
    setup_checkpoint_tables()
    print("Checkpoint tables created successfully.")
