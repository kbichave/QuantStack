"""Tests for backup script structure and behavior."""
import os
import subprocess
import pytest

BACKUP_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "backup.sh"
)

def test_backup_script_exists():
    assert os.path.isfile(BACKUP_SCRIPT)

def test_backup_script_is_executable():
    assert os.access(BACKUP_SCRIPT, os.X_OK)

def test_backup_script_uses_set_euo_pipefail():
    with open(BACKUP_SCRIPT) as f:
        content = f.read()
    assert "set -euo pipefail" in content

def test_backup_script_uses_flock():
    with open(BACKUP_SCRIPT) as f:
        content = f.read()
    assert "flock" in content

def test_backup_script_uses_custom_format():
    with open(BACKUP_SCRIPT) as f:
        content = f.read()
    assert "--format=custom" in content

def test_backup_script_verifies_dump():
    with open(BACKUP_SCRIPT) as f:
        content = f.read()
    assert "pg_restore --list" in content

def test_backup_script_prunes_old_backups():
    with open(BACKUP_SCRIPT) as f:
        content = f.read()
    assert "RETENTION_DAYS" in content
    assert "-mtime" in content
