"""The generated release must remain inside its approved source boundary."""
from pathlib import Path
import hashlib
import json


def test_public_export_manifest_contains_only_approved_paths_and_matching_bytes():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'EXPORT-MANIFEST.json').read_text())
    assert manifest['generated_snapshot'] and not manifest['private_history_included']
    assert 'private_source_commit' not in manifest
    forbidden=('project-context/','docs/plans/','archive/','docs/evaluations/')
    for item in manifest['files']:
        rel=item['path'];path=root/rel
        assert not rel.startswith(forbidden)
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert path.suffix not in ('.env','.jsonl','.sqlite','.sqlite3','.db','.log')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    assert any(item['path'].startswith('amplifier_data/licenses/') for item in manifest['files'])
