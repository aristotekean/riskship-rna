"""Package only GP code, documentation, executed notebook and aggregate results."""
from pathlib import Path
import zipfile


def package():
    root = Path(__file__).resolve().parent
    files = [root / p for p in [
        'gp_shipment_loss.py', 'gp_full_experiment.py', 'gp_report.py',
        'build_gp_notebook.py', 'check_gp_notebook.py', 'package_gp_results.py', 'validate_gp_results.py',
        'requirements-gp.txt', 'README_GP.md', 'tests/test_gp_full.py',
        'docs/GP_EXPERIMENTOS.md', 'docs/GP_ENTRENAMIENTO_COMPLETO.md',
        'notebooks/gp_shipment_loss.ipynb']]
    for folder in [root / 'results' / 'gp_full_v1']:
        files += [p for p in folder.rglob('*') if p.is_file() and
                  'private_models' not in p.parts and p.suffix in ['.json', '.csv', '.txt', '.png']]
    files += [p for p in (root / 'results').glob('gp_smoke_results*') if p.is_file()]
    assert all(p.exists() for p in files), 'A deliverable is missing'
    destination = root / 'riskship_gp_entrenamiento_completo.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            assert path.suffix not in ['.pickle', '.parquet']
            assert 'datos' not in path.relative_to(root).parts
            archive.write(path, path.relative_to(root))
    print({'zip': str(destination), 'files': len(set(files)), 'bytes': destination.stat().st_size})


if __name__ == '__main__':
    package()
