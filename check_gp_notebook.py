"""Execute the generated GP notebook top-to-bottom with this interpreter."""
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    source = root / 'notebooks' / 'gp_shipment_loss.ipynb'
    notebook = nbformat.read(source, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=300,
                            resources={'metadata': {'path': str(source.parent)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    errors = [out for cell in notebook.cells if cell.cell_type == 'code'
              for out in cell.get('outputs', []) if out.output_type == 'error']
    if errors:
        raise AssertionError(errors)
    nbformat.write(notebook, source)
    counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == 'code']
    print({'executed_cells': counts, 'errors': len(errors), 'notebook': str(source)})


if __name__ == '__main__':
    main()
