"""Generate the self-contained GP notebook from the tested Python sources."""
from pathlib import Path
import nbformat as nb


def build():
    root = Path(__file__).resolve().parent
    doc = nb.v4.new_notebook()
    cells = []
    cells.append(nb.v4.new_markdown_cell("""# RiskShip — Programación Genética reproducible

Predicción de `is_lost` al crear el envío. Mismo conjunto y particiones RNA,
métricas por artículo, evolución de árboles DEAP en CPU.

Este notebook contiene el código necesario y puede abrirse fuera del repositorio.
Modo inicial: **smoke**, 30 individuos, 3 generaciones, fitness con todo train.
`MODE='full'` ejecuta las tres configuraciones, selección con validación, cinco
semillas y cinco ablaciones sin dominio. Puede tardar varios minutos.

Los datos son corporativos. La ejecución local usa el Parquet existente.
Para Colab, cargar datos requiere autorización de su propietario; el notebook
no monta Drive ni carga archivos automáticamente. Cambiar `CLOUD_DATA_AUTHORIZED`
solo si ya se cuenta con esa autorización.

El score del árbol no está calibrado. Test se evalúa después de seleccionar
árbol/configuración/umbral; la comparación usa un test ya conocido del RNA.
"""))
    cells.append(nb.v4.new_code_cell("""import importlib.util, os, subprocess, sys
from pathlib import Path
MODE = 'smoke'  # 'full' para el protocolo completo
CLOUD_DATA_AUTHORIZED = False
IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB and not CLOUD_DATA_AUTHORIZED:
    raise RuntimeError('Se requiere autorización para usar datos corporativos en Colab. También puede ejecutarse localmente.')
if IN_COLAB:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'deap==1.4.4', 'pandas==2.3.3', 'numpy==2.5.3',
                           'scikit-learn==1.9.1', 'pyarrow==24.0.0', 'matplotlib==3.11.2'])
print('Python:', sys.version.split()[0], '| Modo:', MODE)
"""))
    cells.append(nb.v4.new_markdown_cell("""## Datos

`DATA_PATH` puede apuntar a un Parquet local. En Colab autorizado se permite
elegir un Parquet o ZIP desde el equipo. Del ZIP solo se lee `dataset.parquet`.
No se extraen las otras exportaciones ni se envían resultados a servicios.
"""))
    cells.append(nb.v4.new_code_cell("""import zipfile
DATA_PATH = next((p for p in [Path('dataset.parquet'), Path('../dataset.parquet')]
                  if p.exists()), Path('dataset.parquet'))
if not DATA_PATH.exists() and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Elegir exactamente un Parquet o ZIP.')
    source = Path(next(iter(uploaded)))
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            members = [n for n in archive.namelist() if n.split('/')[-1] == 'dataset.parquet']
            if len(members) != 1:
                raise ValueError('El ZIP debe contener un solo dataset.parquet.')
            DATA_PATH = Path('dataset.parquet')
            DATA_PATH.write_bytes(archive.read(members[0]))
    else:
        DATA_PATH = source
if not DATA_PATH.exists():
    raise FileNotFoundError('Definir DATA_PATH al dataset.parquet autorizado.')
OUTPUT = Path('gp_colab_results') if IN_COLAB else DATA_PATH.resolve().parent / 'results' / 'notebook_smoke'
OUTPUT.mkdir(parents=True, exist_ok=True)
"""))
    cells.append(nb.v4.new_markdown_cell("""## Implementación

Las siguientes celdas registran dos módulos embebidos para que el notebook sea
autocontenido. El primero comparte filtros y métricas; el segundo incorpora
codificación temporal suavizada, mediana/IQR de train, operaciones protegidas,
elitismo, penalización por nodos y caché de fitness. La codificación de cada
mes usa etiquetas de meses train anteriores; todos los artículos del mismo
envío quedan juntos. Se asume que esas etiquetas históricas estaban disponibles.
La regla retrospectiva de madurez se conserva por comparabilidad con RNA.
"""))
    for module in ["gp_shipment_loss", "gp_full_experiment"]:
        # Embedded module source is a generated literal; no working-directory dependency.
        source = (root / f"{module}.py").read_text(encoding="utf-8")
        cells.append(nb.v4.new_code_cell(
            "import sys, types\n"
            f"embedded = types.ModuleType({module!r})\n"
            f"sys.modules[{module!r}] = embedded\n"
            f"exec(compile({source!r}, {module + '.py'!r}, 'exec'), embedded.__dict__)\n"
            f"print('Módulo cargado: {module}')"))
    cells.append(nb.v4.new_code_cell("""import json, pickle
import numpy as np, pandas as pd
import gp_full_experiment as experiment
train, val, test, split = experiment.load_parts(DATA_PATH)
display(pd.DataFrame(split).drop(columns='shared_shipment_groups', errors='ignore').T)
assert split['shared_shipment_groups'] == 0
print('Train completo:', len(train), '| Validation:', len(val), '| Test:', len(test))
"""))
    cells.append(nb.v4.new_markdown_cell("""## Ejecución

En modo smoke se selecciona un árbol usando validación y luego se evalúa test.
En modo full se fija el plan antes de entrenar, se comparan tres configuraciones
con semillas 42/43, se completa la ganadora a cinco semillas y se ejecutan cinco
ablaciones. La selección queda registrada antes de calcular nuevas métricas test.
No cambiar hiperparámetros según los resultados test.
"""))
    cells.append(nb.v4.new_code_cell("""if MODE == 'full':
    # One worker works in Colab and avoids multiprocessing of embedded modules.
    experiment.full_suite(DATA_PATH, OUTPUT / 'full', workers=1)
elif MODE == 'smoke':
    config = experiment.Config('notebook_smoke_30x3', 'temporal', 30, 3)
    result = experiment.fit_job((str(DATA_PATH), str(OUTPUT), experiment.asdict(config), 42, True))
    with (OUTPUT / result['model_path']).open('rb') as stream:
        model = pickle.load(stream)
    scores = experiment.predict_tree(model['tree'], model['encoder'].transform(test))
    result['test'] = experiment.evaluate(test['is_lost'].to_numpy(int), scores, model['threshold'])
    experiment.json_write(OUTPUT / 'smoke_evaluated.json', result)
    display(pd.DataFrame({'validation': result['validation'], 'test': result['test']}).drop(index='confusion_matrix'))
    print('Árbol:', result['tree'], '| Nodos:', result['nodes'], '| Tiempo:', round(result['fit_seconds'], 2), 's')
    experiment.make_figures(OUTPUT, {True: (scores, result)}, test['is_lost'].to_numpy(int), [result])
else:
    raise ValueError("MODE debe ser 'smoke' o 'full'")
"""))
    cells.append(nb.v4.new_code_cell("""from IPython.display import Image, display
FIGURES = (OUTPUT / 'full' if MODE == 'full' else OUTPUT) / 'figures'
display(Image(filename=str(FIGURES / 'evolution.png')))
display(Image(filename=str(FIGURES / 'pr_and_ranking.png')))
print('Resultados agregados:', OUTPUT)
print('Los modelos con mapas de categorías son privados; no incluirlos en archivos públicos.')
"""))
    cells.append(nb.v4.new_markdown_cell("""## Resultados ampliados guardados

Si el paquete incluye la suite ya ejecutada, se muestra aquí su comparación
agregada de cinco semillas. Estos son resultados anteriores guardados; la prueba
smoke de las celdas anteriores tiene un presupuesto distinto y no los sustituye.
"""))
    cells.append(nb.v4.new_code_cell("""FULL_RESULTS = DATA_PATH.resolve().parent / 'results' / 'gp_full_v1'
comparison = FULL_RESULTS / 'comparison_rna_gp.csv'
if comparison.exists():
    measured = pd.read_csv(comparison)
    display(measured[measured['metric'].isin(['pr_auc', 'roc_auc', 'f1', 'recall_at_20%'])]
            .pivot(index='metric', columns='model', values='mean').round(4))
    display(Image(filename=str(FULL_RESULTS / 'figures' / 'rna_gp_comparison.png')))
else:
    print('Suite ampliada no disponible en este entorno. Ejecutar MODE=full para generarla.')
"""))
    cells.append(nb.v4.new_markdown_cell("""## Alcance de los resultados

Una corrida smoke valida funcionamiento. Para comparar, usar media y desviación
estándar de cinco semillas y la misma partición, con y sin `DOM_DOMAIN_ID`.
Las métricas son por artículo y sobre una muestra case-control; no describen
directamente precisión o probabilidad de producción. RNA usa umbral 0,5 y GP
selecciona F1 en validación, por lo que sus métricas de clasificación no tienen
el mismo punto operativo. PR-AUC y ranking permiten comparar sin ese umbral.

La RNA reportada tiene PR-AUC 0,7496 ± 0,0020 y ROC-AUC 0,8241 ± 0,0020.
Estas cifras proceden del repositorio; este notebook no reentrena la RNA.
"""))
    doc.cells = cells
    doc.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"}}
    destination = root / "notebooks" / "gp_shipment_loss.ipynb"
    nb.write(doc, destination)
    nb.validate(doc)
    print(destination)


if __name__ == "__main__":
    build()
