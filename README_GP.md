# RiskShip GP

Modelo de ranking mediante árboles evolucionados con DEAP. Variable objetivo:
`is_lost` por artículo al crear el envío. Complementa el RNA conservado en
`notebooks/rna_shipment_loss.ipynb` y `exports/riskship_rna_model.*`.

## Entrenamiento con todo train

Desde la raíz de la copia local:

```powershell
uv sync --frozen
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe gp_full_experiment.py --workers 2
```

El comando usa todas las 125.082 filas TRAIN para cada fitness; no incluye
validation/test en entrenamiento. Conserva el mismo corte temporal del RNA.
Evalúa tres configuraciones con dos semillas, completa la seleccionada por
validation a cinco semillas y repite cinco veces sin `DOM_DOMAIN_ID`.
Guarda la selección antes de puntuar test. Puede reutilizar corridas terminadas
del mismo plan; para un experimento nuevo usar `--output results/otro_nombre`.
El plan, las semillas y los hiperparámetros no se deben ajustar mirando test.

Resultados medidos y límites: `docs/GP_ENTRENAMIENTO_COMPLETO.md`.
Tablas: `results/gp_full_v1/final_metrics.csv`, `summary.json`,
`comparison_rna_gp.csv`. Evidencias: subcarpeta `figures/`.
La ejecución inicial de 60×5 sigue registrada aparte, en
`docs/GP_EXPERIMENTOS.md` y `results/gp_smoke_results*`.

## Notebook autocontenido

Abrir `notebooks/gp_shipment_loss.ipynb`. Trae código embebido e inicia en modo
`smoke` (30 individuos, 3 generaciones con train completo). `MODE='full'` ejecuta
la suite completa. El modo smoke no tiene el mismo presupuesto que la suite.
Para regenerarlo a partir del código fuente y verificar ejecución:

```powershell
.\.venv\Scripts\python.exe build_gp_notebook.py
.\.venv\Scripts\python.exe check_gp_notebook.py
```

El notebook fue ejecutado localmente de arriba abajo. Las rutas de Colab están
preparadas pero no se afirma una ejecución remota. Requiere autorización previa
para cargar datos corporativos; no monta Drive ni publica datos. Puede leer un
Parquet o únicamente el Parquet contenido dentro de un ZIP autorizado.

## Usar el árbol entrenado localmente

```python
import json
import pandas as pd
from pathlib import Path
from gp_full_experiment import load_local_model, score_raw_shipments

folder = Path('results/gp_full_v1')
selection = json.loads((folder / 'selection_sealed_before_test.json').read_text())
run = json.loads((folder / (selection['reference_run'] + '.json')).read_text())
model = load_local_model(folder / run['model_path'])  # bundle local confiable
items = pd.read_parquet('envios_autorizados.parquet')
scores = score_raw_shipments(model, items)
at_risk = scores >= model['threshold']
```

El score es una puntuación continua, no una probabilidad calibrada. La decisión
usa un umbral elegido en validation. Los modelos contienen mapas de categorías
corporativas y se guardan solo en `private_models/`, excluido del paquete de
resultados compartible. Los JSON/CSV agregados no contienen SHIPMENT_ID.

## Paquete de cambios

`riskship_gp_entrenamiento_completo.zip` incluye código GP, notebook ejecutado,
documentación y resultados agregados. No incluye datos empresariales, modelos
con mapas de categorías ni artefactos RNA. Al usarlo fuera de esta copia, agregar
localmente el `dataset.parquet` original autorizado. Un entorno exclusivo GP
puede instalar `requirements-gp.txt`, sin instalar AutoGluon.

## Interpretación

Se reportan cinco semillas, media y DE. El test ya se conocía del RNA/smoke:
no es una nueva validación ciega. La selección usa validation exclusivamente.
La madurez de la cohorte reproduce la regla retrospectiva RNA. El target
encoding temporal supone disponibles etiquetas de meses previos; se requiere
auditar su madurez real antes de desplegar. La muestra es case-control, las
métricas son por artículo y el posible artefacto de dominio sigue abierto.
