"""Build aggregate experiment report and verify original files against the ZIP."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


def check_originals(root, archive_path, output):
    checked, changed = 0, []
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = Path(*item.filename.split('/')[1:])
            local = root / relative
            source_hash = hashlib.sha256(archive.read(item)).hexdigest()
            local_hash = hashlib.sha256(local.read_bytes()).hexdigest() if local.exists() else None
            checked += 1
            if source_hash != local_hash:
                changed.append(str(relative))
            if str(relative).startswith(('exports', 'notebooks', 'dataset.parquet')):
                hashes[str(relative)] = {'original': source_hash, 'current': local_hash}
    expected_changes = {'pyproject.toml', 'uv.lock'}
    unexpected = [p for p in changed if p not in expected_changes]
    report = {'zip_name': Path(archive_path).name, 'original_files_checked': checked,
              'changed_original_files': changed, 'unexpected_changes': unexpected,
              'rna_and_data_unchanged': not unexpected, 'sha256': hashes}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if unexpected:
        raise AssertionError(f'Unexpected original changes: {unexpected}')
    return report


def build(root, directory, archive):
    table = pd.read_csv(directory / 'final_metrics.csv')
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    selection = json.loads((directory / 'selection_sealed_before_test.json').read_text(encoding='utf-8'))
    screening = json.loads((directory / 'configuration_selection.json').read_text(encoding='utf-8'))
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    integrity = check_originals(root, archive, directory / 'original_integrity.json')
    reference = json.loads((directory / (selection['reference_run'] + '.json')).read_text(encoding='utf-8'))
    reference_no = json.loads((directory / (selection['reference_no_domain'] + '.json')).read_text(encoding='utf-8'))
    independent = json.loads((directory / 'independent_validation.json').read_text(encoding='utf-8'))
    assert independent['errors'] == 0 and len(independent['checked_runs']) == 10
    tie_table = pd.read_csv(directory / 'ranking_tie_audit.csv')
    ties = ['| Dominio | K | Recall esperado con desempate aleatorio (media) |', '|---|---:|---:|']
    for (included, fraction), group in tie_table.groupby(['include_domain', 'fraction']):
        ties.append(f"| {'Con' if included else 'Sin'} | {fraction:.0%} | {group.expected_recall_random_ties.mean():.4f} |")
    metrics = [('pr_auc', 'PR-AUC', .7496, .0020), ('roc_auc', 'ROC-AUC', .8241, .0020),
               ('recall', 'Recall', .5339, .0136), ('precision', 'Precision', .7568, .0157),
               ('f1', 'F1', .6259, .0044), ('recall_at_1%', 'Recall@1 %', .0321, .0000),
               ('recall_at_5%', 'Recall@5 %', .1581, .0006),
               ('recall_at_10%', 'Recall@10 %', .3019, .0020),
               ('recall_at_20%', 'Recall@20 %', .5060, .0024),
               ('precision_at_1%', 'Precision@1 %', 1., .0),
               ('precision_at_5%', 'Precision@5 %', .9873, .0038),
               ('precision_at_10%', 'Precision@10 %', .9421, .0061),
               ('precision_at_20%', 'Precision@20 %', .7895, .0037)]
    comparison = ['| Métrica test | GP con dominio | GP sin dominio | RNA reportada |',
                  '|---|---:|---:|---:|']
    compare_records = []
    for key, name, rna_mean, rna_std in metrics:
        values = [summary[variant][key] for variant in ['with_domain', 'without_domain']]
        comparison.append(f"| {name} | {values[0]['mean']:.4f} ± {values[0]['std']:.4f} | "
                          f"{values[1]['mean']:.4f} ± {values[1]['std']:.4f} | {rna_mean:.4f} ± {rna_std:.4f} |")
        for variant, value in zip(['GP con dominio', 'GP sin dominio'], values):
            compare_records.append({'model': variant, 'metric': key, **value, 'source': 'measured_current_run'})
        compare_records.append({'model': 'RNA', 'metric': key, 'mean': rna_mean, 'std': rna_std,
                                'source': 'repository_report_not_retrained'})
    pd.DataFrame(compare_records).to_csv(directory / 'comparison_rna_gp.csv', index=False)
    configs = ['| Configuración | Codificación | Población | Generaciones | λ | AP validación (media 42/43) |',
               '|---|---|---:|---:|---:|---:|']
    for cfg in plan['configurations']:
        configs.append(f"| {cfg['name']} | {cfg['encoding']} | {cfg['population']} | {cfg['generations']} | "
                       f"{cfg['penalty']} | {screening['validation_means'][cfg['name']]:.4f} |")
    terminals = '\n'.join(f"- `x{i}`: `{name}`" for i, name in reference['terminal_names'].items())
    fit_total = sum(json.loads(p.read_text(encoding='utf-8'))['fit_seconds']
                    for p in directory.glob('*__seed*.json'))
    gaps = []
    for history_path in directory.glob('*_history.csv'):
        history = pd.read_csv(history_path)
        for generation, delta in zip(history.generation, history.elapsed_seconds.diff()):
            if delta > 60:
                gaps.append({'run': history_path.stem.removesuffix('_history'),
                             'generation': int(generation), 'wall_gap_seconds': float(delta)})
    (directory / 'timing_anomalies.json').write_text(json.dumps(gaps, indent=2), encoding='utf-8')
    report = f'''# RiskShip — GP con todo el conjunto de entrenamiento

## Resultado y alcance

Se ejecutaron tres configuraciones con dos semillas cada una, se completó la
ganadora a cinco semillas (42–46) y se repitió esa misma configuración cinco
veces sin `DOM_DOMAIN_ID`: **14 entrenamientos**. El fitness de cada árbol usa
**las 125.082 filas de train**, sin submuestreo. Los resultados siguientes son
medidos; la RNA procede de las cifras publicadas en el repositorio.

{chr(10).join(comparison)}

GP con dominio alcanza AP test media {summary['with_domain']['pr_auc']['mean']:.4f},
frente a 0,5479 del smoke inicial (una semilla). Sigue por debajo de RNA
(0,7496). La ablación reduce AP media en
{summary['with_domain']['pr_auc']['mean'] - summary['without_domain']['pr_auc']['mean']:.4f};
esto muestra dependencia de dominio, compatible con la preocupación por su
artefacto de extracción. La comparación con el smoke cambia codificación,
presupuesto evolutivo y filas de fitness a la vez: no permite atribuir la mejora
exclusivamente a usar más datos ni probar el efecto aislado de cada técnica.

Media ± desviación estándar muestral (ddof=1). Las métricas son por artículo.
La comparación principal usa ranking/AP/ROC-AUC: RNA clasifica con umbral 0,5,
y GP con umbral F1 óptimo en validación, por lo que los puntos operativos son
distintos. No se reentrenó RNA.

## Datos, integridad y protección del holdout

Mismo Parquet de 262.317 × 58; 165.002 filas después de filtros RNA:
train septiembre 2025–enero 2026 (125.082 filas, 112.303 envíos), validation
febrero 2026 (18.831 filas, 17.297 envíos), test marzo 2026 (21.089 filas,
19.506 envíos). Cero grupos compartidos. Se verifican forma, recuentos y
consistencia de etiqueta por envío en cada ejecución.

Se comprobó SHA-256 de {integrity['original_files_checked']} archivos originales contra el ZIP.
El modelo exportado RNA, su manifiesto, notebook, informe y datos conservan
sus bytes originales. Únicos archivos originales cambiados en la copia:
`pyproject.toml` y `uv.lock`, para añadir DEAP en la fase inicial.
La evidencia está en `original_integrity.json`.

Las configuraciones se fijaron en `plan.json`; se elige la mayor AP media en
validation entre las semillas 42/43. El árbol de cada corrida se elige por AP
de validation entre campeones de train por generación y los cinco mejores
árboles de train al terminar. El umbral maximiza F1 exactamente sobre todos
los umbrales de validation, incluyendo empates de score. Test se puntúa
después de guardar `selection_sealed_before_test.json`.

El test ya había sido observado en el RNA y el smoke test anterior. Por eso
esta es una comparación con holdout fijo, no una validación ciega nueva.

## Técnicas aplicadas

1. Codificación por frecuencia aprendida solo de train como referencia.
2. Codificación objetivo temporal suavizada (50 observaciones equivalentes de
   prior), adicional a frecuencias. Para cada mes train solo usa etiquetas de
   meses train anteriores; primer mes: prior fijo 0,5. Todos los artículos del
   mismo envío permanecen juntos. Validation/test usan mapas de train completo.
3. Imputación por mediana y escalado mediana/IQR de train; recorte ±30 para
   limitar extremos. El score sigue sin ser probabilidad calibrada.
4. Árboles DEAP con suma, resta, multiplicación, división protegida, min/max,
   negación, absoluto, tanh, logaritmo protegido y constantes [-1,1].
5. Selección torneo 4, crossover 0,8 y mutación 0,2 independientes (un hijo
   puede recibir ambos), elitismo de un individuo, altura máxima 8 y máximo
   127 nodos. Fitness: AP(train completo) − λ × nodos.
6. Caché por expresión para no recalcular árboles idénticos. Incluye terminales
   individuales en la población inicial y protege las operaciones intermedias.
7. Selección por validation y ablación completa sin dominio: 13 variables
   fuente frente a 12; en codificación temporal, 19 frente a 17 terminales.

{chr(10).join(configs)}

Configuración elegida: **{selection['configuration']['name']}**. La búsqueda
con dos semillas es limitada; no demuestra optimalidad. Las cinco semillas
posteriores cuantifican variación de entrenamiento, no incertidumbre por
muestreo ni por selección de hiperparámetros.

## Modelo de referencia

Con dominio: semilla {reference['seed']}, elegida por validation AP
{reference['validation_ap']:.4f}, {reference['nodes']} nodos, altura {reference['height']}.

```text
{reference['tree']}
```

{terminals}

Sin dominio: semilla {reference_no['seed']}, validation AP
{reference_no['validation_ap']:.4f}, {reference_no['nodes']} nodos.

Los terminales son variables transformadas; el árbol no es una ecuación directa
sobre pesos y montos crudos. Los encoders y árboles se conservan en
`private_models/` para scoring local y se verifica recarga idéntica en validación.
Estos mapas contienen categorías empresariales; se excluyen del ZIP compartible.

## Entorno y coste

Python {plan['environment']['python']}, {plan['environment']['platform']},
{plan['environment']['logical_cpus']} CPU lógicas; {plan['environment']['workers']}
procesos simultáneos. Versiones: `{json.dumps(plan['environment']['packages'])}`.
Suma de tiempos de evolución y selección de las 14 corridas: {fit_total:.1f} s.
Ese total suma tiempo por corrida con ejecución concurrente; no equivale al
tiempo de pared de la sesión ni incluye la preparación de datos. Cada JSON
también guarda `total_seconds`, y cada CSV registra evolución por generación.
Se detectaron {len(gaps)} intervalos entre generaciones de más de 60 segundos
(ver `timing_anomalies.json`). La causa no fue establecida; estos tiempos de
pared pueden incluir pausas del entorno. No se usan para afirmar ventaja de
velocidad frente a RNA ni como benchmark de CPU.

## Correcciones y comprobaciones

- La prueba inicial tenía una muestra de fitness por fila, sin muestreo por
  envío. La suite nueva evalúa todos los artículos train y no hace submuestreo.
- Se agregó elitismo y selección del árbol por validation; el prototipo elegía
  directamente el mejor fitness train.
- Se sustituyó la búsqueda aproximada de 1.000 umbrales por la solución exacta
  sobre la curva PR de validation.
- Se agregó imputación de mediana train para numéricos faltantes restantes.
- Se verifican aislamiento de etiquetas holdout, exclusión del mes actual en
  codificación, grupos, categorías desconocidas, exclusión de dominio,
  umbral F1 y recarga del modelo. Ver `tests/test_gp_full.py`.
- Se añadió `load_local_model` para cargar desde otro programa los bundles
  creados por multiprocessing en Windows; `score_raw_shipments` permite
  inferencia por artículo sin etiqueta y con el mismo encoder persistido.
- La validación independiente recargó los diez modelos finales, reprodujo
  todas las métricas validation/test con tolerancia 1e-12 y comprobó que
  permutar etiquetas test o introducir columnas de fuga no cambia los scores.

### Empates en ranking

La tabla principal usa el desempate estable del RNA para reproducir su métrica.
Un árbol simbólico puede empatar scores. El orden de extracción puede afectar
las métricas en el límite K; por ello se calculó también la esperanza exacta de
Recall@K bajo desempate aleatorio del grupo en el límite (sin cambiar selección).
Los extremos posibles y tamaños de grupos están en `ranking_tie_audit.csv`.

{chr(10).join(ties)}

`domain_purity_audit.json` registra categorías de dominio grandes y puras por
partición, sin publicar sus nombres. La ablación conserva exactamente los mismos
datos y cortes temporales, y cambia únicamente los terminales derivados de dominio.

## Limitaciones e interpretación

Entrenar con todo train aumenta la evidencia usada en fitness; no garantiza
mejor generalización. Mantener validation/test separados es necesario para
medirla. La muestra case-control y los artefactos de extracción limitan la
validez de producción; no extrapolar directamente PR-AUC, precision o ranking.

La regla de madurez usa etiquetas del dataset completo como el RNA: es una
definición retrospectiva de la cohorte. Se conserva por comparabilidad, no se
presenta como política operativa previamente validada. La codificación temporal
asume disponibilidad de etiquetas de meses anteriores; los tiempos reales de
resolución pueden invalidar ese supuesto. Se requiere reextracción homogénea y
una evaluación con disponibilidad temporal de etiquetas para despliegue.

La ablación elimina una columna sospechosa; no prueba ausencia de todos los
artefactos ni causalidad. Las categorías correlacionadas pueden preservar señal
similar. Solo hay un mes test y la selección reiterada puede sobreajustar a
validation. Una validación en meses adicionales sería una evaluación externa
distinta que requiere datos nuevos.

## Reproducir

```powershell
uv sync --frozen
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe gp_full_experiment.py --workers 2 --output results/gp_full_v1
```

Para un entrenamiento nuevo usar otro directorio de salida. La suite puede
reutilizar corridas completadas del mismo plan. `build_gp_notebook.py` genera
`notebooks/gp_shipment_loss.ipynb`, autocontenido. Su modo smoke ejecuta 30×3
con train completo; el modo full replica el plan completo. Se verificó localmente
con nbclient; no se ejecutó remotamente en Colab.

Las figuras están en `results/gp_full_v1/figures/`: evolución y complejidad,
curvas PR y Recall@K, matrices de confusión y árboles con nombres de terminales.
'''
    (root / 'docs' / 'GP_ENTRENAMIENTO_COMPLETO.md').write_text(report, encoding='utf-8')
    # Comparison plot uses aggregate metrics only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (metric, name, rna_mean, rna_std) in zip(axes, [metrics[0], metrics[1], metrics[8]]):
        values = [summary[v][metric] for v in ['with_domain', 'without_domain']]
        ax.bar(['GP con DOM', 'GP sin DOM', 'RNA'],
               [values[0]['mean'], values[1]['mean'], rna_mean],
               yerr=[values[0]['std'], values[1]['std'], rna_std],
               color=['#3483fa', '#2b9b70', '#888888'], capsize=4)
        ax.set(title=name, ylim=(0, 1))
        ax.tick_params(axis='x', labelrotation=20)
    fig.suptitle('Test por artículo: media ± DE, cinco semillas; RNA reportada')
    fig.tight_layout()
    fig.savefig(directory / 'figures' / 'rna_gp_comparison.png', dpi=180)
    plt.close(fig)
    print(root / 'docs' / 'GP_ENTRENAMIENTO_COMPLETO.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/gp_full_v1'))
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parent, args.results, args.archive)
