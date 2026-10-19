# RiskShip — GP con todo el conjunto de entrenamiento

## Resultado y alcance

Se ejecutaron tres configuraciones con dos semillas cada una, se completó la
ganadora a cinco semillas (42–46) y se repitió esa misma configuración cinco
veces sin `DOM_DOMAIN_ID`: **14 entrenamientos**. El fitness de cada árbol usa
**las 125.082 filas de train**, sin submuestreo. Los resultados siguientes son
medidos; la RNA procede de las cifras publicadas en el repositorio.

| Métrica test | GP con dominio | GP sin dominio | RNA reportada |
|---|---:|---:|---:|
| PR-AUC | 0.6393 ± 0.0338 | 0.5767 ± 0.0160 | 0.7496 ± 0.0020 |
| ROC-AUC | 0.7734 ± 0.0186 | 0.7517 ± 0.0163 | 0.8241 ± 0.0020 |
| Recall | 0.6732 ± 0.0408 | 0.7218 ± 0.0426 | 0.5339 ± 0.0136 |
| Precision | 0.5506 ± 0.0369 | 0.4763 ± 0.0364 | 0.7568 ± 0.0157 |
| F1 | 0.6039 ± 0.0129 | 0.5720 ± 0.0172 | 0.6259 ± 0.0044 |
| Recall@1 % | 0.0270 ± 0.0034 | 0.0246 ± 0.0001 | 0.0321 ± 0.0000 |
| Recall@5 % | 0.1343 ± 0.0127 | 0.1200 ± 0.0019 | 0.1581 ± 0.0006 |
| Recall@10 % | 0.2595 ± 0.0172 | 0.2257 ± 0.0063 | 0.3019 ± 0.0020 |
| Recall@20 % | 0.4450 ± 0.0199 | 0.3994 ± 0.0104 | 0.5060 ± 0.0024 |
| Precision@1 % | 0.8427 ± 0.1061 | 0.7668 ± 0.0021 | 1.0000 ± 0.0000 |
| Precision@5 % | 0.8387 ± 0.0796 | 0.7493 ± 0.0121 | 0.9873 ± 0.0038 |
| Precision@10 % | 0.8099 ± 0.0537 | 0.7043 ± 0.0198 | 0.9421 ± 0.0061 |
| Precision@20 % | 0.6944 ± 0.0311 | 0.6232 ± 0.0162 | 0.7895 ± 0.0037 |

GP con dominio alcanza AP test media 0.6393,
frente a 0,5479 del smoke inicial (una semilla). Sigue por debajo de RNA
(0,7496). La ablación reduce AP media en
0.0626;
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

Se comprobó SHA-256 de 58 archivos originales contra el ZIP.
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

| Configuración | Codificación | Población | Generaciones | λ | AP validación (media 42/43) |
|---|---|---:|---:|---:|---:|
| frequency_100x15 | frequency | 100 | 15 | 0.0001 | 0.5475 |
| temporal_130x18 | temporal | 130 | 18 | 0.0001 | 0.5519 |
| temporal_180x25_regularized | temporal | 180 | 25 | 0.0003 | 0.6061 |

Configuración elegida: **temporal_180x25_regularized**. La búsqueda
con dos semillas es limitada; no demuestra optimalidad. Las cinco semillas
posteriores cuantifican variación de entrenamiento, no incertidumbre por
muestreo ni por selección de hiperparámetros.

## Modelo de referencia

Con dominio: semilla 44, elegida por validation AP
0.6818, 36 nodos, altura 8.

```text
add(add(sub(sub(add(add(x8, sub(x8, x11)), x1), abs(x1)), x11), add(x7, add(add(x15, min(sub(sub(x7, x11), x11), x7)), sub(x10, x11)))), sub(add(x8, min(x14, x15)), x11))
```

- `x0`: `freq__PICKING_TYPE`
- `x1`: `freq__SHP_LG_FACILITY_ID`
- `x2`: `freq__DOM_DOMAIN_ID`
- `x3`: `freq__ORD_CATEGORY_NAME_L1`
- `x4`: `freq__BRAND_NAME`
- `x5`: `freq__TIPO_BULKY`
- `x6`: `past_target__PICKING_TYPE`
- `x7`: `past_target__SHP_LG_FACILITY_ID`
- `x8`: `past_target__DOM_DOMAIN_ID`
- `x9`: `past_target__ORD_CATEGORY_NAME_L1`
- `x10`: `past_target__BRAND_NAME`
- `x11`: `past_target__TIPO_BULKY`
- `x12`: `PESO_TOTAL_KG`
- `x13`: `DIM_MAX_CM`
- `x14`: `VALOR_TOTAL_USD`
- `x15`: `CANTIDAD_ITEMS`
- `x16`: `RATIO_SVC_MES_ANTERIOR`
- `x17`: `ES_BULKY`
- `x18`: `day_of_week`

Sin dominio: semilla 45, validation AP
0.5865, 9 nodos.

Los terminales son variables transformadas; el árbol no es una ecuación directa
sobre pesos y montos crudos. Los encoders y árboles se conservan en
`private_models/` para scoring local y se verifica recarga idéntica en validación.
Estos mapas contienen categorías empresariales; se excluyen del ZIP compartible.

## Entorno y coste

Python 3.12.3, Windows-11-10.0.26200-SP0,
12 CPU lógicas; 2
procesos simultáneos. Versiones: `{"numpy": "2.5.3", "pandas": "2.3.3", "scikit-learn": "1.9.1", "deap": "1.4.4", "pyarrow": "24.0.0", "matplotlib": "3.11.2"}`.
Suma de tiempos de evolución y selección de las 14 corridas: 13663.8 s.
Ese total suma tiempo por corrida con ejecución concurrente; no equivale al
tiempo de pared de la sesión ni incluye la preparación de datos. Cada JSON
también guarda `total_seconds`, y cada CSV registra evolución por generación.
Se detectaron 7 intervalos entre generaciones de más de 60 segundos
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

| Dominio | K | Recall esperado con desempate aleatorio (media) |
|---|---:|---:|
| Sin | 1% | 0.0246 |
| Sin | 5% | 0.1200 |
| Sin | 10% | 0.2257 |
| Sin | 20% | 0.3995 |
| Con | 1% | 0.0270 |
| Con | 5% | 0.1343 |
| Con | 10% | 0.2595 |
| Con | 20% | 0.4449 |

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
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe gp_full_experiment.py --workers 2 --output results/gp_full_v1
```

Para un entrenamiento nuevo usar otro directorio de salida. La suite puede
reutilizar corridas completadas del mismo plan. `build_gp_notebook.py` genera
`notebooks/gp_shipment_loss.ipynb`, autocontenido. Su modo smoke ejecuta 30×3
con train completo; el modo full replica el plan completo. Se verificó localmente
con nbclient; no se ejecutó remotamente en Colab.

Las figuras están en `results/gp_full_v1/figures/`: evolución y complejidad,
curvas PR y Recall@K, matrices de confusión y árboles con nombres de terminales.
