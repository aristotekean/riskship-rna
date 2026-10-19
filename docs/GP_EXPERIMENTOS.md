# RiskShip — auditoría y smoke test de Programación Genética

Este documento registra únicamente la primera prueba pequeña de GP. El modelo
RNA original y sus artefactos no se modifican.

Registro histórico: la ampliación posterior, las cinco semillas, la ablación
sin dominio y el notebook ejecutado están documentados en
`GP_ENTRENAMIENTO_COMPLETO.md`. Los pendientes al final de este registro
describen el estado anterior a esa ampliación.

## Cambios realizados

- Se añadió `gp_shipment_loss.py` como experimento aditivo con DEAP.
- Se añadió `deap` al entorno del proyecto y se actualizó `uv.lock`.
- Se conservó `is_lost`, el filtrado de filas, la regla de madurez y la
  partición temporal del notebook RNA.
- Se verifica que no haya `SHIPMENT_ID` compartidos entre train, validation y
  test.
- Las categorías se codifican por frecuencia aprendida solamente en train;
  categorías desconocidas en validation/test reciben cero.
- Los numéricos se escalan con mediana/IQR calculados solamente en train.
- Cada individuo es un árbol DEAP real con primitivas protegidas, selección
  por torneo, crossover, mutación y límite de altura.
- El fitness es `average_precision - penalty * número_de_nodos`.
- El árbol produce un `risk_score`, no una probabilidad calibrada.
- El umbral se selecciona únicamente en validation maximizando F1.
- Se guardan métricas agregadas, historia de evolución y texto del árbol; no
  se guardan identificadores de envíos ni filas del dataset.

## Smoke test ejecutado

Comando:

```text
uv run python gp_shipment_loss.py --population 60 --generations 5 --fitness-sample 2500 --output results/gp_smoke_results.json
```

Datos y particiones reproducidas desde RNA:

| Partición | Filas | Envíos | Positivos |
|---|---:|---:|---:|
| Train | 125.082 | 112.303 | 38,96 % |
| Validation | 18.831 | 17.297 | 31,64 % |
| Test | 21.089 | 19.506 | 31,21 % |

Filas limpias: 165.002. Envíos compartidos entre particiones: 0.

Árbol ganador de esta corrida:

```text
add(sub(x1, x5), x9)
```

El mapeo de terminales es el orden de `FEATURE_COLUMNS`: `x1` corresponde a
`SHP_LG_FACILITY_ID`, `x5` a `TIPO_BULKY` y `x9` a `CANTIDAD_ITEMS` después de
la codificación/escalado numérico. Es un árbol de smoke test, no un modelo
final.

| Métrica | Validation | Test |
|---|---:|---:|
| PR-AUC | 0,5257 | 0,5479 |
| ROC-AUC | 0,7032 | 0,7245 |
| Recall | 0,6887 | 0,7174 |
| Precision | 0,4529 | 0,4588 |
| F1 | 0,5464 | 0,5597 |
| Recall@1 % | 0,0230 | 0,0243 |
| Recall@5 % | 0,1081 | 0,1164 |
| Recall@10 % | 0,2056 | 0,2176 |
| Recall@20 % | 0,3738 | 0,3821 |

Tiempo: aproximadamente 6,6 segundos en el entorno local de revisión.

## Interpretación

La prueba confirma que el pipeline GP funciona, respeta el split temporal y
produce un árbol evolucionado. No permite concluir que GP sea competitivo con
RNA: usa una población y muestra de fitness pequeñas, una sola semilla y una
codificación inicial por frecuencia. Además, los scores no son probabilidades.

## Pendiente antes de comparar contra RNA

1. Crear la versión autocontenida para Colab y documentar la carga autorizada
   del Parquet/ZIP.
2. Comparar configuraciones de población/generaciones y penalización.
3. Ejecutar cinco semillas sobre el mismo split.
4. Repetir la mejor configuración excluyendo `DOM_DOMAIN_ID`.
5. Reportar medias, desviaciones estándar, coste, complejidad y métricas de
   ranking sin usar test para selección.
