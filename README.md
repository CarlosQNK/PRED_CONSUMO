# Predicción del consumo eléctrico de una vivienda: LSTM vs SARIMA

**Grupo 2** · Redes Neuronales y Deep Learning · Maestría en Estadística con mención en Ciencia de Datos e Inteligencia Artificial · ESPOCH (Cohorte II)

Integrantes: Carlos Cuenca, Wilson Sani, William Allauca y Gustavo Chaglla.

## Enlaces

| Recurso | Dirección |
|---|---|
| Portal web | https://carlosqnk.github.io/PRED_CONSUMO/ |
| Dashboard (R Shiny) | https://sji9ta-carlos-cuenca.shinyapps.io/PRED_CONSUMO/ |
| Datos originales (UCI) | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |

## Pregunta

¿Una red neuronal recurrente LSTM pronostica el consumo eléctrico de la próxima hora con menor error que un modelo estadístico SARIMA?

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| `index.html`, `img/`, `docs/` | Portal web publicado con GitHub Pages |
| `dashboard/app.R` | Código fuente del dashboard en R Shiny (no entrena modelos; lee los resultados) |
| `dashboard/datos/` | Base de datos limpia y resultados en CSV |
| `notebook/` | Cuaderno de Python (Google Colab) con la preparación de datos, el entrenamiento de los modelos y la exportación de resultados |

### Archivos de `dashboard/datos/`

| Archivo | Descripción |
|---|---|
| `serie_horaria.csv` | Serie horaria de energía (kWh): 17.520 horas del 26-11-2008 al 25-11-2010, con la marca de hora imputada |
| `predicciones.csv` | Valor real y pronóstico de cada método en las 2.208 horas de prueba |
| `metricas.csv` | MAE, RMSE y MASE de cada método en prueba |
| `particion.csv` | Fechas y tamaño de entrenamiento, validación y prueba |
| `historial_lstm.csv` | Pérdida por época de la LSTM |
| `sarima_parametros.csv` | Parámetros estimados de SARIMA |
| `error_por_hora.csv`, `perfil_hora.csv`, `perfil_dia.csv` | Resúmenes para los gráficos |

## Datos

Individual Household Electric Power Consumption (Hébrail y Bérard, UCI Machine Learning Repository, licencia CC BY 4.0, DOI 10.24432/C58K54): 2.075.259 mediciones por minuto de una vivienda de Francia, entre diciembre de 2006 y noviembre de 2010. El archivo original (127 MB) no se incluye; el cuaderno lo descarga desde UCI.

Preparación: la potencia activa (kW) se agregó a energía horaria (kWh); las horas con menos de 45 minutos válidos se imputaron con el valor de la misma hora de la semana anterior (368 horas, 2,10 %); se usaron los últimos 24 meses.

## Métodos

| Método | Configuración |
|---|---|
| LSTM | Ventana de 48 horas, LSTM(32) → Dense(1), 4.385 parámetros, Adam, pérdida MSE, parada temprana |
| SARIMA | (1,0,1)(0,1,1) con periodo 24, ajustado con los últimos 90 días de entrenamiento |
| Persistencia | Valor de la hora anterior |
| Estacional | Valor de la misma hora del día anterior |

Partición cronológica: 13.104 horas de entrenamiento, 2.208 de validación y 2.208 de prueba. Todos los métodos se evalúan a una hora vista sobre el mismo periodo de prueba.

## Resultados en prueba

| Modelo | MAE (kWh) | RMSE (kWh) | MASE |
|---|---|---|---|
| SARIMA | 0,3764 | 0,5355 | 0,6040 |
| LSTM | 0,3862 | 0,5367 | 0,6198 |
| Persistencia (1 h) | 0,4516 | 0,6528 | 0,7247 |
| Estacional (24 h) | 0,5773 | 0,8378 | 0,9263 |

SARIMA obtuvo el menor error medio, por un margen de 2,5 %. En las 179 horas de consumo alto, la LSTM tuvo menor error (0,911 frente a 1,045 kWh). No se evaluó la significancia estadística de la diferencia.

## Cómo ejecutar

**Dashboard (R):**

```r
install.packages(c("shiny", "shinydashboard", "plotly", "DT"))
shiny::runApp("dashboard")
```

**Modelos (Python):** abrir `notebook/Grupo2_Consumo_Electrico_LSTM_vs_SARIMA.ipynb` en Google Colab y ejecutar todas las celdas. Al final se descarga `resultados.zip`, cuyo contenido va en `dashboard/datos/`.

## Limitaciones

1. Una sola vivienda; los resultados no se generalizan a otros hogares.
2. Se usaron 24 de los 47 meses disponibles.
3. La LSTM se entrenó una sola vez, con una configuración y una semilla.
4. SARIMA tiene orden fijo, no modela la estacionalidad semanal y se ajustó con 90 días.
5. El dashboard muestra resultados ya calculados; no entrena en línea.
