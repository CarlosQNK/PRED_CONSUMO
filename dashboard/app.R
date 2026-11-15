# =====================================================================
#   MAESTRIA EN ESTADISTICA con mención en Ciencia de Datos e IA (ESPOCH)
#   Asignatura: Redes Neuronales y Deep Learning — Grupo 2
# =====================================================================
#   DASHBOARD R SHINY — PREDICCIÓN DEL CONSUMO ELÉCTRICO: LSTM vs SARIMA
#   (versión interactiva)
#
#   La app NO entrena modelos: lee los resultados que exporta el
#   cuaderno de Colab. Estructura de la carpeta:
#     app_consumo/
#     ├── app.R
#     └── datos/      <- contenido de resultados.zip
#
#   INSTALACIÓN (una sola vez):
#   install.packages(c("shiny", "shinydashboard", "plotly", "DT"))
#   EJECUCIÓN:  shiny::runApp("ruta/a/app_consumo")
# =====================================================================
suppressPackageStartupMessages({
  library(shiny); library(shinydashboard); library(plotly); library(DT)
})

# ---------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------
leer <- function(archivo) {
  ruta <- file.path("datos", archivo)
  if (!file.exists(ruta)) stop(paste("Falta el archivo:", ruta,
                                     "Coloque los CSV dentro de la carpeta datos."))
  read.csv(ruta, stringsAsFactors = FALSE, check.names = FALSE,
           fileEncoding = "UTF-8")
}
met   <- leer("metricas.csv")
pred  <- leer("predicciones.csv")
histo <- leer("historial_lstm.csv")
serie <- leer("serie_horaria.csv")
part  <- leer("particion.csv")
sar   <- leer("sarima_parametros.csv")

DIAS <- c("Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom")
FMT  <- "%Y-%m-%d %H:%M:%S"

con_tiempo <- function(d) {
  fh <- as.POSIXct(d$fecha_hora, format = FMT, tz = "UTC")
  d$dia  <- as.Date(substr(d$fecha_hora, 1, 10))
  d$hora <- as.integer(substr(d$fecha_hora, 12, 13))
  d$dsem <- factor(DIAS[(as.POSIXlt(fh)$wday + 6) %% 7 + 1], levels = DIAS)
  d$mes  <- substr(d$fecha_hora, 1, 7)
  d
}
serie <- con_tiempo(serie)
pred  <- con_tiempo(pred)

# Verificación de integridad y enlace exacto de cada pronóstico a su hora.
requeridas_serie <- c("fecha_hora", "kwh", "imputado")
requeridas_pred <- c("fecha_hora", "real", "lstm", "sarima", "estacional", "persistencia")
if (!all(requeridas_serie %in% names(serie)) ||
    !all(requeridas_pred %in% names(pred))) {
  stop("Los archivos de serie o predicciones no tienen las columnas esperadas.")
}
if (anyDuplicated(serie$fecha_hora) || anyDuplicated(pred$fecha_hora)) {
  stop("Hay fechas y horas duplicadas en los CSV.")
}
idx_prueba <- match(pred$fecha_hora, serie$fecha_hora)
if (anyNA(idx_prueba)) stop("Hay pronósticos sin una hora correspondiente en la serie.")
if (any(!is.finite(pred$real)) || any(!is.finite(serie$kwh)) ||
    any(!is.finite(as.matrix(pred[, c("lstm", "sarima", "estacional", "persistencia")])))) {
  stop("Hay valores no numéricos o no finitos en los resultados.")
}
if (any(!(serie$imputado %in% c(0, 1)))) stop("La columna imputado solo puede contener 0 o 1.")
pred$imputado <- as.integer(serie$imputado[idx_prueba])

ini_val <- as.Date(substr(part$desde[2], 1, 10))
ini_pru <- as.Date(substr(part$desde[3], 1, 10))
serie$conjunto <- ifelse(serie$dia < ini_val, "Entrenamiento",
                         ifelse(serie$dia < ini_pru, "Validación", "Prueba"))

modelos <- c("LSTM" = "lstm", "SARIMA" = "sarima",
             "Estacional (24 h)" = "estacional", "Persistencia (1 h)" = "persistencia")
colores <- c("Real" = "#111827", "LSTM" = "#dc2626", "SARIMA" = "#16a34a",
             "Estacional (24 h)" = "#9ca3af", "Persistencia (1 h)" = "#f59e0b")
nombre_de <- function(k) names(modelos)[modelos == k]

fila <- function(m) met[met$modelo == m, ]
# MASE: escala del pronóstico ingenuo estacional (24 horas) SOLO en entrenamiento.
# Se recalcula desde la serie para evitar redondeos del archivo metricas.csv.
kwh_ent <- serie$kwh[serie$conjunto == "Entrenamiento"]
if (length(kwh_ent) <= 24) stop("Entrenamiento insuficiente para MASE.")
escala_mase <- mean(abs(kwh_ent[25:length(kwh_ent)] -
                             kwh_ent[1:(length(kwh_ent) - 24)]))
if (!is.finite(escala_mase) || escala_mase <= 0) stop("Escala MASE no válida.")
umbral_alto <- as.numeric(quantile(serie$kwh[serie$conjunto == "Entrenamiento" &
                                          serie$imputado == 0], 0.90, na.rm = TRUE))
n_imp_prueba <- sum(pred$imputado == 1)
mejor_epoca <- histo$epoca[which.min(histo$val_loss)]
f_ini <- min(pred$dia); f_fin <- max(pred$dia)
num <- function(x, d = 4) format(round(x, d), nsmall = d, decimal.mark = ",")
ent <- function(x) format(x, big.mark = ".", decimal.mark = ",")

metricas_de <- function(d, cols) {
  if (nrow(d) == 0) return(data.frame())
  do.call(rbind, lapply(cols, function(k) {
    e <- d$real - d[[k]]
    data.frame(Modelo = nombre_de(k), Horas = nrow(d), MAE = mean(abs(e)),
               RMSE = sqrt(mean(e^2)), MASE = mean(abs(e)) / escala_mase,
               Sesgo = mean(e), stringsAsFactors = FALSE)
  }))
}
metricas_originales <- function(d, todas = FALSE) {
  # En la prueba completa se respetan las métricas originales exportadas
  # antes del redondeo de los pronósticos CSV.
  if (todas) {
    out <- data.frame(Modelo = met$modelo, Horas = nrow(d),
                      MAE = met$MAE, RMSE = met$RMSE, MASE = met$MASE,
                      Sesgo = vapply(met$modelo, function(n) {
                        k <- unname(modelos[n]); mean(d$real - d[[k]])
                      }, numeric(1)), stringsAsFactors = FALSE)
    return(out)
  }
  metricas_de(d, unname(modelos))
}
filtrar_tipo <- function(d, tipo) {
  fs <- d$dsem %in% c("Sáb", "Dom")
  if (tipo == "Laborables") d[!fs, ] else if (tipo == "Fin de semana") d[fs, ] else d
}
eje_cat <- function(v) list(title = "", categoryorder = "array", categoryarray = v)

# ---------------------------------------------------------------------
# 2. INTERFAZ (UI)
# ---------------------------------------------------------------------
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Consumo Eléctrico"),
  dashboardSidebar(
    sidebarMenu(
      id = "menu",
      menuItem("1. Datos",                 tabName = "datos",   icon = icon("database")),
      menuItem("2. Análisis exploratorio", tabName = "eda",     icon = icon("chart-bar")),
      menuItem("3. Modelos",               tabName = "modelos", icon = icon("cogs")),
      menuItem("4. Comparación / Mejor",   tabName = "result",  icon = icon("trophy")),
      menuItem("5. Pronósticos",           tabName = "pron",    icon = icon("chart-line")),
      menuItem("6. Análisis de errores",   tabName = "errores", icon = icon("exclamation-triangle")),
      menuItem("7. Consulta por hora",     tabName = "consulta", icon = icon("search")),
      menuItem("Guía metodológica",        tabName = "guia",    icon = icon("book"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$meta(name = "viewport", content = "width=device-width, initial-scale=1"),
      tags$style(HTML(paste0(
        ".box{border-top-width:3px;border-radius:4px}",
        ".small-box h3{font-size:28px}",
        ".box-body{overflow-wrap:anywhere}",
        ".nota-metodo{background:#f3f7fa;border-left:4px solid #3c8dbc;",
        "padding:10px 13px;margin:10px 0 16px;font-size:14px}",
        ".nota-cuidado{background:#fff8eb;border-left:4px solid #f39c12;",
        "padding:10px 13px;margin:10px 0 16px;font-size:14px}"
      )))
    ),
    tabItems(

      # ---------------- 1. DATOS ----------------
      tabItem("datos",
        fluidRow(
          valueBox(ent(nrow(serie)), "Horas analizadas", icon = icon("clock"), color = "blue", width = 3),
          valueBox(ent(part$horas[1]), "Horas de entrenamiento", icon = icon("graduation-cap"), color = "light-blue", width = 3),
          valueBox(ent(part$horas[3]), "Horas de prueba", icon = icon("flask"), color = "green", width = 3),
          valueBox(sum(serie$imputado), "Horas imputadas", icon = icon("exclamation-circle"), color = "yellow", width = 3)),
        fluidRow(
          box(title = "Serie de consumo", width = 4, status = "primary", solidHeader = TRUE,
              p(strong("Individual Household Electric Power Consumption"), " (UCI). ",
                "Potencia activa por minuto agregada a energía horaria (kWh); últimos 24 meses."),
              dateRangeInput("r_serie", "Periodo", start = min(serie$dia), end = max(serie$dia),
                             min = min(serie$dia), max = max(serie$dia), language = "es", separator = " a "),
              radioButtons("agreg", "Agregación", c("Horaria", "Diaria", "Semanal"), selected = "Diaria", inline = TRUE),
              checkboxInput("ver_part", "Sombrear validación (naranja) y prueba (rojo)", TRUE),
              downloadButton("dl_serie", "Descargar serie horaria (.csv)", class = "btn-info"),
              hr(), h4("Partición cronológica"), tableOutput("t_part")),
          box(title = "Consumo de energía", width = 8, status = "primary",
              plotlyOutput("g_serie", height = "430px"))),
        fluidRow(
          box(title = "Horas imputadas por conjunto", width = 12, status = "warning",
              tableOutput("t_imputaciones"),
              div(class = "nota-cuidado",
                  "Las horas imputadas contienen valores sustituidos, no mediciones originales. ",
                  "En la evaluación se comparan ambas opciones sin volver a entrenar modelos."))),
        fluidRow(
          box(title = "Vista previa de los datos", width = 12, status = "primary", DTOutput("tabla_serie")))
      ),

      # ---------------- 2. EDA ----------------
      tabItem("eda",
        fluidRow(
          box(title = "Opciones", width = 3, status = "primary", solidHeader = TRUE,
              selectInput("eda_conj", "Datos", c("Entrenamiento", "Toda la serie")),
              selectInput("eda_grupo", "Promedio por",
                          c("Hora del día" = "hora", "Día de la semana" = "dsem", "Mes" = "mes")),
              radioButtons("eda_tipo", "Tipo de día", c("Todos", "Laborables", "Fin de semana")),
              sliderInput("bins", "Barras del histograma", 10, 80, 40, step = 5)),
          box(title = "Patrón de consumo (kWh medio por hora)", width = 9, status = "primary",
              plotlyOutput("g_perfil", height = "360px"))),
        fluidRow(
          box(title = "Mapa de calor: hora del día × día de la semana", width = 7, status = "primary",
              plotlyOutput("g_calor", height = "380px")),
          box(title = "Distribución del consumo horario", width = 5, status = "primary",
              plotlyOutput("g_hist", height = "380px"))),
        fluidRow(
          box(title = "Resumen estadístico (kWh)", width = 12, status = "info", DTOutput("tab_resumen")))
      ),

      # ---------------- 3. MODELOS ----------------
      tabItem("modelos",
        fluidRow(
          box(title = "Red LSTM", width = 6, status = "danger", solidHeader = TRUE,
              tags$ul(
                tags$li("Entrada: 48 horas anteriores; media y desviación calculadas solamente en entrenamiento."),
                tags$li("Arquitectura: LSTM(32) → Dense(1); 4.385 parámetros."),
                tags$li("Adam (lr = 0,001), pérdida MSE, lote 64."),
                tags$li("Early Stopping sobre val_loss (paciencia 5), con restauración de los mejores pesos.")),
              radioButtons("curva", "Curva", c("Pérdida (MSE)" = "loss", "MAE" = "mae"), inline = TRUE),
              plotlyOutput("g_curvas", height = "330px"),
              strong(textOutput("txt_epocas")),
              div(class = "nota-metodo", "Las curvas se calcularon en datos escalados; ",
                  "no se deben comparar sus valores directamente con el RMSE en kWh de la prueba.")),
          box(title = "SARIMA", width = 6, status = "success", solidHeader = TRUE,
              tags$ul(
                tags$li("Modelo SARIMA(1,0,1)(0,1,1) con periodo 24."),
                tags$li("Parámetros estimados una sola vez con los últimos 90 días de entrenamiento."),
                tags$li("Pronóstico a una hora vista con parámetros fijos (sin reestimar en prueba).")),
              div(class = "nota-cuidado", "La LSTM utiliza un entrenamiento histórico mucho más largo ",
                  "que los 90 días empleados para ajustar SARIMA. Son las configuraciones evaluadas; ",
                  "no una comparación general de todas las LSTM y todos los SARIMA."),
              h4("Parámetros estimados"), DTOutput("tab_sarima"),
              h4("Modelos de referencia"),
              tags$ul(
                tags$li("Estacional: valor de la misma hora del día anterior."),
                tags$li("Persistencia: valor de la hora anterior."))))
      ),

      # ---------------- 4. COMPARACIÓN ----------------
      tabItem("result",
        fluidRow(
          box(width = 4, status = "primary", title = "Criterio de comparación", solidHeader = TRUE,
              selectInput("alcance", "Horas evaluadas",
                          c("Prueba completa (2.208 h)" = "todas",
                            "Solo valores observados (sin imputar el objetivo)" = "observadas")),
              selectInput("metrica", "Métrica para ordenar", c("MAE", "RMSE", "MASE")),
              selectInput("ref", "Comparar contra", met$modelo, selected = "Persistencia (1 h)"),
              helpText("Mismo periodo de prueba y mismos pronósticos; no hay reajuste de modelos.")),
          valueBoxOutput("vb_mejor", width = 8)),
        fluidRow(
          box(title = "Resultados del subconjunto elegido", width = 6, status = "primary",
              textOutput("txt_alcance"), DTOutput("tab_metricas"),
              div(class = "nota-metodo", "MAE y RMSE se expresan en kWh; ",
                  "MASE se divide por el error estacional calculado SOLO en entrenamiento. ",
                  "MASE < 1 no implica necesariamente superar al método estacional ",
                  "en cualquier subperiodo de prueba.")),
          box(title = "Comparación gráfica", width = 6, status = "primary",
              plotlyOutput("g_met", height = "335px"))),
        fluidRow(
          box(title = "Reducción del error frente al modelo de referencia (%)", width = 12,
              status = "info", DTOutput("tab_mejora"))),
        fluidRow(
          box(title = "Análisis de sensibilidad a las horas imputadas", width = 12,
              status = "warning", solidHeader = TRUE,
              div(class = "nota-cuidado",
                  "Se excluyen de la evaluación las horas cuyo consumo real fue sustituido ",
                  "(87 de 2.208). Las predicciones NO se recalculan y sus historias anteriores ",
                  "todavía pueden contener horas imputadas. Por ello, esta es una sensibilidad ",
                  "sobre la variable objetivo, no una validación libre de toda imputación."),
              DTOutput("tab_sensibilidad"),
              p(strong("Lectura de los resultados:")), textOutput("txt_sensibilidad"))),
        fluidRow(
          box(title = "Interpretación comparativa", width = 12, status = "primary",
              uiOutput("interpretacion_comparacion")))
      ),

      # ---------------- 5. PRONÓSTICOS ----------------
      tabItem("pron",
        fluidRow(
          box(title = "Controles", width = 3, status = "primary", solidHeader = TRUE,
              dateRangeInput("rango", "Periodo (dentro de la prueba)", start = f_ini, end = min(f_ini + 2, f_fin),
                             min = f_ini, max = f_fin, language = "es", separator = " a "),
              checkboxGroupInput("mods", "Modelos", choices = modelos, selected = c("lstm", "sarima")),
              radioButtons("ver", "Mostrar", c("Valores (kWh)" = "valor", "Error (real − pronóstico)" = "error")),
              actionButton("inicio3", "Primeros 3 días", icon = icon("calendar"), class = "btn-info"),
              actionButton("todo", "Ver toda la prueba", icon = icon("expand"), class = "btn-success"),
              div(class = "nota-metodo", "Cada punto corresponde a un pronóstico a una hora. ",
                  "Mostrar las 2.208 horas a la vez dificulta apreciar los picos.")),
          box(title = "Pronóstico a una hora vista frente al valor real", width = 9, status = "primary",
              plotlyOutput("g_pred", height = "440px"))),
        fluidRow(
          box(title = "Métricas del periodo seleccionado", width = 12, status = "info",
              textOutput("txt_imput_pred"), DTOutput("tab_rango")))
      ),

      # ---------------- 6. ERRORES ----------------
      tabItem("errores",
        fluidRow(
          box(title = "Opciones", width = 3, status = "primary", solidHeader = TRUE,
              selectInput("e_grupo", "Agrupar el error por",
                          c("Hora del día" = "hora", "Día de la semana" = "dsem", "Fecha" = "dia")),
              radioButtons("e_met", "Métrica", c("MAE", "RMSE"), inline = TRUE),
              radioButtons("e_tipo", "Tipo de día", c("Todos", "Laborables", "Fin de semana")),
              sliderInput("e_horas", "Horas del día", 0, 23, c(0, 23), step = 1),
              checkboxInput("e_solo_obs", "Excluir horas imputadas del objetivo", TRUE),
              selectInput("e_modelo", "Modelo para el detalle", choices = modelos)),
          box(title = "¿Cuándo se equivocan más los modelos?", width = 9, status = "primary",
              plotlyOutput("g_err", height = "400px"))),
        fluidRow(
          box(title = "Pronóstico frente a valor real", width = 6, status = "primary",
              plotlyOutput("g_disp", height = "380px")),
          box(title = "Distribución de los errores (real − pronóstico)", width = 6, status = "primary",
              plotlyOutput("g_resid", height = "380px"))),
        fluidRow(
          box(title = "Consumos altos: ¿qué pasa con los picos?", width = 12, status = "warning",
              div(class = "nota-metodo", textOutput("txt_picos")),
              DTOutput("tab_picos"),
              p("Sesgo = consumo real − pronóstico. Positivo: el modelo subestima. ",
                "Este análisis es descriptivo; no demuestra superioridad estadística.")))
      ),

      # ---------------- 7. CONSULTA ----------------
      tabItem("consulta",
        fluidRow(
          box(title = "Seleccione una hora del periodo de prueba", width = 3, status = "primary", solidHeader = TRUE,
              dateInput("c_fecha", "Fecha", value = f_ini + 3, min = f_ini, max = f_fin, language = "es"),
              sliderInput("c_hora", "Hora", 0, 23, 19, step = 1),
              helpText("Cada modelo pronosticó esa hora con los datos disponibles hasta la hora anterior.")),
          column(9,
            fluidRow(valueBoxOutput("vb_real", width = 4), valueBoxOutput("vb_lstm", width = 4),
                     valueBoxOutput("vb_sarima", width = 4)),
            fluidRow(valueBoxOutput("vb_est", width = 4), valueBoxOutput("vb_per", width = 4),
                     valueBoxOutput("vb_gana", width = 4)))),
        fluidRow(
          box(width = 12, status = "warning", uiOutput("aviso_punto"))),
        fluidRow(
          box(title = "Pronósticos del día seleccionado", width = 12, status = "primary",
              plotlyOutput("g_dia_sel", height = "380px")))
      ),

      # ---------------- GUÍA ----------------
      tabItem("guia",
        fluidRow(
          box(title = "Pregunta y procedimiento", width = 6, status = "primary", solidHeader = TRUE,
              p(strong("Pregunta: "), "¿una LSTM mejora el pronóstico horario de un SARIMA ",
                "en la serie estudiada?"),
              tags$ol(
                tags$li("Datos UCI: Global_active_power en kW. Serie horaria en kWh; ",
                        "se usa el promedio de potencia por hora y se requieren al menos 45 minutos observados."),
                tags$li("Horas sin información suficiente: imputación con la misma hora de ",
                        "la semana anterior; casos restantes con arrastre hacia adelante o atrás."),
                tags$li("Últimos 24 meses: división cronológica en entrenamiento, validación y prueba."),
                tags$li("LSTM: 48 horas previas, 32 unidades; escalado exclusivamente con entrenamiento."),
                tags$li("SARIMA(1,0,1)(0,1,1)[24]: estimado con los últimos 90 días de entrenamiento."),
                tags$li("Dos referencias: persistencia de una hora y valor de 24 horas atrás."),
                tags$li("Evaluación de 2.208 pronósticos a una hora vista: MAE, RMSE y MASE.")),
              p("MASE: error absoluto dividido por el MAE de la referencia estacional ",
                "de 24 horas, calculado en entrenamiento.")),
          box(title = "Limitaciones de la comparación", width = 6, status = "warning", solidHeader = TRUE,
              tags$ol(
                tags$li("Se utilizaron 24 de los aproximadamente 47 meses disponibles."),
                tags$li("LSTM: un solo entrenamiento, una semilla y una configuración; ",
                        "no se compararon varias arquitecturas ni se repitió el ajuste."),
                tags$li("SARIMA: orden fijo, estacionalidad diaria y solo 90 días ",
                        "para estimación, a diferencia del mayor historial de LSTM."),
                tags$li("Las 87 horas de prueba imputadas modifican ligeramente ",
                        "el orden de RMSE entre SARIMA y LSTM."),
                tags$li("Excluir objetivos imputados no elimina posibles valores imputados ",
                        "en el historial de entrada."),
                tags$li("Única vivienda francesa. No se generalizan estos resultados ",
                        "a otros hogares ni a otros horizontes de pronóstico."),
                tags$li("No se demostró superioridad estadística entre SARIMA y LSTM; ",
                        "son errores observados en una única evaluación."),
                tags$li("Shiny presenta predicciones almacenadas; no entrena los modelos en línea.")),
              p("Fuente: Hébrail, G. y Bérard, A. Individual Household Electric Power Consumption. ",
                "UCI Machine Learning Repository. Licencia CC BY 4.0."))),
        fluidRow(
          box(title = "Conclusión basada en los datos exportados", width = 12, status = "info",
              textOutput("txt_conclusion"),
              div(class = "nota-metodo", "La aplicación utiliza únicamente los CSV del proyecto: ",
                  "no introduce simulaciones nuevas ni cambia las predicciones.")))
      )
    )
  )
)

# ---------------------------------------------------------------------
# 3. SERVIDOR
# ---------------------------------------------------------------------
server <- function(input, output, session) {

  # ===== 1. DATOS =====
  serie_sel <- reactive({
    req(input$r_serie[1], input$r_serie[2])
    d <- serie[serie$dia >= input$r_serie[1] & serie$dia <= input$r_serie[2], ]
    validate(need(nrow(d) > 0, "No hay datos en el periodo elegido."))
    d
  })

  output$g_serie <- renderPlotly({
    d <- serie_sel()
    if (input$agreg == "Horaria") {
      x <- d$fecha_hora; y <- d$kwh; yl <- "kWh por hora"
    } else if (input$agreg == "Diaria") {
      a <- aggregate(kwh ~ dia, data = d, FUN = sum)
      x <- as.character(a$dia); y <- a$kwh; yl <- "kWh por día"
    } else {
      d$sem <- as.character(cut(d$dia, "week"))
      a <- aggregate(kwh ~ sem, data = d, FUN = sum)
      x <- a$sem; y <- a$kwh; yl <- "kWh por semana"
    }
    formas <- list()
    if (isTRUE(input$ver_part)) {
      rect <- function(a, b, col) list(type = "rect", xref = "x", yref = "paper", x0 = as.character(a),
                                       x1 = as.character(b), y0 = 0, y1 = 1, fillcolor = col,
                                       opacity = 0.15, line = list(width = 0), layer = "below")
      formas <- list(rect(ini_val, ini_pru, "orange"), rect(ini_pru, max(serie$dia) + 1, "red"))
    }
    plot_ly(x = x, y = y, type = "scatter", mode = "lines",
            line = list(color = "#1d4ed8", width = 1)) %>%
      layout(shapes = formas, xaxis = list(title = "", type = "date"), yaxis = list(title = yl))
  })

  output$t_part <- renderTable({
    # Fechas compactas: evitan el salto de columna en pantallas estrechas.
    data.frame(Conjunto = part$conjunto,
               Desde = substr(part$desde, 1, 10),
               Hasta = substr(part$hasta, 1, 10),
               Horas = format(as.integer(part$horas), big.mark = ".", trim = TRUE),
               check.names = FALSE)
  }, striped = TRUE, spacing = "xs")
  output$t_imputaciones <- renderTable({
    data.frame(Conjunto = c("Entrenamiento", "Validación", "Prueba", "TOTAL"),
      Horas = c(as.integer(part$horas), nrow(serie)),
      `Horas imputadas` = c(vapply(c("Entrenamiento", "Validación", "Prueba"),
                             function(g) as.integer(sum(serie$imputado[serie$conjunto == g] == 1)), integer(1)),
                           sum(serie$imputado == 1)), check.names = FALSE)
  }, striped = TRUE)

  output$tabla_serie <- renderDT({
    d <- serie_sel()[, c("fecha_hora", "kwh", "imputado", "conjunto", "dsem")]
    names(d) <- c("Fecha y hora", "kWh", "Imputado (1 = sí)", "Conjunto", "Día")
    datatable(d, options = list(scrollX = TRUE, pageLength = 8), filter = "top", rownames = FALSE)
  })

  output$dl_serie <- downloadHandler(
    filename = function() "serie_horaria_kwh.csv",
    content  = function(file) write.csv(serie[, c("fecha_hora", "kwh", "imputado", "conjunto")], file, row.names = FALSE))

  # ===== 2. EDA =====
  eda_d <- reactive({
    d <- if (input$eda_conj == "Entrenamiento") serie[serie$conjunto == "Entrenamiento", ] else serie
    d <- filtrar_tipo(d, input$eda_tipo)
    validate(need(nrow(d) > 0, "No hay datos con esos filtros."))
    d
  })

  output$g_perfil <- renderPlotly({
    d <- eda_d()
    a <- aggregate(d$kwh, list(g = d[[input$eda_grupo]]), mean)
    a <- a[!is.na(a$x), ]
    etiq <- as.character(a$g)
    plot_ly(x = etiq, y = a$x, type = "bar", marker = list(color = "#3c8dbc")) %>%
      layout(xaxis = eje_cat(etiq), yaxis = list(title = "kWh medio por hora"))
  })

  output$g_calor <- renderPlotly({
    d <- eda_d()
    m <- tapply(d$kwh, list(d$dsem, d$hora), mean)
    plot_ly(x = as.integer(colnames(m)), y = rownames(m), z = m, type = "heatmap", colorscale = "Blues",
            reversescale = TRUE, colorbar = list(title = "kWh")) %>%
      layout(xaxis = list(title = "Hora del día", dtick = 2),
             yaxis = list(title = "", categoryorder = "array", categoryarray = rev(DIAS)))
  })

  output$g_hist <- renderPlotly({
    plot_ly(x = eda_d()$kwh, type = "histogram", nbinsx = input$bins, marker = list(color = "#3c8dbc")) %>%
      layout(xaxis = list(title = "kWh por hora"), yaxis = list(title = "Frecuencia"))
  })

  output$tab_resumen <- renderDT({
    x <- eda_d()$kwh
    q <- unname(quantile(x, c(0.25, 0.5, 0.75)))
    df <- data.frame(N = length(x), Media = mean(x), DE = sd(x), Min = min(x), Q1 = q[1],
                     Mediana = q[2], Q3 = q[3], Max = max(x))
    datatable(df, options = list(dom = "t"), rownames = FALSE) %>% formatRound(2:8, 3)
  })

  # ===== 3. MODELOS =====
  output$g_curvas <- renderPlotly({
    ent_col <- if (input$curva == "loss") "loss" else "mae"
    val_col <- paste0("val_", ent_col)
    plot_ly() %>%
      add_trace(x = histo$epoca, y = histo[[ent_col]], type = "scatter", mode = "lines+markers",
                name = "Entrenamiento", line = list(color = "#dc2626"), marker = list(color = "#dc2626")) %>%
      add_trace(x = histo$epoca, y = histo[[val_col]], type = "scatter", mode = "lines+markers",
                name = "Validación", line = list(color = "#0891b2"), marker = list(color = "#0891b2")) %>%
      layout(shapes = list(list(type = "line", xref = "x", yref = "paper", x0 = mejor_epoca, x1 = mejor_epoca,
                                y0 = 0, y1 = 1, line = list(dash = "dash", color = "gray"))),
             xaxis = list(title = "Época"), yaxis = list(title = "Datos escalados"),
             legend = list(orientation = "h", y = -0.2))
  })

  output$txt_epocas <- renderText({
    sprintf("Épocas ejecutadas: %d. Mejor época (menor val_loss, línea punteada): %d.", nrow(histo), mejor_epoca)
  })

  output$tab_sarima <- renderDT({
    df <- sar; names(df) <- c("Parámetro", "Valor")
    datatable(df, options = list(dom = "t"), rownames = FALSE) %>% formatRound(2, 4)
  })

  # ===== 4. COMPARACIÓN =====
  datos_evaluacion <- reactive({
    if (input$alcance == "observadas") pred[pred$imputado == 0, ] else pred
  })
  resumen_evaluacion <- reactive({
    metricas_originales(datos_evaluacion(), todas = input$alcance == "todas")
  })
  output$txt_alcance <- renderText({
    d <- datos_evaluacion()
    sprintf("%s horas evaluadas; %s horas imputadas excluidas de este cálculo.",
            ent(nrow(d)), ent(n_imp_prueba - sum(d$imputado)))
  })
  output$vb_mejor <- renderValueBox({
    m <- resumen_evaluacion()
    i <- which.min(m[[input$metrica]])
    valueBox(paste("Menor error observado:", m$Modelo[i]),
             paste(input$metrica, "=", num(m[[input$metrica]][i]),
                   "| No implica superioridad estadística"),
             icon = icon("balance-scale"), color = "purple")
  })
  output$tab_metricas <- renderDT({
    m <- resumen_evaluacion()
    df <- m[, c("Modelo", "Horas", "MAE", "RMSE", "MASE", "Sesgo")]
    datatable(df[order(df[[input$metrica]]), ],
              options = list(dom = "t", scrollX = TRUE), rownames = FALSE) %>%
      formatRound(3:6, 4)
  })
  output$g_met <- renderPlotly({
    m <- resumen_evaluacion(); v <- m[[input$metrica]]
    plot_ly(x = m$Modelo, y = v, type = "bar",
            marker = list(color = unname(colores[m$Modelo])),
            text = sprintf("%.4f", v), textposition = "outside") %>%
      layout(xaxis = eje_cat(m$Modelo),
             yaxis = list(title = if (input$metrica == "MASE") "MASE" else "kWh",
                          range = c(0, max(v) * 1.18)))
  })
  output$tab_mejora <- renderDT({
    m <- resumen_evaluacion()
    r <- m[m$Modelo == input$ref, ]
    req(nrow(r) == 1)
    df <- data.frame(Modelo = m$Modelo,
      `Reducción del MAE (%)` = 100 * (r$MAE - m$MAE) / r$MAE,
      `Reducción del RMSE (%)` = 100 * (r$RMSE - m$RMSE) / r$RMSE,
      check.names = FALSE)
    df <- df[df$Modelo != input$ref, ]
    datatable(df, options = list(dom = "t"), rownames = FALSE) %>% formatRound(2:3, 1)
  })
  sensibilidad <- reactive({
    a <- metricas_originales(pred, todas = TRUE)
    b <- metricas_originales(pred[pred$imputado == 0, ], todas = FALSE)
    merge(a[, c("Modelo", "MAE", "RMSE")], b[, c("Modelo", "MAE", "RMSE")],
          by = "Modelo", suffixes = c("_total", "_observadas"), sort = FALSE)
  })
  output$tab_sensibilidad <- renderDT({
    d <- sensibilidad()
    names(d) <- c("Modelo", "MAE prueba completa", "RMSE prueba completa",
                  "MAE solo observadas", "RMSE solo observadas")
    datatable(d, rownames = FALSE, options = list(dom = "t", scrollX = TRUE)) %>%
      formatRound(2:5, 4)
  })
  output$txt_sensibilidad <- renderText({
    d <- sensibilidad()
    a <- d[d$Modelo == "SARIMA", ]; b <- d[d$Modelo == "LSTM", ]
    sprintf(paste0(
      "En la prueba completa SARIMA tiene menor MAE (%.4f frente a %.4f) y ",
      "menor RMSE (%.4f frente a %.4f). Tras excluir las %d horas imputadas, ",
      "SARIMA conserva el menor MAE (%.4f frente a %.4f), pero LSTM ",
      "presenta un RMSE apenas menor (%.4f frente a %.4f). ",
      "El cambio de orden es muy pequeño: no establece superioridad estadística."),
      a$MAE_total, b$MAE_total, a$RMSE_total, b$RMSE_total, n_imp_prueba,
      a$MAE_observadas, b$MAE_observadas, b$RMSE_observadas, a$RMSE_observadas)
  })
  output$interpretacion_comparacion <- renderUI({
    m <- resumen_evaluacion()
    ma <- m$MAE[m$Modelo == "SARIMA"]; ml <- m$MAE[m$Modelo == "LSTM"]
    tags$div(class = "nota-metodo",
      sprintf("En este subconjunto, SARIMA registra un MAE de %.4f kWh y LSTM de %.4f kWh. ",
              ma, ml),
      sprintf("La diferencia absoluta es de %.4f kWh (%.2f %% respecto a LSTM). ",
              abs(ml-ma), 100 * abs(ml-ma) / ml),
      "Una diferencia numérica no demuestra que un método sea mejor en otros ",
      "periodos o viviendas. Para sostener superioridad estadística se requeriría ",
      "un diseño de comparación e incertidumbre apropiados para series temporales.")
  })

  # ===== 5. PRONÓSTICOS =====
  observeEvent(input$todo, updateDateRangeInput(session, "rango", start = f_ini, end = f_fin))
  observeEvent(input$inicio3, updateDateRangeInput(session, "rango",
                                                   start = f_ini, end = min(f_ini + 2, f_fin)))

  sel <- reactive({
    req(input$rango[1], input$rango[2])
    d <- pred[pred$dia >= input$rango[1] & pred$dia <= input$rango[2], ]
    validate(need(nrow(d) > 0, "No hay datos en el periodo elegido."))
    d
  })

  output$g_pred <- renderPlotly({
    d <- sel()
    p <- plot_ly()
    if (input$ver == "valor")
      p <- add_trace(p, x = d$fecha_hora, y = d$real, type = "scatter", mode = "lines", name = "Real",
                     line = list(color = colores[["Real"]], width = 2.5))
    for (k in input$mods) {
      n <- nombre_de(k)
      y <- if (input$ver == "valor") d[[k]] else d$real - d[[k]]
      p <- add_trace(p, x = d$fecha_hora, y = y, type = "scatter", mode = "lines", name = n,
                     line = list(color = colores[[n]], width = 1.3))
    }
    p %>% layout(xaxis = list(title = "", type = "date"),
                 yaxis = list(title = if (input$ver == "valor") "kWh" else "Error (kWh)"),
                 legend = list(orientation = "h", y = -0.15), hovermode = "x unified")
  })

  output$txt_imput_pred <- renderText({
    d <- sel()
    sprintf("Se visualizan %s pronósticos; %s corresponden a horas con consumo imputado. ",
            ent(nrow(d)), ent(sum(d$imputado == 1)))
  })
  output$tab_rango <- renderDT({
    validate(need(length(input$mods) > 0, "Seleccione al menos un modelo."))
    df <- metricas_de(sel(), input$mods)
    datatable(df[order(df$MAE), ], options = list(dom = "t"), rownames = FALSE) %>%
      formatRound(3:6, 4)
  })

  # ===== 6. ERRORES =====
  err_d <- reactive({
    d <- filtrar_tipo(pred, input$e_tipo)
    if (isTRUE(input$e_solo_obs)) d <- d[d$imputado == 0, ]
    d <- d[d$hora >= input$e_horas[1] & d$hora <= input$e_horas[2], ]
    validate(need(nrow(d) > 0, "No hay datos con esos filtros."))
    d
  })

  output$g_err <- renderPlotly({
    d <- err_d()
    g <- d[[input$e_grupo]]
    g <- if (is.factor(g)) droplevels(g) else g
    f <- if (input$e_met == "MAE") function(e) mean(abs(e)) else function(e) sqrt(mean(e^2))
    p <- plot_ly(); etiq <- NULL
    for (k in unname(modelos)) {
      v <- tapply(d$real - d[[k]], g, f)
      etiq <- names(v); n <- nombre_de(k)
      p <- add_trace(p, x = etiq, y = as.numeric(v), type = "scatter", mode = "lines+markers", name = n,
                     line = list(color = colores[[n]]), marker = list(color = colores[[n]], size = 5))
    }
    p %>% layout(xaxis = eje_cat(etiq), yaxis = list(title = paste(input$e_met, "(kWh)")),
                 legend = list(orientation = "h", y = -0.2), hovermode = "x unified")
  })

  output$g_disp <- renderPlotly({
    d <- err_d(); k <- input$e_modelo; n <- nombre_de(k)
    mx <- max(c(d$real, d[[k]]))
    plot_ly() %>%
      add_trace(x = d$real, y = d[[k]], type = "scatter", mode = "markers", name = n,
                marker = list(color = colores[[n]], size = 4, opacity = 0.35)) %>%
      add_trace(x = c(0, mx), y = c(0, mx), type = "scatter", mode = "lines", name = "Pronóstico perfecto",
                line = list(dash = "dash", color = "gray")) %>%
      layout(xaxis = list(title = "Valor real (kWh)"), yaxis = list(title = paste("Pronóstico", n, "(kWh)")),
             legend = list(orientation = "h", y = -0.2))
  })

  output$g_resid <- renderPlotly({
    d <- err_d(); k <- input$e_modelo; n <- nombre_de(k)
    plot_ly(x = d$real - d[[k]], type = "histogram", nbinsx = 50, marker = list(color = colores[[n]])) %>%
      layout(xaxis = list(title = paste("Error de", n, "(kWh)")), yaxis = list(title = "Frecuencia"))
  })

  # Evaluación descriptiva de los picos reales, con umbral definido en entrenamiento.
  picos <- reactive({
    pred[pred$imputado == 0 & pred$real >= umbral_alto, ]
  })
  output$txt_picos <- renderText({
    sprintf(paste0(
      "Se consideran consumos altos los valores >= %.3f kWh, percentil 90 ",
      "del entrenamiento observado. En prueba hay %s horas no imputadas ",
      "que superan ese umbral. No se eligió el umbral a partir de la prueba."),
      umbral_alto, ent(nrow(picos())))
  })
  output$tab_picos <- renderDT({
    df <- metricas_de(picos(), unname(modelos))
    df <- df[, c("Modelo", "Horas", "MAE", "RMSE", "Sesgo")]
    datatable(df[order(df$MAE), ], options = list(dom = "t"), rownames = FALSE) %>%
      formatRound(3:5, 4)
  })

  # ===== 7. CONSULTA =====
  punto <- reactive({
    req(input$c_fecha)
    d <- pred[pred$dia == input$c_fecha & pred$hora == input$c_hora, ]
    validate(need(nrow(d) == 1, "Seleccione una fecha dentro del periodo de prueba."))
    d
  })

  caja <- function(k, color, icono) {
    renderValueBox({
      d <- punto()
      valueBox(paste(num(d[[k]], 3), "kWh"),
               paste0(nombre_de(k), " · error: ", num(abs(d$real - d[[k]]), 3), " kWh"),
               icon = icon(icono), color = color)
    })
  }
  output$vb_real <- renderValueBox(valueBox(paste(num(punto()$real, 3), "kWh"), "Consumo real",
                                            icon = icon("bolt"), color = "navy"))
  output$vb_lstm   <- caja("lstm", "red", "project-diagram")
  output$vb_sarima <- caja("sarima", "green", "chart-line")
  output$vb_est    <- caja("estacional", "light-blue", "calendar")
  output$vb_per    <- caja("persistencia", "yellow", "clock")
  output$vb_gana <- renderValueBox({
    d <- punto()
    e <- sapply(unname(modelos), function(k) abs(d$real - d[[k]]))
    valueBox(nombre_de(unname(modelos)[which.min(e)]), "Pronóstico más cercano en esta hora",
             icon = icon("trophy"), color = "purple")
  })

  output$aviso_punto <- renderUI({
    d <- punto()
    if (d$imputado == 1) {
      div(class = "nota-cuidado", strong("Atención: "),
          "el valor real de esta hora fue imputado; su error de pronóstico ",
          "no se ha contrastado con una medición original.")
    } else {
      div(class = "nota-metodo", "El valor real de esta hora procede de un ",
          "registro observado. La mejor predicción de una sola hora no ",
          "determina el mejor modelo en todo el periodo.")
    }
  })
  output$txt_conclusion <- renderText({
    a <- metricas_originales(pred, todas = TRUE)
    s <- a[a$Modelo == "SARIMA", ]; l <- a[a$Modelo == "LSTM", ]
    sprintf(paste0("En las 2.208 horas de prueba, SARIMA obtuvo MAE %.4f kWh ",
      "y RMSE %.4f kWh; LSTM, MAE %.4f y RMSE %.4f kWh. ",
      "La ventaja es pequeña y depende de si se incluyen horas imputadas. ",
      "Ambos redujeron el error medio frente a los métodos de referencia. ",
      "No se puede concluir que uno sea universalmente superior."),
      s$MAE, s$RMSE, l$MAE, l$RMSE)
  })

  output$g_dia_sel <- renderPlotly({
    req(input$c_fecha)
    d <- pred[pred$dia == input$c_fecha, ]
    validate(need(nrow(d) > 0, "Seleccione una fecha dentro del periodo de prueba."))
    p <- plot_ly() %>%
      add_trace(x = d$hora, y = d$real, type = "scatter", mode = "lines+markers", name = "Real",
                line = list(color = colores[["Real"]], width = 2.5), marker = list(color = colores[["Real"]]))
    for (k in unname(modelos)) {
      n <- nombre_de(k)
      p <- add_trace(p, x = d$hora, y = d[[k]], type = "scatter", mode = "lines", name = n,
                     line = list(color = colores[[n]], width = 1.3))
    }
    p %>% layout(shapes = list(list(type = "line", xref = "x", yref = "paper", x0 = input$c_hora,
                                    x1 = input$c_hora, y0 = 0, y1 = 1, line = list(dash = "dot", color = "purple"))),
                 xaxis = list(title = "Hora del día", dtick = 1), yaxis = list(title = "kWh"),
                 legend = list(orientation = "h", y = -0.2), hovermode = "x unified")
  })
}

shinyApp(ui, server)
