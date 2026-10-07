# OIIP — Evaluación Volumétrica Probabilística (Monte Carlo)

Aplicación interactiva construida con **Streamlit** para estimar el aceite original en sitio (OOIP) mediante simulación Monte Carlo.

## Fórmula

`
OOIP = 7758 · A · h · φ · (1 − Sw) / Bo
`

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| A | Área de la trampa | acres |
| h | Espesor neto petrolífero | ft |
| φ | Porosidad efectiva | fracción |
| Sw | Saturación de agua connata | fracción |
| Bo | Factor volumétrico del aceite | RB/STB |

## Características

- Distribuciones triangulares y uniformes por variable
- Percentiles petroleros: **P90, P50, P10**
- Histograma de distribución del OOIP
- Análisis de sensibilidad (correlación de rango de Spearman)
- Exportación de resultados en CSV

## Instalación

`ash
pip install -r requirements.txt
`

## Uso

`ash
streamlit run app.py
`

## Convención de percentiles

- **P90** = valor excedido con 90 % de probabilidad (caso bajo / conservador)
- **P50** = caso base
- **P10** = valor excedido con 10 % de probabilidad (caso alto / optimista)
