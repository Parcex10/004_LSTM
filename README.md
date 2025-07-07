# 004_LSTM_Trading

#### Autores: Adrián Herrera, Patrick F. Bárcena y Carlos Moreno

# 📈 Estrategia de Trading con LSTM para NVIDIA (NVDA)

Este proyecto implementa una red neuronal LSTM para analizar series temporales financieras y predecir precios de cierre de NVIDIA (NVDA). La solución integra un flujo completo desde la preparación de datos, construcción y entrenamiento de modelos, hasta la simulación de estrategias de trading y comparación con un enfoque Buy & Hold.

---

## 🎯 Objetivo

Desarrollar un modelo capaz de aprender patrones en los precios históricos de NVIDIA y predecir movimientos futuros. A partir de estas predicciones, se construyó una estrategia de trading activa para evaluar su desempeño frente a mantener la acción a largo plazo (*Buy & Hold*).

---

## 🧪 Resumen del Experimento

Se entrenaron dos arquitecturas de redes neuronales:

- **Modelo Sencillo:** Una sola capa LSTM con Dropout para regularización.
- **Modelo Robusto:** Dos capas LSTM en cascada con Dropout para mayor capacidad de aprendizaje.

Ambos modelos fueron evaluados mediante métricas de regresión y su desempeño como estrategias de trading fue probado en un backtesting comparativo.

---

## 🧠 Técnicas Usadas

- Normalización de datos con MinMaxScaler.
- Creación de secuencias para series temporales (look-back window).
- Arquitecturas LSTM con Keras (TensorFlow backend).
- Backtesting para comparar las estrategias LSTM vs Buy & Hold.
- Guardado de modelos y scaler para predicciones futuras.

---

## ⚙️ Tecnologías Usadas

- Python 3.x
- Pandas / NumPy / Matplotlib
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebooks

---

## 🗂️ Estructura del Proyecto

004_LSTM_Trading/
├── data/               # Dataset original NVDA_10yr.csv
├── models/             # Modelos entrenados y scaler
│   ├── lstm_model_simple.h5
│   ├── lstm_model_robust.h5
│   └── close_price_scaler.save
├── notebooks/          # Notebook principal
│   └── report.ipynb
├── utils/              # Clase TradingLSTM
│   └── trading_lstm.py
├── main.py             # Script de entrada
├── README.md           # Este documento
└── requirements.txt    # Dependencias

---

## 📊 Resultados

| Métrica             | Modelo Sencillo | Modelo Robusto |
|---------------------|------------------|----------------|
| MAE                 | 5.12             | 4.78           |
| RMSE                | 6.84             | 6.21           |
| Falsos Positivos    | 12               | 8              |
| Falsos Negativos    | 7                | 5              |

> El modelo robusto logró un mejor desempeño tanto en las métricas de error como en el backtesting, mostrando una curva de capital más estable y menos señales falsas.

---

## 🤔 Reflexión Final

Como equipo, tuvimos la oportunidad de construir paso a paso un flujo completo de machine learning para trading financiero con datos de NVIDIA. El proyecto nos permitió reforzar conceptos de series temporales, redes neuronales LSTM y análisis financiero aplicado.

Uno de los mayores retos fue ajustar la arquitectura LSTM para evitar el sobreajuste en un mercado tan volátil. El uso de técnicas como Dropout y el diseño de un backtesting sencillo fueron claves para evaluar la viabilidad de la estrategia. 

Lo que más resaltó fue ver cómo un modelo bien entrenado puede adaptarse a tendencias reales y cómo, incluso frente a Buy & Hold, una estrategia basada en predicciones puede capturar movimientos importantes si se configura adecuadamente.

---

## 🚀 Cómo ejecutar

1️⃣ Clona el repositorio:  
```bash
git clone https://github.com/Parcex10/004_LSTM_Trading.git
cd 004_LSTM_Trading

2️⃣ Instala dependencias:

pip install -r requirements.txt
3️⃣ Ejecuta el notebook principal:

jupyter notebook notebooks/report.ipynb
4️⃣ (Opcional) Corre el flujo desde main.py:

python main.py
