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
| MAE                 | 2.66             | 5.18           |
| RMSE                | 4.15             | 7.46           |
| Falsos Positivos    | 147              | 46             |
| Falsos Negativos    | 262              | 296            |

> El modelo sencillo presentó menor error promedio, aunque generó más señales falsas. El modelo robusto fue más conservador, con menos falsas compras pero a costa de omitir muchas oportunidades de compra reales.

---

## 🤔 Reflexión Final

Como equipo, construimos paso a paso un flujo completo de Machine Learning para trading financiero con datos de NVIDIA. Este proyecto no solo reforzó conocimientos técnicos sobre series temporales y redes LSTM, sino que también nos permitió analizar los desafíos de aplicar modelos predictivos en mercados volátiles.

Uno de los mayores retos fue encontrar un balance entre la precisión del modelo y su capacidad para generar señales de compra/venta efectivas. El modelo sencillo demostró ser más reactivo, capturando tendencias pero generando más falsas alertas. En cambio, el modelo robusto, aunque redujo los falsos positivos, resultó ser demasiado conservador, dejando pasar muchas oportunidades.

Lo que más resaltó fue comprobar cómo incluso un modelo relativamente simple puede detectar patrones útiles en los precios y cómo la implementación de un backtesting permite visualizar las fortalezas y debilidades de cada estrategia. Este aprendizaje nos deja una base sólida para optimizar futuros proyectos de predicción financiera.

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
