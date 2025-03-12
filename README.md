# 🏆 Sistema de Modelado de Ajedrez con Aprendizaje Profundo
📅 **Proyecto en Desarrollo | Febrero 2025 - Presente**  
📌 **Implementación de un modelo basado en redes neuronales para la predicción de movimientos en ajedrez.**  

---

## 📖 Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un **modelo de predicción de movimientos en ajedrez** utilizando **redes neuronales convolucionales (CNNs)**.  
Se basa en **un enfoque híbrido de aprendizaje**, combinando:  
✅ **Pre-entrenamiento con un dataset ampliado de partidas históricas.**  
✅ **Generación de partidas sintéticas con Monte Carlo Tree Search (MCTS).**  
✅ **Entrenamiento refinado con partidas evaluadas por Stockfish.**  
✅ **Optimización de arquitectura mediante análisis estadístico.**  

El modelo es capaz de **predecir el siguiente movimiento óptimo en una posición dada** y evaluar la posición del tablero.  

---

## 🚀 Metodología
### 1️⃣ 📊 Selección de la Estructura Óptima  
   - Comparación de diferentes arquitecturas con **bloques residuales**.  
   - Identificación de la mejor estructura basada en **análisis estadístico**.
<p align="center">
    <img src="imgn\model1_2.png" width="30%">
    <img src="imgn\model2.png" width="30%">
    <img src="imgn\model3_1.png" width="30%">
    <img src="imgn\model3_2.png" width="30%">
    <img src="imgn\model3_3.png" width="30%">
    <img src="imgn\model4_1.png" width="30%">
</p>  

### 2️⃣ 📚 Pre-entrenamiento con un Dataset Ampliado  
   - Se utilizaron **millones de partidas de bases de datos públicas** para mejorar la representación de estados de juego.  
   - Transformación de datos en una representación matricial para alimentar el modelo.  

### 3️⃣ 🌲 Generación de Partidas con MCTS  
   - Implementación del algoritmo **Monte Carlo Tree Search (MCTS)** para generar partidas sintéticas.  
   - Mejora del aprendizaje con exploración estratégica de posiciones críticas.  

### 4️⃣ ♟️ Entrenamiento con Partidas de Stockfish  
   - Combinación de partidas MCTS con datos de **Stockfish**, un motor de ajedrez de alto nivel.  
   - Refinamiento del modelo con partidas de alta calidad.  

### 5️⃣ 🛠 Optimización del Aprendizaje  
   - Uso de `ReduceLROnPlateau` para ajustar la tasa de aprendizaje dinámicamente.  
   - Aumento del **batch size** para mejorar la estabilidad del entrenamiento.  
   - **Dropout y Batch Normalization** para evitar sobreajuste.  

---

## 📌 Arquitectura del Modelo
🧠 **Red Neuronal Convolucional (CNN) con una única capa residual**  
🔄 **Tres cabezas de predicción:**  
   - **Predicción de la casilla de origen y destino (`from_square`, `to_square`).**  
   - **Evaluación de la posición con una regresión (`position_eval`).**  
📉 **Optimización con Adam y reducción progresiva del learning rate.**  

---


