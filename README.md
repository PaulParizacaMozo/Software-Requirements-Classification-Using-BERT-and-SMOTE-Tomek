# Plan de Implementación: Clasificación de Requisitos de Software con BERT y SMOTE-Tomek

Este documento describe el plan de implementación para un proyecto de investigación y desarrollo enfocado en la clasificación automática de requisitos de software. El objetivo es construir un pipeline robusto que aborde los desafíos de la **ambigüedad del lenguaje natural** y el **desbalance de clases**, problemas prevalentes en este dominio.

## 1. Definición del Problema

En la ingeniería de requisitos de software, la clasificación automática enfrenta dos obstáculos principales:

1. **Ambigüedad del Lenguaje Natural**: Los requisitos se escriben en lenguaje humano, lo que introduce ambigüedad, sinonimia y dependencia del contexto. Esto dificulta que los modelos tradicionales interpreten y clasifiquen correctamente la intención de un requisito.

2. **Desbalance de Clases**: Los conjuntos de datos de requisitos suelen estar muy desbalanceados. Algunas categorías (ej. requisitos funcionales) son abundantes, mientras que ciertos tipos de requisitos no funcionales (ej. Portabilidad, Legal) están severamente subrepresentados. Esto provoca que los modelos tiendan a ignorar las clases minoritarias, afectando su capacidad de generalización.

Este proyecto propone una solución para mitigar estos dos problemas de manera conjunta.

## 2. Objetivos del Proyecto

### Objetivo General

El objetivo final es **desarrollar y evaluar un modelo de clasificación de requisitos de software** que utilice BERT para la generación de embeddings y SMOTE-Tomek para manejar el desbalance de clases, con el fin de mejorar significativamente la precisión y la capacidad de generalización en comparación con los enfoques tradicionales.

### Objetivos Específicos (Hitos del Plan)

- **Hito 1: Preprocesamiento de Datos**: Implementar un pipeline de preprocesamiento de texto robusto (limpieza, tokenización, lematización).
- **Hito 2: Generación de Embeddings**: Utilizar un modelo BERT preentrenado para transformar los requisitos en vectores semánticos de alta calidad.
- **Hito 3: Balanceo del Dataset**: Aplicar la técnica híbrida SMOTE-Tomek sobre los embeddings para crear un conjunto de datos de entrenamiento balanceado y limpio.
- **Hito 4: Entrenamiento de Clasificadores**: Entrenar y optimizar un modelo de clasificación supervisado (ej. SVM) con los datos procesados.
- **Hito 5: Evaluación y Comparación**: Evaluar rigurosamente el modelo propuesto y compararlo contra una línea base para validar su efectividad.

## 3. Arquitectura Propuesta

La implementación se guiará por la siguiente arquitectura de pipeline modular. Cada fase representa una etapa del plan de trabajo:

### Fase 3.1: Preprocesamiento de Datos

- **Entrada**: Texto de requisitos en bruto.
- **Proceso**: Limpieza, normalización a minúsculas, tokenización, eliminación de *stopwords* y lematización.
- **Salida**: Texto de requisitos limpio y estandarizado.

### Fase 3.2: Generación de Embeddings con BERT

- **Entrada**: Requisitos preprocesados.
- **Proceso**: Carga de un modelo Transformer preentrenado. Generación de un vector de embedding (ej. a partir del token `[CLS]`) para cada requisito.
- **Salida**: Un conjunto de embeddings de alta dimensionalidad.

### Fase 3.3: Manejo del Desbalanceo con SMOTE-Tomek

- **Entrada**: Conjunto de embeddings y sus etiquetas.
- **Proceso**: Aplicación de la estrategia de sobremuestreo sintético (SMOTE) y submuestreo por limpieza de ruido (Tomek Links).
- **Salida**: Un conjunto de datos de embeddings balanceado.

### Fase 3.4: Clasificación Supervisada

- **Entrada**: Dataset de embeddings balanceado.
- **Proceso**: Entrenamiento de un clasificador (SVM, Random Forest, etc.) para mapear los embeddings a las clases de requisitos.
- **Salida**: Un modelo de clasificación entrenado.

### Fase 3.5: Evaluación y Ajuste

- **Entrada**: Modelo entrenado y conjunto de datos de prueba (no visto y sin balancear).
- **Proceso**: Cálculo de métricas (Precisión, Recall, F1-Score) y análisis de la matriz de confusión.
- **Salida**: Reporte de rendimiento y conclusiones sobre la efectividad del modelo.

## 4. Plan de Implementación por Fases

El proyecto se ejecutará de manera incremental para asegurar la calidad y permitir una comparación justa.

- **Fase I: Establecimiento de la Línea Base (Baseline)**
    1. **Análisis Exploratorio de Datos (EDA)**: Estudiar el dataset PROMISE_exp para comprender la distribución de clases y las características del texto.
    2. **Implementación del Modelo Tradicional**: Replicar el enfoque del repositorio base: preprocesamiento estándar, extracción de características con **TF-IDF**, y entrenamiento de clasificadores clásicos (Naive Bayes, SVM).
    3. **Evaluación de la Línea Base**: Documentar el rendimiento del modelo tradicional. Este será el punto de referencia a superar.

- **Fase II: Desarrollo del Modelo Propuesto**
    1. **Implementación del Pipeline**: Codificar cada uno de los módulos de la arquitectura propuesta (Fases 3.1 a 3.5).
    2. **Experimentación y Optimización**: Probar diferentes modelos BERT, ajustar los parámetros de SMOTE-Tomek y optimizar los hiperparámetros del clasificador final.
    3. **Evaluación del Modelo Propuesto**: Medir el rendimiento del nuevo modelo en el mismo conjunto de prueba utilizado en la Fase I.

- **Fase III: Análisis Comparativo y Documentación**
    1. **Comparación de Resultados**: Analizar las métricas obtenidas por el modelo propuesto frente a la línea base.
    2. **Generación de Conclusiones**: Documentar los hallazgos, las limitaciones y las posibles líneas de trabajo futuro.
    3. **Limpieza y Finalización**: Organizar el código, refinar los notebooks y completar la documentación del repositorio.

## 5. Estructura Propuesta del Repositorio

Para mantener el proyecto organizado desde el inicio, se utilizará la siguiente estructura de directorios:

```
.
├── baseline/          # Codigo y datos replicados
├── src/               # Arquitectura nueva
├── models/            # Almacén para los modelos entrenados
├── requirements.txt   # Dependencias del proyecto
└── README.md          # Este plan de implementación
```

Este plan servirá como hoja de ruta para el desarrollo del proyecto. Los hitos y fases definidos guiarán el trabajo para alcanzar los objetivos de manera estructurada.

## 6. Configuración del Entorno de Implementación

Para ejecutar este proyecto, se requiere una configuración de software específica. El entorno está basado en **Python 3.7** y se gestionará con **Conda**.

### Paso 1: Prerrequisitos

Asegúrate de tener instalado **Anaconda** o **Miniconda** ademas de **CUDA** en tu sistema.

### Paso 2: Crear y Activar el Entorno Conda

Usaremos Conda para crear un entorno aislado llamado `myenv` con la versión 3.7 de Python.

1. Abre tu terminal
2. Crea el entorno ejecutando el siguiente comando:

    ```bash
    conda create --name myenv python=3.7 -y
    ```

3. Una vez creado, activa el entorno. Este será el espacio de trabajo para el proyecto.

    ```bash
    conda activate myenv
    ```

### Paso 3: Instalar Dependencias con Pip

Aunque estemos en un entorno Conda, usaremos `pip` y el archivo `requirements.txt` para instalar las versiones exactas de las librerías necesarias para la replicación.

Con el entorno `myenv` activado, ejecuta:

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar Modelos de NLTK

El proyecto requiere modelos específicos de la librería NLTK. Puedes descargarlos ejecutando lo siguiente en tu terminal, con el entorno `myenv` activado:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

# Resultados Replicados

Se ejecutó el código original en mi máquina local para obtener los resultados presentados en las siguientes tablas. A partir de este proceso, se generaron dos conjuntos de resultados: los resultados originales y los resultados replicados, los cuales se muestran a continuación para su comparación y análisis.

### Tabla 1: Resultados Originales

<table>
 <tr>
  <td></td>
  <td colspan="4" align="center">BoW</td>
  <td colspan="4" align="center">TF-IDF</td>
  </tr>
 <tr>
  <td></td>
  <td align="center">Precision</td>
  <td align="center">Recall</td>
  <td align="center">F1-score</td>
  <td align="center">Accuracy</td>
  <td align="center">Precision</td>
  <td align="center">Recall</td>
  <td align="center">F1-score</td>
  <td align="center">Accuracy</td>
  </tr>
 <tr>
  <td align="center">MNB</td>
  <td align="center">0.913</td>
  <td align="center">0.911</td>
  <td align="center">0.911</td>
  <td align="center">0.911</td>
  <td align="center">0.928</td>
  <td align="center">0.927</td>
  <td align="center">0.927</td>
  <td align="center">0.927</td>
  </tr>
 <tr>
  <td align="center">MNB (11-class)</td>
  <td align="center">0.774</td>
  <td align="center">0.741</td>
  <td align="center">0.737</td>
  <td align="center">0.741</td>
  <td align="center">0.740</td>
  <td align="center">0.739</td>
  <td align="center">0.722</td>
  <td align="center">0.738</td>
  </tr>
 <tr>
  <td align="center">MNB (12-class)</td>
  <td align="center">0.789</td>
  <td align="center">0.769</td>
  <td align="center">0.768</td>
  <td align="center">0.769</td>
  <td align="center">0.767</td>
  <td align="center">0.764</td>
  <td align="center">0.744</td>
  <td align="center">0.764</td>
  </tr>
</table>

### Tabla 2: Resultados Replicados

<table>
 <tr>
  <td></td>
  <td colspan="4" align="center">BoW</td>
  <td colspan="4" align="center">TF-IDF</td>
  </tr>
 <tr>
  <td></td>
  <td align="center">Precision</td>
  <td align="center">Recall</td>
  <td align="center">F1-score</td>
  <td align="center">Accuracy</td>
  <td align="center">Precision</td>
  <td align="center">Recall</td>
  <td align="center">F1-score</td>
  <td align="center">Accuracy</td>
  </tr>
 <tr>
  <td align="center">MNB</td>
  <td align="center">0.9137</td>
  <td align="center">0.9123</td>
  <td align="center">0.9121</td>
  <td align="center">0.9123</td>
  <td align="center">0.9312</td>
  <td align="center">0.9298</td>
  <td align="center">0.9299</td>
  <td align="center">0.9298</td>
  </tr>
 <tr>
  <td align="center">MNB (11-class)</td>
  <td align="center">0.7546</td>
  <td align="center">0.7295</td>
  <td align="center">0.7257</td>
  <td align="center">0.7295</td>
  <td align="center">0.7612</td>
  <td align="center">0.7372</td>
  <td align="center">0.7229</td>
  <td align="center">0.7372</td>
  </tr>
 <tr>
  <td align="center">MNB (12-class)</td>
  <td align="center">0.7922</td>
  <td align="center">0.7761</td>
  <td align="center">0.7725</td>
  <td align="center">0.7761</td>
  <td align="center">0.7573</td>
  <td align="center">0.7514</td>
  <td align="center">0.7303</td>
  <td align="center">0.7514</td>
  </tr>
</table>

### Explicación

1. **2-clases:**

   - Los resultados de **BoW** y **TF-IDF** replicados son **idénticos** o **muy cercanos** a los originales. La precisión, el recall, el F1-score y la accuracy son muy similares, con una ligera mejora en **TF-IDF** en los resultados replicados.
2. **11-clases:**

   - Aunque los resultados replicados en **BoW** y **TF-IDF** son un poco **inferiores** a los originales, aún están **relativamente cerca**. Esto se observa especialmente en la precisión y el recall. Es posible que un ajuste en los parámetros del modelo o en el preprocesamiento de los datos pueda mejorar estos resultados.
3. **12-clases:**

   - Los resultados replicados en **BoW** son **similares** a los originales, con una pequeña mejora en la precisión y el recall. Sin embargo, en **TF-IDF**, los resultados replicados muestran una ligera **disminución** en precisión y recall en comparación con los originales.

Los resultados replicados están bastante cerca de los originales, con algunas diferencias menores que podrían estar relacionadas con el preprocesamiento, el modelo o los datos específicos utilizados.

## Propuesta

El pipeline propuesto para la clasificación de requerimientos sigue una secuencia de seis pasos fundamentales, con bucles de retroalimentación para optimizar y refinar los modelos.

1. **Preprocesamiento de Datos**: Limpieza, normalización, tokenización, eliminación de stopwords y lematización de los datos para su entrada al modelo.

2. **Generación de Embeddings con BERT**: Se cargan modelos preentrenados de BERT y se ajustan al dominio para extraer embeddings contextuales. Luego, se aplican técnicas de reducción de dimensionalidad y optimización de recursos.

3. **Manejo del Desbalanceo de Clases**: Se abordan las clases desbalanceadas con técnicas como SMOTE, Tomek Links y ajuste de pesos, para generar un conjunto de datos equilibrado.

4. **Clasificación con Modelos Supervisados**: Modelos como SVM, Bosques Aleatorios y Redes Neuronales se entrenan con los embeddings balanceados para clasificar los requerimientos.

5. **Evaluación del Modelo**: Se calculan métricas como precisión, recall, F1-score y AUC, y se realiza validación cruzada para evaluar el desempeño del modelo.

6. **Ajustes y Optimización**: Refinamiento del modelo mediante la sintonización de hiperparámetros y ajustes iterativos basados en los resultados obtenidos.

Cada paso está diseñado para asegurar que el modelo sea robusto y eficiente, utilizando técnicas avanzadas de procesamiento de lenguaje natural y aprendizaje automático.

![Pipeline del modelo propuesto](resources/pipelineTesis.png)

## Resultados con la Nueva Arquitectura (BERT + SMOTE-Tomek + MLP)

#### 2-clases

| **Modelo**                                    | **Precision** | **Recall** | **F1-score** | **Accuracy** |
| --------------------------------------------- | ------------- | ---------- | ------------ | ------------ |
| BERT + SMOTE-Tomek + MLP - SVM                | 0.8683        | 0.8368     | 0.8269       | 0.8368       |
| BERT + SMOTE-Tomek + MLP - Random Forest      | 0.9808        | 0.9808     | 0.9808       | 0.9808       |
| BERT + SMOTE-Tomek + MLP - MLP  | 0.9888        | 0.9888     | 0.9888       | 0.9888       |

#### 11-clases

| **Modelo**                                    | **Precision** | **Recall** | **F1-score** | **Accuracy** |
| --------------------------------------------- | ------------- | ---------- | ------------ | ------------ |
| BERT + SMOTE-Tomek + MLP - SVM                | 0.9536        | 0.9486     | 0.9492       | 0.9486       |
| BERT + SMOTE-Tomek + MLP - Random Forest      | 0.9947        | 0.9946     | 0.9946       | 0.9946       |
| BERT + SMOTE-Tomek + MLP - MLP  | 0.9949        | 0.9946     | 0.9946       | 0.9946       |

#### 12-clases

| **Modelo**                                    | **Precision** | **Recall** | **F1-score** | **Accuracy** |
| --------------------------------------------- | ------------- | ---------- | ------------ | ------------ |
| BERT + SMOTE-Tomek + MLP - SVM                | 0.9150        | 0.9056     | 0.9068       | 0.9056       |
| BERT + SMOTE-Tomek + MLP - Random Forest      | 0.9953        | 0.9952     | 0.9952       | 0.9952       |
| BERT + SMOTE-Tomek + MLP - MLP  | 0.9826        | 0.9824     | 0.9824       | 0.9824       |

---

### Explicación de Resultados

1. **2-clases:**

   - **MLP Neural Network** muestra el mejor desempeño en todas las métricas (precision, recall, f1-score y accuracy), seguido de **Random Forest** y **SVM**.

2. **11-clases:**

   - **Random Forest** y **MLP Neural Network** presentan resultados muy cercanos en precisión, recall, f1-score y accuracy, con **Random Forest** liderando ligeramente en todas las métricas.

3. **12-clases:**

   - **Random Forest** nuevamente supera a los otros modelos, con **MLP Neural Network** siendo la siguiente opción más eficiente en precisión y recall, aunque con un desempeño ligeramente inferior.

Este formato destaca claramente que todos los modelos están basados en la arquitectura **BERT + SMOTE-Tomek + MLP**, y permite comprender que esta combinación mejora el rendimiento en tareas de clasificación multiclase.
