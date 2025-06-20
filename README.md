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

# Resultados

En esta sección se presentan los detalles del entorno de ejecución, los datos utilizados, y los resultados experimentales obtenidos con el modelo propuesto para la clasificación de requisitos de software.

## Entorno de Ejecución

El modelo fue entrenado y evaluado en un entorno de ejecución con las siguientes especificaciones de hardware y software:

**Hardware:**

- GPU: NVIDIA Quadro T1000.
- Procesador: Intel(R) Core(TM) i7-9850H (12) @ 4.60 GHz.
- Memoria RAM: 32 GB.

**Software:**

- **Sistema Operativo:** Arch Linux.
- **Python:** 3.8 (usando Conda).
- **Librerías principales:**
  - `nltk==3.5`
  - `scikit-learn==0.24.0`
  - `gensim==3.8.3`
  - `catboost==0.26`
  - `tensorflow==2.4.0`
  - `PyTorch`
  - `transformers`

El código fuente completo se encuentra en el siguiente repositorio de GitHub, con el fin de garantizar la reproducibilidad de los resultados:

[https://github.com/PaulParizacaMozo/Software-Requirements-Classification-Using-BERT-and-SMOTE-Tomek](https://github.com/PaulParizacaMozo/Software-Requirements-Classification-Using-BERT-and-SMOTE-Tomek)

## Datos

El conjunto de datos original tera-PROMISE consta de 625 requisitos etiquetados, con 255 requisitos funcionales y 370 no funcionales. Para este trabajo, se utilizó el conjunto de datos ampliado **PROMISE_exp**, que contiene 444 requisitos funcionales y 525 no funcionales. Un 10% de este conjunto de datos se separó para pruebas, mientras que el resto fue procesado utilizando la técnica SMOTE-Tomek para abordar el desbalanceo de clases. Posteriormente, el modelo CatBoost fue utilizado para la clasificación.

**Número de requisitos por tipo en PROMISE_exp**

| Tipo de Requisito | Cantidad |
| :--- | :--- |
| **Requisito Funcional (F)** | **444** |
| Disponibilidad (A) | 31 |
| Legal (L) | 15 |
| Apariencia (LF) | 49 |
| Mantenibilidad (MN) | 24 |
| Operatividad (O) | 77 |
| Rendimiento (PE) | 67 |
| Escalabilidad (SC) | 22 |
| Seguridad (SE) | 125 |
| Usabilidad (US) | 85 |
| Tolerancia a Fallos (FT) | 18 |
| Portabilidad (PO) | 12 |
| **Total** | **969** |

A pesar de la expansión, el conjunto sigue estando desbalanceado, con la clase de portabilidad siendo la menos representada.

**Distribución de clases en PROMISE_exp**
![Distribución de clases en PROMISE_exp](graficos/stacked_plot_classes.png)

## Metodología y Configuración Experimental

El pipeline experimental se diseñó para garantizar una evaluación robusta del modelo de clasificación.

- **Generación de Embeddings:** Se utilizó el modelo **BERT** (`bert-base-uncased`) para convertir cada requisito textual en un vector numérico de 768 dimensiones.
- **Modelo de Clasificación:** Se empleó un clasificador **CatBoost**, un algoritmo de Gradient Boosting.
- **Manejo de Datos y Balanceo:**
  - El conjunto de datos se dividió de forma estratificada en un **conjunto de entrenamiento (90%)** y un **conjunto de prueba (10%)**.
  - La técnica **SMOTE-Tomek** se aplicó exclusivamente al conjunto de entrenamiento.
- **Proceso de Validación y Prueba:**
  - **Validación Cruzada:** Se realizó una validación cruzada de 10 pliegues (k=10) sobre el conjunto de entrenamiento.
  - **Evaluación Final:** Se entrenó un modelo final con todo el conjunto de entrenamiento y se evaluó su rendimiento definitivo sobre el conjunto de prueba aislado.
- **Métricas de Evaluación:** El rendimiento se midió con **Accuracy**, **Precisión**, **Recall** y **F1-Score** (usando promedio ponderado para multiclase).

## Resultados Experimentales

Se llevaron a cabo dos experimentos de clasificación: binaria (Funcional vs. No Funcional) y multiclase (11 tipos de requisitos no funcionales).

### Clasificación Binaria (Funcional vs. No Funcional)

La validación cruzada arrojó un **F1-Score promedio de 97.25% ± 0.59%**. En la evaluación final sobre el conjunto de prueba, el modelo alcanzó un **F1-Score global de 86.36%**.

**Reporte de clasificación final (Test Set) para 2 Clases**

| Clase | Precisión | Recall | F1-Score | Soporte |
| :--- | :---: | :---: | :---: | :---: |
| F | 0.94 | 0.75 | 0.84 | 44 |
| NF | 0.82 | 0.96 | 0.89 | 53 |
| **Promedio Ponderado** | **0.88** | **0.87** | **0.86** | **97** |

**Matriz de confusión para la clasificación de 2 clases**
![Matriz de confusión para la clasificación de 2 clases](graficos/img2.png)

El modelo es particularmente efectivo para identificar requisitos No Funcionales (Recall de 96%).

### Clasificación Multiclase (11 Clases No Funcionales)

La validación cruzada resultó en un **F1-Score promedio de 99.77% ± 0.27%**. Sin embargo, la evaluación sobre el conjunto de prueba final, que mantiene el desbalanceo original, alcanzó un **F1-Score global de 71.66%**.

**Reporte de clasificación final (Test Set) para 11 Clases NF**

| Clase | Precisión | Recall | F1-Score | Soporte |
| :--- | :---: | :---: | :---: | :---: |
| A (Disponibilidad) | 1.00 | 0.67 | 0.80 | 3 |
| FT (Tolerancia a Fallos) | 0.00 | 0.00 | 0.00 | 2 |
| L (Legal) | 1.00 | 1.00 | 1.00 | 1 |
| LF (Apariencia) | 0.60 | 0.60 | 0.60 | 5 |
| MN (Mantenibilidad) | 0.33 | 0.50 | 0.40 | 2 |
| O (Operatividad) | 0.62 | 0.62 | 0.62 | 8 |
| PE (Rendimiento) | 0.64 | 1.00 | 0.78 | 7 |
| PO (Portabilidad) | 0.00 | 0.00 | 0.00 | 1 |
| SC (Escalabilidad) | 1.00 | 0.50 | 0.67 | 2 |
| SE (Seguridad) | 0.92 | 0.92 | 0.92 | 13 |
| US (Usabilidad) | 0.78 | 0.78 | 0.78 | 9 |
| **Promedio Ponderado** | **0.72** | **0.74** | **0.72** | **53** |

**Matriz de confusión para la clasificación de 11 clases NF**
![Matriz de confusión para la clasificación de 11 clases NF](graficos/img11.png)

El modelo muestra un excelente rendimiento en clases con soporte suficiente como Seguridad (SE), pero lucha con clases con muy pocas muestras como Tolerancia a Fallos (FT) y Portabilidad (PO).

# Comparación con Trabajos Relacionados

Los resultados se compararon con modelos de referencia de la literatura.

### Clasificación Binaria (Funcional vs. No Funcional)

**Comparación de rendimiento F/NF en el dataset PROMISE NFR**

| Enfoque | P (F) | R (F) | F1 (F) | P (NF) | R (NF) | F1 (NF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Literatura (10-fold CV)* | | | | | | |
| K. & M. (word feat. no sel.) | .92 | .93 | .93 | .93 | .92 | .92 |
| A. et al. (processed) | .90 | **.97** | **.93** | **.98** | .93 | **.95** |
| NoRBERT (large, ep.10, OS) | .92 | .88 | .90 | .92 | .95 | .93 |
| *Literatura (Otras metodologías)* | | | | | | |
| NoRBERT (large, loPo, US) | .87 | .71 | .78 | .82 | .93 | .87 |
| **Modelo Propuesto** | **.94** | .75 | .84 | .82 | **.96** | .89 |

#### Análisis de Resultados

- **Precisión en Clase Funcional (F):** Nuestro modelo alcanza un **94%**, superando a los demás enfoques.
- **Recall en Clase No Funcional (NF):** El modelo logra un **96%**, el más alto entre los competidores.
- **Trade-off:** Existe un claro trade-off, donde el alto recall en NF se produce a costa de un recall más bajo en F.

### Clasificación Multiclase (11 Clases No Funcionales)

**Comparación de rendimiento para 4 clases de NFR**

| Enfoque | P (US) | R (US) | F1 (US) | P (SE) | R (SE) | F1 (SE) | P (O) | R (O) | F1 (O) | P (PE) | R (PE) | F1 (PE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Literatura (10-fold CV)* | | | | | | | | | | | | |
| K. & M.multi | .65 | .82 | .70 | .81 | .77 | .75 | .81 | .86 | .82 | .86 | .81 | .80 |
| NoRBERTmulti all | .83 | .88 | .86 | .90 | **.92** | .91 | .78 | .84 | .81 | **.92** | .87 | **.90** |
| *Literatura (Otras metodologías)* | | | | | | | | | | | | |
| NoRBERTmulti all (loPo) | .62 | **.84** | .71 | .75 | .86 | .80 | **.75** | .76 | **.75** | **.92** | .67 | .77 |
| **Modelo Propuesto** | .78 | .78 | .78 | **.92** | **.92** | **.92** | .62 | .62 | .62 | .64 | **1.00** | .78 |

#### Análisis de Resultados

- **Dominio en Seguridad (SE):** Nuestro modelo, con **92%** en P, R y F1, supera a todos los demás enfoques en esta clase.
- **Identificación en Rendimiento (PE):** Se alcanza un **recall perfecto de 100%**, un resultado notable.

# Conclusiones y Trabajo Futuro

### Conclusiones Principales

1. **Eficacia del Pipeline:** La combinación de BERT, SMOTE-Tomek y CatBoost es altamente eficaz, logrando un F1-Score de **86.36%** en la clasificación binaria.
2. **Rendimiento en Clases Críticas:** El modelo destaca en la clasificación multiclase, logrando un rendimiento de vanguardia en **Seguridad (92% F1)** y un recall perfecto en **Rendimiento (100%)**.
3. **Impacto de la Metodología:** La separación de un conjunto de prueba final fue crucial para obtener una medida realista del rendimiento y evitar la fuga de datos.
4. **Desafío del Desbalanceo:** A pesar de SMOTE, el desbalanceo extremo sigue siendo un desafío en la evaluación final con datos del mundo real.

### Trabajo Futuro

- **Optimización de Hiperparámetros:** Aplicar `GridSearchCV` o Búsqueda Bayesiana podría mejorar el rendimiento.
- **Fine-Tuning de BERT:** Realizar un fine-tuning específico del dominio podría generar embeddings de mayor calidad.
- **Aumentación de Datos Avanzada:** Explorar técnicas como la retrotraducción para las clases más minoritarias.
- **Análisis de Interpretabilidad:** Integrar herramientas como SHAP para entender las predicciones del modelo.
