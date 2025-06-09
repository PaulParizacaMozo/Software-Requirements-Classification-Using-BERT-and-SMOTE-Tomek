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
├── data/              # Datasets raw y procesados
├── notebooks/         # Jupyter Notebooks para exploración y desarrollo incremental
├── src/               # Scripts modulares (.py) del pipeline final
├── models/            # Almacén para los modelos entrenados
├── reports/           # Figuras, gráficos y reportes de evaluación
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
