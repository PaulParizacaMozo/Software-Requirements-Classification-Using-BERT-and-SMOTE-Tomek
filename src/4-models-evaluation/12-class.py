
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Crear el directorio de salida para los resultados
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar el dataset balanceado generado previamente
df = pd.read_csv('../3-smoteTomek/output/dataset_balanced.csv', sep=',', header=0)

# En este caso, las clases son tanto 'F' como las 11 clases NF
df['_class_'] = df['_class_'].apply(lambda x: x if x in ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO'] else None)

# Eliminar las filas con clases no definidas (si las hubiera)
df = df.dropna(subset=['_class_'])

# Separar características (X) y etiquetas (y)
X = df.drop('_class_', axis=1)
y = df['_class_']

# Codificar las etiquetas en números (si es necesario)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Abrir archivo para guardar resultados en formato CSV
output_file = os.path.join(output_dir, 'results_classification_12_classes.csv')

# Inicializar el archivo CSV con los encabezados
with open(output_file, "w") as file:
    file.write("Model,Precision,Recall,F1-score,Accuracy\n")
    
    # Modelos a utilizar
    models = {
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "MLP Neural Network": MLPClassifier(max_iter=300)
    }
    
    # Configuración de K-Fold Cross Validation (10 pliegues)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Iterar a través de cada modelo
    for model_name, model in models.items():
        print(f"Evaluando el modelo {model_name}...")
        
        fold = 1
        avg_precision, avg_recall, avg_f1, avg_accuracy = 0, 0, 0, 0
        
        for train_index, test_index in kf.split(X, y_encoded):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            
            # Entrenamiento del modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas de desempeño
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            
            # Acumular resultados por pliegue
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            avg_accuracy += accuracy
            
            fold += 1
        
        # Promediar resultados de los pliegues
        avg_precision /= 10
        avg_recall /= 10
        avg_f1 /= 10
        avg_accuracy /= 10
        
        # Guardar resultados en archivo CSV
        file.write(f"{model_name},{avg_precision},{avg_recall},{avg_f1},{avg_accuracy}\n")
        
        # Imprimir resultados por consola (opcional)
        print(f"{model_name}:")
        print(f"Precisión: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1-Score: {avg_f1:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}\n")

print(f"Resultados de clasificación guardados en: {output_file}")
