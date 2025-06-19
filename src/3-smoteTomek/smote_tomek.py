
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split

# Crear el directorio de salida si no existe
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Leer los embeddings generados por BERT
df_embeddings = pd.read_csv('../2-preprocessing/output/dataset_embeddings.csv', sep=',', header=0)

# Separar las características (embeddings) y las etiquetas (clases)
X = df_embeddings.drop('_class_', axis=1)
y = df_embeddings['_class_']

# Aplicar SMOTE (Synthetic Minority Over-sampling Technique) para balancear el dataset
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Aplicar Tomek Links para eliminar ejemplos cercanos a los límites de decisión
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_smote, y_smote)

# Crear un nuevo DataFrame con los datos balanceados
df_balanced = pd.DataFrame(X_tomek, columns=X.columns)
df_balanced['_class_'] = y_tomek

# Guardar el dataset balanceado en un archivo CSV dentro de la carpeta de salida
balanced_csv_path = os.path.join(output_dir, 'dataset_balanced.csv')
df_balanced.to_csv(balanced_csv_path, sep=',', header=True, index=False, quotechar='"', doublequote=True)

print(f"Dataset balanceado guardado en: {balanced_csv_path}")

