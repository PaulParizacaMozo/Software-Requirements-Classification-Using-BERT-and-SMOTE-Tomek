import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Cargar el modelo BERT preentrenado y su tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función para convertir el texto a embeddings con BERT
def process_requirement_text_with_bert(text):
    # Tokenización con BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Obtener las salidas del modelo BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extraer el embedding del token [CLS] (representación del texto completo)
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()  # Seleccionamos el embedding [CLS]
    
    return cls_embedding

# Cargar el archivo de datos
df = pd.read_csv('../1-exploratory-analysis/data/PROMISE_exp.csv', sep=',', header=0, quotechar='"', doublequote=True)

# Eliminar la columna de 'ProjectID' (si es que no se usa)
del df['ProjectID']

# Aplicar el proceso de generación de embeddings a cada requerimiento
embeddings_list = df['RequirementText'].apply(process_requirement_text_with_bert)

# Convertir la lista de embeddings en un DataFrame
embeddings_df = pd.DataFrame(embeddings_list.tolist())

# Agregar la columna de clases (_class_)
embeddings_df['_class_'] = df['_class_']

# Guardar los embeddings en un archivo CSV
embeddings_df.to_csv('./output/dataset_embeddings.csv', sep=',', header=True, index=False, quotechar='"', doublequote=True)

print("Embeddings generados y guardados en './output/dataset_embeddings.csv'")

