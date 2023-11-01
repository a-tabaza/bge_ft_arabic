# not async ik but when I created and commited this file the script was running I couldn't pause it

from llama_index.embeddings import OptimumEmbedding
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

df = pd.read_parquet('ar-en-final-fuzzy-deduplicated.parquet')

# OptimumEmbedding.create_and_save_optimum_model(
#     "BAAI/bge-large-en-v1.5", "./bge_onnx"
# )

embed_model = OptimumEmbedding(folder_name="./bge_onnx")

n = 10_000
df_splits = [df[i:i+n][['en','en_translated']] for i in range(0,df.shape[0],n)]

def similarity(text1, text2):
    embedding1 = embed_model.get_text_embedding(text1)
    embedding2 = embed_model.get_text_embedding(text2)
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm

for idx, split in enumerate(tqdm(df_splits)):
    split['similarity'] = split.progress_apply(lambda x: similarity(x['en'], x['en_translated']), axis=1)
    split.to_parquet(f'similarity/ar-en-final-fuzzy-deduplicated-similarity-{idx}.parquet')
