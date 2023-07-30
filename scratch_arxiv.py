import numpy as np
import pandas as pd
import json
import os
from embeddings.sgpt_embedder import SGPTEmbedder
from vectordb.vector_db import VectorDB

filepath = 'data/raw/csv/imdb/movies_metadata.csv'
raw_data = pd.read_csv(filepath)

embedder = SGPTEmbedder()
embedder.max_seq_length = 1000
embedding_dims = embedder.get_sentence_embedding_dimension()
vector_index = VectorDB(n_dims=embedding_dims)

