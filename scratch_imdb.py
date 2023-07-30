import numpy as np
import pandas as pd
import json
import os
from embeddings.sgpt_embedder import SGPTEmbedder
from vectordb.vector_db import VectorDB

filepath = 'data/embeddings/embedded_plots.csv'
raw_data = pd.read_csv(filepath)

embedder = SGPTEmbedder()
embedder.max_seq_length = 1000
embedding_dims = embedder.get_sentence_embedding_dimension()
vector_index = VectorDB(n_dims=embedding_dims)

cols_to_drop = ['index', 'release year', 'title', 'origin/ethnicity', 'director',
                'cast', 'genre', 'wiki page', 'plot']
idx_series = raw_data['index']
title_series = raw_data['title']
embedded_df = raw_data.drop(cols_to_drop, axis=1)


vector_index.add_vectors(
    vectors=embedded_df.values,
    ids=idx_series.values,
    metadata=title_series.values
)

query_str = 'toys that come to life'
embezzlement_embedded = embedder.encode(query_str)
embezzlement_neighbors = vector_index.get_neighbors(embezzlement_embedded, k=5, return_metadata=True)

for neighbor in embezzlement_neighbors:
    print(raw_data.iloc[neighbor['id']]['plot'])