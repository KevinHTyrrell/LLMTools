import numpy as np
import pandas as pd
import faiss
from typing import List, Union

class VectorDB:
    def __init__(
            self,
            n_dims: int
    ):
        self._n_dims            = n_dims,
        self._index             = faiss.IndexIDMap(faiss.IndexFlatL2(n_dims))
        self._curr_id           = 0
        self._index_built       = False
        self._id_id_map         = {}
        self._id_vector_map     = {}
        self._id_metadata_map   = {}

    def add_vectors(
            self,
            vectors: np.ndarray,
            ids: Union[List, np.ndarray, pd.Series] = None,
            metadata: List[str] = None
    ):
        if ids is not None:
            assert len(vectors) == len(ids), 'IDS AND VECTORS MUST BE 1 TO 1'
        if metadata is not None:
            assert len(vectors) == len(metadata), 'METADATA AND VECTORS MUST BE 1 TO 1'

        id_list = []
        for i in range(len(vectors)):
            vector = vectors[i, :]
            id_internal = self._curr_id
            id_list.append(id_internal)
            self._id_vector_map[id_internal] = vector
            if ids is not None:
                self._id_id_map[id_internal] = ids[i]
            if metadata is not None:
                self._id_metadata_map[id_internal] = metadata[i]
            self._curr_id += 1
        self._index.add_with_ids(vectors, id_list)
        self._index_built = True

    def get_vector(self, id):
        assert self._index_built, 'INDEX MUST CONTAIN VECTORS BEFORE ACCESS'
        internal_id = self._id_id_map[id]
        return self._id_vector_map[internal_id]

    def get_neighbors(
            self,
            vector = None,
            id = None,
            k: int = 10,
            return_metadata: bool = False
    ):
        assert (vector is not None) or (id is not None), 'MUST PROVIDE VECTOR OR VECTOR_ID'
        if vector is None:
            internal_id = self._id_id_map[id]
            vector = self._id_vector_map[internal_id]
        vector = np.expand_dims(vector, axis=0) if len(vector.shape) == 1 else vector
        distances, internal_vector_ids = self._index.search(vector, k=k)
        distances, internal_vector_ids = distances.flatten(), internal_vector_ids.flatten()
        matching_vectors = [self._id_vector_map[id] for id in internal_vector_ids]
        if return_metadata:
            assert len(self._id_metadata_map) > 0, 'NO METADATA PROVIDED'
            metadata_list = [self._id_metadata_map[id] for id in internal_vector_ids]
        vector_info_list = []
        for i in range(len(matching_vectors)):
            vector_info = {
                'distance': distances[i],
                'vector': matching_vectors[i],
            }
            if return_metadata:
                vector_info.update({'metadata': metadata_list[i]})
            vector_info_list.append(vector_info)
        return vector_info_list

    def add_metadata(self, ids, metadata: List[str]):
        for i in range(len(ids)):
            internal_id = self._id_id_map[ids[i]]
            self._id_metadata_map[internal_id] = metadata[i]

# n_dims = 768
# n_vectors = 100
# vector_content = np.random.random(n_vectors * n_dims)
# vector_reshaped = np.reshape(vector_content, newshape=(-1, n_dims))
# metadata_list = np.arange(len(vector_reshaped)).astype(str)
# index = VectorDB(n_dims)
# index.add_vectors(vectors=vector_reshaped, metadata=metadata_list.tolist())
# output = index.get_neighbors(vector_reshaped[0], return_metadata=True)
