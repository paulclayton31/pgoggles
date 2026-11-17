"""Turns raw frames (numpy ndarrays) into facial embedding vectors via insightface."""

import numpy as np
from insightface.app import FaceAnalysis

from ..db.schema import EMBEDDING_DIM


class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0) -> None:
        self._app = FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=ctx_id)

    def embed(self, frame: np.ndarray) -> list[float] | None:
        faces = self._app.get(frame)
        if not faces:
            return None
        embedding = faces[0].normed_embedding
        assert embedding.shape[0] == EMBEDDING_DIM
        return embedding.tolist()
