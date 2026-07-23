"""BGE-M3 임베더 — 7곳에 흩어진 로드/encode/enc_lock/sv 람다를 단일화. 스레드 안전."""
import threading
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models


class Embedder:
    def __init__(self, model="BAAI/bge-m3", use_fp16=True, device=None):
        kw = {"devices": device} if device else {}
        self.m = BGEM3FlagModel(model, use_fp16=use_fp16, **kw)
        self._lk = threading.Lock()

    def encode(self, texts, dense=True, sparse=True, **kw):
        with self._lk:  # BGE-M3 encode는 GPU 상태공유 → 락으로 동시호출 보호
            return self.m.encode(texts, return_dense=dense, return_sparse=sparse, **kw)

    def dense(self, texts, **kw):
        return self.encode(texts, dense=True, sparse=False, **kw)["dense_vecs"]

    @staticmethod
    def sparse_vec(lw):
        return models.SparseVector(
            indices=[int(k) for k in lw], values=[float(v) for v in lw.values()]
        )
