"""하이브리드 검색기 — 9~16곳에 복붙되던 hybrid() 단일화. dense+sparse 서버측 RRF."""
from qdrant_client import QdrantClient, models
from rag.core import config


class Retriever:
    def __init__(self, embedder, url=config.QDRANT_URL, collection=config.COLLECTION,
                 prefetch=config.PREFETCH):
        self.e = embedder
        self.cli = QdrantClient(url=url)
        self.C = collection
        self.prefetch = prefetch

    def hybrid(self, q, k=config.TOP_K, query_filter=None):
        """dense top-N + sparse top-N을 서버측 RRF로 융합. 포인트 리스트 반환."""
        e = self.e.encode([q])
        return self.cli.query_points(
            self.C,
            prefetch=[
                models.Prefetch(query=e["dense_vecs"][0].tolist(), using="dense", limit=self.prefetch),
                models.Prefetch(query=self.e.sparse_vec(e["lexical_weights"][0]), using="sparse", limit=self.prefetch),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, query_filter=query_filter, with_payload=True,
        ).points

    def pages(self, q, k=config.TOP_K, query_filter=None):
        """검색 결과가 걸친 페이지 번호 집합(small-to-big 부모 조회용)."""
        pts = self.hybrid(q, k, query_filter)
        return sorted({p for x in pts for p in (x.payload.get("pages") or [])})
