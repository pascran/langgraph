import os
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
COLL="silson_v1"
client=QdrantClient(url="http://localhost:6333")
print("INDEXED_POINTS:", client.count(COLL).count)
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
def sv(lw): return models.SparseVector(indices=[int(k) for k in lw.keys()],values=[float(v) for v in lw.values()])
def search(q,limit=2):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return client.query_points(COLL,prefetch=[
        models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=20),
        models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=20)],
        query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
Q=["비만으로 입원하면 보상되나?","청약철회는 며칠 이내에 가능한가?","입원의료비 보험가입금액 최고한도는?","자동차보험에서 보상받는 의료비는?"]
for q in Q:
    print("\nQ:",q)
    for r in search(q):
        t=r.payload['text'].replace('\n','').strip()[:65]
        print(f"  p{r.payload['page']} rrf={r.score:.3f} | {t}")
print("\nSEARCH_OK")
