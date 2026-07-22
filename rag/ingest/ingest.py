import os, uuid
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
import fitz
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models

PDF="data/db_silson_2014.pdf"; COLL="silson_v1"
# 1) parse (텍스트레이어 베이스라인; OCR 경로는 나중에 교체)
doc=fitz.open(PDF)
pages=[(i+1,p.get_text()) for i,p in enumerate(doc)]
# 2) chunk (구조인지 재귀분할, 한국어 구분자)
from langchain_text_splitters import RecursiveCharacterTextSplitter
sp=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,separators=["\n\n","\n",". "," ",""])
chunks=[]
for pageno,text in pages:
    if not text.strip(): continue
    for c in sp.split_text(text):
        c=c.strip()
        if len(c)<30: continue
        chunks.append({"text":c,"page":pageno})
print("CHUNKS", len(chunks))
# 3) embed dense+sparse
model=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
texts=[c["text"] for c in chunks]
emb=model.encode(texts,return_dense=True,return_sparse=True,batch_size=16,max_length=512)
# 4) qdrant (dense cosine + sparse)
client=QdrantClient(url="http://localhost:6333")
try: client.delete_collection(COLL)
except: pass
client.create_collection(COLL,
    vectors_config={"dense":models.VectorParams(size=1024,distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse":models.SparseVectorParams()})
def sv(lw): return models.SparseVector(indices=[int(k) for k in lw.keys()],values=[float(v) for v in lw.values()])
pts=[models.PointStruct(id=str(uuid.uuid4()),
        vector={"dense":emb['dense_vecs'][i].tolist(),"sparse":sv(emb['lexical_weights'][i])},
        payload={"text":c["text"],"page":c["page"],"chunk_index":i}) for i,c in enumerate(chunks)]
for k in range(0,len(pts),128): client.upsert(COLL,pts[k:k+128])
print("INDEXED", client.count(COLL).count)
# 5) 하이브리드 RRF 검색 데모
q="통원 시 상급종합병원 외래 공제금액은 얼마인가?"
qe=model.encode([q],return_dense=True,return_sparse=True)
res=client.query_points(COLL,prefetch=[
        models.Prefetch(query=qe['dense_vecs'][0].tolist(),using="dense",limit=20),
        models.Prefetch(query=sv(qe['lexical_weights'][0]),using="sparse",limit=20)],
    query=models.FusionQuery(fusion=models.Fusion.RRF),limit=3,with_payload=True)
print("QUERY:",q)
for r in res.points:
    print(f"  [p{r.payload['page']} score={r.score:.3f}] {r.payload['text'][:75].strip()}")
print("INGEST_DEMO_OK")
