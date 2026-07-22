import os, re, json, time
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
cli=QdrantClient(url="http://localhost:6333"); C="silson_v2_sem"
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')][:12]
qv=[m.encode([g['question']],return_dense=True)['dense_vecs'][0].tolist() for g in golden]

print("=== B) Payload 인덱스 생성 ===")
for fld,sch in [("dambo",models.PayloadSchemaType.KEYWORD),("btype",models.PayloadSchemaType.KEYWORD),("pages",models.PayloadSchemaType.INTEGER)]:
    try: cli.create_payload_index(C,field_name=fld,field_schema=sch); print(" +인덱스:",fld,sch.value)
    except Exception as e: print(" ",fld,"err",str(e)[:50])
info=cli.get_collection(C)
print(" 컬렉션 포인트:",info.points_count,"| payload_schema:",list((info.payload_schema or {}).keys()))

def dsearch(v,ef=None,exact=False,k=10,flt=None):
    return cli.query_points(C,query=v,using="dense",limit=k,search_params=models.SearchParams(hnsw_ef=ef,exact=exact),query_filter=flt,with_payload=False).points

print("\n=== A) ef_search 튜닝: exact(정답) 대비 recall@10 + 지연 ===")
gt=[set(p.id for p in dsearch(v,exact=True,k=10)) for v in qv]  # brute-force ground truth
for ef in [8,16,32,64,128,256]:
    t0=time.time(); rec=0
    for _ in range(3):
        for i,v in enumerate(qv):
            ids=set(p.id for p in dsearch(v,ef=ef,exact=False,k=10)); 
    for i,v in enumerate(qv):
        ids=set(p.id for p in dsearch(v,ef=ef,exact=False,k=10)); rec+=len(ids&gt[i])/10
    dt=(time.time()-t0)/(3*len(qv))*1000
    print(f"  hnsw_ef={ef:4d}  recall@10={rec/len(qv):.3f}  avg {dt:.2f}ms/q")

print("\n=== B) Payload 인덱스 활용: 필터검색 데모 ===")
# 표(btype=table)만 검색
r=dsearch(qv[0],ef=128,k=3,flt=models.Filter(must=[models.FieldCondition(key="btype",match=models.MatchValue(value="table"))]))
full=cli.retrieve(C,ids=[p.id for p in r],with_payload=True)
print(" [btype=table 필터] top3 →",[(x.payload.get('pages'),x.payload.get('btype')) for x in full])
# 특정 페이지대만
r2=cli.query_points(C,query=qv[0],using="dense",limit=3,query_filter=models.Filter(must=[models.FieldCondition(key="pages",range=models.Range(gte=40,lte=48))]),with_payload=True).points
print(" [pages 40~48 특약구간 필터] top3 →",[x.payload.get('pages') for x in r2])
print("TUNE_DONE")
