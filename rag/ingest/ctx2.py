import os, re, json, uuid, glob
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from concurrent.futures import ThreadPoolExecutor
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from openai import OpenAI
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
cli=QdrantClient(url="http://localhost:6333")
llm=OpenAI(base_url="http://localhost:8001/v1",api_key="x")
sv=lambda lw:models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
PAGES={}
for f in glob.glob('data/ocr/p*.md'):
    n=int(re.search(r'p(\d+)',f).group(1)); PAGES[n]=open(f,encoding='utf-8').read()
sm=lambda t:re.sub(r'<[^>]+>',' ',t)
pts,_=cli.scroll("silson_v2_sem",limit=2000,with_payload=True)
chunks=[{"text":p.payload['text'],"dambo":p.payload.get('dambo'),"pages":p.payload.get('pages') or []} for p in pts]
SYS="주어진 약관 페이지 맥락을 참고해, 아래 [부분]이 약관에서 어떤 담보종목·조항·표에 관한 것인지 검색이 잘 되도록 핵심어 위주 한 문장(30자 내외)으로 situating. 문장만 출력."
def gc(c):
    win=" ".join(sm(PAGES.get(p,'')) for p in c['pages'][:2])[:2000]
    try:
        r=llm.chat.completions.create(model="Qwen/Qwen3-8B-AWQ",messages=[{"role":"system","content":SYS},{"role":"user","content":f"[페이지 맥락]\n{win}\n\n[부분]\n{c['text'][:600]}"}],temperature=0,max_tokens=48,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
        return r.choices[0].message.content.strip().replace(chr(10),' ')[:100]
    except Exception: return ""
with ThreadPoolExecutor(max_workers=4) as ex: ctxs=list(ex.map(gc,chunks))
for c,x in zip(chunks,ctxs): c['ctx']=x; c['ctext']=f"{x} {c['text']}"
print("CONTEXTS_DONE",len(chunks),"| 샘플:",chunks[10]['ctx'][:55])
emb=m.encode([c['ctext'] for c in chunks],return_dense=True,return_sparse=True,batch_size=16,max_length=512)
C="silson_v2_ctx2"
try: cli.delete_collection(C)
except: pass
cli.create_collection(C,vectors_config={"dense":models.VectorParams(size=1024,distance=models.Distance.COSINE)},sparse_vectors_config={"sparse":models.SparseVectorParams()})
P=[models.PointStruct(id=str(uuid.uuid4()),vector={"dense":emb['dense_vecs'][i].tolist(),"sparse":sv(emb['lexical_weights'][i])},payload={"text":c['text'],"ctx":c['ctx'],"pages":c['pages']}) for i,c in enumerate(chunks)]
for k in range(0,len(P),128): cli.upsert(C,P[k:k+128])
print("INDEXED",cli.count(C).count)
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
for g in golden:
    mm=re.search(r'p(\d+)',g['source']); g['gp']=int(mm.group(1)) if mm else None
def hyb(q,coll,limit=5):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points(coll,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
pg=lambda p:(p.payload.get('pages') or [])
def metr(coll,K=(1,3,5)):
    hit={k:0 for k in K}; mrr=0
    for g in golden:
        res=hyb(g['question'],coll); ranks=[i+1 for i,p in enumerate(res) if g['gp'] in pg(p)]; f=ranks[0] if ranks else None
        for k in K:
            if f and f<=k: hit[k]+=1
        mrr+=(1/f) if f else 0
    n=len(golden); return {**{f"hit@{k}":round(hit[k]/n,3) for k in K},"MRR":round(mrr/n,3)}
print("SEM(코사인급락)      :", metr("silson_v2_sem"))
print("CTX2(+페이지맥락)    :", metr("silson_v2_ctx2"))
print("CTX2_DONE")
