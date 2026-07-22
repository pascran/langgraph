import os, re, json, uuid, torch, numpy as np
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client import QdrantClient, models
B=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
B=[b for b in B if b['type'] not in ('header','page_number')]; B.sort(key=lambda b:(b['page'],b['order']))
strip=lambda h:re.sub(r'<[^>]+>',' ',h)
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
# 문장 스트림
sents=[]
for b in B:
    if b['type']=='table': sents.append(('__T__',b)); continue
    for s in re.split(r'(?<=[.다])\s+',strip(b['content'])):
        s=s.strip()
        if len(s)>=10: sents.append((s,b))
txts=[s for s,_ in sents if s!='__T__']
E=m.encode(txts,return_dense=True,batch_size=32,max_length=256)['dense_vecs']
d=[1-float(np.dot(E[i],E[i+1])) for i in range(len(E)-1)]; thr=float(np.percentile(d,88)) if d else 1
chunks=[]; cur=[]; ei=0; prev=None
def flush():
    global cur
    if cur:
        txt=' '.join(x[0] for x in cur).strip()
        if len(txt)>=30: chunks.append({"text":txt,"meta":{**cur[-1][1]['meta'],"pages":sorted(set(x[1]['page'] for x in cur)),"btype":"text"}})
    cur=[]
for s,b in sents:
    if s=='__T__': flush(); prev=None; chunks.append({"text":strip(b['content']),"meta":{**b['meta'],"pages":[b['page']],"btype":"table"}}); continue
    v=E[ei]; ei+=1
    if cur and prev is not None and (1-float(np.dot(v,prev)))>thr and sum(len(x[0]) for x in cur)>200: flush()
    cur.append((s,b)); prev=v
    if sum(len(x[0]) for x in cur)>700: flush(); prev=None
flush()
print("SEM_CHUNKS",len(chunks),"표",sum(1 for c in chunks if c['meta']['btype']=='table'),"| thr=%.3f"%thr)
emb=m.encode([c['text'] for c in chunks],return_dense=True,return_sparse=True,batch_size=16,max_length=512)
cli=QdrantClient(url="http://localhost:6333"); C="silson_v2_sem"
try: cli.delete_collection(C)
except: pass
cli.create_collection(C,vectors_config={"dense":models.VectorParams(size=1024,distance=models.Distance.COSINE)},sparse_vectors_config={"sparse":models.SparseVectorParams()})
sv=lambda lw:models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
pts=[models.PointStruct(id=str(uuid.uuid4()),vector={"dense":emb['dense_vecs'][i].tolist(),"sparse":sv(emb['lexical_weights'][i])},payload={"text":c['text'],"dambo":c['meta'].get('dambo'),"pages":c['meta'].get('pages'),"btype":c['meta'].get('btype')}) for i,c in enumerate(chunks)]
for k in range(0,len(pts),128): cli.upsert(C,pts[k:k+128])
print("INDEXED_SEM",cli.count(C).count)
# ===== eval sem (hybrid + rerank) =====
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
for g in golden:
    mm=re.search(r'p(\d+)',g['source']); g['gp']=int(mm.group(1)) if mm else None
rtok=AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3'); rmod=AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3',dtype=torch.float16).cuda().eval()
def rrs(q,docs):
    with torch.no_grad():
        inp=rtok([[q,x] for x in docs],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda'); return rmod(**inp).logits.view(-1).float().cpu().tolist()
def hybrid(q,limit=20):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
pg=lambda p:(p.payload.get('pages') or [])
def metrics(fn,K=(1,3,5)):
    hit={k:0 for k in K}; mrr=0; n=len(golden)
    for g in golden:
        res=fn(g['question']); ranks=[i+1 for i,p in enumerate(res) if g['gp'] in pg(p)]; f=ranks[0] if ranks else None
        for k in K:
            if f and f<=k: hit[k]+=1
        mrr+=(1/f) if f else 0
    return {**{f"hit@{k}":round(hit[k]/n,3) for k in K},"MRR":round(mrr/n,3)}
def hr(q):
    c=hybrid(q,20)
    if not c: return []
    s=rrs(q,[p.payload['text'] for p in c]); return [c[i] for i in sorted(range(len(c)),key=lambda i:-s[i])[:5]]
print("SEM 하이브리드:", metrics(lambda q:hybrid(q,5)))
print("SEM +리랭커  :", metrics(hr))
print("SEM_EVAL_DONE")
