import os, re, json, uuid, numpy as np
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
B=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
B=[b for b in B if b['type'] not in ('header','page_number')]; B.sort(key=lambda b:(b['page'],b['order']))
strip=lambda h:re.sub(r'<[^>]+>',' ',h)
LIST=re.compile(r'^\s*([①-⑳]|\d{1,2}[.)]|[가-힣]\.)')  # ①-⑳, 1., 가.
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
sents=[]
for b in B:
    if b['type']=='table': sents.append(('__T__',b)); continue
    for s in re.split(r'(?<=[.다])\s+',strip(b['content'])):
        s=s.strip()
        if len(s)>=2: sents.append((s,b))
txts=[s for s,_ in sents if s!='__T__']
E=m.encode(txts,return_dense=True,batch_size=32,max_length=256)['dense_vecs']
d=[1-float(np.dot(E[i],E[i+1])) for i in range(len(E)-1)]; thr=float(np.percentile(d,88)) if d else 1
chunks=[]; cur=[]; ei=0; prev=None
def flush():
    global cur
    if cur:
        txt=' '.join(x[0] for x in cur).strip(); pages=sorted(set(x[1]['page'] for x in cur))
        if len(txt)>=30: chunks.append({"text":txt,"meta":{**cur[-1][1]['meta'],"pages":pages,"btype":"text"}})
        elif chunks and chunks[-1]['meta'].get('btype')=='text':
            chunks[-1]['text']=(chunks[-1]['text']+' '+txt).strip(); chunks[-1]['meta']['pages']=sorted(set(chunks[-1]['meta']['pages']+pages))
    cur=[]
for s,b in sents:
    if s=='__T__': flush(); prev=None; chunks.append({"text":strip(b['content']),"meta":{**b['meta'],"pages":[b['page']],"btype":"table"}}); continue
    v=E[ei]; ei+=1; is_list=bool(LIST.match(s))
    if cur and prev is not None and not is_list and (1-float(np.dot(v,prev)))>thr and sum(len(x[0]) for x in cur)>200: flush()  # FIX: 리스트 항목은 분리금지
    cur.append((s,b)); prev=v
    if sum(len(x[0]) for x in cur)>700 and not is_list: flush(); prev=None   # FIX: 리스트 중간 절단 금지
flush()
bic=[c for c in chunks if '비만' in c['text']]
print("CHUNKS",len(chunks),"| 비만청크:",len(bic),"| 비만청크에 '보상하지않' 포함:",any('보상하지 않' in c['text'] or '보상하지않' in c['text'] for c in bic))
if bic: print("  비만청크:",bic[0]['text'][:170].replace(chr(10)," "))
emb=m.encode([c['text'] for c in chunks],return_dense=True,return_sparse=True,batch_size=16,max_length=512)
cli=QdrantClient(url="http://localhost:6333"); C="silson_v2_sem"
sv=lambda lw:models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
try: cli.delete_collection(C)
except: pass
cli.create_collection(C,vectors_config={"dense":models.VectorParams(size=1024,distance=models.Distance.COSINE)},sparse_vectors_config={"sparse":models.SparseVectorParams()})
P=[models.PointStruct(id=str(uuid.uuid4()),vector={"dense":emb['dense_vecs'][i].tolist(),"sparse":sv(emb['lexical_weights'][i])},payload={"text":c['text'],"dambo":c['meta'].get('dambo'),"pages":c['meta'].get('pages'),"btype":c['meta'].get('btype')}) for i,c in enumerate(chunks)]
for k in range(0,len(P),128): cli.upsert(C,P[k:k+128])
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
for g in golden:
    mm=re.search(r'p(\d+)',g['source']); g['gp']=int(mm.group(1)) if mm else None
def hyb(q,limit=5):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
pg=lambda p:(p.payload.get('pages') or []); hit={1:0,3:0,5:0}; mrr=0
for g in golden:
    res=hyb(g['question']); ranks=[i+1 for i,p in enumerate(res) if g['gp'] in pg(p)]; f=ranks[0] if ranks else None
    for k in (1,3,5):
        if f and f<=k: hit[k]+=1
    mrr+=(1/f) if f else 0
n=len(golden); print("INDEXED",cli.count(C).count,"| 골든:",{f"hit@{k}":round(hit[k]/n,3) for k in (1,3,5)},"MRR",round(mrr/n,3))
print("FIX2_DONE")
