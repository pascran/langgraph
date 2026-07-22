import os, re, json, uuid
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
B=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
B=[b for b in B if b['type'] not in ('header','page_number')]
B.sort(key=lambda b:(b['page'],b['order']))
strip=lambda h: re.sub(r'<[^>]+>',' ',h)
sec=lambda b:(b['meta'].get('dambo') or b['meta'].get('johang') or b['meta'].get('gwan') or 'general')
CH,OV=600,100
def split(t):
    t=re.sub(r'\s+',' ',t).strip()
    if len(t)<=CH: return [t] if len(t)>=30 else []
    sents=re.split(r'(?<=[.다]) ',t); out=[]; cur=''
    for s in sents:
        if len(cur)+len(s)>CH and cur: out.append(cur.strip()); cur=cur[-OV:]
        cur+=s+' '
    if len(cur.strip())>=30: out.append(cur.strip())
    return out
chunks=[]; buf=[]; bmeta=None; bpages=set(); csec=None
def emit():
    global buf,bpages
    if buf:
        for p in split(' '.join(buf)):
            chunks.append({"text":p,"meta":{**(bmeta or {}),"pages":sorted(bpages),"btype":"text"}})
    buf=[]; bpages=set()
for b in B:
    if b['type']=='table':
        emit(); chunks.append({"text":strip(b['content']),"meta":{**b['meta'],"pages":[b['page']],"btype":"table"}}); continue
    s=sec(b)
    if buf and s!=csec: emit()
    csec=s; bmeta=b['meta']; buf.append(strip(b['content'])); bpages.add(b['page'])
    if sum(len(x) for x in buf)>CH: emit()
emit()
print("CHUNKS",len(chunks),"| 표청크",sum(1 for c in chunks if c['meta']['btype']=='table'))
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
emb=m.encode([c['text'] for c in chunks],return_dense=True,return_sparse=True,batch_size=16,max_length=512)
cli=QdrantClient(url="http://localhost:6333")
try: cli.delete_collection("silson_v2")
except: pass
cli.create_collection("silson_v2",vectors_config={"dense":models.VectorParams(size=1024,distance=models.Distance.COSINE)},sparse_vectors_config={"sparse":models.SparseVectorParams()})
sv=lambda lw: models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
def payload(c):
    md=c['meta']; return {"text":c['text'],"dambo":md.get('dambo'),"johang":md.get('johang'),"gwan":md.get('gwan'),"pages":md.get('pages'),"btype":md.get('btype')}
pts=[models.PointStruct(id=str(uuid.uuid4()),vector={"dense":emb['dense_vecs'][i].tolist(),"sparse":sv(emb['lexical_weights'][i])},payload=payload(c)) for i,c in enumerate(chunks)]
for k in range(0,len(pts),128): cli.upsert("silson_v2",pts[k:k+128])
print("INDEXED",cli.count("silson_v2").count)
def search(q,limit=2):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points("silson_v2",prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=20),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=20)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
for q in ["비만으로 입원하면 보상되나?","입원의료비 보험가입금액 최고한도는?","자동차보험에서 보상받는 의료비는?"]:
    print("\nQ:",q)
    for r in search(q):
        print(f"  p{r.payload.get('pages')} 담보={r.payload.get('dambo')} | {r.payload['text'][:58].strip()}")
print("CHUNK_INDEX_DONE")
