import os, re, json, torch
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client import QdrantClient, models
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
for g in golden:
    mm=re.search(r'p(\d+)',g['source']); g['gp']=int(mm.group(1)) if mm else None
cli=QdrantClient(url="http://localhost:6333")
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
# 리랭커: transformers 직접 (FlagReranker는 tf5.x 비호환)
rtok=AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
rmod=AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3',torch_dtype=torch.float16).cuda().eval()
def rerank_scores(q,docs):
    with torch.no_grad():
        inp=rtok([[q,d] for d in docs],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        return rmod(**inp).logits.view(-1).float().cpu().tolist()
sv=lambda lw: models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
COLL="silson_v2"
def hybrid(q,limit=20):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points(COLL,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=limit,with_payload=True).points
pg=lambda p:(p.payload.get('pages') or [])
def metrics(fn,K=(1,3,5)):
    hit={k:0 for k in K}; mrr=0; n=len(golden)
    for g in golden:
        res=fn(g['question']); ranks=[i+1 for i,p in enumerate(res) if g['gp'] in pg(p)]
        f=ranks[0] if ranks else None
        for k in K:
            if f and f<=k: hit[k]+=1
        mrr+=(1/f) if f else 0
    return {**{f"hit@{k}":round(hit[k]/n,3) for k in K},"MRR":round(mrr/n,3)}
def hyb_rr(q):
    cand=hybrid(q,20)
    if not cand: return []
    s=rerank_scores(q,[p.payload['text'] for p in cand])
    return [cand[i] for i in sorted(range(len(cand)),key=lambda i:-s[i])[:5]]
base=metrics(lambda q:hybrid(q,5)); rer=metrics(hyb_rr)
print("BASELINE (하이브리드 RRF)    :", base)
print("+RERANKER (BGE-v2-m3 top20→5):", rer)
print("ΔMRR:", round(rer['MRR']-base['MRR'],3), "| Δhit@3:", round(rer['hit@3']-base['hit@3'],3), "| Δhit@1:", round(rer['hit@1']-base['hit@1'],3))
from collections import defaultdict
byd=defaultdict(lambda:[0,0])
for g in golden:
    res=hybrid(g['question'],3)
    byd[g['difficulty']][0]+=any(g['gp'] in pg(p) for p in res); byd[g['difficulty']][1]+=1
print("난이도별 hit@3(base):", {k:f"{v[0]}/{v[1]}" for k,v in byd.items()})
print("EVAL_DONE")
