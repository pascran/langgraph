import os, sys, json, re, threading, torch
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m, cli, PAGES, chat, sv
from qdrant_client import models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
os.makedirs('data/ragas',exist_ok=True)
C="silson_v2_sem"
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
rtok=AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
rmod=AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3',dtype=torch.float16).cuda().eval()
enc_lock=threading.Lock(); rr_lock=threading.Lock()
def enc(q):
    with enc_lock: return m.encode([q],return_dense=True,return_sparse=True)
def dense(q,k=5):
    e=enc(q); pts=cli.query_points(C,query=e['dense_vecs'][0].tolist(),using="dense",limit=k,with_payload=True).points
    return [{"text":p.payload['text'],"pages":p.payload.get('pages')} for p in pts]
def hybrid(q,k=5):
    e=enc(q); pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
    return [{"text":p.payload['text'],"pages":p.payload.get('pages')} for p in pts]
def rerank(q,cands,k=5):
    with rr_lock:
        inp=rtok([[q,c['text']] for c in cands],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        with torch.no_grad(): sc=rmod(**inp).logits.view(-1).float().cpu().tolist()
    return [cands[i] for i in sorted(range(len(cands)),key=lambda i:-sc[i])[:k]]
def parent(docs):
    pages=sorted({p for d in docs for p in (d['pages'] or [])})
    return [f"[p{n}] "+PAGES.get(n,'')[:3000] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 면책(보상안함)인지 보상인지 명확히. 없으면 '약관에서 확인 불가'."
def gen(ctx,q): return chat(SYS,"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}")
def L0(q): d=dense(q,5); c=[x['text'] for x in d]; return gen(c,q),c
def L1(q): d=hybrid(q,5); c=[x['text'] for x in d]; return gen(c,q),c
def L2(q): d=hybrid(q,5); c=parent(d); return gen(c,q),c
def L3(q): d=rerank(q,hybrid(q,20),5); c=parent(d); return gen(c,q),c
def L4(q):
    d=hybrid(q,5); tries=0; kept=[]
    while True:
        kept=[x for x in d if chat("문서가 질문 주제에 관련 있으면 yes 아니면 no. 한 단어.",f"[질문]{q}\n[문서]{x['text'][:400]}",4).lower().startswith('y')]
        if kept or tries>=2: break
        q2=chat("검색되게 약관 용어로 다르게 1문장 재작성. 질문만.",q+f" (재{tries+1})",48); d=hybrid(q2,5); tries+=1
    docs=kept or d; c=parent(docs); return gen(c,q),c
LAD={"L0":L0,"L1":L1,"L2":L2,"L3":L3,"L4":L4}
out={}
for name,fn in LAD.items():
    def run(g):
        a,c=fn(g['question']); return {"question":g['question'],"answer":a,"contexts":c,"ground_truth":g['answer']}
    with ThreadPoolExecutor(max_workers=6) as ex: out[name]=list(ex.map(run,golden))
    print(name,"done",len(out[name]),flush=True)
json.dump(out,open('data/ragas/ladder.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print("LADDER_DONE", {k:len(v) for k,v in out.items()})
