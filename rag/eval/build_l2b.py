import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m, chat, sv, cli
from qdrant_client import models
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
C="silson_v2_sem"
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
# 부모 페이지: 표 블록은 HTML 구조 보존, 텍스트는 태그 제거
_bl=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
_pg=defaultdict(list)
for _b in sorted(_bl,key=lambda b:(b['page'],b['order'])):
    if _b['type'] in ('header','page_number'): continue
    if _b['type']=='table': _pg[_b['page']].append("[표]\n"+_b['content'].strip())
    else: _pg[_b['page']].append(re.sub(r'<[^>]+>',' ',_b['content']).strip())
PAGES_ST={p:'\n'.join(v) for p,v in _pg.items()}
enc_lock=threading.Lock()
def enc(q):
    with enc_lock: return m.encode([q],return_dense=True,return_sparse=True)
def embed(txts):
    with enc_lock: return m.encode(txts,return_dense=True)['dense_vecs']
def hybrid(q,k=5):
    e=enc(q); pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
    return [{"text":p.payload['text'],"pages":p.payload.get('pages')} for p in pts]
def parent(docs):
    pages=sorted({p for d in docs for p in (d['pages'] or [])})
    return [f"[p{n}]\n"+PAGES_ST.get(n,'')[:3500] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행과 열을 정확히 대응시켜 수치를 정확히 읽어라. 면책/보상 명확히. 없으면 '약관에서 확인 불가'."
def gen(ctx,q): return chat(SYS,"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}")
def jparse(s):
    mm=re.search(r'\{[^{}]*\}',s,re.S)
    if mm:
        for c in (mm.group(0),mm.group(0).replace("'",'"')):
            try: return json.loads(c)
            except: pass
    return None
def faith(a,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(chat("답변을 단순 사실문장으로 나눈 뒤 각 문장이 문맥에서 추론가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[답변]{a}\n\n형식만: {{\"supported\":정수,\"total\":정수}}",100))
    return (r['supported']/r['total']) if r and r.get('total') else float('nan')
def crecall(gt,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(chat("정답을 사실문장으로 나눈 뒤 각 문장이 문맥에 귀속가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[정답]{gt}\n\n형식만: {{\"attributable\":정수,\"total\":정수}}",100))
    return (r['attributable']/r['total']) if r and r.get('total') else float('nan')
def acorr(a,gt):
    r=jparse(chat("답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.",f"[정답]{gt}\n[답변]{a}\n형식만: {{\"tp\":정수,\"fp\":정수,\"fn\":정수}}",80)); f1=float('nan')
    if r:
        tp,fp,fn=r.get('tp',0),r.get('fp',0),r.get('fn',0); d=tp+0.5*(fp+fn); f1=tp/d if d else 0.0
    v=embed([a,gt]); sim=float(np.dot(v[0],v[1])); return (0.75*f1+0.25*sim) if f1==f1 else 0.25*sim
def one(g):
    d=hybrid(g['question'],5); ctx=parent(d); a=gen(ctx,g['question'])
    return {"q":g['question'],"type":g.get('type'),"cr":crecall(g['answer'],ctx),"faith":faith(a,ctx),"ac":acorr(a,g['answer'])}
with ThreadPoolExecutor(max_workers=6) as ex: rows=list(ex.map(one,golden))
json.dump(rows,open('data/ragas/L2b_custom.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(rs,k,f=lambda r:True): xs=[r[k] for r in rs if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
gt={g['question']:g.get('type') for g in golden}
L2=[{**r,"type":gt.get(r['q'])} for r in json.load(open('data/ragas/L2_custom.json',encoding='utf-8'))]
tab=lambda r:r['type']=='table'
print("           전체ac  표type_ac  전체cr  전체faith")
print("L2  (평문표) ", mn(L2,'ac'),  mn(L2,'ac',tab),  mn(L2,'cr'),  mn(L2,'faith'))
print("L2b (구조표) ", mn(rows,'ac'),mn(rows,'ac',tab),mn(rows,'cr'),mn(rows,'faith'))
print("L2B_DONE")
