import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
m=BGEM3FlagModel("BAAI/bge-m3",use_fp16=False,devices="cpu")
cli=QdrantClient(url="http://localhost:6333")
sv=lambda lw: models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
from qdrant_client import models
from openai import OpenAI
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
GEN=OpenAI(base_url="http://localhost:8001/v1",api_key="x"); GM="bottlecapai/ThinkingCap-Qwen3.6-27B-FP8"
CRIT=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"
C="silson_v2_sem"; golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
def html2md(h):
    rows=[]
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>',h,re.S|re.I):
        cs=[re.sub(r'<[^>]+>',' ',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',row,re.S|re.I)]
        if cs: rows.append(cs)
    if len(rows)<2: return re.sub(r'<[^>]+>',' ',h)
    o=[]
    for i,r in enumerate(rows):
        o.append('| '+' | '.join(r)+' |')
        if i==0: o.append('|'+'|'.join(['---']*len(r))+'|')
    return '\n'.join(o)
_bl=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
pg=defaultdict(list)
for b in sorted(_bl,key=lambda b:(b['page'],b['order'])):
    if b['type'] in ('header','page_number'): continue
    pg[b['page']].append("[표]\n"+html2md(b['content']) if b['type']=='table' else re.sub(r'<[^>]+>',' ',b['content']).strip())
PMD={p:'\n'.join(v) for p,v in pg.items()}
lock=threading.Lock()
def enc(q):
    with lock: return m.encode([q],return_dense=True,return_sparse=True)
def embed(t):
    with lock: return m.encode(t,return_dense=True)['dense_vecs']
def hybrid(q,k=5):
    e=enc(q); pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
    return sorted({p for x in pts for p in (x.payload.get('pages') or [])})
def parent(pages): return [f"[p{n}]\n"+PMD.get(n,'')[:3500] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행/열을 정확히 읽어라. 계산이 필요하면 단계적으로 계산해 최종 수치를 명확히 제시하라. 면책/보상 명확히. 없으면 '약관에서 확인 불가'."
def gen27(ctx,q):
    r=GEN.chat.completions.create(model=GM,messages=[{"role":"system","content":SYS},{"role":"user","content":"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}"}],temperature=0.6,top_p=0.95,max_tokens=2048)
    return (r.choices[0].message.content or "").strip()
def jchat(sy,us,mx=120):
    r=CRIT.chat.completions.create(model=CM,messages=[{"role":"system","content":sy},{"role":"user","content":us}],temperature=0,max_tokens=mx,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    return r.choices[0].message.content.strip()
def jparse(s):
    mm=re.search(r'\{[^{}]*\}',s,re.S)
    if mm:
        for c in (mm.group(0),mm.group(0).replace("'",'"')):
            try: return json.loads(c)
            except: pass
    return None
def faith(a,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(jchat("답변을 단순 사실문장으로 나눈 뒤 각 문장이 문맥에서 추론가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[답변]{a}\n\n형식만: {{\"supported\":정수,\"total\":정수}}"))
    return min(r['supported']/r['total'],1.0) if r and r.get('total') else float('nan')
def crecall(gt,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(jchat("정답을 사실문장으로 나눈 뒤 각 문장이 문맥에 귀속가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[정답]{gt}\n\n형식만: {{\"attributable\":정수,\"total\":정수}}"))
    return min(r['attributable']/r['total'],1.0) if r and r.get('total') else float('nan')
def acorr(a,gt):
    r=jparse(jchat("답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.",f"[정답]{gt}\n[답변]{a}\n형식만: {{\"tp\":정수,\"fp\":정수,\"fn\":정수}}",80)); f1=float('nan')
    if r:
        tp,fp,fn=r.get('tp',0),r.get('fp',0),r.get('fn',0); d=tp+0.5*(fp+fn); f1=tp/d if d else 0.0
    v=embed([a,gt]); return (0.75*f1+0.25*float(np.dot(v[0],v[1]))) if f1==f1 else 0.25*float(np.dot(v[0],v[1]))
def one(g):
    ctx=parent(hybrid(g['question'],5)); a=gen27(ctx,g['question'])
    return {"q":g['question'],"type":g.get('type'),"a":a,"cr":crecall(g['answer'],ctx),"faith":faith(a,ctx),"ac":acorr(a,g['answer'])}
with ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(one,golden))
json.dump(rows,open('data/ragas/e2e_27b.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(k,f=lambda r:True): xs=[r[k] for r in rows if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
print("=== 27B thinking 생성 + 32B 심판 (L2c 구성) ===")
print(f"전체 ac={mn('ac')} cr={mn('cr')} faith={mn('faith')}")
print(f"표type ac={mn('ac',lambda r:r['type']=='table')}")
print("--- 계산/추론 질문 확인 ---")
for r in rows:
    if '자기부담' in r['q'] or '3만원' in r['q']:
        print(f"[ac={r['ac']:.2f} f={r['faith']:.2f}] {r['q'][:40]}\n   답변: {r['a'][:120]}")
print("E2E_DONE")
