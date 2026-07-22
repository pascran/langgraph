import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m, chat, sv, cli
from openai import OpenAI
from qdrant_client import models
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
critic=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"
def jchat(sy,us,mx=120):
    r=critic.chat.completions.create(model=CM,messages=[{"role":"system","content":sy},{"role":"user","content":us}],temperature=0,max_tokens=mx,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    return r.choices[0].message.content.strip()
C="silson_v2_sem"; golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
def html2rec(h):   # ① 행 단위 레코드: "행키: 열=값, 열=값"
    rows=[]
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>',h,re.S|re.I):
        cells=[re.sub(r'<[^>]+>',' ',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',row,re.S|re.I)]
        if cells: rows.append(cells)
    if len(rows)<2: return re.sub(r'<[^>]+>',' ',h)
    hdr=rows[0]; out=[]
    for r in rows[1:]:
        key=r[0] if r else ''
        pairs=[f"{hdr[j]}={r[j]}" for j in range(1,len(r)) if j<len(hdr) and r[j]]
        out.append(f"- {key}: "+", ".join(pairs))
    return f"표(행별, 기준={hdr[0]}):\n"+"\n".join(out)
_bl=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
pg=defaultdict(list)
for b in sorted(_bl,key=lambda b:(b['page'],b['order'])):
    if b['type'] in ('header','page_number'): continue
    pg[b['page']].append("[표]\n"+html2rec(b['content']) if b['type']=='table' else re.sub(r'<[^>]+>',' ',b['content']).strip())
PREC={p:'\n'.join(v) for p,v in pg.items()}
enc_lock=threading.Lock()
def enc(q):
    with enc_lock: return m.encode([q],return_dense=True,return_sparse=True)
def embed(t):
    with enc_lock: return m.encode(t,return_dense=True)['dense_vecs']
def hybrid(q,k=5):
    e=enc(q); pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
    return sorted({p for x in pts for p in (x.payload.get('pages') or [])})
def parent(pages): return [f"[p{n}]\n"+PREC.get(n,'')[:3500] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행별 레코드로 제공되니 해당 행의 값을 정확히 읽어라. 면책/보상 명확히. 없으면 '약관에서 확인 불가'."
def gen(ctx,q): return chat(SYS,"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}")
def one(g):
    ctx=parent(hybrid(g['question'],5)); a=gen(ctx,g['question'])
    return {"type":g.get('type'),"gt":g['answer'],"a":a,"ctx":ctx}
with ThreadPoolExecutor(max_workers=6) as ex: D=list(ex.map(one,golden))
print("GEN_DONE",len(D),flush=True)
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
def sc(it): return {"type":it['type'],"cr":crecall(it['gt'],it['ctx']),"faith":faith(it['a'],it['ctx']),"ac":acorr(it['a'],it['gt'])}
with ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(sc,D))
json.dump(rows,open('data/ragas/L6rec_custom.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(rs,k,f=lambda r:True): xs=[r[k] for r in rs if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
tab=lambda r:r['type']=='table'
st=json.load(open('data/ragas/stack32_scores.json',encoding='utf-8'))
print("            전체ac  표ac   cr     faith")
for k in ['L2b','L2c','L5']:
    print(f"{k:11s} {mn(st[k],'ac')} {mn(st[k],'ac',tab)} {mn(st[k],'cr')} {mn(st[k],'faith')}")
print(f"L6 records  {mn(rows,'ac')} {mn(rows,'ac',tab)} {mn(rows,'cr')} {mn(rows,'faith')}")
print("RECORDS_DONE")
