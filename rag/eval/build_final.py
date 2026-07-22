import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m, chat, sv, cli   # chat=생성 8B :8001
from openai import OpenAI
from qdrant_client import models
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
critic=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"  # 심판 32B :8002
def jchat(sy,us,mx=120):
    r=critic.chat.completions.create(model=CM,messages=[{"role":"system","content":sy},{"role":"user","content":us}],temperature=0,max_tokens=mx,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    return r.choices[0].message.content.strip()
C="silson_v2_sem"; golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
def html2md(h):
    rows=re.findall(r'<tr[^>]*>(.*?)</tr>',h,re.S|re.I); out=[]
    for i,row in enumerate(rows):
        cells=[re.sub(r'<[^>]+>',' ',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',row,re.S|re.I)]
        if not cells: continue
        out.append('| '+' | '.join(cells)+' |')
        if i==0: out.append('|'+'|'.join(['---']*len(cells))+'|')
    return '\n'.join(out) if out else re.sub(r'<[^>]+>',' ',h)
_bl=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
def build_pages(tblfmt):
    pg=defaultdict(list)
    for b in sorted(_bl,key=lambda b:(b['page'],b['order'])):
        if b['type'] in ('header','page_number'): continue
        if b['type']=='table': pg[b['page']].append("[표]\n"+(html2md(b['content']) if tblfmt=='md' else b['content'].strip()))
        else: pg[b['page']].append(re.sub(r'<[^>]+>',' ',b['content']).strip())
    return {p:'\n'.join(v) for p,v in pg.items()}
PST=build_pages('html'); PMD=build_pages('md')
enc_lock=threading.Lock()
def enc(q):
    with enc_lock: return m.encode([q],return_dense=True,return_sparse=True)
def embed(t):
    with enc_lock: return m.encode(t,return_dense=True)['dense_vecs']
def hybrid(q,k=5):
    e=enc(q); pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
    return sorted({p for x in pts for p in (x.payload.get('pages') or [])})
def parent(pm,pages): return [f"[p{n}]\n"+pm.get(n,'')[:3500] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행과 열을 정확히 대응시켜 수치를 정확히 읽어라. 면책/보상 명확히. 없으면 '약관에서 확인 불가'."
VSYS="다음 [답변]의 수치·금액·기간·비율이 [문맥]의 표·조항과 정확히 일치하는지 검토하라. 틀린 값이 있으면 문맥 기준으로 교정하고 근거 [pN]을 유지한 최종 답변만 출력. 맞으면 그대로 출력."
def gen(ctx,q): return chat(SYS,"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}")
def verify(a,ctx,q): return chat(VSYS,"[문맥]\n"+"\n\n".join(ctx)[:3500]+f"\n\n[질문]{q}\n\n[답변]{a}")
def one(g):
    pages=hybrid(g['question'],5); ch=parent(PST,pages); cm=parent(PMD,pages)
    ab=gen(ch,g['question']); ac=gen(cm,g['question']); a5=verify(ac,cm,g['question'])
    return {"q":g['question'],"type":g.get('type'),"gt":g['answer'],"L2b":[ab,ch],"L2c":[ac,cm],"L5":[a5,cm]}
with ThreadPoolExecutor(max_workers=6) as ex: D=list(ex.map(one,golden))
json.dump(D,open('data/ragas/stack32_answers.json','w',encoding='utf-8'),ensure_ascii=False)
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
def scfg(key):
    def sc(it): a,ctx=it[key]; return {"type":it['type'],"cr":crecall(it['gt'],ctx),"faith":faith(a,ctx),"ac":acorr(a,it['gt'])}
    with ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(sc,D))
    return rows
res={}
for key in ['L2b','L2c','L5']:
    rows=scfg(key); res[key]=rows
    def mn(k,f=lambda r:True): xs=[r[k] for r in rows if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
    tab=lambda r:r['type']=='table'
    print(f"{key} (32B심판)  전체ac={mn('ac')}  표ac={mn('ac',tab)}  cr={mn('cr')}  faith={mn('faith')}",flush=True)
    json.dump({k:v for k,v in res.items()},open('data/ragas/stack32_scores.json','w',encoding='utf-8'),ensure_ascii=False)
print("FINAL_DONE")
