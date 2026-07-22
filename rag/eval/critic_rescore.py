import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
critic=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"
def jchat(sy,us,mx=120):
    r=critic.chat.completions.create(model=CM,messages=[{"role":"system","content":sy},{"role":"user","content":us}],temperature=0,max_tokens=mx,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    return r.choices[0].message.content.strip()
enc_lock=threading.Lock()
def embed(t):
    with enc_lock: return m.encode(t,return_dense=True)['dense_vecs']
def jparse(s):
    mm=re.search(r'\{[^{}]*\}',s,re.S)
    if mm:
        for c in (mm.group(0),mm.group(0).replace("'",'"')):
            try: return json.loads(c)
            except: pass
    return None
def faith(a,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(jchat("답변을 단순 사실문장으로 나눈 뒤 각 문장이 문맥에서 추론가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[답변]{a}\n\n형식만: {{\"supported\":정수,\"total\":정수}}"))
    return (r['supported']/r['total']) if r and r.get('total') else float('nan')
def crecall(gt,ctx):
    c="\n".join(ctx)[:4000]; r=jparse(jchat("정답을 사실문장으로 나눈 뒤 각 문장이 문맥에 귀속가능한지 세어 JSON만.",f"[문맥]\n{c}\n\n[정답]{gt}\n\n형식만: {{\"attributable\":정수,\"total\":정수}}"))
    return (r['attributable']/r['total']) if r and r.get('total') else float('nan')
def acorr(a,gt):
    r=jparse(jchat("답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.",f"[정답]{gt}\n[답변]{a}\n형식만: {{\"tp\":정수,\"fp\":정수,\"fn\":정수}}",80)); f1=float('nan')
    if r:
        tp,fp,fn=r.get('tp',0),r.get('fp',0),r.get('fn',0); d=tp+0.5*(fp+fn); f1=tp/d if d else 0.0
    v=embed([a,gt]); return (0.75*f1+0.25*float(np.dot(v[0],v[1]))) if f1==f1 else 0.25*float(np.dot(v[0],v[1]))
lad=json.load(open('data/ragas/ladder.json')); res={}
for name in ['L0','L1','L2','L3','L4']:
    def one(it): return {"cr":crecall(it['ground_truth'],it['contexts']),"faith":faith(it['answer'],it['contexts']),"ac":acorr(it['answer'],it['ground_truth'])}
    with ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(one,lad[name]))
    def mn(k): xs=[r[k] for r in rows if r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
    res[name]={"context_recall":mn('cr'),"faithfulness":mn('faith'),"answer_correctness":mn('ac')}
    json.dump(res,open('data/ragas/critic_scores.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print(name,res[name],flush=True)
print("CRITIC_RESCORE_DONE")
