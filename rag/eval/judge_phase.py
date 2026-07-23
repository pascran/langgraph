import os, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'; os.environ['CUDA_VISIBLE_DEVICES']=''
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=False,devices='cpu')
CRIT=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"
rows=json.load(open('data/ragas/ans_27b.json',encoding='utf-8'))
lock=threading.Lock()
def emb(t):
    with lock: return m.encode(t,return_dense=True)['dense_vecs']
def jchat(us,mx=120):
    r=CRIT.chat.completions.create(model=CM,messages=[{"role":"user","content":us}],temperature=0,max_tokens=mx,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    return r.choices[0].message.content.strip()
def jp(s):
    mm=re.search(r'\{[^{}]*\}',s,re.S)
    if mm:
        for c in (mm.group(0),mm.group(0).replace("'",'"')):
            try: return json.loads(c)
            except: pass
    return None
def score(it):
    a,ctx,gt=it['a'],it['ctx'],it['gt']; c="\n".join(ctx)[:4000]
    rf=jp(jchat(f"답변을 단순 사실문장으로 나눈 뒤 각 문장이 문맥에서 추론가능한지 세어 JSON만.\n[문맥]\n{c}\n[답변]{a}\n형식만:{{\"supported\":정수,\"total\":정수}}"))
    faith=min(rf['supported']/rf['total'],1.0) if rf and rf.get('total') else float('nan')
    rc=jp(jchat(f"정답을 사실문장으로 나눈 뒤 각 문장이 문맥에 귀속가능한지 세어 JSON만.\n[문맥]\n{c}\n[정답]{gt}\n형식만:{{\"attributable\":정수,\"total\":정수}}"))
    cr=min(rc['attributable']/rc['total'],1.0) if rc and rc.get('total') else float('nan')
    ra=jp(jchat(f"답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.\n[정답]{gt}\n[답변]{a}\n형식만:{{\"tp\":정수,\"fp\":정수,\"fn\":정수}}",80)); f1=float('nan')
    if ra:
        tp,fp,fn=ra.get('tp',0),ra.get('fp',0),ra.get('fn',0); d=tp+0.5*(fp+fn); f1=tp/d if d else 0.0
    v=emb([a,gt]); ac=(0.75*f1+0.25*float(np.dot(v[0],v[1]))) if f1==f1 else 0.25*float(np.dot(v[0],v[1]))
    return {"type":it['type'],"cr":cr,"faith":faith,"ac":ac}
with ThreadPoolExecutor(max_workers=3) as ex: sc=list(ex.map(score,rows))
json.dump(sc,open('data/ragas/scores_27b.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(k,f=lambda r:True): xs=[r[k] for r in sc if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
tab=lambda r:r['type']=='table'
print("=== 27B thinking (L2c 구성, 32B 심판) ===")
print(f"전체 ac={mn('ac')} cr={mn('cr')} faith={mn('faith')}  | 표type ac={mn('ac',tab)}")
print("=== 8B 기준선(stack32 L2c): ac=0.644 cr=0.949 faith=0.938 표ac=0.604 ===")
print("JUDGE_DONE")
