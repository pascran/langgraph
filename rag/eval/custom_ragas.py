import os, sys, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
sys.path.insert(0,'.')
from rag.graph.agentic_rag import m, chat
from concurrent.futures import ThreadPoolExecutor
import numpy as np
lad=json.load(open('data/ragas/ladder.json'))
enc_lock=threading.Lock()
def embed(txts):
    with enc_lock: return m.encode(txts,return_dense=True)['dense_vecs']
def jparse(s):
    mm=re.search(r'\{[^{}]*\}',s,re.S)
    if mm:
        for c in (mm.group(0), mm.group(0).replace("'",'"')):
            try: return json.loads(c)
            except: pass
    return None
def faith(ans,ctx):   # RAGAS faithfulness: 답변 주장 중 문맥근거 비율
    c="\n".join(ctx)[:4000]
    r=jparse(chat("답변을 단순 사실문장으로 나눈 뒤, 각 문장이 문맥에서 추론가능한지 세어 JSON만 출력.",f"[문맥]\n{c}\n\n[답변]{ans}\n\n형식만: {{\"supported\":정수,\"total\":정수}}",100))
    return min(r['supported']/r['total'],1.0) if r and r.get('total') else float('nan')  # 클램프: supported>total 방지
def crecall(gt,ctx):  # RAGAS context_recall: 정답 문장 중 문맥귀속 비율
    c="\n".join(ctx)[:4000]
    r=jparse(chat("정답을 사실문장으로 나눈 뒤, 각 문장이 문맥에 귀속가능한지 세어 JSON만 출력.",f"[문맥]\n{c}\n\n[정답]{gt}\n\n형식만: {{\"attributable\":정수,\"total\":정수}}",100))
    return min(r['attributable']/r['total'],1.0) if r and r.get('total') else float('nan')  # 클램프
def acorr(ans,gt):    # RAGAS answer_correctness: 0.75*F1(사실) + 0.25*의미유사도
    r=jparse(chat("답변과 정답을 사실 단위로 비교. tp=양쪽공통, fp=답변에만, fn=정답에만. JSON만.",f"[정답]{gt}\n[답변]{ans}\n형식만: {{\"tp\":정수,\"fp\":정수,\"fn\":정수}}",80))
    f1=float('nan')
    if r:
        tp,fp,fn=r.get('tp',0),r.get('fp',0),r.get('fn',0); d=tp+0.5*(fp+fn); f1=tp/d if d else 0.0
    v=embed([ans,gt]); sim=float(np.dot(v[0],v[1]))
    return (0.75*f1+0.25*sim) if f1==f1 else 0.25*sim
def one(it):
    return {"q":it['question'],"cr":crecall(it['ground_truth'],it['contexts']),"faith":faith(it['answer'],it['contexts']),"ac":acorr(it['answer'],it['ground_truth'])}
res={}
for name,items in lad.items():
    with ThreadPoolExecutor(max_workers=6) as ex: rows=list(ex.map(one,items))
    def mean(k): xs=[r[k] for r in rows if r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
    res[name]={"context_recall":mean("cr"),"faithfulness":mean("faith"),"answer_correctness":mean("ac")}
    json.dump(rows,open(f'data/ragas/{name}_custom.json','w',encoding='utf-8'),ensure_ascii=False)
    json.dump(res,open('data/ragas/custom_scores.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
    print(name,res[name],flush=True)
print("CUSTOM_DONE")
