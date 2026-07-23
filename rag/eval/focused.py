import os, sys, json, re
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'; os.environ['CUDA_VISIBLE_DEVICES']=''
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from openai import OpenAI
from collections import defaultdict
import numpy as np
m=BGEM3FlagModel("BAAI/bge-m3",use_fp16=False,devices="cpu"); cli=QdrantClient(url="http://localhost:6333"); C="silson_v2_sem"
sv=lambda lw: models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
GEN=OpenAI(base_url="http://localhost:8001/v1",api_key="x"); GM="bottlecapai/ThinkingCap-Qwen3.6-27B-FP8"
CRIT=OpenAI(base_url="http://localhost:8002/v1",api_key="x"); CM="Qwen/Qwen3-32B-AWQ"
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
def html2md(h):
    rows=[[re.sub(r'<[^>]+>',' ',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',row,re.S|re.I)] for row in re.findall(r'<tr[^>]*>(.*?)</tr>',h,re.S|re.I)]
    rows=[r for r in rows if r]
    if len(rows)<2: return re.sub(r'<[^>]+>',' ',h)
    o=[]
    for i,r in enumerate(rows):
        o.append('| '+' | '.join(r)+' |'); 
        if i==0: o.append('|'+'|'.join(['---']*len(r))+'|')
    return '\n'.join(o)
_bl=[json.loads(l) for l in open('data/parsed/blocks_kie.jsonl',encoding='utf-8')]
pg=defaultdict(list)
for b in sorted(_bl,key=lambda b:(b['page'],b['order'])):
    if b['type'] in ('header','page_number'): continue
    pg[b['page']].append("[표]\n"+html2md(b['content']) if b['type']=='table' else re.sub(r'<[^>]+>',' ',b['content']).strip())
PMD={p:'\n'.join(v) for p,v in pg.items()}
def parent(q):
    e=m.encode([q],return_dense=True,return_sparse=True)
    pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=5,with_payload=True).points
    pages=sorted({p for x in pts for p in (x.payload.get('pages') or [])})
    return [f"[p{n}]\n"+PMD.get(n,'')[:3500] for n in pages], e
def gen27(ctx,q):
    r=GEN.chat.completions.create(model=GM,messages=[{"role":"system","content":"약관 문맥에만 근거해 답하고 근거 [pN]. 계산 필요시 단계적으로 계산해 최종 수치 명시. 없으면 확인불가."},{"role":"user","content":"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}"}],temperature=0.6,top_p=0.95,max_tokens=1536)
    return (r.choices[0].message.content or "").strip()
def acorr(a,gt):
    r=CRIT.chat.completions.create(model=CM,messages=[{"role":"user","content":f"답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.\n[정답]{gt}\n[답변]{a}\n형식만: {{\"tp\":정수,\"fp\":정수,\"fn\":정수}}"}],temperature=0,max_tokens=80,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    mm=re.search(r'\{[^{}]*\}',r.choices[0].message.content,re.S); f1=float('nan')
    if mm:
        try:
            d=json.loads(mm.group(0)); tp,fp,fn=d.get('tp',0),d.get('fp',0),d.get('fn',0); den=tp+0.5*(fp+fn); f1=tp/den if den else 0.0
        except: pass
    return f1
OLD={"자기부담":0.49,"3만원":0.16}
targets=[g for g in golden if any(w in g['question'] for w in ['자기부담','3만원','공제금액','최고한도'])][:5]
print("=== 실패했던/표 질문 27B 재생성 (F1 정확도, 32B 심판) ===")
for g in targets:
    ctx,_=parent(g['question']); a=gen27(ctx,g['question']); f1=acorr(a,g['answer'])
    old=next((v for k,v in OLD.items() if k in g['question']),None)
    print(f"\n[27B F1={f1:.2f}{' (8B였음 '+str(old)+')' if old else ''}] {g['question'][:45]}")
    print(f"  정답: {g['answer'][:60]}")
    print(f"  27B답변: {a[:130]}")
print("FOCUSED_DONE")
