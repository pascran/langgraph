import os, json, re, threading
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from openai import OpenAI
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True); cli=QdrantClient(url="http://localhost:6333"); C="silson_v2_sem"
GEN=OpenAI(base_url="http://localhost:8001/v1",api_key="x"); GM="bottlecapai/ThinkingCap-Qwen3.6-27B-FP8"
sv=lambda lw: models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
golden=[json.loads(l) for l in open('data/golden/silson_golden.jsonl',encoding='utf-8')]
def html2md(h):
    rows=[[re.sub(r'<[^>]+>',' ',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',row,re.S|re.I)] for row in re.findall(r'<tr[^>]*>(.*?)</tr>',h,re.S|re.I)]
    rows=[r for r in rows if r]
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
def parent(q):
    with lock: e=m.encode([q],return_dense=True,return_sparse=True)
    pts=cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=5,with_payload=True).points
    pages=sorted({p for x in pts for p in (x.payload.get('pages') or [])})
    return [f"[p{n}]\n"+PMD.get(n,'')[:3500] for n in pages] or ["(없음)"]
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행/열 정확히 읽어라. 계산 필요시 단계적으로 계산해 최종 수치 명시. 없으면 '약관에서 확인 불가'."
def one(g):
    ctx=parent(g['question'])
    r=GEN.chat.completions.create(model=GM,messages=[{"role":"system","content":SYS},{"role":"user","content":"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{g['question']}"}],temperature=0.6,top_p=0.95,max_tokens=1536)
    return {"q":g['question'],"type":g.get('type'),"gt":g['answer'],"ctx":ctx,"a":(r.choices[0].message.content or "").strip()}
with ThreadPoolExecutor(max_workers=4) as ex: rows=list(ex.map(one,golden))
json.dump(rows,open('data/ragas/ans_27b.json','w',encoding='utf-8'),ensure_ascii=False)
print("GEN_DONE",len(rows),"| 빈답변:",sum(1 for r in rows if not r['a']))
