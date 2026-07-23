import os, json, re, threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
CRIT=OpenAI(base_url="http://localhost:8002/v1",api_key="x",max_retries=2,timeout=90); CM="Qwen/Qwen3-32B-AWQ"
D=json.load(open('data/ragas/stack32_answers.json',encoding='utf-8'))
A27={r['q']:r['a27'] for r in json.load(open('data/ragas/answers_27b.json',encoding='utf-8')) if r.get('a27')}
sub=[it for it in D if it['q'] in A27]   # 27B 실답 있는 문항만
print(f"공정비교 대상: {len(sub)}/{len(D)} 문항 (27B 실답 있음)",flush=True)
def f1(a,gt):
    r=CRIT.chat.completions.create(model=CM,messages=[{"role":"user","content":f"답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.\n[정답]{gt}\n[답변]{a}\n형식만:{{\"tp\":정수,\"fp\":정수,\"fn\":정수}}"}],temperature=0,max_tokens=80,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    mm=re.search(r'\{[^{}]*\}',r.choices[0].message.content,re.S)
    if mm:
        try:
            d=json.loads(mm.group(0)); tp,fp,fn=d.get('tp',0),d.get('fp',0),d.get('fn',0); den=tp+0.5*(fp+fn); return tp/den if den else 0.0
        except: pass
    return float('nan')
def one(it):
    return {"type":it.get('type'),"q":it['q'],"f8":f1(it['L2c'][0],it['gt']),"f27":f1(A27[it['q']],it['gt'])}
with ThreadPoolExecutor(max_workers=4) as ex: R=list(ex.map(one,sub))
json.dump(R,open('data/ragas/proof_scores.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(k,f=lambda r:True): xs=[r[k] for r in R if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
tab=lambda r:r['type']=='table'
print(f"\n=== 최종증명 ({len(sub)}문항, 동일 컨텍스트, 32B F1 심판) ===")
print(f"8B  F1: 전체={mn('f8')}  표={mn('f8',tab)}")
print(f"27B F1: 전체={mn('f27')}  표={mn('f27',tab)}")
print("--- 27B 개선 top ---")
for r in sorted(R,key=lambda r:(r['f27'] or 0)-(r['f8'] or 0),reverse=True)[:5]:
    print(f"  8B={r['f8']:.2f}→27B={r['f27']:.2f}  {r['q'][:36]}")
print("BATCH_JUDGE_DONE")
