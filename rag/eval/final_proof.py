import os, json, re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
GEN=OpenAI(base_url="http://localhost:8001/v1",api_key="x",max_retries=1,timeout=150); GM="bottlecapai/ThinkingCap-Qwen3.6-27B-FP8"
CRIT=OpenAI(base_url="http://localhost:8002/v1",api_key="x",max_retries=1,timeout=90); CM="Qwen/Qwen3-32B-AWQ"
D=json.load(open('data/ragas/stack32_answers.json',encoding='utf-8'))
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행/열 정확히 읽어라. 계산 필요시 단계적으로 계산해 최종 수치 명시. 없으면 '약관에서 확인 불가'."
def gen27(ctx,q):
    r=GEN.chat.completions.create(model=GM,messages=[{"role":"system","content":SYS},{"role":"user","content":"[문맥]\n"+"\n\n".join(ctx)+f"\n\n[질문]{q}"}],temperature=0.6,top_p=0.95,max_tokens=1536)
    return (r.choices[0].message.content or "").strip()
def f1(a,gt):
    r=CRIT.chat.completions.create(model=CM,messages=[{"role":"user","content":f"답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.\n[정답]{gt}\n[답변]{a}\n형식만:{{\"tp\":정수,\"fp\":정수,\"fn\":정수}}"}],temperature=0,max_tokens=80,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    mm=re.search(r'\{[^{}]*\}',r.choices[0].message.content,re.S)
    if mm:
        try:
            d=json.loads(mm.group(0)); tp,fp,fn=d.get('tp',0),d.get('fp',0),d.get('fn',0); den=tp+0.5*(fp+fn); return tp/den if den else 0.0
        except: pass
    return float('nan')
def one(it):
    q=it['q']; gt=it['gt']; ctx=it['L2c'][1] if isinstance(it['L2c'],list) else it['L2c']
    a8=it['L2c'][0] if isinstance(it['L2c'],list) else ""
    a27=gen27(ctx,q)
    return {"type":it.get('type'),"q":q,"f1_8b":f1(a8,gt),"f1_27b":f1(a27,gt),"a27":a27,"gt":gt}
with ThreadPoolExecutor(max_workers=3) as ex: R=list(ex.map(one,D))
json.dump(R,open('data/ragas/final_proof.json','w',encoding='utf-8'),ensure_ascii=False)
def mn(k,f=lambda r:True): xs=[r[k] for r in R if f(r) and r[k]==r[k]]; return round(sum(xs)/len(xs),3) if xs else None
tab=lambda r:r['type']=='table'
print("=== 최종증명: 8B vs 27B thinking (동일 L2c 컨텍스트, 32B F1 심판) ===")
print(f"8B  F1 전체={mn('f1_8b')}  표={mn('f1_8b',tab)}")
print(f"27B F1 전체={mn('f1_27b')}  표={mn('f1_27b',tab)}")
print("--- 개선 큰 질문 top ---")
for r in sorted(R,key=lambda r:(r['f1_27b'] or 0)-(r['f1_8b'] or 0),reverse=True)[:4]:
    print(f"  8B={r['f1_8b']:.2f}→27B={r['f1_27b']:.2f}  {r['q'][:38]}")
print("PROOF_DONE")
