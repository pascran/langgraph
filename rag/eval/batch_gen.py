import os, json, threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
GEN=OpenAI(base_url="http://localhost:8001/v1",api_key="x",max_retries=1,timeout=120); GM="bottlecapai/ThinkingCap-Qwen3.6-27B-FP8"
D=json.load(open('data/ragas/stack32_answers.json',encoding='utf-8'))
OUT='data/ragas/answers_27b.json'
done={}
if os.path.exists(OUT):
    try: done={r['q']:r for r in json.load(open(OUT,encoding='utf-8'))}
    except: done={}
SYS="약관 문맥에만 근거해 간결히 답하고 근거를 [pN]로. 표는 행/열 정확히 읽어라. 계산 필요시 단계적으로 계산해 최종 수치 명시. 없으면 '약관에서 확인 불가'."
lk=threading.Lock()
def gen27(it):
    q=it['q']
    if q in done and done[q].get('a27'): return
    try:
        r=GEN.chat.completions.create(model=GM,messages=[{"role":"system","content":SYS},{"role":"user","content":"[문맥]\n"+"\n\n".join(it['L2c'][1])+f"\n\n[질문]{q}"}],temperature=0.6,top_p=0.95,max_tokens=1536)
        a=(r.choices[0].message.content or "").strip(); st="ok"
    except Exception as e:
        a=""; st=str(e)[:40]
    with lk:
        done[q]={"q":q,"type":it.get('type'),"gt":it['gt'],"a27":a}
        json.dump(list(done.values()),open(OUT,'w',encoding='utf-8'),ensure_ascii=False)
        print(f"{len(done)}/{len(D)} [{st}] a_len={len(a)}",flush=True)
with ThreadPoolExecutor(max_workers=2) as ex: list(ex.map(gen27,D))
print("BATCH_GEN_DONE",sum(1 for v in done.values() if v.get('a27')))
