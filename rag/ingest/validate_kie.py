import os, re, json, fitz
from collections import Counter, defaultdict
blocks=[json.loads(l) for l in open('data/parsed/blocks.jsonl',encoding='utf-8')]
def plain(b): return re.sub(r'<[^>]+>',' ',b['content']) if b['type']=='table' else b['content']
# ===== CER (OCR vs 텍스트레이어, 공백제거) =====
try:
    from rapidfuzz.distance import Levenshtein as LV; lev=LV.distance
except Exception:
    def lev(a,b):
        m,n=len(a),len(b); dp=list(range(n+1))
        for i in range(1,m+1):
            prev=dp[0]; dp[0]=i
            for j in range(1,n+1):
                cur=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=cur
        return dp[n]
norm=lambda s: re.sub(r"\s+","",s)
doc=fitz.open('data/db_silson_2014.pdf')
ocr_pg=defaultdict(list)
for b in blocks:
    if b['type'] in ('text','title','table','ref_text'): ocr_pg[b['page']].append(plain(b))
rows=[]; TR=TE=0
for i in range(len(doc)):
    p=i+1; ref=norm(doc[i].get_text())
    if len(ref)<30: continue
    hyp=norm(''.join(ocr_pg.get(p,[]))); d=lev(ref,hyp)
    rows.append((p,len(ref),d/max(1,len(ref)))); TR+=len(ref); TE+=d
print(f"CER_AGG {TE/max(1,TR):.4f}  (문자정확도 {1-TE/max(1,TR):.2%})  pages={len(rows)}")
print("WORST", [(p,round(c,3)) for p,_,c in sorted(rows,key=lambda x:-x[2])[:6]])
print("BEST ", [(p,round(c,3)) for p,_,c in sorted(rows,key=lambda x:x[2])[:4]])
# ===== KIE (구조+필드) =====
gwan=re.compile(r'제\s*(\d+)\s*관'); johang=re.compile(r'^\s*(\d{1,2})\.\s*\(([^)]{2,25})\)')
dambo=re.compile(r'(신상해입원|상해통원|신질병입원|질병통원)')
money=re.compile(r'[\d,]+\s*(?:만|천)?\s*원'); pct=re.compile(r'\d+\s*%'); period=re.compile(r'\d+\s*(?:일|년|개월|회|건)')
cur={'gwan':None,'johang':None,'dambo':None}; enr=[]; ks=Counter()
for b in sorted(blocks,key=lambda x:(x['page'],x['order'])):
    t=plain(b)
    if (m:=gwan.search(t)): cur['gwan']=f"제{m.group(1)}관"
    if (m:=johang.search(t)): cur['johang']=f"{m.group(1)}.{m.group(2).strip()}"
    if (m:=dambo.search(t)): cur['dambo']=m.group(1)
    f={'money':money.findall(t),'pct':pct.findall(t),'period':period.findall(t)}
    for k,v in f.items():
        if v: ks[k]+=len(v)
    e=dict(b); e['meta']=dict(cur); e['fields']=f; enr.append(e)
with open('data/parsed/blocks_kie.jsonl','w',encoding='utf-8') as w:
    for e in enr: w.write(json.dumps(e,ensure_ascii=False)+"\n")
print("KIE_FIELDS", dict(ks))
covered=sum(1 for e in enr if e['meta']['dambo'] or e['meta']['gwan'])
print(f"메타태깅 블록: {covered}/{len(enr)}")
for e in enr:
    if e['meta'].get('dambo') and e['fields']['money']:
        print(f"SAMPLE p{e['page']} meta={e['meta']} money={e['fields']['money'][:4]}"); break
print("VALIDATE_KIE_DONE")
