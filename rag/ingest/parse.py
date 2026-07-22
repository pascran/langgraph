import os, re, json, glob
from collections import Counter
os.makedirs('data/parsed', exist_ok=True)
pat = re.compile(r'<\|det\|>\s*(\w+)\s*\[([\d,\s]+)\]\s*<\|/det\|>\s*(.*)', re.S)
files = sorted(glob.glob('data/ocr/p*.md'), key=lambda f: int(re.search(r'p(\d+)', f).group(1)))
blocks=[]
for f in files:
    page=int(re.search(r'p(\d+)',f).group(1))
    txt=open(f,encoding='utf-8').read()
    order=0
    for part in re.split(r'(?=<\|det\|>)', txt):
        m=pat.search(part.strip())
        if not m: continue
        typ,bbox,content=m.group(1),m.group(2),m.group(3).strip()
        if not content: continue
        blocks.append({"page":page,"order":order,"type":typ,
            "bbox":[int(x) for x in re.findall(r'\d+',bbox)],"content":content})
        order+=1
with open('data/parsed/blocks.jsonl','w',encoding='utf-8') as w:
    for b in blocks: w.write(json.dumps(b,ensure_ascii=False)+'\n')
print("PAGES",len(files),"BLOCKS",len(blocks))
print("TYPES",dict(Counter(b['type'] for b in blocks)))
tbl=[b for b in blocks if b['type']=='table']
print("TABLES",len(tbl))
if tbl: print("표 예시(p{}):".format(tbl[0]['page']), tbl[0]['content'][:120])
print("PARSE_DONE")
