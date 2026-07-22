import re, json
import lxml.html as LH
from apted import APTED, Config
from rapidfuzz.distance import Levenshtein

class N:
    def __init__(s,tag,content=""): s.tag=tag; s.content=content; s.children=[]
def html_tree(html):
    root=N("root")
    try: t=LH.fromstring(html).xpath("//table")[0]
    except Exception: return root
    tab=N("table")
    for tr in t.xpath(".//tr"):
        r=N("tr")
        for td in tr.xpath(".//td|.//th"):
            r.children.append(N("td", re.sub(r"\s+","",td.text_content() or "")))
        tab.children.append(r)
    root.children.append(tab); return root
def size(n): return 1+sum(size(c) for c in n.children)
class Cfg(Config):
    def rename(s,a,b):
        if a.tag!=b.tag: return 1.0
        if a.tag=="td" and (a.content or b.content):
            return Levenshtein.distance(a.content,b.content)/max(len(a.content),len(b.content),1)
        return 0.0
    def children(s,n): return n.children
def teds(pred,true):
    tp,tt=html_tree(pred),html_tree(true)
    d=APTED(tp,tt,Cfg()).compute_edit_distance()
    return 1 - d/max(size(tp),size(tt))
def struct(html):
    try: t=LH.fromstring(html).xpath("//table")[0]
    except: return (0,0)
    rows=t.xpath(".//tr"); return (len(rows), max((len(r.xpath('.//td|.//th')) for r in rows),default=0))

# 기준정답: 공제금액 선택형 (내가 약관 읽고 작성한 정답 구조)
REF_SUNTAEK = """<table>
<tr><td>구분</td><td>항목</td><td>공제금액</td></tr>
<tr><td>선택형</td><td>외래의료비 의원 치과의원 한의원 조산원 보건소 보건의료원 보건지소 보건진료소</td><td>1만원</td></tr>
<tr><td></td><td>종합병원 병원 치과병원 한방병원 요양병원</td><td>1만5천원</td></tr>
<tr><td></td><td>종합전문요양기관 또는 상급종합병원</td><td>2만원</td></tr>
<tr><td>약제의료비 처방조제비</td><td>약국 한국희귀의약품센터에서의 처방 조제</td><td>8천원</td></tr></table>"""

blocks=[json.loads(l) for l in open('data/parsed/blocks.jsonl',encoding='utf-8')]
tables=[b for b in blocks if b['type']=='table']
# 선택형 공제금액 OCR 표 찾기: 8천원 + 1만원 포함하고 "20%"는 없는(선택형) 표
def score_match(t):
    c=t['content']; return ('8천원' in c) + ('1만원' in c) + ('2만원' in c) - ('20%' in c)*0.5
cand=sorted(tables,key=score_match,reverse=True)[:3]
print("=== 후보 표(선택형 공제금액) TEDS ===")
for t in cand:
    sc=teds(t['content'],REF_SUNTAEK)
    print(f"p{t['page']}: TEDS={sc:.3f}  구조(행,열)OCR={struct(t['content'])} vs REF={struct(REF_SUNTAEK)}")
# 전체 표 구조 통계
print("=== 표 50개 구조 통계 ===")
from collections import Counter
shapes=Counter(struct(t['content']) for t in tables)
print("최다 (행,열):", shapes.most_common(6))
print("빈 표(파싱실패):", sum(1 for t in tables if struct(t['content'])==(0,0)))
print("TEDS_DONE")
