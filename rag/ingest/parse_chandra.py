import re, json, glob, sys
LABEL={'Text':'text','Caption':'title','Section-Header':'title','Table':'table',
       'List-Group':'text','Page-Header':'header','Page-Footer':'page_number','Diagram':'figure'}
def parse_html(html, page):
    blocks=[]; order=0
    for m in re.finditer(r'<div\s+data-bbox="([^"]*)"\s+data-label="([^"]*)"\s*>(.*?)</div>', html, re.S):
        bbox,label,inner=m.group(1),m.group(2),m.group(3)
        t=LABEL.get(label,'text')
        if label=='Table':
            mt=re.search(r'(<table.*?</table>)',inner,re.S)
            content=mt.group(1) if mt else re.sub(r'<[^>]+>',' ',inner).strip(); t='table'
        else:
            content=re.sub(r'<[^>]+>',' ',inner); content=re.sub(r'\s+',' ',content).strip()
        if not content: continue
        blocks.append({"page":page,"order":order,"type":t,"bbox":bbox,"label":label,"content":content}); order+=1
    return blocks
if __name__=="__main__":
    # 검증: bake p21
    html=open('data/ocr_bake/chandra2_p21.png.md',encoding='utf-8').read()
    bl=parse_html(html,21)
    from collections import Counter
    print("블록수:",len(bl),"| 타입:",dict(Counter(b['type'] for b in bl)))
    for b in bl:
        if b['type']=='table':
            print(f"\n[표 블록] rows={b['content'].count('<tr')} rowspan={b['content'].count('rowspan')} colspan={b['content'].count('colspan')}")
            print("표 텍스트 발췌:", re.sub(r'<[^>]+>',' ',b['content'])[:120])
            break
    print("\n[텍스트 블록 샘플]")
    for b in bl[:4]:
        if b['type']!='table': print(f"  ({b['label']}) {b['content'][:70]}")
