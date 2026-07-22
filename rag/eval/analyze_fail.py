import json
for cfg in ['L2','L4']:
    rows=json.load(open(f'data/ragas/{cfg}_custom.json',encoding='utf-8'))
    b={'ok':0,'검색실패':0,'생성환각':0,'코퍼스/부분':0}; fails=[]
    for r in rows:
        ac,cr,f=r['ac'],r['cr'],r['faith']
        if ac>=0.6: b['ok']+=1
        elif cr<0.6: b['검색실패']+=1; fails.append((r['q'][:26],'검색',round(cr,2),round(f,2),round(ac,2)))
        elif f<0.7: b['생성환각']+=1; fails.append((r['q'][:26],'생성',round(cr,2),round(f,2),round(ac,2)))
        else: b['코퍼스/부분']+=1; fails.append((r['q'][:26],'코퍼스',round(cr,2),round(f,2),round(ac,2)))
    print(f"=== {cfg}: {b}")
    for x in fails[:7]: print(f"   {x[1]:4s} cr={x[2]} f={x[3]} ac={x[4]}  {x[0]}")
