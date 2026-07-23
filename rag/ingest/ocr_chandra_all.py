import os, torch, gc, glob, re
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
torch.backends.cudnn.enabled=False
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
os.makedirs('data/ocr_chandra',exist_ok=True)
mid="datalab-to/chandra-ocr-2"
prompt="이 페이지의 모든 텍스트와 표를 정확히 추출하라. 표는 HTML <table>로 행과 열 구조(rowspan/colspan 병합 포함)를 정확히 유지하라. 원문 순서를 지켜라."
proc=AutoProcessor.from_pretrained(mid,trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(mid,trust_remote_code=True,dtype=torch.bfloat16,device_map='cuda').eval()
pages=sorted(glob.glob('data/pages/p*.png'),key=lambda p:int(re.search(r'p(\d+)',p).group(1)))
print("PAGES",len(pages),flush=True)
for pg in pages:
    n=int(re.search(r'p(\d+)',pg).group(1)); out=f'data/ocr_chandra/p{n}.md'
    if os.path.exists(out) and os.path.getsize(out)>0: print(f"p{n} skip",flush=True); continue
    img=Image.open(pg).convert('RGB')
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
    inp.pop('token_type_ids',None)
    with torch.no_grad(): o=model.generate(**inp,max_new_tokens=4096,do_sample=False)
    txt=proc.decode(o[0][inp['input_ids'].shape[1]:],skip_special_tokens=True)
    open(out,'w',encoding='utf-8').write(txt); print(f"p{n} len {len(txt)}",flush=True)
print("OCR_ALL_DONE")
