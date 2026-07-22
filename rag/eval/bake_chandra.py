import os, torch, gc, traceback
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
torch.backends.cudnn.enabled=False   # GB10 conv2d 우회
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
PAGES=['data/pages/p21.png','data/pages/p23.png']
mid="datalab-to/chandra-ocr-2"; tag="chandra2"
prompt="이 페이지의 모든 텍스트와 표를 마크다운으로 변환하라. 표는 행과 열 구조를 정확히 유지하라."
try:
    proc=AutoProcessor.from_pretrained(mid,trust_remote_code=True)
    model=AutoModelForImageTextToText.from_pretrained(mid,trust_remote_code=True,dtype=torch.bfloat16,device_map='cuda').eval()
    for pg in PAGES:
        img=Image.open(pg).convert('RGB')
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
        inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        inp.pop('token_type_ids',None)
        with torch.no_grad(): out=model.generate(**inp,max_new_tokens=4096,do_sample=False)
        txt=proc.decode(out[0][inp['input_ids'].shape[1]:],skip_special_tokens=True)
        open(f"data/ocr_bake/{tag}_{os.path.basename(pg)}.md","w",encoding="utf-8").write(txt)
        print(tag,pg,"len",len(txt),flush=True)
except Exception as e:
    print(tag,"FAIL",str(e)[:150],flush=True); traceback.print_exc()
print("CHANDRA_DONE")
