import os, io, contextlib
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
import torch
torch.backends.cudnn.enabled=False
import fitz
from transformers import AutoModel, AutoTokenizer
os.makedirs('data/ocr',exist_ok=True); os.makedirs('data/pages',exist_ok=True); os.makedirs('data/ocr_out',exist_ok=True)
doc=fitz.open('data/db_silson_2014.pdf')
tok=AutoTokenizer.from_pretrained('baidu/Unlimited-OCR',trust_remote_code=True)
model=AutoModel.from_pretrained('baidu/Unlimited-OCR',trust_remote_code=True,use_safetensors=True,dtype=torch.bfloat16).eval().cuda()
print("MODEL_LOADED",flush=True)
for i in range(len(doc)):
    pno=i+1; img=f'data/pages/p{pno}.png'
    if not os.path.exists(img): doc[i].get_pixmap(dpi=200).save(img)
    buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            model.infer(tok,prompt='<image>document parsing.',image_file=img,output_path='data/ocr_out',base_size=1024,image_size=640,crop_mode=True,max_length=32768)
    except Exception as e:
        buf.write(f"[OCR_ERROR p{pno}: {e}]")
    out=buf.getvalue()
    open(f'data/ocr/p{pno}.md','w',encoding='utf-8').write(out)
    print(f"OCR p{pno}: {len(out)} chars",flush=True)
print("OCR_ALL_DONE",flush=True)
