import os
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
import torch
torch.backends.cudnn.enabled = False   # GB10 conv2d 엔진 오류 회피
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True)
model = AutoModel.from_pretrained('baidu/Unlimited-OCR', trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()
print("MODEL_LOADED")
os.makedirs('data/ocr_out', exist_ok=True)
res = model.infer(tok, prompt='<image>document parsing.', image_file='data/pages/p21.png',
        output_path='data/ocr_out', base_size=1024, image_size=640, crop_mode=True, max_length=32768)
print("INFER RESULT TYPE:", type(res))
if isinstance(res,str): print("=== OCR 출력(앞부분) ===\n"+res[:1200])
print("=== output_path 파일 ===")
for f in os.listdir('data/ocr_out'): print(" ", f)
print("OCR_TEST_DONE")
