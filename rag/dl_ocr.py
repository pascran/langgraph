from huggingface_hub import snapshot_download
print("DOWNLOADING baidu/Unlimited-OCR", flush=True)
snapshot_download("baidu/Unlimited-OCR")
print("OCR_DONE", flush=True)
