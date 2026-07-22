import fitz, os
os.makedirs("data/pages", exist_ok=True)
doc=fitz.open("data/db_silson_2014.pdf")
for pno in [19,21,26]:
    pix=doc[pno-1].get_pixmap(dpi=200)
    pix.save(f"data/pages/p{pno}.png")
    print("rendered", f"p{pno}.png", pix.width,"x",pix.height)
