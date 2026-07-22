import os
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from openai import OpenAI
m=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)
cli=QdrantClient(url="http://localhost:6333")
llm=OpenAI(base_url="http://localhost:8001/v1",api_key="x")
sv=lambda lw:models.SparseVector(indices=[int(k) for k in lw],values=[float(v) for v in lw.values()])
def retrieve(q,k=5,C="silson_v2_sem"):
    e=m.encode([q],return_dense=True,return_sparse=True)
    return cli.query_points(C,prefetch=[models.Prefetch(query=e['dense_vecs'][0].tolist(),using="dense",limit=30),models.Prefetch(query=sv(e['lexical_weights'][0]),using="sparse",limit=30)],query=models.FusionQuery(fusion=models.Fusion.RRF),limit=k,with_payload=True).points
SYS="당신은 보험약관 전문가입니다. 반드시 아래 [문맥]에만 근거해 답하세요. 문맥에 없으면 '약관에서 확인되지 않습니다'라고 하세요. 답변 끝에 근거를 [pN]로 표기하고 간결히."
Q=["통원 시 상급종합병원 외래 공제금액은?","비만으로 입원하면 보상되나?","입원의료비 보험가입금액 최고한도는?","청약철회는 며칠 이내에 가능한가?"]
for q in Q:
    ctx=retrieve(q,5)
    body="\n\n".join(f"(p{c.payload.get('pages')}) {c.payload['text']}" for c in ctx)
    r=llm.chat.completions.create(model="Qwen/Qwen3-8B-AWQ",messages=[{"role":"system","content":SYS},{"role":"user","content":f"[문맥]\n{body}\n\n[질문] {q}"}],temperature=0,max_tokens=256,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    print("Q:",q); print("A:",r.choices[0].message.content.strip()[:280]); print()
print("GEN_DEMO_OK")
