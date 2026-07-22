import os, sys, types, json
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
# shim: ragas가 최상단에서 import하는 제거된 langchain 모듈들을 더미로 대체
for name in ['langchain_community.chat_models.vertexai']:
    mod=types.ModuleType(name); mod.ChatVertexAI=type('ChatVertexAI',(),{}); sys.modules[name]=mod
import ragas; print("ragas", ragas.__version__)
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
llm=LangchainLLMWrapper(ChatOpenAI(base_url="http://localhost:8001/v1",api_key="x",model="Qwen/Qwen3-8B-AWQ",temperature=0,timeout=180,model_kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking":False}}}))
emb=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name='BAAI/bge-m3',model_kwargs={'device':'cuda'}))
lad=json.load(open('data/ragas/ladder.json'))
items=lad['L0'][:3]
ds=EvaluationDataset.from_list([{"user_input":it['question'],"response":it['answer'],"retrieved_contexts":it['contexts'],"reference":it['ground_truth']} for it in items])
r=evaluate(ds,metrics=[LLMContextRecall(),Faithfulness()],llm=llm,embeddings=emb,run_config=RunConfig(max_workers=4))
df=r.to_pandas(); print("cols:",list(df.columns))
print(df[[c for c in df.columns if c in ('context_recall','faithfulness')]].mean())
print("PROBE_OK")
