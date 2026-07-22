import os, sys, types, json, warnings
warnings.filterwarnings('ignore')
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
for name in ['langchain_community.chat_models.vertexai']:
    mod=types.ModuleType(name); mod.ChatVertexAI=type('ChatVertexAI',(),{}); sys.modules[name]=mod
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, LLMContextRecall, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
llm=LangchainLLMWrapper(ChatOpenAI(base_url="http://localhost:8001/v1",api_key="x",model="Qwen/Qwen3-8B-AWQ",temperature=0,timeout=180,model_kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking":False}}}))
emb=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name='BAAI/bge-m3',model_kwargs={'device':'cuda'}))
lad=json.load(open('data/ragas/ladder.json'))
mets=[LLMContextRecall(),Faithfulness(),AnswerCorrectness()]
MCOL=['context_recall','faithfulness','answer_correctness']
scores={}
for name,items in lad.items():
    ds=EvaluationDataset.from_list([{"user_input":it['question'],"response":it['answer'],"retrieved_contexts":it['contexts'],"reference":it['ground_truth']} for it in items])
    r=evaluate(ds,metrics=mets,llm=llm,embeddings=emb,run_config=RunConfig(max_workers=10),show_progress=False)
    df=r.to_pandas(); cols=[c for c in MCOL if c in df.columns]
    scores[name]={c:round(float(df[c].mean()),3) for c in cols}
    df[['user_input']+cols].to_json(f'data/ragas/{name}_perq.json',orient='records',force_ascii=False)
    json.dump(scores,open('data/ragas/scores.json','w'),ensure_ascii=False,indent=1)
    print(name,scores[name],flush=True)
print("RAGAS_DONE")
