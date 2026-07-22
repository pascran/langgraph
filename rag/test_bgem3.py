import os
os.environ['HF_HOME']='/home/coreedge-dev/langgraph/.hf'
from FlagEmbedding import BGEM3FlagModel
m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
q = '실손보험 통원 시 병원별 공제금액은 얼마인가?'
out = m.encode([q], return_dense=True, return_sparse=True)
print('DENSE_DIM', len(out['dense_vecs'][0]))
print('SPARSE_TERMS', len(out['lexical_weights'][0]))
print('BGE_M3_OK')
