"""RAGAS 스타일 지표(직접 구현) — 9벌 복붙되던 faith/crecall/acorr/jparse의 단일 정본.

모든 지표는 LLM 호출을 `judge`(콜러블: prompt->str)로, 임베딩을 `embed`(콜러블: list[str]->vecs)로
주입받는다. 덕분에 실제 모델 없이 목(mock)으로 단위 테스트가 가능하다.
정의: faithfulness=근거문장/총문장, context_recall=귀속문장/총문장, answer_correctness=0.75·F1+0.25·의미유사도.
"""
import re
import json

NAN = float("nan")


def jparse(s):
    """LLM 출력에서 첫 JSON 오브젝트 추출(작은따옴표 폴백 포함). 실패 시 None."""
    if not s:
        return None
    mm = re.search(r"\{[^{}]*\}", s, re.S)
    if not mm:
        return None
    for cand in (mm.group(0), mm.group(0).replace("'", '"')):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _ratio(r, num_key):
    if r and r.get("total"):
        return min(r[num_key] / r["total"], 1.0)  # 클램프: LLM이 num>total로 세는 경우 방지
    return NAN


def faithfulness(answer, contexts, judge):
    c = "\n".join(contexts)[:4000]
    r = jparse(judge(
        "답변을 단순 사실문장으로 나눈 뒤 각 문장이 문맥에서 추론가능한지 세어 JSON만.\n"
        f'[문맥]\n{c}\n[답변]{answer}\n형식만: {{"supported":정수,"total":정수}}'
    ))
    return _ratio(r, "supported")


def context_recall(gt, contexts, judge):
    c = "\n".join(contexts)[:4000]
    r = jparse(judge(
        "정답을 사실문장으로 나눈 뒤 각 문장이 문맥에 귀속가능한지 세어 JSON만.\n"
        f'[문맥]\n{c}\n[정답]{gt}\n형식만: {{"attributable":정수,"total":정수}}'
    ))
    return _ratio(r, "attributable")


def _f1(r):
    if not r:
        return NAN
    tp, fp, fn = r.get("tp", 0), r.get("fp", 0), r.get("fn", 0)
    den = tp + 0.5 * (fp + fn)
    return tp / den if den else 0.0


def answer_f1(answer, gt, judge):
    """사실단위 F1만(임베딩 불필요). 8B vs 27B 배치 비교에 사용."""
    r = jparse(judge(
        "답변과 정답을 사실 단위로 비교. tp=공통,fp=답변에만,fn=정답에만. JSON만.\n"
        f'[정답]{gt}\n[답변]{answer}\n형식만: {{"tp":정수,"fp":정수,"fn":정수}}'
    ))
    return _f1(r)


def answer_correctness(answer, gt, judge, embed):
    """0.75·F1(사실) + 0.25·의미유사도(임베딩 코사인). embed=list[str]->정규화 벡터."""
    import numpy as np  # lazy: 순수 테스트에서 numpy 없이 위 함수들만 임포트 가능
    f1 = answer_f1(answer, gt, judge)
    v = embed([answer, gt])
    sim = float(np.dot(v[0], v[1]))
    return (0.75 * f1 + 0.25 * sim) if f1 == f1 else 0.25 * sim
