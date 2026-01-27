import re
import json
import time
import requests
import pandas as pd
import os

# =========================
# 설정 (Environment Setup)
# =========================
API_KEY = "U2FsdGVkX1+UNdm0t4n/19cwdwd2s7fqeCxsGdgYF7RoDyGd9zviOKsBeYQScNasSaG0j/EmT0D5VPK3gu5IGD/lXnFpGPCcVJyxPSHRVtZ6o9Uk82bpM/YZoYxSYwzHhgAPrs7ESRoRApXoSbqr/jDJr1dKSczvHE3z8ytAW4hueNOiikMUW2xQPWIqu6vc"
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "gpt-4o-mini-2024-07-18"

TEST_CSV_PATH = "./dev.csv" 
OUT_CSV_PATH = "./submission_drj.csv"
OUT_LOG_PATH = "./agent_reasoning_log.txt"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 소자 품질 검사 전문가다. "
    "이미지를 분석하여 구조적 결함(chipping, bending, bridge 등)을 찾아내야 한다. "
    "사소한 그림자나 미세한 질감 변화는 정상으로 간주하고, 명백한 파손에 집중하라."
)

# =========================
# 유틸리티 함수
# =========================
def _post_chat(messages, timeout=90):
    payload = {"model": MODEL, "messages": messages, "stream": False}
    try:
        r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"API Error: {r.status_code} | {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API Call Failed: {e}")
        raise

def _safe_json_extract(s: str) -> dict:
    try:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m: return json.loads(m.group(0))
        return json.loads(s)
    except:
        return {}

# =========================
# DRJ (Describe-Reason-Judge) 로직
# =========================

def describe_image(img_url: str) -> str:
    """1단계: 객관적 묘사 (Describe)"""
    prompt = (
        "이미지 속 반도체 소자의 외관을 객관적으로 묘사해줘. "
        "결함 여부를 판단하지 말고 다음 항목에 대해서만 사실대로 적어줘.\n"
        "1. 패키지 본체 상단 자국이나 파손 여부\n"
        "2. 리드(금속 다리)의 정렬 및 휨 상태\n"
        "3. 납땜 부위의 뭉침이나 이물질"
    )
    return _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

def reason_and_judge(img_url: str, description: str) -> dict:
    """2 & 3단계: 기술적 추론 및 최종 판정 (Reason & Judge)"""
    prompt = (
        f"당신은 앞서 작성된 다음 묘사를 바탕으로 최종 판정을 내려야 합니다.\n\n"
        f"[묘사 내용]:\n{description}\n\n"
        "판단 기준:\n"
        "- 본체 모서리가 크게 깨져나갔거나(Chipped), 리드가 서로 붙었거나, 심하게 휘었으면 '비정상'입니다.\n"
        "- 단순히 표면이 약간 거칠거나 그림자가 있는 것은 '정상'입니다.\n\n"
        "반드시 아래 JSON 형식으로만 최종 답변을 하세요.\n"
        "{\n"
        "  \"label\": 0 또는 1, # 0:정상, 1:비정상\n"
        "  \"reason\": \"최종 판정 결과 사유를 '비정상이고 그 이유는 ~다' 또는 '정상이고 그 이유는 ~다'의 형식으로 작성\"\n"
        "}"
    )
    
    content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])
    return _safe_json_extract(content)

def drj_agent(img_url: str) -> dict:
    try:
        # 1. 시각적 묘사
        desc = describe_image(img_url)
        # 2. 추론 및 판정
        result = reason_and_judge(img_url, desc)
        
        # 만약 JSON 추출 실패 시 기본값
        if not result:
            result = {"label": 0, "reason": "정상이고 그 이유는 특이사항이 발견되지 않았기 때문이다."}
            
        return result
    except Exception as e:
        return {"label": 0, "reason": f"정상이고 그 이유는 분석 중 오류({e})가 발생하여 기본값으로 처리되었기 때문이다."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    logs = []
    
    print(f"--- DRJ 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 분석 중...", end=" ", flush=True)
        
        # 에이전트 실행
        res = drj_agent(img_url)
        
        # 결과 및 로그 저장
        results.append({"id": _id, "label": res['label']})
        log_entry = f"ID: {_id} | 결과: {'비정상' if res['label']==1 else '정상'}\n사유: {res['reason']}\n{'-'*50}\n"
        logs.append(log_entry)
        
        print(f"-> {res['label']}")
        print(f"   [사유]: {res['reason']}")
        
        time.sleep(0.5) # Rate limit 방지
        
    # 결과 CSV 저장
    pd.DataFrame(results).to_csv(OUT_CSV_PATH, index=False)
    
    # 상세 사유 로그 저장
    with open(OUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(logs)
        
    print(f"\n✅ 분석 완료!")
    print(f"- 제출 파일: {OUT_CSV_PATH}")
    print(f"- 상세 사유 로그: {OUT_LOG_PATH}")

if __name__ == "__main__":
    main()
