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
OUT_CSV_PATH = "./submission_drj_scoring.csv"
OUT_LOG_PATH = "./agent_drj_scoring_log.txt"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 소자 품질 검사 전문가다. "
    "이미지를 분석하여 구조적 결함뿐만 아니라 미세한 품질 저하 요소까지 찾아내어 점수화해야 한다."
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
# DRJ (Describe-Reason-Score) 로직
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

def reason_and_score(img_url: str, description: str) -> dict:
    """2 & 3단계: 기술적 추론 및 품질 점수 산정 (Reason & Score)"""
    prompt = (
        f"당신은 앞서 작성된 다음 묘사를 바탕으로 반도체 소자의 품질 점수를 0.0에서 1.0 사이로 산정해야 합니다.\n\n"
        f"1.0은 가장 완벽한 상태, 0.0은 가장 이상이 심한 상태를 의미합니다.\n\n"
        f"[묘사 내용]:\n{description}\n\n"
        "점수 산정 기준:\n"
        "- 1.0 (완벽): 결함이 전혀 없음. 본체와 리드가 매우 정교하고 깨끗함.\n"
        "- 0.8~0.9 (우수): 미세한 표면 노이즈나 무시할 수 있는 수준의 흔적이 있으나 기능상 완벽함.\n"
        "- 0.5~0.7 (보통): 눈에 띄는 자격이나 약간의 정렬 불균형이 있으나 심각한 파손은 아님.\n"
        "- 0.2~0.4 (불량): 명확한 파손(Chipped), 리드 휨(Bending), 또는 브릿지(Bridge) 현상이 관찰됨.\n"
        "- 0.0~0.1 (심각): 형태가 심하게 일그러졌거나 리드 프레임이 손실되는 등 치명적인 결함.\n\n"
        "반드시 아래 JSON 형식으로만 최종 답변을 하세요.\n"
        "{\n"
        "  \"score\": 0.0 ~ 1.0 사이의 실수,\n"
        "  \"reason\": \"최종 점수 산정 사유를 논리적으로 작성\"\n"
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

def drj_scoring_agent(img_url: str) -> dict:
    try:
        # 1. 시각적 묘사
        desc = describe_image(img_url)
        # 2. 추론 및 점수 산정
        result = reason_and_score(img_url, desc)
        
        if not result or 'score' not in result:
            result = {"score": 1.0, "reason": "특이사항이 발견되지 않아 기본 점수로 처리됨."}
            
        return result
    except Exception as e:
        return {"score": 1.0, "reason": f"분석 중 오류({e}) 발생으로 기본 점수 처리됨."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    logs = []
    
    print(f"--- DRJ 스코어링 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 점수 산정 중...", end=" ", flush=True)
        
        # 에이전트 실행
        res = drj_scoring_agent(img_url)
        
        # 결과 및 로그 저장
        results.append({"id": _id, "score": res['score']})
        log_entry = f"ID: {_id} | 점수: {res['score']}\n사유: {res['reason']}\n{'-'*50}\n"
        logs.append(log_entry)
        
        print(f"-> {res['score']}")
        print(f"   [사유]: {res['reason']}")
        
        time.sleep(0.5)
        
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
