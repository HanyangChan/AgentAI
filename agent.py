import re
import json
import time
import random
import requests
import pandas as pd
import os

# =========================
# 설정 (Environment Setup)
# =========================
# 사용자는 솔트룩스 LUXIA PLATFORM API Key를 환경 변수 'LUXIA_API_KEY'에 설정해야 합니다.
API_KEY = "U2FsdGVkX1+UNdm0t4n/19cwdwd2s7fqeCxsGdgYF7RoDyGd9zviOKsBeYQScNasSaG0j/EmT0D5VPK3gu5IGD/lXnFpGPCcVJyxPSHRVtZ6o9Uk82bpM/YZoYxSYwzHhgAPrs7ESRoRApXoSbqr/jDJr1dKSczvHE3z8ytAW4hueNOiikMUW2xQPWIqu6vc"
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "gpt-4o-mini-2024-07-18"

# 경로 설정
TEST_CSV_PATH = "./dev.csv" 
OUT_PATH = "./submission_agent.csv"

# API 호출 헤더
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# 라벨 정의
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# =========================
# 시스템 프롬프트 (System Prompts)
# =========================
SYSTEM_ROLE = (
    "너는 반도체 소자(SOT-23 등) 품질 검사 전문가다. "
    "이미지를 분석하여 미세한 결함(Chiiped package, Bent lead, Solder bridge 등)을 찾아내야 한다. "
    "반드시 요청된 형식(JSON 또는 지정된 텍스트)으로만 답변한다."
)

# =========================
# 유틸리티 함수 (Utility Functions)
# =========================
def _post_chat(messages, timeout=90):
    payload = {"model": MODEL, "messages": messages, "stream": False}
    try:
        r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"API Error: status={r.status_code}, body={r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Post Chat Error: {e}")
        raise

def _safe_json_extract(s: str) -> dict:
    try:
        # JSON 블록 찾기
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return json.loads(s)
    except Exception:
        raise ValueError(f"JSON Parsing Failed: {s[:200]}")

# =========================
# 에이전트 단계별 로직 (Agent Stages)
# =========================

def initial_guess(img_url: str) -> dict:
    """1단계: 이미지 초기 분석 및 판정"""
    prompt = (
        "이미지를 보고 반도체 소자의 결함 여부를 판단해줘. "
        "결함이 있다면 구체적으로 무엇인지 설명하고, 최종 라벨을 결정해.\n"
        "출력 형식(JSON):\n"
        "{\n"
        "  \"description\": \"이미지에 대한 상세 묘사\",\n"
        "  \"defects\": [\"결함1\", \"결함2\"], # 없으면 빈 리스트\n"
        "  \"label\": 0 또는 1, # 0:정상, 1:비정상\n"
        "  \"confidence\": 1~10 # 판단 신뢰도\n"
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

def critical_review(img_url: str, first_result: dict) -> str:
    """2단계: 비판적 검토 (Devil's Advocate)"""
    label_str = "정상(0)" if first_result['label'] == 0 else "비정상(1)"
    prompt = (
        f"앞선 분석에서는 이 이미지를 '{label_str}'으로 판정했어. (근거: {first_result['description']})\n"
        "하지만 만약 이 판정이 '틀렸다'고 가정한다면, 어떤 미세한 증거를 놓쳤을 가능성이 있을까? "
        "이미지를 다시 한번 아주 세밀하게(줌인해서 보듯) 살펴보고, 반대 의견을 제시해줘."
    )
    
    content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])
    return content

def final_confirmation(first_result: dict, review_comment: str) -> int:
    """3단계: 최종 의사결정"""
    prompt = (
        "아래 두 가지 상반된 의견이나 분석 결과를 종합하여 최종 판정을 내려줘.\n\n"
        f"[1차 분석]: {json.dumps(first_result, ensure_ascii=False)}\n"
        f"[비판적 검토]: {review_comment}\n\n"
        "최종 결과는 반드시 아래 JSON 형식으로만 답변해.\n"
        "{\"final_label\": 0 또는 1}"
    )
    
    content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt},
    ])
    res = _safe_json_extract(content)
    return int(res.get("final_label", first_result['label']))

# =========================
# 메인 실행 함수 (Main Logic)
# =========================
def classify_with_correction(img_url: str) -> int:
    try:
        # 1. 초기 분석
        res1 = initial_guess(img_url)
        
        # 신뢰도가 매우 높고 정상인 경우 바로 종료 (선택 사항)
        # if res1['label'] == 0 and res1['confidence'] >= 9:
        #     return 0
            
        # 2. 비판적 검토
        review = critical_review(img_url, res1)
        
        # 3. 최종 확정
        final_label = final_confirmation(res1, review)
        return final_label
        
    except Exception as e:
        print(f"Error during agent process: {e}")
        return LABEL_NORMAL # Fallback

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("경고: LUXIA_API_KEY 환경 변수가 설정되지 않았습니다.")
        # return

    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    
    for i, row in test_df.iterrows():
        img_url = row['img_url']
        _id = row['id']
        
        print(f"[{i+1}/{len(test_df)}] Processing {_id}...", end=" ", flush=True)
        label = classify_with_correction(img_url)
        print(f"-> {label}")
        
        results.append({"id": _id, "label": label})
        time.sleep(0.5) # API Rate Limit 방지
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"결과 저장 완료: {OUT_PATH}")

if __name__ == "__main__":
    main()
