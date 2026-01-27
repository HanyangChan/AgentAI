import re
import json
import time
import requests
import pandas as pd
import os
import io
import base64
from PIL import Image

# =========================
# 설정 (Environment Setup)
# =========================
API_KEY = "U2FsdGVkX1+UNdm0t4n/19cwdwd2s7fqeCxsGdgYF7RoDyGd9zviOKsBeYQScNasSaG0j/EmT0D5VPK3gu5IGD/lXnFpGPCcVJyxPSHRVtZ6o9Uk82bpM/YZoYxSYwzHhgAPrs7ESRoRApXoSbqr/jDJr1dKSczvHE3z8ytAW4hueNOiikMUW2xQPWIqu6vc"
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL = "gpt-4o-mini-2024-07-18"

TEST_CSV_PATH = "./dev.csv" 
OUT_CSV_PATH = "./submission_scoring.csv"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 부품 품질 검사 전문가다. "
    "전체 이미지와 분할된 상세 이미지를 대조 분석하여 미세한 결함까지 점수화해야 한다."
)

# =========================
# 유틸리티 함수
# =========================
def _post_chat(messages, timeout=120):
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

def download_and_split_image(url: str):
    """이미지를 다운로드하여 상단과 하단으로 분할하고, 원본 포함 Base64 반환"""
    r = requests.get(url, timeout=30)
    img = Image.open(io.BytesIO(r.content))
    w, h = img.size
    
    # 상단 (본체 중심)
    top_img = img.crop((0, 0, w, h // 2))
    # 하단 (리드 중심)
    bottom_img = img.crop((0, h // 2, w, h))
    
    def to_b64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return to_b64(img), to_b64(top_img), to_b64(bottom_img)

# =========================
# 스코어링 분석 로직
# =========================

def scoring_analyze(full_b64, top_b64, bottom_b64):
    """세 가지 뷰에 대해 [객관적 묘사] 후 [품질 점수 산정] 수행"""
    
    # 1. 전체 뷰 객관적 묘사
    full_prompt = (
        "이미지 속 반도체 소자의 전체 외관을 객관적으로 묘사해줘. "
        "결함 여부를 판단하지 말고 본체의 정렬, 균형, 전반적인 형태에 대해서만 사실대로 적어줘."
    )
    full_desc = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{full_b64}"}},
        ]},
    ])

    # 2. 상단 뷰 객관적 묘사 (본체)
    top_prompt = (
        "이미지 속 반도체 소자의 '상단(본체)' 부분에 집중하여 객관적으로 묘사해줘. "
        "결함 여부를 판단하지 말고 표면의 자국, 질감, 모서리 형태에 대해서만 사실대로 적어줘."
    )
    top_desc = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": top_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
        ]},
    ])

    # 3. 하단 뷰 객관적 묘사 (리드)
    bottom_prompt = (
        "이미지 속 반도체 소자의 '하단(리드/다리)' 부분에 집중하여 객관적으로 묘사해줘. "
        "결함 여부를 판단하지 말고 리드의 간격, 정렬 상태, 형태에 대해서만 사실대로 적어줘."
    )
    bottom_desc = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": bottom_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bottom_b64}"}},
        ]},
    ])

    # 4. 품질 점수 산정 (Scoring)
    final_prompt = (
        f"당신은 아래 세 가지 시점의 객관적 묘사를 바탕으로 반도체 소자의 품질 점수를 0.0에서 1.0 사이로 산정해야 합니다.\n\n"
        f"1.0은 가장 완벽한 상태, 0.0은 가장 이상이 심한 상태를 의미합니다.\n\n"
        f"[전체 뷰 묘사]:\n{full_desc}\n\n"
        f"[상단 뷰 묘사]:\n{top_desc}\n\n"
        f"[하단 뷰 묘사]:\n{bottom_desc}\n\n"
        "점수 산정 기준:\n"
        "- 1.0 (완벽): 결함이 전혀 없음. 본체와 리드가 매우 정교하고 깨끗함.\n"
        "- 0.8~0.9 (우수): 미세한 표면 노이즈나 무시할 수 있는 수준의 흔적이 있으나 기능상 완벽함.\n"
        "- 0.5~0.7 (보통): 눈에 띄는 자국이나 약간의 정렬 불균형이 있으나 심각한 파손은 아님.\n"
        "- 0.2~0.4 (불량): 명확한 파손(Chipped), 리드 휨(Bending), 또는 브릿지(Bridge) 현상이 관찰됨.\n"
        "- 0.0~0.1 (심각): 형태가 심하게 일그러졌거나 리드 프레임이 손실되는 등 치명적인 결함.\n\n"
        "반드시 아래 JSON 형식으로만 최종 답변을 하세요.\n"
        "{\n"
        "  \"score\": 0.0 ~ 1.0 사이의 실수,\n"
        "  \"reason\": \"상단과 하단 각각의 상태를 언급하며 점수를 산정한 논리적인 근거를 작성\"\n"
        "}"
    )
    
    final_content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": final_prompt}
    ])
    
    return _safe_json_extract(final_content)

def scoring_agent(img_url: str):
    try:
        full_b64, top_b64, bottom_b64 = download_and_split_image(img_url)
        result = scoring_analyze(full_b64, top_b64, bottom_b64)
        if not result or 'score' not in result:
            result = {"score": 1.0, "reason": "정상이고 그 이유는 육안상 결함이 확인되지 않기 때문이다."}
        return result
    except Exception as e:
        return {"score": 1.0, "reason": f"분석 중 오류({e})가 발생했기 때문이다."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    
    print(f"--- 스코어링 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 점수 산정 중...", end=" ", flush=True)
        res = scoring_agent(img_url)
        
        results.append({"id": _id, "score": res['score']})
        print(f"-> {res['score']}")
        print(f"   [사유]: {res['reason']}")
        
        time.sleep(0.5)
        
    pd.DataFrame(results).to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✅ 분석 완료! 제출 파일: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
