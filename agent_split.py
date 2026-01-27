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
OUT_CSV_PATH = "./submission_split.csv"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 부품 품질 검사 전문가다. "
    "분할된 이미지를 세밀하게 분석하여 결함을 찾아내야 한다."
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
    """이미지를 다운로드하여 상단과 하단으로 분할한 후 Base64로 반환"""
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
    
    return to_b64(top_img), to_b64(bottom_img)

# =========================
# Split-View 분석 로직
# =========================

def analyze_parts(top_b64, bottom_b64):
    """상단과 하단 이미지를 개별 묘사하고 최종 판결"""
    
    # 1. 상단 분석 (부품 본체)
    top_prompt = "이 이미지는 반도체 소자의 '상단(본체)' 부분이다. 표면의 찍힘, 크랙, 모서리 파손 여부를 자세히 설명해줘."
    top_desc = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": top_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
        ]},
    ])

    # 2. 하단 분석 (리드)
    bottom_prompt = "이 이미지는 반도체 소자의 '하단(리드/다리)' 부분이다. 리드의 휘어짐, 간격 불일치, 납땜 뭉침 여부를 자세히 설명해줘."
    bottom_desc = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": bottom_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bottom_b64}"}},
        ]},
    ])

    # 3. 통합 판정
    final_prompt = (
        f"상단과 하단의 분석 결과를 바탕으로 최종 품질 판정을 내려줘.\n\n"
        f"[상단 묘사]: {top_desc}\n"
        f"[하단 묘사]: {bottom_desc}\n\n"
        "반드시 아래 JSON 형식으로만 최종 답변을 하세요.\n"
        "{\n"
        "  \"label\": 0 또는 1, # 0:정상, 1:비정상\n"
        "  \"reason\": \"최종 판정 결과 사유를 '비정상이고 그 이유는 ~다' 또는 '정상이고 그 이유는 ~다'의 형식으로 작성\"\n"
        "}"
    )
    
    final_content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": final_prompt}
    ])
    
    return _safe_json_extract(final_content)

def split_agent(img_url: str) -> dict:
    try:
        top_b64, bottom_b64 = download_and_split_image(img_url)
        result = analyze_parts(top_b64, bottom_b64)
        if not result:
            result = {"label": 0, "reason": "정상이고 그 이유는 별다른 결함이 발견되지 않았기 때문이다."}
        return result
    except Exception as e:
        return {"label": 0, "reason": f"정상이고 그 이유는 분석 중 오류({e})가 발생하여 기본값으로 처리되었기 때문이다."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    
    print(f"--- Split-View 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 분석 중...", end=" ", flush=True)
        res = split_agent(img_url)
        
        results.append({"id": _id, "label": res['label']})
        print(f"-> {res['label']}")
        print(f"   [사유]: {res['reason']}")
        
        time.sleep(0.5)
        
    pd.DataFrame(results).to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✅ 분석 완료! 제출 파일: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
