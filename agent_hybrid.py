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
OUT_CSV_PATH = "./submission_hybrid_final.csv"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 부품 품질 검사 전문가다. "
    "전체 이미지와 분할된 상세 이미지를 대조 분석하여 미세한 결함까지 찾아내야 한다."
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
# 하이브리드 DRJ 분석 로직
# =========================

def hybrid_analyze(full_b64, top_b64, bottom_b64):
    """세 가지 뷰에 대해 [객관적 묘사] 후 [통합 추론 및 판정] 수행"""
    
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

    # 4. 통합 추론 및 판정 (Reason & Judge)
    final_prompt = (
        f"당신은 아래 세 가지 시점의 객관적 묘사를 바탕으로 '상단(본체)'과 '하단(리드)' 각각에 대해 품질 판정을 내리고 이를 종합해야 합니다.\n\n"
        f"[전체 뷰 묘사]:\n{full_desc}\n\n"
        f"[상단 뷰 묘사]:\n{top_desc}\n\n"
        f"[하단 뷰 묘사]:\n{bottom_desc}\n\n"
        "판단 기준:\n"
        "1. 상단의 경우: 본체 모서리가 크게 깨져나갔거나(Chipped), 표면에 명확한 파손 자국이 있는 경우 '비정상'입니다.\n"
        "2. 하단의 경우: 리드가 서로 붙었거나(Bridge), 심하게 비틀어진 경우(Bending), 리드 프레임이 변형된 경우(Forming Issue), 모든 구멍에 리드가 안들어간 경우 '비정상'입니다.\n"
        "3. 공통: 미세한 표면 거칠기, 그림자, 사진 노이즈는 '정상'입니다.\n\n"
        "최종 판정은 상단이나 하단 중 어느 한 곳이라도 비정상이면 '비정상(1)'으로 판정하세요.\n\n"
        "반드시 아래 JSON 형식으로만 최종 답변을 하세요.\n"
        "{\n"
        "  \"label\": 0 또는 1,\n"
        "  \"reason\": \"상단과 하단 각각의 상태를 언급하며 '비정상이고 그 이유는 ~다' 또는 '정상이고 그 이유는 ~다'의 형식으로 작성\"\n"
        "}"
    )
    
    final_content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": final_prompt}
    ])
    
    return _safe_json_extract(final_content)

def hybrid_agent(img_url: str):
    try:
        full_b64, top_b64, bottom_b64 = download_and_split_image(img_url)
        result = hybrid_analyze(full_b64, top_b64, bottom_b64)
        if not result:
            result = {"label": 0, "reason": "정상이고 그 이유는 육안상 결함이 확인되지 않기 때문이다."}
        return result
    except Exception as e:
        return {"label": 0, "reason": f"정상이고 그 이유는 분석 중 오류({e})가 발생했기 때문이다."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    results = []
    
    print(f"--- 하이브리드 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 분석 중...", end=" ", flush=True)
        res = hybrid_agent(img_url)
        
        results.append({"id": _id, "label": res['label']})
        print(f"-> {res['label']}")
        print(f"   [사유]: {res['reason']}")
        
        time.sleep(0.5)
        
    pd.DataFrame(results).to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✅ 분석 완료! 제출 파일: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
