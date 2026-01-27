import re
import json
import time
import requests
import pandas as pd
import os
import io
import base64
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# 설정 (Environment Setup)
# =========================
API_KEY = "U2FsdGVkX1+UNdm0t4n/19cwdwd2s7fqeCxsGdgYF7RoDyGd9zviOKsBeYQScNasSaG0j/EmT0D5VPK3gu5IGD/lXnFpGPCcVJyxPSHRVtZ6o9Uk82bpM/YZoYxSYwzHhgAPrs7ESRoRApXoSbqr/jDJr1dKSczvHE3z8ytAW4hueNOiikMUW2xQPWIqu6vc"
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL_LLM = "gpt-4o-mini-2024-07-18"

TEST_CSV_PATH = "./dev.csv" 
OUT_CSV_PATH = "./submission_resnet_vlm.csv"
RESNET_WEIGHTS_PATH = "resnet18_fewshot.pth"

HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2

# =========================
# 시스템 역할 설정
# =========================
SYSTEM_ROLE = (
    "너는 반도체 부품 품질 검사 전문가다. "
    "전체 이미지와 분할된 상세 이미지, 그리고 영상 분석 모델(ResNet)의 예측 결과를 종합하여 최종 판정을 내려야 한다."
)

# =========================
# ResNet 모델 로드
# =========================
def load_resnet_model(weights_path):
    model = models.resnet18(weights=None) # pre-trained=False since we load our weights
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    if os.path.exists(weights_path):
        print(f"Loading ResNet weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    else:
        print(f"Warning: {weights_path} not found. Using randomly initialized model (for demo).")
    
    model = model.to(DEVICE)
    model.eval()
    return model

# =========================
# 유틸리티 함수
# =========================
def _post_chat(messages, timeout=120):
    payload = {"model": MODEL_LLM, "messages": messages, "stream": False}
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

def preprocess_image(url: str):
    """이미지 다운로드, RGB 변환 및 Crop (Top/Bottom)"""
    r = requests.get(url, timeout=30)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    w, h = img.size
    
    # 상단 (본체 중심)
    top_img = img.crop((0, 0, w, h // 2))
    # 하단 (리드 중심)
    bottom_img = img.crop((0, h // 2, w, h))
    
    # ResNet용 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    top_tensor = transform(top_img).unsqueeze(0).to(DEVICE)
    bottom_tensor = transform(bottom_img).unsqueeze(0).to(DEVICE)
    
    def to_b64(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "full_t": img_tensor, "top_t": top_tensor, "bottom_t": bottom_tensor,
        "full_b64": to_b64(img), "top_b64": to_b64(top_img), "bottom_b64": to_b64(bottom_img)
    }

# =========================
# 하이브리드 추론 로직
# =========================

def hybrid_resnet_llm_analyze(model_resnet, img_data):
    """ResNet의 예측 결과와 LLM의 시각적 판단을 결합"""
    
    # 1. ResNet Inference (전체, 상단, 하단 각각 수행 가능하나 여기서는 예시로 전체/상단/하단 각각 판단)
    with torch.no_grad():
        out_full = model_resnet(img_data["full_t"])
        out_top = model_resnet(img_data["top_t"])
        out_bottom = model_resnet(img_data["bottom_t"])
        
        prob_full = torch.softmax(out_full, dim=1)[0]
        prob_top = torch.softmax(out_top, dim=1)[0]
        prob_bottom = torch.softmax(out_bottom, dim=1)[0]
        
        # label 0: 정상, 1: 비정상 (사용자 코드 클래스 NUM_CLASSES=2에 따름)
        pred_full = int(prob_full.argmax())
        pred_top = int(prob_top.argmax())
        pred_bottom = int(prob_bottom.argmax())
        
        conf_full = float(prob_full[pred_full])
        conf_top = float(prob_top[pred_top])
        conf_bottom = float(prob_bottom[pred_bottom])

    resnet_report = (
        f"- 전체 이미지 ResNet 예측: {'비정상(1)' if pred_full == 1 else '정상(0)'} (신뢰도: {conf_full:.2f})\n"
        f"- 상단 이미지 ResNet 예측: {'비정상(1)' if pred_top == 1 else '정상(0)'} (신뢰도: {conf_top:.2f})\n"
        f"- 하단 이미지 ResNet 예측: {'비정상(1)' if pred_bottom == 1 else '정상(0)'} (신뢰도: {conf_bottom:.2f})"
    )

    # 2. LLM Final Judgment (Reason & Judge)
    prompt = (
        f"당신에게는 반도체 부품 이미지와 영상 분석 모델(ResNet)의 예측 결과가 주어집니다.\n\n"
        f"[ResNet 모델 분석 리포트]:\n{resnet_report}\n\n"
        "이미지 1: 전체 외관\n"
        "이미지 2: 상단(본체 중심)\n"
        "이미지 3: 하단(리드 중심)\n\n"
        "판단 기준:\n"
        "1. ResNet의 판단이 '비정상'이더라도 실제 이미지에서 결함(모서리 깨짐, 리드 휨 등)이 명확하지 않으면 '정상'으로 판단할 수 있습니다.\n"
        "2. 반대로 ResNet이 '정상'이라 하더라도 육안상 치명적인 결함이 보인다면 '비정상'으로 판단하세요.\n"
        "3. 상단(본체)의 깨짐(Chipped)이나 하단 리드의 브릿지/변형은 주요 결함입니다.\n\n"
        "반드시 아래 JSON 형식으로만 답변하세요.\n"
        "{\n"
        "  \"label\": 0 또는 1,\n"
        "  \"reason\": \"ResNet 결과와 본인의 시각적 분석을 대조하여 최종 결론을 내린 이유를 작성\"\n"
        "}"
    )
    
    final_content = _post_chat([
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data['full_b64']}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data['top_b64']}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data['bottom_b64']}"}},
        ]},
    ])
    
    return _safe_json_extract(final_content)

def agent_runner(model_resnet, img_url: str):
    try:
        img_data = preprocess_image(img_url)
        result = hybrid_resnet_llm_analyze(model_resnet, img_data)
        if not result:
            result = {"label": 0, "reason": "정상이고 그 이유는 최종 판단 결과가 누락되어 기본값으로 설정했기 때문이다."}
        return result
    except Exception as e:
        return {"label": 0, "reason": f"정상이고 그 이유는 처리 중 오류({e})가 발생했기 때문이다."}

# =========================
# 메인 실행
# =========================
def main():
    test_df = pd.read_csv(TEST_CSV_PATH)
    model_resnet = load_resnet_model(RESNET_WEIGHTS_PATH)
    
    results = []
    print(f"--- 하이브리드 ResNet+LLM 에이전트 구동 시작 (총 {len(test_df)}개) ---")
    
    for i, row in test_df.iterrows():
        _id = row['id']
        img_url = row['img_url']
        
        print(f"[{i+1}/{len(test_df)}] {_id} 분석 중...", end=" ", flush=True)
        res = agent_runner(model_resnet, img_url)
        
        results.append({"id": _id, "label": res.get('label', 0)})
        print(f"-> {res.get('label', 0)}")
        print(f"   [사유]: {res.get('reason', 'N/A')}")
        
        time.sleep(0.5)
        
    pd.DataFrame(results).to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✅ 분석 완료! 제출 파일: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
