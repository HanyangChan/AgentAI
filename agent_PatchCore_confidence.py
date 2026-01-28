import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import requests
import base64
import io
import json

# =========================
# 설정 (Configuration)
# =========================
API_KEY = "U2FsdGVkX1+UNdm0t4n/19cwdwd2s7fqeCxsGdgYF7RoDyGd9zviOKsBeYQScNasSaG0j/EmT0D5VPK3gu5IGD/lXnFpGPCcVJyxPSHRVtZ6o9Uk82bpM/YZoYxSYwzHhgAPrs7ESRoRApXoSbqr/jDJr1dKSczvHE3z8ytAW4hueNOiikMUW2xQPWIqu6vc"
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create"
MODEL_LLM = "gpt-4o-mini-2024-07-18"

TRAIN_CSV_PATH = "./dev.csv"
TEST_CSV_PATH = "./test/test.csv"
OUT_CSV_PATH = "./submission_confidence.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIDENCE_THRESHOLD = 0.9  # 0.8에서 0.9로 상향하여 재시도 로직 유도

# =========================
# 1. PatchCore 구현 (Feature Extractor)
# =========================
class PatchCore(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        self.backbone.to(DEVICE)
        
        self.features = []
        self.layers_to_extract = ['layer2', 'layer3']
        for name, module in self.backbone.named_children():
            if name in self.layers_to_extract:
                module.register_forward_hook(self.hook)
        
        self.memory_bank = None
        self.nbrs = None

    def hook(self, module, input, output):
        self.features.append(output)

    def forward_features(self, x):
        self.features = []
        _ = self.backbone(x)
        return self.features

    def embed(self, x):
        with torch.no_grad():
            features = self.forward_features(x.to(DEVICE))
        
        target_size = features[0].shape[-2:]
        processed_features = []
        for f in features:
            f = torch.nn.AvgPool2d(3, 1, 1)(f)
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            processed_features.append(f)
            
        combined = torch.cat(processed_features, dim=1)
        return combined

    def fit(self, train_loader):
        print(">>> [System] PatchCore Memory Bank 구축 중...")
        embedding_vectors = []
        
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0] 
            features = self.embed(imgs)
            f = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            embedding_vectors.append(f.cpu().numpy())
        
        self.memory_bank = np.concatenate(embedding_vectors, axis=0)
        
        if self.memory_bank.shape[0] > 10000:
            indices = np.random.choice(self.memory_bank.shape[0], 10000, replace=False)
            self.memory_bank = self.memory_bank[indices]
            
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
        self.nbrs.fit(self.memory_bank)
        print(f">>> [System] 학습 완료. Memory Bank Size: {self.memory_bank.shape}")

    def predict(self, img_tensor):
        features = self.embed(img_tensor.unsqueeze(0))
        f = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        query_vectors = f.cpu().numpy()
        distances, _ = self.nbrs.kneighbors(query_vectors)
        anomaly_score = np.max(distances)
        return anomaly_score, distances, features.shape[-2:]

# =========================
# 2. 유틸리티 함수
# =========================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def check_missing_component(img_tensor, normal_brightness_range):
    """
    이미지의 절대값 평균(Normalised Absolute Mean)을 계산하여 소자 누락 여부를 판단
    """
    # 1. 전역 절대값 평균 (Global Deviation)
    curr_brightness = torch.mean(torch.abs(img_tensor)).item()
    
    # 2. 중앙 영역 절대값 평균 (Local Deviation - 소자 위치 강조)
    # 224x224에서 중앙 100x100 영역
    center_brightness = torch.mean(torch.abs(img_tensor[:, 62:162, 62:162])).item()
    
    (min_g, max_g), (min_c, max_c) = normal_brightness_range
    if min_g == 0 and max_g == 0: return False, curr_brightness
    
    # 임계치 (0.9, 1.1) 적용 - 전역 또는 중앙 영역 중 하나라도 크게 벗어나면 불량으로 간주
    if (curr_brightness < min_g * 0.9 or curr_brightness > max_g * 1.1 or
        center_brightness < min_c * 0.9 or center_brightness > max_c * 1.1):
        return True, curr_brightness
    return False, curr_brightness

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_image_from_url(url):
    try:
        if url.startswith("http"):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            img = Image.open(response.raw).convert("RGB")
        else:
            img = Image.open(url).convert("RGB")
        return img
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        return None

# =========================
# 3. LLM Agent (VLM)
# =========================
def query_llm_agent(original_img, is_retry=False):
    # 크롭 영역 정의 (224x224 기준)
    # 1. 상단 (헤드 강조): (0, 0, 224, 140)
    head_crop = original_img.resize((224, 224)).crop((0, 0, 224, 140))
    # 2. 하단 (리드 강조): (0, 100, 224, 224)
    leads_crop = original_img.resize((224, 224)).crop((0, 100, 224, 224))
    
    img_b64 = image_to_base64(original_img)
    head_b64 = image_to_base64(head_crop)
    leads_b64 = image_to_base64(leads_crop)
    
    retry_context = ""
    if is_retry:
        retry_context = "\n[주의] 이전 판단의 확신도가 낮아 재검토가 필요합니다. 세 장의 이미지(원본, 상단, 하단)를 더 정밀하게 시각적으로 분석해주세요."

    prompt = f"""
    당신은 반도체 소자 결함을 탐지하는 전문가 AI입니다.{retry_context}
    
    제공된 세 장의 이미지는 다음과 같습니다:
    1. 전체 이미지
    2. 소자 헤드(상단) 확대 크롭
    3. 은색 리드(하단) 확대 크롭

    [임무]
    이미지를 시각적으로 정밀하게 분석하여 '정상(0)'인지 '불량(1)'인지 판단하고, 그 확신도(confidence)를 함께 제공하세요.
    - 헤드부분에 흠집이나 균열, 하얀색 내용물의 노출이 있는지 상단 크롭을 통해 확인하세요.
    - 3개의 은색 리드가 검은색 원 중앙에 정확히 정렬되어 있는지 하단 크롭을 통해 확인하세요.
    - 트랜지스터의 부재나 심각한 위치 이탈을 체크하세요.

    결과는 반드시 JSON 형식으로만 답하세요. 
    confidence는 0.0에서 1.0 사이의 실수여야 합니다. (예: 0.95)
    
    형식: {{"label": 0 또는 1, "confidence": float, "reason": "이유"}}
    """

    data = {
        "model": MODEL_LLM,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{head_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{leads_b64}"}}
                ]
            }
        ],
        "temperature": 0.0
    }

    try:
        resp = requests.post(BRIDGE_URL, headers={"apikey": API_KEY, "Content-Type": "application/json"}, json=data)
        result = resp.json()['choices'][0]['message']['content']
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return json.loads(result)
    except Exception as e:
        print(f"LLM 에러: {e}")
        return {"label": 0, "confidence": 0.0, "reason": f"LLM Error: {str(e)}"}

# =========================
# 메인 실행 로직
# =========================
def main():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    patchcore = PatchCore()
    transform = get_transform()
    
    # PatchCore는 Memory Bank 구축 및 참조용으로 유지 (필요시 사용 가능)
    print(f">>> [System] {TRAIN_CSV_PATH}에서 정상 데이터 로드 중...")
    train_images = []
    exclude_ids = ["DEV_008", "DEV_009", "DEV_014", "DEV_017"]
    train_subset = train_df[~train_df['id'].isin(exclude_ids)]
    
    for _, row in train_subset.iterrows():
        img = load_image_from_url(row['img_url'])
        if img: train_images.append(transform(img))
    
    if len(train_images) > 0:
        train_tensor = torch.stack(train_images)
        train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=4)
        patchcore.fit(train_loader)

        # 정상 이미지의 절대값 평균 범위 산출 (전역 및 중앙)
        global_vals = []
        center_vals = []
        for imgs in train_loader:
            for j in range(imgs.size(0)):
                global_vals.append(torch.mean(torch.abs(imgs[j])).item())
                center_vals.append(torch.mean(torch.abs(imgs[j][:, 62:162, 62:162])).item())
        
        normal_brightness_range = ((min(global_vals), max(global_vals)), 
                                   (min(center_vals), max(center_vals)))
        print(f">>> [System] 정상 범위 (전역): {normal_brightness_range[0][0]:.4f} ~ {normal_brightness_range[0][1]:.4f}")
        print(f">>> [System] 정상 범위 (중앙): {normal_brightness_range[1][0]:.4f} ~ {normal_brightness_range[1][1]:.4f}")
    else:
        normal_brightness_range = ((0, 0), (0, 0))

    results = []
    THRESHOLD_NORMAL = 2.0
    THRESHOLD_DEFECT = 2.5

    print(f"--- 분석 시작 (총 {len(test_df)}개) ---")

    for i, row in test_df.iterrows():
        _id = row['id']
        img = load_image_from_url(row['img_url'])
        if not img:
             results.append({"id": _id, "label": 0, "confidence": 0.0, "reason": "Image Load Failed"})
             continue

        img_tensor = transform(img).to(DEVICE)
        
        # [0단계] Rule-based: 소자 누락 체크
        is_missing, brightness = check_missing_component(img_tensor, normal_brightness_range)
        
        final_label = 0
        final_confidence = 1.0  # 명확한 경우 1.0
        final_reason = ""

        if is_missing:
            final_label = 1
            final_reason = f"Rule-based: 소자 누락 감지 (현재 밝기: {brightness:.4f})"
            print(f"[{_id}] {final_reason}")
        else:
            score, _, _ = patchcore.predict(img_tensor)

            if score < THRESHOLD_NORMAL:
                final_label = 0
                final_reason = f"PatchCore 확신 (Normal, Score: {score:.2f})"
                print(f"[{_id}] {final_reason}")
                
            elif score > THRESHOLD_DEFECT:
                final_label = 1
                final_reason = f"PatchCore 확신 (Defect, Score: {score:.2f})"
                print(f"[{_id}] {final_reason}")
            
            else:
                print(f"[{_id}] 판단 보류 (Score: {score:.2f}) -> 상/하단 크롭 후 LLM 호출...")
                llm_res = query_llm_agent(img)
                
                final_label = llm_res.get('label', 0)
                final_confidence = llm_res.get('confidence', 0.0)
                final_reason = f"LLM 판단: {llm_res.get('reason')}"

                if final_confidence < CONFIDENCE_THRESHOLD:
                    print(f"   ㄴ 확신도 부족 ({final_confidence:.2f}) -> 재판단 수행 중...")
                    retry_res = query_llm_agent(img, is_retry=True)
                    
                    if retry_res.get('confidence', 0.0) >= final_confidence:
                        final_label = retry_res.get('label', final_label)
                        final_confidence = retry_res.get('confidence', final_confidence)
                        final_reason = f"LLM 재판단: {retry_res.get('reason')}"
                        print(f"   ㄴ 재판단 결과: {final_label} (Conf: {final_confidence:.2f})")
                    else:
                        print(f"   ㄴ 재판단 확신도가 더 낮아 기존 결과 유지.")
                
                print(f"   ㄴ 최종 결과: {final_label} ({final_reason})")

        results.append({
            "id": _id, 
            "label": final_label,
            "confidence": final_confidence,
            "reason": final_reason
        })

    # 결과 저장 (Submission 형식: id, label)
    out_df = pd.DataFrame(results)
    
    # 1. 제출용 CSV (id, label 만 포함)
    submission_df = out_df[['id', 'label']]
    submission_df.to_csv(OUT_CSV_PATH, index=False)
    print(f"\n제출용 결과 저장됨: {OUT_CSV_PATH}")

    # 2. 분석용 상세 CSV (id, label, confidence, reason 포함)
    reasoning_csv_path = OUT_CSV_PATH.replace(".csv", "_detailed.csv")
    out_df.to_csv(reasoning_csv_path, index=False)
    print(f"분석용 상세 결과 저장됨: {reasoning_csv_path}")

if __name__ == "__main__":
    main()
