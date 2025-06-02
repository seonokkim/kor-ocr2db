import os
from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
from models.azure_document_intelligence import AzureDocumentIntelligenceModel
import json
from pathlib import Path

def load_test_data(test_dir: str, label_dir: str):
    """테스트 데이터 로드"""
    images = []
    ground_truth = []
    
    # 첫 번째 이미지와 라벨만 로드
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                # 라벨 파일 경로 생성
                label_path = (Path(label_dir) / Path(root).relative_to(test_dir) / file).with_suffix('.json')
                
                if label_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            # 텍스트와 바운딩 박스 추출
                            annotations = []
                            for anno in label_data.get('annotations', []):
                                text = anno.get('annotation.text', '')
                                bbox = anno.get('annotation.bbox', [0, 0, 0, 0])
                                annotations.append((text, bbox))
                            ground_truth.append(annotations)
                            images.append(img)
                        break  # 첫 번째 이미지만 로드
                break  # 첫 번째 이미지만 로드
        break  # 첫 번째 이미지만 로드
    
    return images, ground_truth

def evaluate_azure_model():
    """Azure Document Intelligence 모델 평가"""
    # 모델 초기화
    model = AzureDocumentIntelligenceModel(mode='read')
    
    # 테스트 데이터 로드
    test_dir = "data/test/images"
    label_dir = "data/test/labels"
    images, ground_truth = load_test_data(test_dir, label_dir)
    
    if not images:
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    # 첫 번째 이미지에 대해 평가
    image = images[0]
    gt = ground_truth[0]
    
    # 예측 수행
    predictions = model(image)
    
    # 결과 출력
    print("\n=== Azure Document Intelligence 모델 평가 결과 ===")
    print("\n예측 결과:")
    for text, bbox in predictions:
        print(f"텍스트: {text}")
        print(f"바운딩 박스: {bbox}")
    
    print("\n정답:")
    for text, bbox in gt:
        print(f"텍스트: {text}")
        print(f"바운딩 박스: {bbox}")

if __name__ == "__main__":
    evaluate_azure_model() 