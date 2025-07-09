# 거리 이미지 기반 유사도 평가 시스템
<br>

## 💬 개요
TripletNet 기반 모델을 활용하여 거리 이미지를 비교하고, 유사도 정량화 및 시각화하는 시스템입니다. <br>
시각적으로 유사한 보행환경을 자동으로 추출할 수 있으며, 유사 환경 추천, 도시 분석 등에 활용할 수 있습니다.
<br><br>

## 📌 목표
- Triplet Loss 기반 임베딩 공간 구성
- 거리 이미지 간 유사도 시각화
<br><br>

## 🗂️ 데이터
- **출처** : 사전 연구 과정에서 직접 수집
- **구성** : 총 2,000장의 거리 이미지 + 해당 이미지에 대응하는 보행성 점수 2,000개
- **전처리** : 점수를 기준으로 Triplet 생성 (Anchor-Positive-Negative)
<br><br>

## 🔍 모델 및 방법
- **사용 기술** : Python, TensorFlow, scikit-learn, OpenCV 
- **모델 구성** :
  - 사전 학습된 ResNet50을 Backbone으로 사용
  - Triplet Loss 기반 학습
  - Cosine Similarity를 통한 이미지 간 유사도 계산
- **학습 환경**:
  - Optimizer : SGD(learning rate=0.001)
  - Epochs : 300
  - Batchsize : 64
  - EarlyStopping 적용
  <br><br>

## 📊 주요 결과
- TripletNet 기반 임베딩 공간 시각화 (t-SNE 활용)
- 시각적으로 유사한 환경이 효과적으로 추출됨
- 거리 유사도 예시:
  | Anchor Image | Positive Image | Negative Image |
  |--------------|----------------|----------------|
  | ![]() | ![]() | ![]() |
- 향후 유사 환경 검색 서비스 또는 보행환경 개선 우선순위 분석 시스템에 적용 가능성을 제시함
<br><br>

## 🔁 회고
Triplet 구성 방식에 따라 학습 성능이 크게 달라지는 결과를 보며, 데이터 샘플링의 중요성을 체감함
