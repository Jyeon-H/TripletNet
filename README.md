# 거리 이미지 기반 유사도 평가 시스템

## 💬 개요
TripletNet 기반 모델을 활용하여 거리 이미지를 비교하여 유사도를 정량화하고,  
시각적으로 유사한 환경을 추출하는 시스템입니다.<br><br>

## 📌 목표
- Triplet Loss 기반 임베딩 공간 구성
- 거리 이미지 간 유사도 시각화
- 실생활 적용 예 : 보행환경 유사도 탐색, 이미지 기반 위치 추천 등
<br><br>
## 🗂️ 데이터
- **출처** : 사전 연구 시 수집
- **구성** : 총 2,000장의 거리 이미지 (전주시 100개 지점, 각 지점 20장씩)
             이미지에 해당하는 보행성 점수 2,000개
- **전처리** : 점수를 기준으로 Triplet 생성 (Anchor-Positive-Negative)
<br><br>
## 🔍 모델 및 방법
- **사용 기술** : Python, TensorFlow, scikit-learn, OpenCV 
- **모델** :
  - 사전 학습된 ResNet50을 Backbone으로 사
  - Triplet Loss 사용
  - Cosine Similarity 기반 유사도 계산
- **학습 환경**:
  - Optimizer : SGD(lr=0.001)
  - Epochs : 300
  - Batchsize : 64
  - EarlyStopping 적용


