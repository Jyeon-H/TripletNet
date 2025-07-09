# 거리 이미지 기반 유사도 평가 시스템
<br>

## 💬 개요
TripletNet 기반 모델을 활용하여 거리 이미지를 비교하고, 유사도 정량화 및 시각화하는 시스템입니다. <br>
시각적으로 유사한 보행환경을 자동으로 추출할 수 있으며, 유사 환경 추천, 도시 분석 등에 활용할 수 있습니다.
<br><br>

## 📌 목적
- 기존 이미지 평가 방식의 문제점 개선
- Triplet 기반 딥러닝 아키텍처 설계 및 평가 지표 확보
- 이미지 간 유사도를 정량적으로 측정 가능한 방식 제안
<br><br>

## 🗂️ 데이터
- 본 프로젝트에 사용된 데이터는 사전 연구에서 수집되어 전달받은 자료로, <br>데이터 제공자의 요청 및 연구 윤리 준수에 따라 외부에 공개할 수 없습니다.
- **구성** : 21,138장의 거리 이미지 + 각 이미지에 대응하는 보행성 점수 21,138개
- **전처리** : 점수를 기준으로 Triplet 생성 (Anchor-Positive-Negative)
<br><br>

## 🔍 모델 및 방법
- **사용 기술** : Python, TensorFlow, scikit-learn, OpenCV 
- **모델 구성** :
![img](https://github.com/user-attachments/assets/686f9316-9229-49d2-aa4a-f4a2a3aa0ffa)
  - 사전 학습된 ResNet50을 Backbone으로 사용
  - Triplet Loss 기반 임베딩 학습
  - Cosine Similarity를 통한 이미지 간 유사도 계산
  - 학습 후 정량 평가 및 시각화
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
  | ![Image](https://github.com/user-attachments/assets/ce6e2df5-8118-4cc9-a9fc-069606e6a77e) | ![Image](https://github.com/user-attachments/assets/a41ff1b8-ff16-49d6-9d0c-415c4c958dd0) |  ![Image](https://github.com/user-attachments/assets/18ecac2d-f876-456b-8c14-831dd67357fa)|
- 향후 유사 환경 검색 서비스 또는 보행환경 개선 우선순위 분석 시스템에 적용 가능성을 제시함
<br><br>

## 🔁 회고
Triplet 구성 방식에 따라 학습 성능이 크게 달라지는 결과를 보며, 데이터 샘플링의 중요성을 체감함
