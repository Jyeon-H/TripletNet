# 거리 이미지 기반 유사도 평가 시스템
<br>

## 💬 개요
ResNet 기반 TripletNet를 활용하여 거리 이미지 유사도 정량화 및 시각화 시스템
<br><br>

## 📌 목표
- 유사 이미지 비교를 위한 딥러닝 아키텍처 설계
- 이미지 간 유사도를 정량적으로 측정 가능한 모델 구현
- 유사한 보행환경을 가진 이미지 추출 및 시각화
<br><br>

## 🙋🏻‍♀️ 수행 역할
- 전체 파이프라인 설계 및 단독 구현 (데이터 수집 -> 모델 학습 -> 성능 평가)
- ResNet 기반 Triplet Network 모델 구현
- Triplet Loss 설계 및 margin에 따른 성능 비교
- 정량 평가(Cosine Similarity)와 정성 평가(유사 이미지 결과 시각화)
<br><br>

## 🗂️ 데이터
- **구성** : 21,138장의 거리 이미지 <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 각 이미지에 대응하는 보행성 점수 21,138개
- **전처리** : 점수를 기준으로 Triplet 무작위 생성 (Anchor-Positive-Negative)
  |-|Anchor|Positive|Negative|
  |---|---|---|---|
  |Range|Random extractor|Anchor+-5|Anchor+-30|
  |Example|65 score|60~70 score|0\~35 or 95\~100 score|
  |sample|<img width="235" height="247" alt="image" src="https://github.com/user-attachments/assets/bc91f698-65c3-47ec-bd7e-a6c346ed3fa9" />|<img width="236" height="247" alt="image" src="https://github.com/user-attachments/assets/0c03045c-6f53-4511-98ab-a61beeceb3ad" />|<img width="236" height="248" alt="image" src="https://github.com/user-attachments/assets/9fb5f399-8d76-43f2-8d31-2151590d60e1" />|
- 본 프로젝트에 사용된 데이터는 사전 연구에서 수집되어 전달받은 자료로, 데이터 제공자의 요청에 따라 공개할 수 없습니다.
<br><br>

## 🔍 모델 및 방법
- **주요 기술** : Python, TensorFlow, ResNet50, Triplet Loss, OpenCV, Pandas 
- **모델 구성** :
![img](https://github.com/user-attachments/assets/686f9316-9229-49d2-aa4a-f4a2a3aa0ffa)
  - 사전 학습된 ResNet50을 Backbone으로 사용
  - Triplet Loss 기반 임베딩 학습 <br><img width="500" height="35" alt="image" src="https://github.com/user-attachments/assets/ac4f175a-b037-4079-860e-56aa222fa97f" />
  - Cosine Similarity를 통한 이미지 간 유사도 계산
- **학습 환경**:
  - *Margin* : 1.0, 0.5, 0.3
  - *Optimizer* : SGD (learning rate=0.001)
  - *Epochs* : 300
  - *Batchsize* : 64
  - *EarlyStopping* 적용
- **평가 방법** :
  - 평가 지표를 활용한 성능 평가
  - 유사 이미지 추출 결과 시각화
  <br><br>

## 📊 주요 결과
- Margin에 따른 성능 비교
  |Margin|F1-score|Accuracy|
  |---|---|---|
  |1.0|0.9158|0.8447|
  |0.5|0.9059|0.8281|
  |0.3|0.8771|0.7812|
- **TripletNetwork(1.0)** 성능
  |Precision|Recall|F1-score|Accuracy|
  |---|---|---|---|
  |1.0|0.8446|0.9158|0.8447|
- 유사한 거리 이미지 시각화 예시
  | Test Image | Fisrt Similar | Second Similar |
  |--------------|----------------|----------------|
  | <img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/0928e5b8-a18f-40a4-ae80-463837a454f9" /> | <img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/bc46c716-04ce-4aa1-9b92-a1d8c8b49a2f" /> | <img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/548129b2-2912-4bb7-bd82-2ef702c86564" /> |
  |<img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/ba1190bc-6432-47c4-8703-87fa2e162900" />|<img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/ef7de523-12b6-4465-97ce-2b6ce7d5f3ce" /> | <img width="290" height="280" alt="image" src="https://github.com/user-attachments/assets/6dd28aa4-17c8-4cdb-b854-f6d7e6e70131" /> |
- 향후 유사 환경 검색 서비스 또는 보행환경 개선 우선순위 분석 시스템에 적용 가능성을 제시함.
<br><br>

## 🔁 회고
Margin 값을 조정함에 따라 모델 성능이 변화하는 것을 관찰하였고, 이를 통해 하이퍼파라미터 튜닝과 반복적인 실험의 중요성을 직접 체감함.<br> 
Precision이 1.0으로 나타난 것을 통해 데이터 불균형이 모델 학습에 미치는 영향을 파악하게 되었고, 데이터 샘플링 젼략의 중요성을 인식할 수 있었음.
