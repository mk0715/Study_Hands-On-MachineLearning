## 3.1 MNIST
- 학습용 데이터로 많이 쓰임. 

## 3.2 이진 분류기 훈련
- 분류 모델을 하나 선택함. 여기서는 SGDClassifier을 선택. 
- SGD가 한번에 하나씩 훈련 샘플을 독립적으로 처리하기 때문에 매우 큰 데이터셋을 효율적으로 처리할 수 있음. 참고로 SGD는 훈련하는 데 무작위성을 사용하므로, 재현을 위해서는 random_state 매개변수 지정.

## 3.3 성능 측정
### 3.3.1 교차 검증을 사용한 정확도 측정
- 교차 검증에는 두가지 방법이 있음. 
- 1. scikit-learn의 cross_val_score 함수사용.
- 2. StratifiedKFold를 사용하여 직접 교차 검증 구현. 
- 두번째 방법은 scikit-learn이 제공하는 기능보다 더 많이 검증과정을 제어할 때 사용.
- 정확도는 불균형 Dataset(한쪽 class가 월등히 많은 경우)일 때 성능 측정 지표로 선호하지 않음.

### 3.3.2 오차행렬(confusion matrix)
- 오차행렬을 만들기 위해서는 실제 타깃과 비교할 수 있도록 예측값을 만들어야함. cross_val_predict 함수 사용. 
- cf. cross_val_score은 점수 반환, cross_val_predict는 예측값 반환
- confusion matrix의 행은 실제 클래스, 열은 예측한 클래스 

![다운로드](https://user-images.githubusercontent.com/75361137/148682479-16e331db-3d44-4827-9ac8-a7113c1f665c.jpg)
- 실제와 예측값이 동일하면 True, 다르면 False가 붙고, 예측값이 Positive면 □P, Negative면 □N로 나타냄.

![20220109_223054](https://user-images.githubusercontent.com/75361137/148684505-fe5f26ea-a8eb-4d90-b1dd-76fd595e3d83.png)

![20220109_223104](https://user-images.githubusercontent.com/75361137/148684516-3ee544fb-62d4-48e4-b086-c19873092d9d.png)

- Precision : 양성 예측의 정확도. 분류기가 다른 모든 양성 샘플을 무시하기 때문에 유용하지 않음. 주로 recall과 같이 사용
- Recall : 분류기가 정확하게 감지한 양성 샘플의 비율.

### 3.3.3 정밀도와 재현율(Precision and Recall)
![20220109_223632](https://user-images.githubusercontent.com/75361137/148684520-1eec4638-df07-4fe8-b8b5-8b3daad08ecb.png)

- F1 score : Precision과 Recall의 조화 평균. 두 분류기를 비교할 때 유용하게 사용.
- 상황에 따라 Precision이 중요할 때도 있고, Recall이 중요할 때도 있음.
- 이 두가지를 모두 높게 얻을 수는 없기에, Precision이 오르면 Recall이 내려가고 그 반대도 마찬가지 -> 이를 Precision/Recall trade-off라고 함.

### 3.3.4 정밀도/재현율 트레이드오프(Precision/Recall trade-off)
![ch3fig4](https://user-images.githubusercontent.com/75361137/148684822-9a204439-35c9-4fc7-b1b9-369adb9c2576.png)

- 결정 임곗값(Threshold)에 따라 Precision과 Recall이 달라짐. Threshold가 올라갈 수록 Precision이 커지고, Recall은 작아짐.
- 기초적으로는 Threshold가 올라가면 FP가 감소하고, FN이 증가함.
- 몇 가지 성질로 풀어보자면, 
- Threshold 증가 -> Precision 증가, Recall 감소.
- Threshold 감소 -> Precision 감소, Recall 증가.
- FP 감소 & FN 증가 -> Precision 증가
- FP 증가 & FN 감소 -> Recall 증가 
- 결과적으로 두 지표는 Trade-off 관계이며, 상황에 따라 Precision이 중요한 상황과 Recall이 중요한 상황이 있음.
- 예시) 1. 암환자를 구별할 때, Threshold를 낮춰서 Recall을 높이는 것이 좋음. 그 이유는 실제로 암에 걸리지 않은 환자가 나올 수는 있으나, 실제로 암에 걸린 환자는 구별이 확실히 가능하기 때문.
- 예시) 2. 판사가 재판을 할 때, Threshold를 높여서 Precision을 높이는 것이 좋음. 무고한 사람이 죄를 선고 받으면 안되기 때문.

### 3.3.5 ROC 곡선
![1_7wxQ1ZymPqM-nj0ZCF1KHA](https://user-images.githubusercontent.com/75361137/148691728-5d440614-0377-4912-98a8-eb9028fb7e02.png)

- ROC 곡선은 거짓양성비율 즉, 양성으로 잘못 분류된 음성 샘플의 비율(FPR)에 대한 진짜양성비율(TPR)의 곡선.
- 곡선 아래의 면적이 AUC.
- ROC 곡선은 PR(Precision/Recall)곡선과 비슷한데, 일반적으로 양성 클래스가 드물거나 거짓 음성보다 거짓 양성이 더 중요할 때 PR곡선을 사용하고 그 외에는 ROC 곡선을 사용함.

## 3.4 다중분류
- 다중 분류기는 둘 이상의 클래스를 구별함.
- 대표적으로 SGDClassifier, RandomForestClassifier, Naive Bayes 같은 알고리즘은 여러 개의 클래스를 직접 처리.
- 반면, LogisticRegression, SupportVectorMachine(SVC) 등의 알고리즘은 이진 분류만 가능. 그러나 이를 여러 개 사용하여 다중 클래스를 분류하기도 함.
- 예를 들어, 특정 숫자 하나를 구분하는 이진 분류기 10개로 클래스가 10개인 숫자 이미지 분류 시스템을 만들 수 있음. 각 분류기의 결정 점수 중 가장 높은 것을 클래스로 선택하는 방식. -> OvR(one-versus-the-rest)전략 또는 OvA(one-versus-all)이라고 정의.
- 또 다른 방식으로 각 숫자의 조합마다 이진 분류기를 훈련. 이는 OvO(one-versus-one)전략이라고 함. 총 분류기는 Nx(N-1)/2개 필요.
- SVC등의 일부 알고리즘은 훈련세트의 크기에 민감해서 작은 훈련세트에서 많은 분류기를 훈련시키는 것이 빠르기에 OvO를 선호하나, 대부분 OvR을 선호.

## 3.5 에러분석
- 성능을 향상시키기 위해 만들어진 에러의 종류를 분석.
- 분류기는 보통 이미지의 위치나 회전 방향에 매우 민감하기에, 이미지를 중앙에 위치시키고 회전되어 있지 않도록 전처리 하면 에러가 줄어들 수 있음.(MNIST)

## 3.6 다중 레이블 분류
- 여러 개의 이진 꼬리표를 출력하는 분류 시스템.
- 예를 들어, 같은 사진에 사람이 여러명 나오면 [1,0,1] 등으로 출력.
- 다중 레이블 분류를 평가하는 방법은 많은데, 각 레이블의 F1 score을 구하고 평균 점수를 계산.
- 레이블에 클래스의 지지도(샘플 수)를 가중치로 주고 분류기 마다 점수를 다르게 주기도 함.

## 3.7 다중 출력 분류
- 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화 한 것.(값을 두개 이상 가지는 것)
