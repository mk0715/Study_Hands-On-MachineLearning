## 개요
- 가장 좋은 모델 하나만 사용하는게 아니라, 이미 만든 괜찮은 일련의 예측기(분류나 회귀 모델)를 연결하여 더 좋은 예측기를 만드는 방법이 앙상블.
- 그 종류에는 투표기반(Voting), 배깅(Bootstrap Aggregationg, Bagging), 부스팅(Boosting), 스태킹(Stacking)등이 있음.
- 결정트리(Decision Tree)의 앙상블이 랜덤포레스트(RandomForest). -> 모든 개별 트리의 예측을 구하고 가장 많은 선택을 받은 클래스를 예측으로 삼음. 

## 7.1 투표 기반 분류기(Voting)
- 어떤 데이터셋에 대해 여러 분류기(로지스틱 회귀, SVM, 랜덤포레스트 등등)을 훈련.

![image](https://user-images.githubusercontent.com/75361137/150737040-b3b27205-1d98-452e-b0fe-d55513d35a07.png)

- 여기서 더 좋은 분류기를 만들기 위해 각 분류기의 예측을 모아 다수결 투표로 정하는 것을 직접 투표(hard voting)분류기.

![image](https://user-images.githubusercontent.com/75361137/150737756-beb3dfd1-f3b1-4711-a864-3590234a167b.png)
- 훈련데이터가 많고 다양하다면, 직접 투표 분류기가 개별 분류기 중 가장 뛰어난 것보다 정확도가 높은 경우가 많음. -> 대수의 법칙

### *앙상블 방법은 예측기가 가능한 서로 독립적일 때 최고의 성능 발휘 -> 다양한 분류기를 얻는 방법은 각기 다른 알고리즘으로 학습. -> 매우 다른 종류의 오차를 만들 가능성이 높아 모델의 정확도를 향상시킴.*
### *투표식 분류기는 앙상블에 포함된 분류기들 사이의 독립성이 전제되면 개별 분류기보다 정확하나, 독립성이 보장되지 못하면 더 낮아질 수 있음.*

- 모든 분류기가 범주마다의 예측확률을 예측할 수 있으면(predict_proba() 메서드가 있으면) 개별 분류기의 예측을 평균해서 확률이 가장 높은 범주를 예측. -> 간접 투표(soft voting)방식.

![image](https://user-images.githubusercontent.com/75361137/150740784-8d98f2b5-4d27-4961-89d4-ac466793cce2.png)
- 직접 투표 방식과 차이점은 직접 투표는 범주가 무엇인지에 따라 결정하고, 간접 투표는 해당 범주에 속할 확률을 평균내어 평균이 가장 높은 것에 결정.
- 즉, 단순히 개별 분류기의 예측결과만을 고려하지 않고 높은 확률값을 반환하는 모델의 비중을 고려할 수 있기 때문에 성능이 좋음.
- cf) SVC에서는 클래스 확률을 제공하지 않으므로, probability=True옵션을 주어 predict_proba() 메서드를 사용.


## 7.2 배깅과 페이스팅(Bagging & Pasting)
- 같은 알고리즘을 사용하고 훈련세트의 서브셋을 무작위로 구성하여 학습시킬 때, 훈련세트에서 중복을 허용하여 샘플링하면 배깅(Bootstrap Aggregating, Bagging), 중복을 허용하지 않으면 페이스팅(pasting).

![image](https://user-images.githubusercontent.com/75361137/150742319-03ff07a4-9f1b-45c9-ac58-c047ccd8e476.png)
- 예측기들이 훈련을 마치면 앙상블은 예측을 모아 새로운 샘플에 대한 예측 생성.
- 분류일 때는 통계적 최빈값(like hard voting), 회귀일 때는 평균(mean) 계산. -> 이 수집함수를 통과하면 개별 예측기보다 편향과 분산이 모두 감소.
- 일반적으로 앙상블 모형은 원본 데이터셋으로 하나의 예측기를 훈련시킬 때보다 분산이 감소.

### 7.2.1 사이킷런의 배깅과 페이스팅
- BaggingClassifier으로 배깅 및 페이스팅 사용.
- bootstrap=True로 지정하면 배깅, False로 지정하면 페이스팅 사용.

![image](https://user-images.githubusercontent.com/75361137/150743101-ecae15e9-7bcb-4e63-9b79-a849fb13d1c7.png)
- 결정경계를 통해 앙상블의 예측이 더 일반화가 잘 이루어짐.
- 부트스트래핑(중복을 허용한 샘플링)은 각 예측기가 학습하는 서브셋에 다양성을 증가 시킴. -> 편향은 배깅이 더 높으나 예측기 간의 상관관계를 줄여주기 때문에 분산은 더 낮음.
- 일반적으로 배깅이 더 나은 모델을 만드나, 여유가 있다면 두개 다 검증해보는 것이 좋음.

### 7.2.2 oob 평가
- 배깅 방식은 중복을 허용하여 훈련샘플에서 선택되지 않는 샘플이 나오는데, 평균적으로 63%정도 샘플되고 나머지 37%는 선택되지 않음. -> 이 37%이 oob(out-of-bag)샘플.
- 예측기가 훈련되는 동안에 oob 샘플은 사용하지 않으며, 앙상블의 평가는 각 예측기의 oob 평가를 평균하여 얻음.
- Scikit-learn에서는 oob_score=True로 지정하여 사용.
- oob 샘플에 대한 결정함수의 값은 oob_decision_function_ 으로 확인 가능.

## 7.3 랜덤 패치와 랜덤 서브스페이스
- BaggingClassifier은 변수 샘플링 지원. -> max_feature, bootstrap_features로 조절 가능.
- 랜덤 패치 방식(Random Patches method) : 훈련 변수와 샘플을 모두 샘플링
- 랜덤 서브스페이스 방식(Random Subspaces method) : 훈련 샘플은 모두 사용(bootstrap=False, max_samples=1), 변수는 샘플링(bootstrap_features=True, max_features=0~1값).
- 변수를 샘플링하면 더 다양한 예측기를 만들고 편향을 늘리지만 분산은 줄임.

## 7.4 랜덤 포레스트(RandomForest)
- 배깅 또는 페이스팅을 적용한 결정 트리의 앙상블로 랜덤 포레스트를 사용할수도 있지만 RandomForestClassifier(RandomForestRegressor)로 사용.
- RandomForestClassifier은 (예외가 있지만) DecisionTreeClassifier와 BaggingClassifier의 매개변수를 모두 가지고 있음.
- 예외로는 splitter(항상 best), presort(항상 False), max_samples(항상 1), base_estimator(항상 지정된 매개변수를 사용한 DecisionTreeClassifier)
- 랜덤포레스트는 트리의 노드를 분할할 때 전체 특성 중 최선의 특성을 찾는 대신 무작위로 선택한 특성 후보 중 최적의 특성을 찾는 식으로 무작위성을 더 주입.
- 이는 트리를 더욱 다양하게 만들고 편향을 손해보는 대신에 분산을 낮추어 전체적으로 더 훌륭한 모델을 만듬.

### 7.4.1 엑스트라 트리(ExtraTree)
- 랜덤포레스트의 핵심은 트리를 더욱 무작위하게 만들기 위해 최적의 임곗값을 찾는 대신 후보 특성을 사용해 무작위로 분할하여 최상의 분할을 선택.
- 이 원리를 더욱 극단적으로 무작위하게 사용한 것이 익스트림 랜덤트리(Extremely Randomized Trees), 줄여서 엑스트라 트리(Extra-Trees).
- 일반적인 랜덤 포레스트 보다 속도가 훨씬 빠름.
- Scikit-learn에서는 ExtraTressClassifier(ExtraTreesRegressor)로 사용.

### 7.4.2 특성 중요도(Feature_importances_)
- 랜덤 포레스트의 또 다른 장점은 변수의 상대적 중요도를 측정하기 쉽다는 것.
- 어떤 변수를 사용한 노드가 전체 트리에 대해 평균적으로 불순도를 얼마나 감소시키는지 확인하여 측정. -> 가중치 평균이며 각 노드의 가중치는 연관된 훈련 샘플 수.
- DecisionTree의 변수 중요도 = (현재 노드의 샘플 비율 x 불순도) - (왼쪽 자식 노드의 샘플 비율 x 불순도) - (오른쪽 자식 노드의 샘플 비율 x 불순도)
- 변수 중요도의 합이 1이 되도록 전체 합으로 나누어 정규화.
- 샘플 비율이란 (해당 노드의 샘플 수) / (전체 샘플 수), 랜덤 포레스트의 변수 중요도는 (각 DecisionTree의 변수 중요도의 합) / (트리 수).
- Scikit-learn에서는 훈련 후 변수마다 이 중요도를 자동으로 계산하고 전체 합이 1이 되도록 정규화. -> feature_importances_ 변수에 저장되어 있음.

![image](https://user-images.githubusercontent.com/75361137/150757554-64dd38a8-dfa3-4180-a105-6a195e061097.png)

## 7.5 부스팅
- 부스팅은 약한 학습기를 여러개 연결하여 강한 학습기를 만드는 앙상블 방법. -> 앞의 모델을 보완하면서 일련의 예측기를 학습.
- 여러가지 존재하지만 가장 인기 있는 것은 에이다부스트(AdaBoost), 그레이디언트 부스팅(Gradient Boosting)

## Bagging과 Boosting 차이점
- Bagging은 병렬적으로 학습, Boosting은 순차적으로 학습.
- Boosting은 가중치를 사용.
- 사용 시 개별 결정 트리의 낮은 성능이 문제이면 Boosting 사용, Overfitting이 문제이면 Bagging 사용.

### 7.5.1 에이다부스트(AdaBoost)
- AdaBoost는 Adaptive Boosting의 줄임말로 이전 모델이 과소적합했던 훈련 샘플의 가중치를 더 높이는 방식. -> 새로운 예측기는 학습하기 어려운 샘플에 점점 더 맞춰짐.

![image](https://user-images.githubusercontent.com/75361137/150758260-a4ccb050-7ec0-433e-9be2-79d8bfa88f28.png)
- 과정
- 1. 기반이 되는 첫 번째 분류기를 훈련 세트에서 훈련시키고 예측값을 생성.
- 2. 잘못 분류된 훈련 샘플의 가중치를 상대적으로 높임.
- 3. 업데이트 된 가중치를 사용해 훈련 세트에서 두번째 분류기를 훈련시키고 예측값 생성.
- 4. 2~3 과정 반복

![image](https://user-images.githubusercontent.com/75361137/150759888-1e53a930-c32b-484b-b579-24746b70f2a0.png)
- 첫번째부터 다섯번째까지 학습을 진행할수록 더 정확하게 분류. -> learning_rate가 0.5인 경우가 더 일반화 가능.
- 경사하강법과 비슷한 방식. -> 경사하강법은 비용 함수를 최소화하도록 파라미터를 조정하고, AdaBoost는 예측이 정확해지도록 앙상블에 가중치를 조정.
- 모든 예측기가 훈련을 마치면 배깅이나 페이스팅과 비슷하게 예측을 만듬. 그러나 가중치가 적용된 훈련 세트의 전반적인 정확도에 따라 예측기마다 다른 가중치 적용.
- 연속된 학습 기법은 각 예측기가 훈련되고 평가된 후에 학습할 수 있기 때문에 병렬화가 불가능. -> 배깅만큼 확장성이 높지 않음.

- 알고리즘 자세히 들여다 보는 것은 pass.(p.259~261)

- Scikit-learn에서는 AdaBoostClassifier(AdaBoostRegressor)을 사용. algorithm으로는 SAMME.R 사용(Default). -> SAMME도 사용 가능.
- AdaBoost가 훈련 세트에 과대적합되면 추정기 수를 줄이거나 추정기의 규제를 더 강하게 주는 방법 고려.

### 7.5.2 그레이디언트 부스팅(Gradient Boosting)
- Gradient Boosting도 AdaBoost처럼 이전 예측기의 오차를 보정하도록 예측기를 순차적으로 추가하지만, 샘플의 가중치를 수정하는게 아니라 이전 예측기가 만든 잔여오차(residual error)에 새로운 예측기를 학습.
- 회귀 문제로 이를 적용시켜봄.

![image](https://user-images.githubusercontent.com/75361137/150763965-c559abc5-aaf2-42c0-a54e-63a3ebc19fb9.png)
- 왼쪽열은 세개의 트리에 대한 개별 예측, 오른쪽 열은 각각의 예측기의 잔여오차로 재학습하여 예측값을 더한 앙상블 모델.
- 결정 트리를 회귀 문제에 이용하면 예측값을 특정 구간에 대한 평균값을 이용하므로 계단 형태를 띄게 됨.
- 트리가 앙상블에 추가될수록 예측이 점점 더 좋아지는 것을 볼 수 있음. -> 이 과정을 그래디언트 부스티드 회귀트리(Gradient Boosted Regression Tree, GBRT).
- Scikit-learn에서는 GradientBoostingRegressor을 사용.
- learning_rate 매개변수로 각 트리의 기여도를 조절. -> 0.1처럼 낮게 설정하면 앙상블을 훈련세트에 학습시키기 위해 필요한 트리는 많아지지만 성능은 좋아짐.

![image](https://user-images.githubusercontent.com/75361137/150764697-9ac2ea17-4399-498a-bfab-3c19af4cff40.png)
- 왼쪽은 훈련세트를 학습하기에 트리가 충분하지 않고, 오른쪽은 트리가 너무 많아 과대적합. 트리수는 n_estimators로 조절.
- 이 경우에 최적의 트리수를 구하기 위해 staged_predict() 메서드를 사용.
- 이 메서드는 훈련의 각 단계(트리가 1개일때, 2개일때..)에서 앙상블에 의해 만들어진 예측기를 순회하는 반복자(iterator)을 반환.

![image](https://user-images.githubusercontent.com/75361137/150765178-943d4bec-714c-4aee-b050-dc34d1be5baa.png)
- 위 방법처럼 120개의 트리를 모두 훈련시키고 최적의 수를 찾기 위해 검증오차를 분석하는 대신, 조기 종료를 사용.
- warm_start=True로 설정하면 fit() 메서드를 호출할 때 기존 트리를 유지하고 훈련을 추가. -> n번 반복동안 검증 오차가 향상 되지 않으면 훈련 중지 설정.
- subsample 변수로 각 트리가 훈련할 때 사용할 훈련 샘플의 비율 지정. -> 이는 편향이 높아지는 대신 분산이 낮아지며, 훈련속도가 향상 됨. -> 확률적 그레이디언트 부스팅(Stochastic Gradient Boosting)
- loss 매개변수를 통해 Gradient Boosting에 다른 비용함수를 사용 가능. (GradientBoostingClassifier의 Default는 deviance, GradientBoostingRegressor의 Default는 ls)

### 7.5.3 XGBoost
- 익스트림 그레이디언트 부스팅(Extreme Gradient Boosting)의 약자.
- 이 패키지의 목표는 매우 빠른 속도, 확장성, 이식성.
- import xgboost를 하여 사용.
- 자동 조기 종료(early_stopping_rounds)와 같은 여러 가지 좋은 기능 제공. 

### *이외에도 히스토그램 기반 그레이디언트 부스팅인 LightGBM등이 있음.*

## 7.6 스태킹
- 앙상블에 속한 모든 예측기의 예측을 취합하는 함수를 사용하는 대신 취합하는 모델을 훈련시키고자 하는 것에서 시작.

![image](https://user-images.githubusercontent.com/75361137/150767376-d0dd7344-80f2-49a1-90f5-92ef15760ade.png)
- 3개의 예측기가 새로운 샘플에 대해 각 예측값을 반환했고, 블렌더(Blender) 또는 메타 학습기(meta learner)라는 마지막 예측기가 3개의 예측을 학습해 최종 예측값 반환.

![image](https://user-images.githubusercontent.com/75361137/150767661-33751fea-aee0-41ee-82bc-a128a4edbe5e.png)

- 블렌더는 일반적으로 홀드 아웃 세트를 사용하여 학습.
- 1. 먼저 훈련 세트는 2개의 subset으로 나눔.
- 2. subset1은 첫 번째 레이어의 예측기들을 훈련시키는데 사용.

![image](https://user-images.githubusercontent.com/75361137/150767879-7860979f-7153-4170-8299-6be6307c2460.png)

- 3. 훈련된 첫번째 레이어의 예측기로 subset2에 대한 예측을 생성.(subset이지만 훈련에 사용이 안되어있어 test셋 처럼 사용가능)
- 4. 생성된 세개의 예측값의 타깃값(y)은 그대로 쓰고 앞에서 예측한 값(ŷ)을 입력 변수로 사용하는 새로운 훈련세트 생성(3차원).
- 5. 블렌더가 새로운 훈련 세트로 학습.(즉, 첫번째 레이어의 예측 3개를 이용해 y를 예측하도록 학습)

- 이런 방식으로 여러개의 블렌더를 훈련 시키는 것도 가능. ex) 하나는 LinearRegression, 하나는 RandomForestRegressor..

![image](https://user-images.githubusercontent.com/75361137/150768657-be3c1a34-c58a-43ab-b420-937e1b06405d.png)

- 블렌더 만의 레이어가 또 만들어지므로 이렇게 하기 위해서는 훈련 세트를 3개의 subset으로 분리.
- subset1은 첫번째 레이어의 예측기들을 훈련.
- subset2는 첫번째 예측기들의 예측을 만들어 블렌더 레이어의 예측기를 위한 훈련세트를 생성.(여기까지는 위랑 똑같음)
- subset3는 두번째 예측기들의 예측을 만들어 세번째 레이어를 훈련시키기 위한 훈련세트를 만드는데 사용.(subset1이랑 똑같지만 두번째 예측기)
- Scikit-learn은 Stacking 지원하지 않음. -> 직접 구현.

