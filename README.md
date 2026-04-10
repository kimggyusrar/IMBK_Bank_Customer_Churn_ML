# 프로젝트명 : 고객 이탈 분류 ML 및 인사이트 분석

---
## period : 2026.04.10 

## Tech Stack

### Language & Environment
* Language: Python
* Environment: Jupyter Notebook, Google Colab

### Data Science & ML
* Data Analysis: Pandas, Numpy, Matplotlib, Seaborn
* AutoML & Library: PyCaret, Optuna
* Machine Learning Models:
  * Gradient Boosting, AdaBoost, Decision Tree, QDA
  * Ensemble: Stacking Classifier (Meta-model: Logistic Regression)
* XAI (Explainable AI): SHAP (TreeExplainer)

## Data source
* kaggle Bank Customer Churn Dataset : row: 10000, col:12

## Data preprocessing
데이터의 품질을 높이고 모델의 왜곡을 방지하기 위해 단계별 전처리를 수행했습니다.
Feature Cleaning: 고유값인 `customer_id`를 제거하여 모델의 과적합(Overfitting) 방지.
2. Encoding: 범주형 변수인 `country`, `gender`에 `LabelEncoder`를 적용하여 수치화.
3. Data Splitting: `stratify=y` 설정을 통해 이탈/비이탈 클래스 비율을 유지하며 Train/Valid 세트 분할 (8:2).
4. Scaling: `StandardScaler`를 적용하여 피처 간 단위 차이를 조정, 특히 메타 모델(Logistic Regression)의 수렴 속도 개선.

## Exploratory Data Analysis (EDA) & Interpretation
데이터의 특성을 파악하고 이탈(Churn)에 영향을 미치는 주요 변수를 식별하기 위해 심층 분석을 수행했습니다.

### 1. 타겟 변수 분포 (Target Distribution)
<img width="865" height="630" alt="image" src="https://github.com/user-attachments/assets/69aaa9d0-847a-4c69-becd-227a5caf3324" />

* 현황: 비이탈 고객(0)이 이탈 고객(1)보다 압도적으로 많은 **데이터 불균형(Imbalanced Data)** 상태임을 확인했습니다.
* 전처리: 모델 학습 시 `stratify` 옵션을 적용하고, 성능 지표로 Accuracy 대신 **F1-Score**를 우선순위로 두어 예측 신뢰도를 높였습니다.

### 2. 변수 간 상관관계 (Correlation Analysis)
<img width="818" height="717" alt="image" src="https://github.com/user-attachments/assets/d273f13a-396b-45e5-840e-6e8c908952e3" />

* Heatmap 분석: `churn`과 가장 높은 양의 상관관계를 보이는 변수는 **나이(Age)**로 나타났습니다.
* 해석: 연령대가 높을수록 이탈 위험이 커지는 경향이 있으며, 이는 은퇴 후 자산 이동이나 상품 만족도 변화와 관련이 있을 것으로 추정됩니다.

### 3. 주요 변수별 상세 분석

#### Age (연령)
<img width="817" height="565" alt="image" src="https://github.com/user-attachments/assets/48206d01-3882-4cc4-9e63-2efbd1dc3c01" />

* KDE Plot 분석: 40대 중후반부터 이탈 고객의 밀도가 급격히 높아지는 양상을 보입니다.
* 인사이트: 고연령층 고객을 유지하기 위한 전용 멤버십이나 건강 관리 연계 금융 상품 등의 리텐션 전략이 필요합니다.

#### Balance (잔액)
* 분석 결과: 잔액이 높을수록 이탈률이 오히려 높게 나타나는 역설적인 현상이 발견되었습니다.
* 인사이트: 고액 자산가는 금리에 민감하여 상품 만기 시 타 은행으로 자금을 이동할 가능성이 큽니다. 만기 전 재예치 혜택 제공 등 선제적 대응이 요구됩니다.

#### Geography (국가별 특성)
* 분석 결과: 인코딩 데이터 분석 결과, **독일(Germany)** 고객의 이탈률이 다른 국가(프랑스, 스페인)에 비해 현저히 높습니다.
* 인사이트: 이는 당시 독일의 경제 상황이나 지역 내 경쟁 은행의 공격적인 마케팅 등의 외부 요인이 작용했을 가능성이 큼을 시사합니다.

#### Number of Products & IsActiveMember
* 분석 결과: 보유 상품 수가 많고 활동적인 회원(Active Member)일수록 이탈률이 낮습니다.
* 인사이트: 고객이 은행의 다양한 서비스를 동시에 이용하도록 유도(Cross-selling)하는 것이 이탈 방지에 핵심적인 역할을 합니다.

## AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value
### SHAP Value Analysis Process
* Explainer: 트리 기반 모델에 최적화된 `TreeExplainer`를 활용하여 피처별 기여도 산출.
* Visualization: `summary_plot`을 통해 각 변수가 예측값(이탈 확률)에 미치는 영향력을 시각화.
* Stacking F1: 0.5816485225505443
<img width="945" height="662" alt="1" src="https://github.com/user-attachments/assets/d3efbc5b-4f63-43be-b662-4f2319667ef7" />

### 핵심 분석 결과 (Core Results)
1. Age (결정적 요인): SHAP 분석 결과 나이가 많을수록 이탈 확률에 압도적인 양(+)의 영향을 미침. 40대 중후반 고객층의 이탈 방지가 비즈니스의 핵심 과제임을 증명.
2. Number of Products: 보유 상품 수가 적을수록 이탈 위험이 급격히 상승함. 다각화된 금융 상품 이용(Cross-selling)이 충성도 유지의 핵심 지표임을 확인.
3. Geography (Germany): 타 국가 대비 독일 고객의 기여도가 높게 나타나며, 지역적 특성에 따른 이탈 위험이 실존함을 데이터로 입증.
4. Balance (High-Value Asset): 고액 잔액 보유자가 이탈 확률이 높게 나타나는 현상을 포착. 이는 자산 이동성이 높은 고액 자산가들을 위한 전담 리텐션 전략의 필요성을 시사함.

---

##  Project Conclusion
본 프로젝트는 AutoML → Optuna Tuning → Stacking으로 이어지는 고도화된 모델링 프로세스를 통해 예측 성능을 확보했으며, SHAP 분석을 통해 데이터 기반의 구체적인 비즈니스 액션 아이템(고연령층 케어, 고액 자산가 리텐션 등)을 도출하는 성과를 거두었습니다.


##  Strategic Insights & Recommendations

SHAP 기반 사후 분석과 모델의 예측 결과를 바탕으로, 은행의 고객 이탈 방지를 위한 3가지 핵심 비즈니스 전략을 제안합니다.

### 1. 연령별 맞춤형 리텐션(Retention) 프로그램 강화
* Insight: 분석 결과 고연령층(40대 중후반 이상) 고객이 이탈에 가장 취약한 타겟임이 증명되었습니다.
* Action: 
  * 중장년층을 위한 연금 연계 금융 상품 및 자산 관리 서비스 라인업 확대.
  * 디지털 뱅킹에 익숙하지 않은 세대를 위한 오프라인 밀착 케어 서비스나 사용 편의성을 높인 시니어 전용 앱 UI 제공.

### 2. 고액 자산가 대상 '로열티 잠금(Lock-in)' 전략
* Insight: 잔액(Balance)이 높은 고객일수록 이탈 확률이 상승하는 양상을 보입니다. 이는 고액 자산가들이 금리 및 혜택 변화에 민감하게 반응하여 상품 만기 시 자금을 이동시키기 때문으로 분석됩니다.
* Action: 
  * 고액 잔액 유지 고객에게 우대 금리뿐만 아니라 비금융 서비스(문화, 의료, 세무 상담 등)를 결합한 VIP 멤버십 강화.
  * 상품 만기 전 재예치 시 추가 인센티브를 제공하는 선제적 프로모션 실행.

### 3. 다상품 이용 유도 및 활성 회원 전환 (Cross-selling)
* Insight: 보유 상품 수(Number of Products)가 적거나 비활성 상태인 고객의 이탈 위험이 매우 높습니다.
* Action: 
  * 신규 고객이 2개 이상의 상품을 이용하도록 유도하는 번들링 패키지(예: 급여통장+적금+카드 결합) 마케팅 전개.
  * 활동성이 낮은 고객에게 정기적인 맞춤형 금융 리포트를 발송하거나, 앱 접속 시 혜택 알림 등을 통해 활성 회원(Active Member)으로의 전환 유도.

### 4. 국가별 특화 마케팅 (Germany Focus)
* Insight: 타 국가 대비 독일 지역 고객의 이탈률이 유독 높은 특이점(Outlier)이 발견되었습니다.
* Action: 
  * 독일 내 경쟁 은행의 상품 경쟁력을 벤치마킹하여 해당 지역 전용 우대 상품 출시.
  * 현지 정치·경제 상황에 맞춘 맞춤형 금융 메시징 및 지역 특화 프로모션 운영.

---

## Conclusion
본 프로젝트는 **데이터 전처리 - 모델링(Stacking) - 최적화(Optuna) - 해석(SHAP)**으로 이어지는 머신러닝의 전 과정을 체계적으로 수행했습니다. 

단순히 예측 성능(F1-Score)을 높이는 것에 그치지 않고, 이탈의 핵심 원인을 데이터로 증명하고 이를 해결하기 위한 **구체적인 비즈니스 액션 아이템을 도출**했다는 점에서 큰 의의가 있습니다.
