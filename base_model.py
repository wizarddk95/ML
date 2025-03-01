import numpy as np
import lightgbm as lgbm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

def default_models(X, y, cv=5, scoring='accuracy'):
    """
    다양한 분류 모델을 비교하는 함수
    - 기본적으로 StratifiedKFold를 사용하여 클래스 비율을 유지하면서 교차 검증 수행
    - 사용자가 원하는 평가지표(scoring)를 변경 가능 (기본값: 'accuracy')
    
    Parameters:
        X (numpy.ndarray or pandas.DataFrame): 특징 데이터
        y (numpy.ndarray or pandas.Series): 타겟 데이터
        cv (int, optional): StratifiedKFold의 폴드 개수 (기본값: 5)
        scoring (str, optional): 모델 평가 기준 (기본값: 'accuracy')

    Returns:
        dict: 모델별 교차 검증 결과
    """

    models = {
        "LogisticRegression": LogisticRegression(),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_jobs=-1),
        "LightGBM": lgbm.LGBMClassifier(verbose=-1),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)  # ✅ StratifiedKFold 적용

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=True, n_jobs=-1)

        train_score_mean = np.round(scores["train_score"].mean(), 4)  # 소수점 4자리
        test_score_mean = np.round(scores["test_score"].mean(), 4)

        print(f"\n===== {name} 분류 모델 결과 =====")
        print(f"훈련 평균 {scoring}: {train_score_mean}")
        print(f"검증 평균 {scoring}: {test_score_mean}")
        print("====================================")
