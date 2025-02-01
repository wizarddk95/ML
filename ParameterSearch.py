# =======================
# 기본 모델 평가
# =======================
def default_models(X, y):
    models = {
        "LogisticRegression": LogisticRegression(),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "CatBoost": CatBoostClassifier(),
        'LightGBM': lgbm.LGBMClassifier(verbose=-1),
        'XGBoost': xgb.XGBClassifier()
    }

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=5, scoring='f1', return_train_score=True)
        print(f'\n===== {name} 분류 모델 결과 =====')
        print("훈련 평균 f1 점수:", scores['train_score'].mean())
        print("검증 평균 f1 점수:", scores['test_score'].mean())


# =======================
# 랜덤서치
# =======================
def random_search(model, X, y):
    if 'lightgbm' in str(type(model)):
        param_distributions = {
            'num_leaves': randint(20, 150),  # 트리의 리프 노드 최대 개수 (커질수록 복잡한 모델)
            'max_depth': randint(3, 20),  # 트리의 최대 깊이
            'learning_rate': uniform(0.005, 0.3),  # 학습률 범위 확장 (너무 작은 값은 수렴 속도 저하)
            'n_estimators': randint(100, 1500),  # 부스팅 반복 횟수 (트리 개수)
            'subsample': uniform(0.7, 0.3),  # 데이터 샘플링 비율 (과적합 방지)
            'colsample_bytree': uniform(0.7, 0.3),  # 특성 샘플링 비율 (과적합 방지)
            'min_split_gain': uniform(0, 1),  # 새로운 노드를 생성하기 위한 최소 손실 감소 값
            'min_child_samples': randint(5, 200),  # 리프 노드 최소 샘플 수 (과적합 방지)
            'reg_alpha': uniform(0.0, 0.5),  # L1 정규화 (Lasso)
            'reg_lambda': uniform(0.0, 1),  # L2 정규화 (Ridge)
            'scale_pos_weight': uniform(1.0, 10.0)  # 클래스 불균형 조정
        }

    elif 'xgboost' in str(type(model)):
        param_distributions = {
            'max_depth': randint(3, 15),  # 트리의 최대 깊이 (일반적으로 3~15)
            'learning_rate': uniform(0.01, 0.3),  # 학습률 (0.01~0.3)
            'n_estimators': randint(100, 1000),  # 부스팅 반복 횟수 (100~1000)
            'subsample': uniform(0.6, 0.4),  # 데이터 샘플링 비율 (0.6~1.0)
            'colsample_bytree': uniform(0.6, 0.4),  # 트리 생성 시 사용할 특성 샘플링 비율 (0.6~1.0)
            'min_child_weight': randint(1, 20),  # 리프 노드의 최소 가중치 (1~20)
            'gamma': uniform(0, 5),  # 리프 노드 분할에 필요한 최소 손실 감소 (0~5)
            'reg_alpha': uniform(0, 1),  # L1 정규화 (Lasso)
            'reg_lambda': uniform(1, 10),  # L2 정규화 (Ridge)
            'scale_pos_weight': randint(1, 10)  # 클래스 불균형 조정
        }

    elif 'catboost' in str(type(model)):
        param_distributions = {
            'depth': randint(3, 12),  # 트리의 최대 깊이 (CatBoost 기본값: 6)
            'iterations': randint(100, 1500),  # 부스팅 반복 횟수
            'learning_rate': uniform(0.005, 0.3),  # 학습률 (너무 낮으면 학습이 느려짐)
            'l2_leaf_reg': uniform(0.0, 5.0),  # L2 정규화 (Ridge) (기본값: 3.0)
            'border_count': randint(10, 255),  # 숫자형 피처를 처리하는 구간 개수
            'bagging_temperature': uniform(0, 1),  # 배깅 강도 조절 (샘플링 다양성 증가)
            'subsample': uniform(0.7, 0.3),  # 샘플링 비율 (과적합 방지)
            'colsample_bylevel': uniform(0.7, 0.3),  # 트리 수준별 특성 샘플링 비율
            'scale_pos_weight': uniform(1.0, 10.0)  # 클래스 불균형 조정
        }

    elif 'RandomForest' in str(type(model)):
        param_distributions_rf = {
            'n_estimators': randint(100, 1500),  # 트리 개수 (너무 크면 학습 시간 증가)
            'max_depth': randint(3, 20),  # 트리의 최대 깊이
            'min_samples_split': randint(2, 20),  # 노드 분할을 위한 최소 샘플 수
            'min_samples_leaf': randint(1, 20),  # 리프 노드의 최소 샘플 수
            'max_features': uniform(0.5, 0.5),  # 사용할 특성 비율 (1.0 = 모든 특성 사용)
            'bootstrap': [True, False],  # 부트스트랩 샘플링 여부
            'class_weight': ['balanced', 'balanced_subsample', None]  # 클래스 불균형 조정
        }

    else:
        print("지원하지 않는 모델 유형")
        return

    # RandomizedSearchCV 설정
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=100,  # 랜덤 탐색 반복 횟수
        scoring='f1',  # F1-스코어 기준 최적화
        cv=5,  # 5-폴드 교차 검증
        verbose=1,
        n_jobs=-1,  # 병렬 처리
    )

    # 랜덤 서치 실행
    random_search.fit(X, y)

    # 최적 하이퍼파라미터 및 성능 출력
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X)
    print("최적 하이퍼파라미터:", random_search.best_params_)
    print("\n훈련세트 F1-스코어:", f1_score(y, y_pred))
    print("최적 검증세트 F1-스코어:", random_search.best_score_)

    # 모델 반환
    return best_model


# =======================
# 그리드 서치
# =======================
def grid_search(model, param, X, y):

    # GridSearchCV 설정
    grid_search = GridSearchCV(
        model,
        param_grid=param,
        scoring='f1',                   # F1 스코어를 기준으로 최적화
        cv=5,                           # 5-폴드 교차 검증
        verbose=1,
        n_jobs=-1                       # 병렬 처리
    )

    # 모델 학습
    grid_search.fit(X, y)

    # 최적 하이퍼파라미터 및 성능 출력
    print("최적 하이퍼파라미터:", grid_search.best_params_)

    # 최적 모델로 테스트 세트 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    print("\n훈련세트 F1-스코어:", f1_score(y, y_pred))
    print("검증세트 F1-스코어:", grid_search.best_score_)
    
    return best_model
    
# =======================
# 특성 중요도 시각화
# =======================
def feat_importance(model):

    if 'lightgbm' in str(type(model)):
        lgbm.plot_importance(best_model, max_num_features=15, importance_type='gain')  # 'split' or 'gain'
        plt.title("LightGBM Feature Importance")
        plt.show()

    elif 'xgboost' in str(type(model)):
        xgb.plot_importance(best_model, max_num_features=15, importance_type='gain')  # 'weight', 'gain', 'cover'
        plt.title("XGBoost Feature Importance")
        plt.show()

    elif 'catboost' in str(type(model)):
        feature_importance = model.get_feature_importance()
        indices = np.argsort(feature_importance)[::-1][:15]

        plt.barh(range(len(indices)), feature_importance[indices], align='center')
        plt.yticks(range(len(indices)), np.array(feature_names)[indices] if feature_names else indices)
        plt.xlabel("Feature Importance")
        plt.title("CatBoost Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()

    elif 'RandomForest' in str(type(model)):
        feature_importance = model.get_feature_importance()
        indices = np.argsort(feature_importance)[::-1][:15]

        plt.barh(range(len(indices)), feature_importance[indices], align='center')
        plt.yticks(range(len(indices)), np.array(feature_names)[indices] if feature_names else indices)
        plt.xlabel("Feature Importance")
        plt.title("RandomForest Feature Importance")
        plt.gca().invert_yaxis()  # 가장 중요한 특성이 위로 가도록 정렬
        plt.show()

    else:
        print("지원하지 않는 모델 유형")


