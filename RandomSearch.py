def random_search(model, X, y):
    if 'lightgbm' in str(type(model)):
        # =============================
        # LGBM 하이퍼 파라미터 범위 설정
        # =============================
        param_distributions = {
            'num_leaves': randint(20, 150),              # 트리의 리프 노드 최대 개수 (커질수록 복잡한 모델)
            'max_depth': randint(3, 20),                 # 트리의 최대 깊이
            'learning_rate': uniform(0.005, 0.3),        # 학습률 범위 확장 (너무 작은 값은 수렴 속도 저하)
            'n_estimators': randint(100, 1500),          # 부스팅 반복 횟수 (트리 개수)
            'subsample': uniform(0.7, 0.3),              # 데이터 샘플링 비율 (과적합 방지)
            'colsample_bytree': uniform(0.7, 0.3),       # 특성 샘플링 비율 (과적합 방지)
            'min_split_gain': uniform(0, 1),             # 새로운 노드를 생성하기 위한 최소 손실 감소 값
            'min_child_samples': randint(5, 200),        # 리프 노드 최소 샘플 수 (과적합 방지)
            'reg_alpha': uniform(0.0, 0.5),              # L1 정규화 (Lasso)
            'reg_lambda': uniform(0.0, 1),               # L2 정규화 (Ridge)
            'scale_pos_weight': uniform(1.0, 10.0)       # 클래스 불균형 조정
        }
    elif 'xgboost' in str(type(model)):
        # ================================
        # XGBoost 하이퍼 파라미터 범위 설정 
        # ================================
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

    # RandomizedSearchCV 설정
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=100,                     # 랜덤 탐색 반복 횟수
        scoring='f1',                   # F1-스코어 기준 최적화
        cv=5,                           # 5-폴드 교차 검증
        verbose=1,
        n_jobs=-1,                      # 병렬 처리
    )

    # 랜덤 서치 실행
    random_search.fit(X, y)

    # 최적 하이퍼파라미터 및 성능 출력
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X)
    print("최적 하이퍼파라미터:", random_search.best_params_)
    print("\n훈련세트 F1-스코어:", f1_score(y, y_pred))
    print("최적 검증세트 F1-스코어:", random_search.best_score_)

    #모델 반환
    return best_model
