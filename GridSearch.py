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
