import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta = None
        self.threshold = threshold

    def sigmoid(self, z):
        """
        로지스틱(Logistic) 함수 
        = 시그모이드(Sigmoid) 함수
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        로지스틱 회귀 학습
        """
        m, n = X.shape
        X_b = np.c_[X, np.ones((m, 1))] # 절편 추가
        self.theta = np.zeros((n + 1, 1)) # 가중치 초기화
        y = y.reshape(-1, 1) # (m,) → (m, 1) 변환

        for _ in range(self.n_iter):
            z = X_b.dot(self.theta) # 선형 조합
            h = self.sigmoid(z) # 시그모이드
            gradients = (1 / m) * X_b.T.dot(h - y) # 경사 계산
            self.theta -= self.learning_rate * gradients # 가중치 업데이트

        return self

    def predict_proba(self, X):
        """
        확률 예측
        """
        X_b = np.c_[X, np.ones((X.shape[0], 1))]
        return self.sigmoid(X_b.dot(self.theta))

    def predict(self, X):
        """
        이진 분류 결과 예측 (0 또는 1)
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)
