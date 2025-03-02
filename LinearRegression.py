import numpy as np

# 1. 정규방정식
class LinearRegression:
    def __init__(self):
        self.theta = None # 회귀 계수

    def fit(self, X, y):
        """
        선형 회귀 학습 (정규방정식 활용)
        :param X: 입력 데이터 (feature)
        :param y: 타겟 값
        :return: 학습된 모델
        """
        # X에 절편을 위한 1 추가
        X_b = np.c_[X, np.ones((X.shape[0], 1))]
        # 정규방정식 계산
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        return self

    def predict(self, X):
        """
        학습된 모델을 사용하여 예측
        """
        X_b = np.c_[X, np.ones((X.shape[0], 1))]

        return X_b.dot(self.theta)

# 2. 경사하강법
class LinearRegressionGD:
    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.learning_rate = learning_rate # 학습률
        self.n_iter = n_iter # 반복횟수
        self.theta = None

    def fit(self, X, y):
        """
        경사 하강법을 이용한 학습 (가중치 업데이트)
        """
        m, n = X.shape

        X_b = np.c_[X, np.ones((m, 1))]

        # theta 초기화 (0 또는 랜덤)
        self.theta = np.zeros((n + 1, 1))

        # 경사하강법 수행
        for _ in range(self.n_iter):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(self.theta) - y) # 기울기 계산
            self.theta -= self.learning_rate * gradients # 가중치 업데이트

        return self

    def predict(self, X):
        """
        학습된 모델을 사용하여 예측
        """
        X_b = np.c_[X, np.ones((X.shape[0], 1))]

        return X_b.dot(self.theta)
