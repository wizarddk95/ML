import numpy as np
import cvxopt
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, kernel='linear', C=None, degree=3, gamma=None, coef0=1):
        """
        SVM 모델을 초기화
        - kernel: 사용할 커널 종류 ('linear', 'poly', 'rbf', 'sigmoid')
        - C: 소프트 마진 (Soft Margin) 설정 (None이면 하드 마진)
        - degree: 다항식 커널에서 차수 (poly 커널용)
        - gamma: RBF, 다항식, 시그모이드 커널의 γ 값 (None이면 1/n_features)
        - coef0: 다항식, 시그모이드 커널의 상수항
        """
        self.kernel_name = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = self._choose_kernel(kernel)

    def _choose_kernel(self, kernel):
        """ 커널 선택 함수 """
        if kernel == 'linear':
            return self._linear_kernel
        elif kernel == "poly":
            return self._polynomial_kernel
        elif kernel == "rbf":
            return self._rbf_kernel
        elif kernel == "sigmoid":
            return self._sigmoid_kernel
        else:
            raise ValueError("지원하지 않는 커널!")

    def _linear_kernel(self, x1, x2):
        """ 선형 커널 """
        return np.dot(x1, x2)

    def _polynomial_kernel(self, x1, x2):
        """ 다항식 커널 """
        return (np.dot(x1, x2) + self.coef0) ** self.degree

    def _rbf_kernel(self, x1, x2):
        """ RBF (Radial Basis Function) 커널 """
        if self.gamma is None:
            self.gamma = 1 / x1.shape[0]
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def _sigmoid_kernel(self, x1, x2):
        """ 시그모이드 커널 """
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

    def fit(self, X, y):
        """
        SVM 학습 함수
        주어진 X (입력 데이터)와 y (레이블)로 최적화 수행
        """
        n_samples, n_features = X.shape

        # 커널 행렬(Gram Matrix) 생성
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Quadratic Programming (QP) 최적화 문제 설정
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        # 제약 조건 설정 (소프트 마진 지원)
        if self.C is None: # 하드 마진 SVM
            G = cvxopt.matrix(np.diag(-np.ones(n_samples)))
            h = cvxopt.matrix(np.zeros(n_samples))
        else: # 소프트 마진 SVM
            G = cvxopt.matrix(np.vstack((np.diag(-np.ones(n_samples)), np.identity(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # QP 최적화 수행
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support Vector 찾기
        sv = alphas > 1e-5
        self.alpha = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        # w, b 계산 (w는 선형 커널에서만 사용 가능)
        if self.kernel_name == "linear":
            self.w = np.sum(self.alpha[:, None] * self.sv_y[:, None] * self.sv_X, axis=0)
        else:
            self.w = None # 비선형 커널에서는 w를 사용할 수 없음

        self.b = np.mean(self.sv_y - np.sum(self.alpha[:, None] * self.sv_y[:, None] * K[sv, sv], axis=0))
        
        return self

    def predict(self, X):
        """
        새로운 데이터 X에 대해 예측을 수행하는 함수
        """
        if self.kernel_name == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, sv_y, sv_x in zip(self.alpha, self.sv_y, self.sv_X):
                    s += alpha * sv_y * self.kernel(X[i], sv_x)
                y_pred[i] = s
            return np.sign(y_pred + self.b)

    def plot(self, X, y):
        """
        학습된 SVM 모델을 시각화하는 함수 (2D 데이터만 가능)
        """
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        plt.scatter(self.sv_X[:, 0], self.sv_X[:, 1], s=200, facecolors='none', edgecolors='k', linewidth=2)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                Z[i, j] = np.sum([alpha * sv_y * self.kernel(np.array([xx[i, j], yy[i, j]]), sv_x) 
                                  for alpha, sv_y, sv_x in zip(self.alpha, self.sv_y, self.sv_X)])
        Z = Z + self.b
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.title(f"SVM Decision Boundary ({self.kernel_name} Kernel)")
        plt.show()
            
