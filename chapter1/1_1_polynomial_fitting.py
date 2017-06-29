# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# ランダムなデータの再現性を取るためにseed値を固定
np.random.seed(0)


def calc_y(x, w):
    return np.array([np.sum([w[i]*(x[n]**i) for i in range(w.size)]) for n in range(x.size)])


def estimate(x, t, M):
    A = np.array([[(x**(i+j)).sum() for j in range(M+1)] for i in range(M+1)])
    T = np.array([(x**(i)*t).sum() for i in range(M+1)])
    return np.linalg.solve(A, T)


def main():
    # x軸の点の数
    N = 1000

    # サンプル数
    n = 300

    # モデルの次数
    M = 9

    # sin(2πx)の関数を作る
    x_true = np.linspace(0, 1, N)
    y_true = np.sin(2*np.pi*x_true)

    # サンプル点を計算する
    x_sample = np.array([i/n+0.05 for i in range(n)])
    y_sample = []
    for i in range(n):
        # 真の関数に正規分布に従うノイズをかける
        y_sample.append(np.sin(2*np.pi*x_sample[i]) + np.random.randn()*0.3)
    y_sample = np.array(y_sample)

    # wの計算
    w = estimate(x_sample, y_sample, M)
    y_data = calc_y(x_true, w)

    # 描画
    plt.plot(x_true, y_true)
    plt.scatter(x_sample, y_sample)
    plt.plot(x_true, y_data)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-2., 2])
    plt.show()

    

if __name__ == "__main__":
    main()
