import numpy as np
import matplotlib.pyplot as plt

def plotData(X1, X2, X3, X4):
    plt.scatter(X1[:, 0], X1[:, 1], marker='+', c='k', label='$\omega_1$')
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c='y', label='$\omega_2$')
    plt.scatter(X3[:, 0], X3[:, 1], marker='v', c='r', label='$\omega_3$')
    plt.scatter(X4[:, 0], X4[:, 1], marker='*', c='b', label='$\omega_4$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.show()

def batchPerceptron(omega1, omega2):
    w = np.array([0, 0, 0])
    rho = 0.01
    theta = 10**(-3)
    t = 0
    max_iter = 10**4
    l = np.size(omega1, 0)
    omega1 = np.append(omega1, np.ones((l, 1)), 1)
    omega2 = np.append(omega2, np.ones((l, 1)), 1)
    N = np.size(omega1, 1)

    while t < max_iter:
        t = t+1
        grad = np.zeros((1, N))
        for i in range(l):
            if np.dot(omega1[i, :], w)*1 <= 0:
                grad = grad + rho*(-1)*omega1[i, :]
                grad = np.reshape(grad,(1,N))
        for i in range(l):
            if np.dot(omega2[i, :], w)*(-1) <= 0:
                grad = grad + rho*(+1)*omega2[i, :]
                grad = np.reshape(grad,(1,N))

        if np.linalg.norm(grad) < theta:
            break

        w = w - grad
        w = np.squeeze(w)

    print("Number of iterations:",t)
    return w

def plotTwoClassesBound(omega1, omega2, w, name1, name2):
    x1_max = max(omega1.max(0)[0], omega2.max(0)[0])
    x1_min = min(omega1.min(0)[0], omega2.min(0)[0])
    x1 = np.linspace(x1_min, x1_max, 1000)
    x2 = (1/w[1])*(-w[0]*x1 - w[2])
    plt.scatter(omega1[:, 0], omega1[:, 1], marker='+', c='k', label='$\omega_'+name1+'$')
    plt.scatter(omega2[:, 0], omega2[:, 1], marker='o', c='y', label='$\omega_'+name2+'$')
    plt.plot(x1,x2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.show()

omega1 = np.array([[0.1, 1.1],
                   [6.8, 7.1],
                   [-3.5, -4.1],
                   [2, 2.7],
                   [4.1, 2.8],
                   [3.1, 5],
                   [-0.8, -1.3],
                   [0.9, 1.2],
                   [5, 6.4],
                   [3.9, 4]])

omega2 = np.array([[7.1, 4.2],
                   [-1.4, -4.3],
                   [4.5, 0],
                   [6.3, 1.6],
                   [4.2, 1.9],
                   [1.4, -3.2],
                   [2.4, -4],
                   [2.5, -6.1],
                   [8.4, 3.7],
                   [4.1, -2.2]])

omega3 = np.array([[-3, -2.9],
                   [0.5, 8.7],
                   [2.9, 2.1],
                   [-0.1, 5.2],
                   [-4, 2.2],
                   [-1.3, 3.7],
                   [-3.4, 6.2],
                   [-4.1, 3.4],
                   [-5.1, 1.6],
                   [1.9, 5.1]])

omega4 = np.array([[-2, -8.4],
                   [-8.9, 0.2],
                   [-4.2, -7.7],
                   [-8.5, -3.2],
                   [-6.7, -4],
                   [-0.5, -9.2],
                   [-5.3, -6.7],
                   [-8.7, -6.4],
                   [-7.1, -9.7],
                   [-8, -6.3]])
                   

plotData(omega1, omega2, omega3, omega4)

print("Batch Perceptron Algorithm for classes 1 and 2")
w = batchPerceptron(omega1, omega2)
print("w",w)
plotTwoClassesBound(omega1,omega2,w,"1","2")

print("Batch Perceptron Algorithm for classes 2 and 3")
w = batchPerceptron(omega2, omega3)
print("w",w)
plotTwoClassesBound(omega2,omega3,w,"2","3")

print("Batch Perceptron Algorithm for classes 3 and 4")
w = batchPerceptron(omega3, omega4)
print("w",w)
plotTwoClassesBound(omega3,omega4,w,"3","4")
