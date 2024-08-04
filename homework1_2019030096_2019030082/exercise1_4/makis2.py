import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plotContours(mean1, mean2, sigma1, sigma2):
    # Creating a grid of evenly spaced points in two-dimensional space
    x1 = np.arange(-2, 8, 0.2)
    x2 = np.arange(-2, 8, 0.2)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.ravel(), X2.ravel()))

    # Calculate probabilities for each class
    p_x_w1 = multivariate_normal(mean=mean1, cov=sigma1).pdf(X).reshape(len(x2), len(x1))
    p_x_w2 = multivariate_normal(mean=mean2, cov=sigma2).pdf(X).reshape(len(x2), len(x1))

    # Plotting the bounded probabilities and decision boundaries
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.contour(X1, X2, p_x_w1, 20, colors='b', alpha=0.5)
    ax.contour(X1, X2, p_x_w2, 20, colors='r', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Projection of the Gaussian functions Countours of each class onto x-y plane')
    plt.grid(True)
    plt.show()
    plt.show()

def plotContours_with_Boundaries(mean1, mean2, sigma1, sigma2):
    # Creating a grid of evenly spaced points in two-dimensional space
    x1 = np.arange(-2, 8, 0.2)
    x2 = np.arange(-2, 8, 0.2)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.ravel(), X2.ravel()))

    # Calculate probabilities for each class
    p_x_w1 = multivariate_normal(mean=mean1, cov=sigma1).pdf(X).reshape(len(x2), len(x1))
    p_x_w2 = multivariate_normal(mean=mean2, cov=sigma2).pdf(X).reshape(len(x2), len(x1))

    # Plotting the bounded probabilities and decision boundaries
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.contour(X1, X2, p_x_w1, 20, colors='b', alpha=0.5)
    ax.contour(X1, X2, p_x_w2, 20, colors='r', alpha=0.5)

    # C
    P_Class1 = [0.1, 0.25, 0.5, 0.75, 0.9]
    colors = ['r', 'g', 'b', 'y', 'k']

    # For each P_Class1 plot the resulting decision boundary
    for i, p1 in enumerate(P_Class1):
        p2 = 1 - p1
        decision_boundary = (np.log(p1) - np.log(p2)) + (np.log(p_x_w1) - np.log(p_x_w2))
        ax.contour(X1, X2, decision_boundary, levels=[0], colors=colors[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Projection of the Gaussian functions Countours of each class with Decision Boundaries')
    plt.grid(True)
    plt.show()

# Defining means and sigmas from the exercise's data
mean1 = [2, 3]
mean2 = [4, 4]
sigma1 = [[2, 0.5], [0.5, 1]]
sigma2 = [[1.5, -0.3], [-0.3, 0.8]]



####################  DO THE EXERCISE  ##################

# Question 2: Plot contours of the distributions
plotContours(mean1, mean2, sigma1, sigma2)

# Question 3: Plot decision boundaries for different P(ω1) values
plotContours_with_Boundaries(mean1, mean2, sigma1, sigma2)

# Question 4: Plot decision boundaries with the same  P(ω) values
sigma_common= [[1.2, 0.4], [0.4, 1.2]]
plotContours_with_Boundaries(mean1, mean2, sigma_common, sigma_common)



################  Παρατηρησεις και Σχολια  ######################

# 3)
#   Αρχικά παρατηρείται ότι για διαφορετικές τιμές πινάκων Σ προκύπτει ως όριο απόφασης μια καμπύλη λόγω της διαφορετικότητας των δύο καμπανών
#   Εφόσον στο παράδειγμα μας δίνεται ότι οι πιθανότητες των δύο Gaussian συναρτήσεων είναι αντίθετες ,  όσο αυξάνεται η τιμή του P(w1) ,
#   τόσο το σύνορο απόφασης μετακινείται προς την κλάση ω1 δίνοντας περισσότερη σημασία σε αυτή.

# 4)
#  Στην περίπτωση που οι τιμές των πινάκων Σ είναι ίδιες  προκύπτει ότι το όριιο απόφασης είναι μία ευθεία γραμμή καθώς οι δύο καμπάνες είναι ίδιες και διαφέρουν μόνος ως προς το κέντρο τους
