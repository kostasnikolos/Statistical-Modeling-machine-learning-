import numpy as np
import matplotlib.pyplot as plt


def calculateMLE_1D(data):
    # calculate μmle= 1/N*Σ(xi)
    mean= np.mean(data, axis=0)
    
    # calculate σ^2mle = 1/Ν*Σ(xi- μ)^2
    sigma= np.var(data,axis=0 )
    
    return mean, sigma
    
    
def calculateMLE(data):
    # calculate μmle= [μ1,μ2,μ3] where μ=1/N*Σ(xi) 

    mean= np.mean(data, axis=0)
    
    # calculate the covariance matrix Σ = 1/N*Σ(yi - μy)(xi - μx) with vars in the diagram and covs at the others where var(x) = 1/Ν*Σ(xi- μ)^2 and cov(x1,x2)= (x1 - μ)*(χ2 - μ)
    covariance_matrix= np.cov(data, rowvar= False , bias= True)
    
    return mean, covariance_matrix

def calculateMLE_separate_data(data):
    # calculate μmle= 1/N*Σ(xi)

    mean= np.mean(data, axis=0)
    
    
    
    # calculate the covariance matrix Σ = 1/N*Σ(yi - μy)(xi - μx) with vars in the diagram  Σ= diag(σ1^2,σ2^2,σ3^2)
    
    # calulate the variances of every xi first
    variances= np.var(data, axis=0)

    # put the variances in the covariance matrix with all others zero
    covariance_matrix= np.diag(variances)

    return mean, covariance_matrix
    
# firstly import the samples from class 1 and 2
samples_w1 = np.array([
    [0.42, -0.087, 0.58],
    [-0.2, -3.3, -3.4],
    [1.3, -0.32, 1.7],
    [0.39, 0.71, 0.23],
    [-1.6, -5.3, -0.15],
    [-0.029, 0.89, -4.7],
    [-0.23, 1.9, 2.2],
    [0.27, -0.3, -0.87],
    [-1.9, 0.76, -2.1],
    [0.87, -1, -2.6]
])

samples_w2 = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063]
])


#1)
# Ο τύπος για την υπολογισμό του μmle για 1-D κανονική κατανομή είναι  μmle= 1/N*Σ(xi) ενώ για το σ^2mle = 1/Ν*Σ(xi- μ)^2 
# χωρίζουμε τα δεδομένα στα 3 διαφορετικά χαρακτηριστικά και υπολογίζουμε για το κάθε ένα ξεχωριστά
# calculate μmle and σ^2mle for x1 features
x1_w1= samples_w1[:, 0]
mean_x1_w1 , sigma_x1_w1 = calculateMLE_1D(x1_w1)
print("\n1) 1-D distirbution ")
print("mean xx1_w1: ",mean_x1_w1)
print("sigma x1_w1",sigma_x1_w1)

# calculate μmle and σ^2mle for x2 features
x2_w1= samples_w1[:, 1]
mean_x2_w1 , sigma_x2_w1 = calculateMLE_1D(x2_w1)

print("\nmean x2_w1:",mean_x2_w1)
print("mean sigma x2_w1:",sigma_x2_w1)

# calculate μmle and σ^2mle for x3 features
x3_w1= samples_w1[:, 2]
mean_x3_w1 , sigma_x3_w1 = calculateMLE_1D(x3_w1)
print("\nmean x3_w1:",mean_x3_w1)
print("sigma x3_w1:",sigma_x3_w1)

#2)
# Χωρίζουμε τα δεδομένα σε ζευγάρια δύο χαρακτηριστικών και υπολογίζουμε για το κάθε ένα τις παραμέτρους:  Μέση τιμή μ= ( μχ , μy)  και πίνακας συνδιακύμανσης Σ = 1/N*Σ(xi - μx)(xi - μx)T
# όπου χρησιμοποιούνται η διακύμανση var(x) = 1/Ν*Σ(xi- μ)^2  και οι συνδιακυμάνσεις μεταξύ των μεταβλητών cov(x1,x2)= (x1 - μ)*(χ2 - μ)
#split the w1 data in 3 different pairs
pairs = [

    samples_w1[:, [0, 1] ], samples_w1[:, [0, 2] ], samples_w1[:, [1, 2] ]
    
]

# caclulate the paraters mean and variance for the 2D pairs of class w1
paramaters_2D= [ calculateMLE(pair) for pair in pairs]
print("\n2 2-D couples of variables) ")
for i in range (0,3):
    print(f"For the {i+1}-th couple of variables (2-D distirbution) the mean (2x1) is : {paramaters_2D[i][0]} and the covariance (2x2) is:{paramaters_2D[i][1]}")
    variance_2D= [paramaters_2D[i][1][0][0],paramaters_2D[i][1][1][1]]
    print(f" The variances of the xi's are: {variance_2D}\n")
    
#3)
# Πλέον χρησιμοποιούμε και τα 3 χαρακτηριστικά που δίνονται στα δεδομένα μας υπολογίζοντας τις παραμέτρους της 3-D κανονικής κατανομής που
# προκύπτει όπως και στο προηγούμενο ερώτημα

# the way of calculating the parrametes for the 3D gaussian is the same as the 2D
parameters_3D = calculateMLE(samples_w1)
variance_3D= [parameters_3D[1][0][0],parameters_3D[1][1][1],parameters_3D[1][2][2]]
print("\n3) 3-D distirbution")
print(f"For 3-D distirbution the mean (3x1) is : {parameters_3D[0]} and the covariance (3x3) is:{parameters_3D[1]}")
print(f"The variances of the xi's are: {variance_3D}")

#4)
# Για το συγκεκριμένο ερώτημα υποθέτουμε ότι το 3-D μοντέλο μας έιναι διαχωρίσιμο και τα 3 χαρακτηριστικά xi i= 1,2,3 της κλάσης ω2
# δεν συσχετίζονται μεταξύ τους .Επομένως οι συνδιακυμάνσεις cov(x1,x2) είναι ίσες με το μηδέν και ο συνολικός 
# πίνακας συνδιακύμανσης είναι διαγώνιος , αποτελούμενος μόνο από την διακύμανση των χαρακτηριστικών Σ= diag(σ1^2,σ2^2,σ3^2).
# Επομενώς υπολογίζουμε τη μέση τιμή μmle= [x1, x2 ,x3] όπως προηγουμένως και την πίνακα συνδιακύμανσης ως Σ= [var(x1) , var(x2) ,var(x3)]

print("\n4) 3-D distirbution wirh seperated data")
parameters_separate_data= calculateMLE_separate_data(samples_w2)
print(f"For 3-D distirbution the mean (3x1) is : {parameters_separate_data[0]} and the covariance (3x3) is:{parameters_separate_data[1]}")
