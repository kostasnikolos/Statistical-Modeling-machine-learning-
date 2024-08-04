import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

# Randomly select initial centroids from the dataset.
def initialize_centroids(X, K):
    ### YOUR CODE HERE

    centroids = 0
    return centroids


# Finds the closest centroid for each sample
def find_closest_centroids(X, centroids):
    ### YOUR CODE HERE

    K = centroids.shape[0]
    samples = X.shape[0]
    idx = samples*[0]
    for j in range(samples):
        distances = np.zeros((K, 1))
        for i in range(K):
            distances[i] = np.linalg.norm(X[j,:] - centroids[i,:])**2
        idx[j] = np.argmin(distances)
    return idx

# Compute the mean of samples assigned to each centroid
def compute_centroids(X, idx, K):
    ### YOUR CODE HERE

    C_k = np.zeros((K, 1))
    mu_k = np.zeros((K, X.shape[1]))
    for i in range(X.shape[0]):
        C_k[idx[i]] += 1
        mu_k[idx[i], :] += X[i, :]

    centroids = np.divide(mu_k, C_k)
    return centroids

# K-means algorithm for a specified number of iterations
def run_kmeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Initialize centroids randomly
def kmeans_init_centroids(X, K):
    ### YOUR CODE HERE

    idx = np.random.randint(X.shape[0], size=K)
    initial_centroids = X[idx, :]
    return initial_centroids

# Load the image
image = io.imread('Fruit2.png')
#image = resize(image, (256, 256))  # Resize for faster processing, if needed
#rows, cols, dims = image.shape

# Size of the image
img_size = image.shape

# Normalize image values in the range 0 - 1
### YOUR CODE HERE

image = image / image.max()
print("image.shape",image.shape)


# Reshape the image to be a Nx3 matrix (N = num of pixels)
X = image.reshape(img_size[0] * img_size[1], img_size[2])

# Perform K-means clustering
K = 64
max_iters = 10

# Initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters)


# K-Means Image Compression
print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = find_closest_centroids(X, centroids)

# Recover the image from the indices
X_recovered = centroids[idx]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], img_size[2])

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()
