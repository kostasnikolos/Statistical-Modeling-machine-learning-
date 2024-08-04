"""
Author: Nikolos Konstantinos, Bekos Konstantinos
Date: 19/5/2024
Description: This script creates a simple Bayesian Classifier in order to classify the number 1,2 given in images (pixels) 
             by creating 3 featuers (aspect-ratio, sum of foreground pixels, centroid of pixels)
"""


import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


class MyBayesClassifier:
  def __init__(self):
    self.class_priors = {}
    self.class_stats = {}

  def train(self, X, y):
    """
    Train the classifier under the assumption of Gaussian distributions:
      calculate priors and Gaussian distribution parameters for each class.

    Args:
    X (pd.DataFrame): DataFrame with features.
    y (pd.Series): Series with target class labels.
    """
    self.classes_ = np.unique(y)
    
    # number of total samples
    total_samples = len(X)
    for class_label in self.classes_:
      # Filter data by class
      X_class = X[y == class_label]
      
      

      # Calculate prior probability for the class
      self.class_priors[class_label] = len(X_class)/ total_samples


      # Calculate mean and covariance for the class
      # Adding a small value to the covariance for numerical stability
      mean = X_class.mean()       

      cov = X_class.cov() + np.eye(X_class.shape[1]) * 1e-8  # Adding small value for numerical stability
      self.class_stats[class_label] = {'mean': mean, 'cov': cov}



  def predict(self, X):
    """
    Predict class labels for each test sample in X.

    Args:
    X (pd.DataFrame): DataFrame with features to predict.

    Returns:
    np.array: Predicted class labels.
    """
    # predict for the features of every image
    predictions = [self._predict_instance(x) for x in X.to_numpy()]
    return np.array(predictions)

  def _predict_instance(self, x):
    """
    Private helper to predict the class for a single instance.

    Args:
    x (pd.Series): A single data point's features.

    Returns:
    The predicted class label.
    """
    posteriors = []

    # Calculate the posterior probability for each class
    # posterior= likelihood*prior
    for class_label in self.classes_:
        likelihood = multivariate_normal.pdf(x, mean=self.class_stats[class_label]['mean'], cov=self.class_stats[class_label]['cov'])
        posterior = likelihood * self.class_priors[class_label]
        posteriors.append(posterior)
    


    # Choose the class with the highest posterior probability
    prediction = self.classes_[np.argmax(posteriors)]
    return prediction

  def _calculate_likelihood_1D(self, x, mean, cov):
    """
    Calculate the Gaussian likelihood of the data x given class statistics.

    Args:
    x (pd.Series): Features of the data point.
    mean (pd.Series): Mean features for the class.
    cov (pd.DataFrame): Covariance matrix of the features for the class.

    Returns:
    float: The likelihood value.
    """
    # create likelihood  aka the class density p(x/wi)
    likelihood = multivariate_normal.pdf(x, mean=mean, cov=cov)
    return likelihood


# Calculate the bounding box to test
def calculate_bounding_box(image):
  # Find non-zero foreground pixels
  nonzero_pixels = np.nonzero(image)
  # Check if there are any foreground pixels
  if nonzero_pixels[0].size == 0:
    return np.nan  # Return NaN if no foreground pixels found

  # Get minimum and maximum coordinates of foreground pixels
  # find min,max row of a nonzero pixel
  min_row = np.min(nonzero_pixels[0])
  max_row = np.max(nonzero_pixels[0])
  
  # find min,max column of a nonzero pixel
  min_column = np.min(nonzero_pixels[1])
  max_column = np.max(nonzero_pixels[1])

  return min_column, min_row, max_column, max_row

# Function to calculate aspect ratio
def aspect_ratio(image):
  """Calculates the aspect ratio of the bounding box around the foreground pixels."""
  try:
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)

    # Find non-zero foreground pixels
    # returns 2 matrices ,[0] for the rows, [1] for the columns
    nonzero_pixels = np.nonzero(img)

    # Check if there are any foreground pixels
    if nonzero_pixels[0].size == 0:
      return np.nan  # Return NaN if no foreground pixels found

    # Get minimum and maximum coordinates of foreground pixels
    # find min,max row of a nonzero pixel
    min_row = np.min(nonzero_pixels[0])
    max_row = np.max(nonzero_pixels[0])
    
    # find min,max column of a nonzero pixel
    min_column = np.min(nonzero_pixels[1])
    max_column = np.max(nonzero_pixels[1])
    
    # Calculate bounding box dimensions
    # calculate minimum boundind box dimension
    
    # calculate  height given the max row pixel and the min row pixel
    height = max_row - min_row +1  
    # calculate with given the max and min column pixel
    width = max_column - min_column + 1

    # Calculate aspect ratio
    aspect_ratio = width / height

    return aspect_ratio

  except (KeyError, ValueError) as e:
    print(f"Error processing image in row {image.name}: {e}")
    return np.nan  # Return NaN for rows with errors

def foreground_pixels(image):
  """
  Calculate the pixel density of the image, defined as the
  count of non-zero pixels

  Args:
  image (np.array): A 1D numpy array representing the image.

  Returns:
  int: The pixel density of the image.
  """
  try:
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)

    # Find non-zero foreground pixels
    nonzero_pixels =  np.count_nonzero(img)
    if nonzero_pixels == 0:
      print(f"Warning: Couldn't find nonzero pixels on  {image.name}")
      return np.nan  # Return NaN if no foreground pixels found
  except (KeyError, ValueError) as e:
    print(f"Error processing image in row  {image.name}: {e}")
    return np.nan  # Return NaN for rows with errors

  return nonzero_pixels

def calculate_centroid(image):
    """
    Calculate the normalized centroid (center of mass) of the image.

    Returns:
    tuple: The (x, y) coordinates of the centroid normalized by image dimensions.
    """
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)
    rows, cols = img.shape
    # rows= number of rows in an image
    # cols = number of cols in an image
    
    # total sum of values for all pixels
    total_mass = np.sum(img)
    # (each column) * (sum of pixels in column) / (total mass)
    x_center = np.sum(np.arange(cols) * np.sum(img, axis=0)) / total_mass
    # (each row) * (sum of pixels in row) / (total mass)
    y_center = np.sum(np.arange(rows) * np.sum(img, axis=1)) / total_mass
    ## Create a single scalar as a centroid feature using x+(y * w) where w is the width of the image  , width = #of cols
    centroid = x_center + (y_center * cols)
    return centroid

def min_max_scaling(X, min_val=-1, max_val=1):
  """Scales features to a range between min_val and max_val."""
  #perfrom min max scaling xi^ = min + (max-min)* (xi-min)
  X_scaled = min_val + (max_val - min_val)*(X - X.min()) / (X.max() - X.min())
  return X_scaled


def visualize_bounding_box(image, color='red'):
  """Visualizes the bounding box around the digit in an image."""
  bbox = calculate_bounding_box(image)

  # Create a drawing object
  sample_image_img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
  scaling = 10
  sample_image_XL = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)

  draw = ImageDraw.Draw(sample_image_img)
  # Draw the rectangle with desired fill color and outline (optional)
  draw.rectangle(bbox, outline=color, width=1)

  sample_image_XL.show()
  sample_image_XL_bbox = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)
  sample_image_XL_bbox.show()




##############################################################################
######    MAIN - CREATE FEATURES - TRAIN (NAIVE) BAYES CLASSIFIER
##############################################################################
def main():

  # Read the training samples from the corresponding file
  nTrainSamples = 10000 # specify 'None' if you want to read the whole file
  df_train = pd.read_csv('data/mnist_train.csv', delimiter=',', nrows=nTrainSamples)
  df_train = df_train[df_train['label'].isin([1, 2])] # Get samples from the selected digits only
  target_train = df_train.label
  data_train = df_train.iloc[:, 1:]

  # Read the test samples from the corresponding file
  nTestSamples = 1000 # specify 'None' if you want to read the whole file
  df_test = pd.read_csv('data/mnist_test.csv', delimiter=',', nrows=nTestSamples)
  df_test = df_test[df_test['label'].isin([1, 2])] # Get samples from the selected digits only
  target_test = df_test.label
  data_test = df_test.iloc[:, 1:]


















  #################### Create the features #############################
  # Calculate aspect ratio as the first feature
  df_train['aspect_ratio'] = data_train.apply(aspect_ratio, axis=1)
  
  ######## EXERCISE 5.1) Print min,max aspect ratio
  print("Minimum aspect ratio (train):", df_train['aspect_ratio'].min())
  print("Maximum aspect ratio (train):", df_train['aspect_ratio'].max())
  
  
   ####### EXERCISE 5.2) Draw sample images with the bounding box calculated in order to make sure aspect ratio is correct

  # Draw 2 sample images from the training data to make sure aspect ratio is correct
  for sample in range (2):
    sample_image = data_train.iloc[sample].values.reshape(28, 28)
    visualize_bounding_box(sample_image)

  
  ######## EXERCISE 5.3 Apply min-max  scaling to aspect ratio
  df_train['aspect_ratio'] = min_max_scaling(df_train['aspect_ratio'])




  ######## EXERCISE 5.8 Calculate the number of non-zero pixels as the second feature
  df_train['fg_pixels'] = data_train.apply(foreground_pixels, axis=1)
  df_train['fg_pixels'] = min_max_scaling(df_train['fg_pixels'])

  ######## EXERCISE 5.9 Calculate the centroid feature as the third feature
  df_train['centroid'] = data_train.apply(calculate_centroid, axis=1)
  df_train['centroid'] = min_max_scaling(df_train['centroid'])





  # # Create the repsective features for the test samples
  df_test['aspect_ratio'] = data_test.apply(aspect_ratio, axis=1)
  df_test['aspect_ratio'] = min_max_scaling(df_test['aspect_ratio'])

  df_test['fg_pixels'] = data_test.apply(foreground_pixels, axis=1)
  df_test['fg_pixels'] = min_max_scaling(df_test['fg_pixels'])

  df_test['centroid'] = data_test.apply(calculate_centroid, axis=1)
  df_test['centroid'] = min_max_scaling(df_test['centroid'])



  ######## EXERCISE   Train the Bayesian Classifier with the features created from train data , predict the values of the test data and calculate accuracy of the classifier
  
  
 
  # Define the features to use for both train and test in this experiment
  features = ["aspect_ratio", "fg_pixels", "centroid"]
  trainData = df_train[features]
  
  # Create the Classifier object and train the Gaussian parameters (prior, mean, cov)
  classifier = MyBayesClassifier()
  ######## EXERCISE 5.5 Train the classifier
  classifier.train(trainData,target_train)
  
  ######## EXERCISE 5.6 Predict on the test samples (for the given feature set)
  test_data = df_test[features]
  predictions = classifier.predict(test_data)
  
  ######## EXERCISE 5.7 Calculate accuracy as an example of validation
  accuracy = accuracy_score(target_test, predictions)
  print("Classification accuracy using all 3 features:", accuracy)

  
###########################################################
###########################################################
if __name__ == "__main__":
  main()
  
# ##################### Παρατηρήσεις & Σχόλια #############################
# ερωτημα 2) Παρατηρείται άπό το αποτέλεσμα ότι ο υπολογισμός του aspect ratio είναι σωστός στην εικόνα που τυπώνεται ο αριθμός περικλείται σωστά από το bounding box και
#            το bounding box υπολογίζεται με τα ίδια δεδομένα με τα οποία υπολογίζεται και το aspect ratio.
# ερώτημα 6) Αφού  εκπαιδευτεί ο Bayes Classifier με τα χαρακτηριστικά των training data που δημιουργήθηκαν (aspect-ratio), χρησιμοποιούνται τα υπόλοιπα δεδομένα ως test
# ακρίβειας του classifier δίνοντας ένα ικανοποιητικό score:0,954545.
# ερώτημα 10) Προστίθονται δύο νέα χαρακτηριστικά για την εκπαίδευση του  Bayesian Classifier δίνοντας ακόμη καλύτερα αποτελέσματα με score:0,96694 που σημαίνει ότι και
# τα δύο νέα χαρακτηριστικά λειτουργούν σωστά αυξάνοντας την ακρίβεια του ταξινομητή.