# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'SVM Data set.csv'
df = pd.read_csv(file_path)

# Extract features and labels
X = df.iloc[:, 1:4]  # Extracting features from the dataset
X = pd.get_dummies(X, columns=['Gender'])  # Converting categorical 'Gender' column to dummy variables

# Splitting the data into training and cross-validation sets
Xc = X.iloc[360:,:].values
X = X.iloc[:360,:].values
yc = df.iloc[360:, -1].values
y = df.iloc[:360, -1].values

# Convert labels to -1 and 1
y = np.where(y <= 0, -1, 1)
yc = np.where(yc <= 0, -1, 1)

# Function to normalize the data
def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.max(x,axis=0)-np.min(x,axis=0)
    return (x - mean) /std

# Plotting the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], marker="x", c=y)
plt.show()

# %%
# Load the test dataset
df_test = pd.read_csv('test.csv')

# Extract features and labels for the test set
Xt = df_test.iloc[:, :3]
Xt = pd.get_dummies(Xt, columns=['Gender'])
Xt = Xt.iloc[:,:].values
yt = df_test.iloc[:, -1].values
yt = np.where(yt <= 0, -1, 1)  # Convert labels to -1 and 1
yt
print('svm without kernal')
# %%
# Define SVM function without kernel
def svm(X, y, alpha, lambda_param, epochs):
    # Ensure X and y are NumPy arrays of float64 type
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    m, n = X.shape  # m: number of samples, n: number of features
    
    # Initialize weights and bias
    w = np.random.randn(n) * np.sqrt(2/m)
    b = np.random.randn(1) * np.sqrt(2/m)
    
    # Training process
    for epoch in range(epochs):
        for j in range(m):
            if y[j] * (np.dot(X[j], w) + b) >= 1:
                w -= alpha * (2 * lambda_param * w)
            else:
                w -= alpha * (2 * lambda_param * w) - alpha * (X[j] * y[j])
                b += alpha * y[j]
    
    # Predictions on training data
    pred = np.sign(np.dot(X, w) + b)
    print(f'Train Accuracy: {np.mean(pred == y) * 100:.2f}%')
    return w, b

# Normalize the data
x_nor = normalize(X) 
xc_nor = normalize(Xc)
xt_nor = normalize(Xt)

# Train the SVM model
w, b = svm(x_nor, y, alpha=0.06, lambda_param=0.001, epochs=100) 

# Predictions on cross-validation and test sets
pred_cv = np.sign(np.dot(xc_nor, w) + b)
pred_test = np.sign(np.dot(xt_nor, w) + b)

# Calculate and print accuracies
print(f'Cross-validation Accuracy: {np.mean(pred_cv == yc) * 100:.2f}%')
print(f'Test Accuracy: {np.mean(pred_test == yt) * 100:.2f}%')

# %%
print('svm with kernal')
# Define Gaussian  kernel function
def rbf_kernel(X1, X2, gamma=0.1):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

# Define SVM function with Gaussian  kernel
def svm_kernel(X, y, alpha=0.01, lambda_param=0.01, epochs=1000, gamma=0.5):
    m = X.shape[0]
    K = rbf_kernel(X, X, gamma)
    
    alpha_vec = np.zeros(m)
    b = 0
    
    # Training process
    for epoch in range(epochs):
        for i in range(m):
            if y[i] * (np.dot(K[i], alpha_vec * y) + b) < 1:
                alpha_vec[i] += alpha * (1 - y[i] * (np.dot(K[i], alpha_vec * y) + b)) - alpha * lambda_param * alpha_vec[i]
                b += alpha * y[i]
            else:
                alpha_vec[i] -= alpha * lambda_param * alpha_vec[i]
    
    return alpha_vec, b

# Define prediction function
def predict(X_train, X_test, alpha_vec, y_train, b, gamma=0.5):
    K_test = rbf_kernel(X_test, X_train, gamma)
    predictions = np.sign(np.dot(K_test, alpha_vec * y_train) + b)
    return predictions

# Train the SVM with RBF kernel
alpha_vec, b = svm_kernel(X, y, alpha=0.04, lambda_param=0.0001, epochs=1000, gamma=0.01)

# Predict on the cross-validation and test sets
y_pred_cv = predict(X, Xc, alpha_vec, y, b, gamma=0.01)
y_pred_train = predict(X, X, alpha_vec, y, b, gamma=0.01)
y_pred_test = predict(X, Xt, alpha_vec, y, b, gamma=0.01)

# Calculate and print accuracies
print(f'Train Accuracy: {np.mean(y_pred_train == y) * 100:.2f}%')
print(f'Cross-validation Accuracy: {np.mean(y_pred_cv == yc) * 100:.2f}%')
print(f'Test Accuracy: {np.mean(y_pred_test == yt) * 100:.2f}%')

# %%
print(" Task 3")
# Define functions to calculate metrics

def metric(y, pred):
    tp = fp = fn = tn = 0
    n = y.shape[0]  # Number of elements in the first dimension

    for i in range(n):
        if y[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1
                
    return tp, fp, fn, tn                       

def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def r2_error(y, pred):
    mean_y = np.mean(y)
    ss_tot = np.sum((y - mean_y) ** 2)
    ss_res = np.sum((y - pred) ** 2)
    return 1 - (ss_res / ss_tot)

# Calculate and print metrics for training set
tp, fp, fn, tn = metric(y, y_pred_train)
print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
print(f'Accuracy: {accuracy(tp, fp, fn, tn)}')
print(f'Precision: {precision(tp, fp)}')
print(f'Recall: {recall(tp, fn)}')
print(f'F1 Score: {f1_score(tp, fp, fn)}')
print(f'R2 Error: {r2_error(y, y_pred_train)}')

# %%
