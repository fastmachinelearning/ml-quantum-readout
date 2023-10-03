import os 
import time 
import numpy as np 
import matplotlib.pyplot as plt
# classifiers
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# metrics
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

#################################
# Helper funcs
#################################
def split_data(X, y, split):
    if split == 1:
        return X, y
    index = int(len(X) * split)
    X = X[:index]
    y = y[:index]
    return X, y

def plot_confusion_matrix(conf_matrix, labels, name):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    num_classes = len(labels)
    plt.xticks(np.arange(num_classes), labels, rotation=45)
    plt.yticks(np.arange(num_classes), labels)

    # Add labels to the plot
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if i == j else 'black')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(f'conf-matrix-{name}.png')

#################################
# Load data 
#################################
data_dir = '../data/new-raw-data'
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)

X, y = split_data(X_train, y_train, 0.01)
X_test, y_test = split_data(X_train, y_train, 0.04)

#################################
# Setup  
#################################
classifiers = [
    ('Logistic Regression', LogisticRegression, 0.1),
    # ('Least Squares', RidgeClassifier, 0.1),
    # ('SVM', SVC, 0.1),
    ('Random Forest', RandomForestClassifier, 0.1),
    ('Decision Tree', DecisionTreeClassifier, 0.1),
    # ('KNeighbors', KNeighborsClassifier, 0.1),
    # ('Naive Bayes', CategoricalNB, 0.1),
    # ('Gaussian', GaussianProcessClassifier, 0.1),
    # Train a BDT
    ('Gradient Boosting', GradientBoostingClassifier, 0.1),
]

clf_params = {
    'Logistic Regression': {},
    'Least Squares': {},
    'SVM': {'gamma': 'auto'},
    'Random Forest': {'max_depth':6, 'random_state': 0},
    'Decision Tree': {'random_state':0},
    'KNeighbors': {'n_neighbors':16},
    'Naive Bayes': {},
    'Gaussian': {'kernel':1.0 * RBF(1.0), 'random_state': 0},
    'Gradient Boosting': {}
}

#################################
# Fit and evaluate classifiers  
#################################

plt.figure(0, figsize=(8, 6))
colors = ["#DB4437", "#4285F4", "#F4B400", "#0F9D58", "purple"]
color_idx = 0

for name, model, split in classifiers:
    clf = model(**clf_params[name]).fit(X, y)
    
    # Training accuracy  
    train_start = time.time()
    train_score = clf.score(X, y)
    train_end = time.time()

    # Test accuracy 
    test_start = time.time()
    test_score = clf.score(X_test, y_test)
    test_end = time.time()
    
    print(name)
    print(f'\tTrain score ({train_end-train_start:.3}s): {train_score}')
    print(f'\tTest score ({test_end-test_start:.3}s): {test_score}')

    # Other metrics 
    y_pred = clf.predict_proba(X_test)[:, 1]
    clf_roc_auc = roc_auc_score(y_test, y_pred)

    y_pred = clf.predict(X_test)
    clf_avg_precision = average_precision_score(y_test, y_pred)

    print(f'\tAverage Precision: {clf_avg_precision}')
    print(f'\tROC AUC: {clf_roc_auc}')

    # Plot confusion matrix 
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix, ['grounded', 'excited'], name.replace(' ', ''))

    # Compute ROC curve parameters
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(0)
    plt.plot(fpr, tpr, color=colors[color_idx], lw=2, label='{} (area = {:.2f})'.format(name, roc_auc))
    color_idx += 1

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
plt.savefig(f'roc.png')
