# Applied Machine Learning

A collection of 14 Jupyter notebooks covering applied machine learning topics. Each notebook is self-contained with written explanations, data preprocessing, model training, and evaluation.

## Overview

| Notebook | Dataset(s) | Key Techniques |
|----------|------------|----------------|
| `ML-Fundamentals-and-Iris-Classification.ipynb` | Iris, Wine | Naive Bayes, Decision Tree, learning curves |
| `Feature-Types-and-Admission-Correlation-Analysis.ipynb` | Iris, Digits, Diabetes130US, 20 Newsgroups, Tourism | Feature exploration (numerical, nominal, date, text, image), precision/recall/F1, correlation matrix from scratch |
| `Suicide-Rates-EDA-and-Binary-Classification.ipynb` | Suicide Rates | Correlation analysis, one-hot encoding, Random Forest, 10-fold CV |
| `Fake-News-Detection-TF-IDF-and-ROC-Analysis.ipynb` | Fake/True News | TF-IDF, Decision Tree / Random Forest / MLP classifier comparison, ROC curve operating point selection |
| `Titanic-Survival-Prediction.ipynb` | Titanic | Missing value imputation, SVM, Naive Bayes, Random Forest, Stratified CV, Kaggle submission |
| `Heart-Disease-Ensemble-Learning.ipynb` | Heart Disease | Ensemble of weak learners (NB, SVC, MLP, DT), subsampling ratios, ensemble vs. single-model comparison |
| `Suicide-Rates-Linear-Regression.ipynb` | Suicide Rates | Linear regression, one-hot vs. numerical encoding, MAE, handling unseen categories |
| `Clustering-and-Anomaly-Detection-KMeans-DBSCAN.ipynb` | Synthetic clustering dataset | K-Means, DBSCAN, anomaly detection, Decision Tree post-clustering |
| `Clustering-and-Anomaly-Detection-v2.ipynb` | Synthetic clustering dataset | Alternate anomaly detection approach using centroid distance thresholding |
| `Reinforcement-Learning-Nim-Game-Q-Learning.ipynb` | Nim game | Q-learning, reward/penalty design, Q-table updates, comparison vs. random and optimal (Guru) players |
| `Network-Intrusion-Detection-CICIDS.ipynb` | CICIDS 2017 (8 traffic captures) | Network flow classification, port feature engineering, Random Forest / DT / KNN across multiple attack types |
| `Association-Rules-Alice-and-MNIST-Neural-Network.ipynb` | Alice in Wonderland (NLTK), MNIST | Apriori / mlxtend association rules, from-scratch two-hidden-layer MLP with backprop (95% accuracy on MNIST) |
| `Credit-Card-Fraud-Detection-Imbalanced-Classes.ipynb` | Credit Card Fraud | Class imbalance handling, F1-score focus, GridSearchCV, custom PyTorch MLP with dropout |
| `Intel-Image-Classification-PyTorch-MLP.ipynb` | Intel Image (14k scenes) | Custom PyTorch MLP on raw pixels, ReLU/Sigmoid activations |
| `Intel-Image-Classification-CNN-TensorFlow.ipynb` | Intel Image (14k scenes) | Keras CNN with Conv2D/MaxPooling, L2 regularization, Dropout, Batch Normalization |

## Datasets

| File | Description |
|------|-------------|
| `Admission_Predict.csv` | Graduate admission features (GRE, TOEFL, CGPA, etc.) |
| `suicideRates.csv` | Global suicide rates 1987–2014 by country, age, sex |
| `Fake.csv` / `True.csv` | Labeled news articles for fake news detection |
| `TitanicData/` | Kaggle Titanic train/test split |
| `heart_dataset.csv` | Heart disease clinical features |
| `clustering_synthetic_dataset.csv` | 2D synthetic dataset with 3 clusters |
| `creditcard.csv` | Anonymized credit card transactions (PCA-transformed features, highly imbalanced) |
| `TrafficLabelling/` | CICIDS 2017 network traffic captures with attack labels |
| `Intel_Image/` | 14k labeled scene images (buildings, forest, glacier, mountain, sea, street) |
| `EP_datasets/mnist/` | MNIST handwritten digit images (60k train / 10k test) |

## Dependencies

```
scikit-learn
pandas
numpy
matplotlib
seaborn
mlxtend
nltk
torch
tensorflow
opencv-python (cv2)
graphviz
scipy
tqdm
```

## Structure

```
.
├── ML-Fundamentals-and-Iris-Classification.ipynb
├── Feature-Types-and-Admission-Correlation-Analysis.ipynb
├── Suicide-Rates-EDA-and-Binary-Classification.ipynb
├── Fake-News-Detection-TF-IDF-and-ROC-Analysis.ipynb
├── Titanic-Survival-Prediction.ipynb
├── Heart-Disease-Ensemble-Learning.ipynb
├── Suicide-Rates-Linear-Regression.ipynb
├── Clustering-and-Anomaly-Detection-KMeans-DBSCAN.ipynb
├── Clustering-and-Anomaly-Detection-v2.ipynb
├── Reinforcement-Learning-Nim-Game-Q-Learning.ipynb
├── Network-Intrusion-Detection-CICIDS.ipynb
├── Association-Rules-Alice-and-MNIST-Neural-Network.ipynb
├── Credit-Card-Fraud-Detection-Imbalanced-Classes.ipynb
├── Intel-Image-Classification-PyTorch-MLP.ipynb
├── Intel-Image-Classification-CNN-TensorFlow.ipynb
├── Module1.ipynb          # Supplemental module
├── predictions_pohane.csv # Titanic Kaggle submission
├── wt_tree.dot / .png     # Weighted decision tree visualization
└── tree.dot / .png        # Decision tree visualization
```
