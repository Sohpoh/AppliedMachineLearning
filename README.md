# Applied Machine Learning — Assignments

A collection of 14 Jupyter notebook assignments covering the core topics of an applied machine learning course. Each notebook (`Pohane-AssignN.ipynb`) is self-contained with written explanations, data preprocessing, model training, and evaluation.

## Assignment Overview

| # | Topic | Dataset(s) | Key Techniques |
|---|-------|------------|----------------|
| 1 | ML Fundamentals & Intro to Classification | Iris, Wine | Naive Bayes, Decision Tree, learning curves |
| 2 | Feature Types & Evaluation Metrics | Iris, Digits, Diabetes130US, 20 Newsgroups, Tourism | Feature exploration (numerical, nominal, date, text, image), precision/recall/F1, correlation matrix from scratch |
| 3 | EDA & Prototype Classification | Suicide Rates | Correlation analysis, one-hot encoding, Random Forest, 10-fold CV |
| 4 | NLP & ROC Analysis | Fake/True News | TF-IDF, Decision Tree / Random Forest / MLP classifier comparison, ROC curve operating point selection |
| 5 | Titanic Survival Prediction | Titanic | Missing value imputation, SVM, Naive Bayes, Random Forest, Stratified CV, Kaggle submission |
| 6 | Ensemble Learning | Heart Disease | Ensemble of weak learners (NB, SVC, MLP, DT), subsampling ratios, ensemble vs. single-model comparison |
| 7 | Regression | Suicide Rates | Linear regression, one-hot vs. numerical encoding, MAE, handling unseen categories |
| 8 | Clustering & Anomaly Detection | Synthetic clustering dataset | K-Means, DBSCAN, anomaly detection, Decision Tree post-clustering |
| 9 | Reinforcement Learning | Nim game | Q-learning, reward/penalty design, Q-table updates, comparison vs. random and optimal (Guru) players |
| 10 | Cybersecurity Intrusion Detection | CICIDS 2017 (8 traffic captures) | Network flow classification, port feature engineering, Random Forest / DT / KNN across multiple attack types |
| 11 | Association Rules & Custom Neural Network | Alice in Wonderland (NLTK), MNIST | Apriori / mlxtend association rules, from-scratch two-hidden-layer MLP with backprop (95% accuracy on MNIST) |
| 12 | Imbalanced Classification & PyTorch | Credit Card Fraud | Class imbalance handling, F1-score focus, GridSearchCV, custom PyTorch MLP with dropout |
| 13 | Image Classification with PyTorch | Intel Image (14k scenes) | Custom PyTorch MLP on raw pixels, ReLU/Sigmoid activations |
| 14 | Convolutional Neural Networks with TensorFlow | Intel Image (14k scenes) | Keras CNN with Conv2D/MaxPooling, L2 regularization, Dropout, Batch Normalization |

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
├── Pohane-Assign1.ipynb   # ML fundamentals
├── Pohane-Assign2.ipynb   # Feature types & metrics
├── ...
├── Pohane-Assign14.ipynb  # CNNs with TensorFlow
├── Module1.ipynb          # Supplemental module
├── predictions_pohane.csv # Titanic Kaggle submission
├── wt_tree.dot / .png     # Weighted decision tree visualization
└── tree.dot / .png        # Decision tree visualization
```
