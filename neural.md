# Neural Network — ML Lab Exam Guide 🧠

Neural Networks are one of the important topics in your ML lab exam, especially because your professor mentioned **epochs, accuracy, graphs, and ensembling**.

Let's learn the concept, Python implementation, important functions, and how to improve model performance.

## 1. What is a Neural Network?

A **Neural Network** is a machine learning model inspired by the way biological neurons process information. It learns patterns from data through interconnected computational units called neurons.

A basic neural network contains three types of layers:



- **Input layer:** Receives the features from the dataset.
- **Hidden layer(s):** Learn patterns and relationships in the data.
- **Output layer:** Produces the final prediction.

For example, to classify flowers:
- Input features: sepal length, sepal width, petal length, petal width.
- Hidden layers: Learn useful combinations of those features.
- Output layer: Predicts the flower species.

### How does it learn?

1. The network receives input features.
2. It calculates a prediction.
3. It compares the prediction with the actual answer using a loss function.
4. It adjusts its weights to reduce the loss.
5. It repeats this process over the training data.

This learning process uses **backpropagation and an optimization algorithm**.

---

## 2. Complete Python code for your lab

For a typical tabular classification dataset, scikit-learn provides `MLPClassifier`, which implements a feedforward neural network trained using backpropagation.

This example assumes the dataset is `data.csv` and the target column is `target`.

```python
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 2. Load dataset
df = pd.read_csv("data.csv")

# 3. Explore dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# 4. Handle missing values (simple baseline)
df = df.dropna()

# 5. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 6. Encode categorical input features
X = pd.get_dummies(X, drop_first=True)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# 10. Train model
model.fit(X_train, y_train)

# 11. Predict
y_pred = model.predict(X_test)

# 12. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 13. Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Neural Network - Confusion Matrix")
plt.show()

# 14. Plot training loss curve
plt.plot(model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss")
plt.show()
```

**Important:** `dropna()` is only a simple baseline. If missing values are substantial, use appropriate imputation. This code is for classification; regression requires `MLPRegressor` and regression metrics.

---

## 3. Understand the important parameters

| Parameter | Meaning |
|---|---|
| `MLPClassifier()` | Creates a neural network classifier |
| `hidden_layer_sizes=(100,)` | One hidden layer containing 100 neurons |
| `activation="relu"` | Uses ReLU activation in hidden layers |
| `solver="adam"` | Uses the Adam optimization algorithm |
| `max_iter=500` | Allows up to 500 training iterations |
| `random_state=42` | Helps reproduce results |
| `model.fit()` | Trains the neural network |
| `model.predict()` | Predicts classes |
| `model.loss_curve_` | Stores the loss at each recorded iteration |

### What does `hidden_layer_sizes=(100,)` mean?

It means the network has **one hidden layer with 100 neurons**.

For example:

```python
hidden_layer_sizes=(50,)
```

One hidden layer with 50 neurons.

```python
hidden_layer_sizes=(100, 50)
```

Two hidden layers:
- First hidden layer: 100 neurons.
- Second hidden layer: 50 neurons.

More neurons or layers do not automatically mean better accuracy.

---

## 4. Epochs — Very important for your exam

Your professor mentioned epochs, so understand this carefully.

An **epoch** is one complete pass through the entire training dataset.

Suppose your training dataset contains 1,000 samples.

- After processing all 1,000 samples once: one epoch.
- After processing all 1,000 samples five times: five epochs.

During training, the network updates its weights to reduce its loss.

### Epoch vs. iteration

These terms are related but not identical.

- **Epoch:** One complete pass through the training dataset.
- **Iteration:** One optimizer update or training step.

For mini-batch training, one epoch usually contains multiple iterations.

In scikit-learn's `MLPClassifier`, `max_iter` generally controls the maximum number of training iterations over the dataset—roughly epochs for common stochastic solvers such as Adam. Training can stop earlier if the convergence criteria are met.

You can inspect the training:

```python
print("Iterations:", model.n_iter_)
print("Final Loss:", model.loss_)
```

Here:
- `n_iter_` shows how many iterations were actually performed.
- `loss_` shows the final training loss.

**Remember:** Increasing the number of epochs does not guarantee higher test accuracy. Too much training can cause overfitting.

---

## 5. What is an activation function?

An activation function introduces nonlinearity, allowing a neural network to learn more complex patterns.

Common activation functions include:

| Activation | Main idea |
|---|---|
| ReLU | Outputs zero for negative inputs and the input itself for positive inputs |
| Sigmoid | Maps values to a range between 0 and 1 |
| Tanh | Maps values to a range between −1 and 1 |
| Softmax | Converts output scores into a probability distribution across classes |

For `MLPClassifier`, the `activation` parameter controls hidden-layer activation. The output activation is chosen automatically based on the classification problem.

For your lab, `activation="relu"` is a reasonable starting point.

---

## 6. Why is feature scaling important?

Neural networks often train more effectively when features are on comparable scales.

For example:
- Age: 18–60
- Salary: 20,000–200,000

Without scaling, the very different feature magnitudes can make optimization more difficult.

Use:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Exam rule:** Fit the scaler on training data only. Apply the fitted scaler to the test data to avoid data leakage.

---

## 7. Graphs for your lab exam

Two useful graphs are the confusion matrix and training loss curve.

### Graph 1: Confusion matrix

```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Neural Network Confusion Matrix")
plt.show()
```

It shows correct and incorrect predictions for each class.

### Graph 2: Training loss curve

```python
plt.plot(model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

The loss curve shows how the training loss changes during learning.

- **Loss decreasing:** The model is generally fitting the training data better.
- **Loss becoming stable:** Training may be approaching convergence.
- **Low training loss but poor test accuracy:** The model may be overfitting or the data may be difficult to generalize from.

A decreasing training loss alone does not prove that test accuracy is improving.

---

## 8. How to improve accuracy

If your neural network gives low accuracy, try these steps:

1. **Check preprocessing:** Handle missing values and encode categorical features correctly.
2. **Scale features:** Usually important for neural networks.
3. **Adjust hidden layers:** Compare `(50,)`, `(100,)`, and `(100, 50)`.
4. **Adjust training iterations:** Increase `max_iter` if the model has not converged.
5. **Try regularization:** Adjust `alpha` to control model complexity.
6. **Use the correct evaluation metrics:** Check the confusion matrix and classification report, not only accuracy.

For example:

```python
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    alpha=0.001,
    max_iter=700,
    random_state=42
)
```

This is an alternative configuration to experiment with, not a guaranteed improvement.

For reliable tuning, select parameters using cross-validation on the training set and reserve the test set for final evaluation.

---

## 9. Neural Network vs. Random Forest

| Feature | Neural Network | Random Forest |
|---|---|---|
| Basic structure | Layers of connected neurons | Collection of Decision Trees |
| Feature scaling | Usually important | Usually unnecessary |
| Important parameters | Layers, neurons, activation, `max_iter`, `alpha` | `n_estimators`, `max_depth`, `min_samples_leaf` |
| Training | Iterative weight optimization | Builds multiple trees |
| Main graph | Loss curve | Feature importance |
| Overfitting control | Regularization, architecture, early stopping | Tree depth, leaf size, other tree restrictions |

Both are ensemble-like in the broad sense of combining components? **No:** Random Forest is specifically an ensemble of trees, whereas a standard feedforward neural network is a single model composed of interconnected layers.

---

## 10. The essential code to remember

```python
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

plt.plot(model.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
```

Before training, make sure you have split the data and scaled the features correctly.

### Quick exam question

Your Neural Network achieves 99% training accuracy but only 70% test accuracy. What is the most likely problem?

**A.** Underfitting  
**B.** Overfitting  
**C.** Feature scaling always causes errors  
**D.** The model needs unlimited epochs

**Correct answer: B — Overfitting.**

The model has learned the training data very well but does not generalize as effectively to unseen data.

---

### Your ML lab revision status

You've now covered the core ideas behind:

1. Logistic Regression
2. Decision Tree
3. SVM
4. Random Forest
5. Neural Network

**Your next priority should be preprocessing, evaluation metrics, graphs, and ensembling.** These are reusable across multiple algorithms, so mastering them will help you adapt quickly to whichever dataset and model your professor gives you.
