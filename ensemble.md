# Ensembling in Machine Learning — Lab Exam Guide

Since you've learned Logistic Regression, Decision Tree, SVM, Random Forest, and Neural Networks, you're ready to understand **ensembling**.

The main idea is simple:

**Instead of relying on one model, combine multiple models to make a prediction.**

For your lab exam, focus on:
1. What ensembling is.
2. Types of ensembling.
3. How to implement a Voting Classifier.
4. How to compare individual models with an ensemble.
5. How to evaluate accuracy and plot graphs.

---

# 1. What is ensembling?

Suppose your task is to predict whether a student will pass or fail.

You train three models:

| Model | Prediction |
|---|---|
| Logistic Regression | Pass |
| Decision Tree | Pass |
| SVM | Fail |

Instead of trusting only one model, you combine their predictions.

With majority voting:
- Pass receives 2 votes.
- Fail receives 1 vote.

**Final prediction: Pass.**

This is the basic idea of ensembling.

Ensembling can improve accuracy when models make complementary errors, but it does not guarantee better performance.

---

# 2. Three types of ensembling

You should understand these three methods for your viva.

## A. Bagging

**Bagging = Bootstrap Aggregating**

Multiple models train independently on different bootstrap samples of the training data. Their predictions are then combined.

Example: Random Forest.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

Here, the Random Forest builds 100 decision trees.

## B. Boosting

Models are trained sequentially, with later models focusing on errors or weaknesses from earlier stages.

Examples:
- AdaBoost
- Gradient Boosting

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=42)
model.fit(X_train, y_train)
```

## C. Voting

Different classification models make predictions, and their outputs are combined.

For example:
- Logistic Regression
- Decision Tree
- Random Forest

Scikit-learn provides `VotingClassifier` to implement this.

**For your lab exam, start by mastering Voting Classifier.**

---

# 3. How to implement ensembling step by step

Let's assume your dataset is already loaded and preprocessed.

You have:

```python
X_train
X_test
y_train
y_test
```

These represent your training features, testing features, training labels, and testing labels.

### Step 1: Import the libraries

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
```

### Step 2: Create the individual models

```python
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Why use `StandardScaler` for Logistic Regression?

Because Logistic Regression often benefits from features on comparable scales.

Why not scale the Decision Tree and Random Forest inputs?

Tree-based models generally do not need feature scaling.

Using a pipeline ensures that scaling is learned from the training data when the model is fitted.

### Step 3: Create the Voting Classifier

```python
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    voting="hard"
)
```

Understand the important parameters:

| Parameter | Meaning |
|---|---|
| `estimators` | The models to combine |
| `"lr"` | Name assigned to Logistic Regression |
| `"dt"` | Name assigned to Decision Tree |
| `"rf"` | Name assigned to Random Forest |
| `voting="hard"` | Combines predicted class labels using majority voting |
| `voting="soft"` | Combines predicted class probabilities |

### Step 4: Train the ensemble

```python
ensemble.fit(X_train, y_train)
```

**Important:** `VotingClassifier.fit()` fits clones of the supplied estimators. You do not need to train the individual models separately before fitting the ensemble.

### Step 5: Make predictions

```python
y_pred = ensemble.predict(X_test)
```

### Step 6: Calculate accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Ensemble Accuracy:", accuracy)
```

That's the complete basic implementation!

---

# 4. Hard voting vs. soft voting

This is an important viva question.

## Hard voting

Each model predicts a class label.

Example:

| Model | Prediction |
|---|---|
| Logistic Regression | Pass |
| Decision Tree | Pass |
| Random Forest | Fail |

The majority class wins.

```python
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    voting="hard"
)
```

## Soft voting

Each model predicts probabilities for each class. The ensemble averages these probabilities, optionally using weights, and chooses the class with the highest combined probability.

Suppose the models predict the probability of passing:

| Model | Probability of Pass |
|---|---:|
| Logistic Regression | 0.80 |
| Decision Tree | 0.60 |
| Random Forest | 0.70 |

The average probability is:

\[
P(\text{Pass})=\frac{0.80+0.60+0.70}{3}=0.70
\]

The combined probability of passing is 70% in this simplified example.

Use:

```python
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("rf", rf)
    ],
    voting="soft"
)
```

Soft voting requires each estimator to support `predict_proba()`. These three models support it with their default configurations.

**Exam tip:** Hard voting combines class labels; soft voting combines probabilities.

---

# 5. How to check whether ensembling improves accuracy

Your professor may ask you to compare models.

You can train the individual models and the ensemble, then compare their test accuracy.

```python
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Ensemble": VotingClassifier(
        estimators=[
            ("lr", lr),
            ("dt", dt),
            ("rf", rf)
        ],
        voting="hard"
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name}: {acc:.4f}")
```

Example output — **illustrative numbers only**:

```text
Logistic Regression: 0.84
Decision Tree:       0.80
Random Forest:       0.88
Ensemble:            0.90
```

In this example, the ensemble achieves the highest accuracy.

However, your actual results will depend on the dataset. An ensemble can perform worse than its strongest individual model, especially when the models make similar errors or weaker models introduce noise.

For formal model selection, use cross-validation on the training data. Keep the test set for final evaluation rather than repeatedly choosing models based on test accuracy.

---

# 6. Plot an accuracy comparison graph

Your professor mentioned that graphs carry marks, so this is useful.

```python
import matplotlib.pyplot as plt

names = list(results.keys())
accuracies = list(results.values())

plt.bar(names, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Individual Models vs Ensemble")
plt.ylim(0, 1)

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
```

This graph compares the accuracy of each individual model with the ensemble.

You can also plot a confusion matrix for the final ensemble:

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    models["Ensemble"].predict(X_test)
)

plt.title("Ensemble Confusion Matrix")
plt.show()
```

---

# 7. What if your professor gives you a different model?

You can replace one of the estimators.

For example, to combine Logistic Regression, SVM, and Random Forest:

```python
ensemble = VotingClassifier(
    estimators=[
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )),
        ("svm", make_pipeline(
            StandardScaler(),
            SVC(probability=True)
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ],
    voting="soft"
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

Why use `SVC(probability=True)`?

Because soft voting needs probability predictions, and SVC does not enable `predict_proba()` by default.

Scaling is included for SVM and Logistic Regression, but not Random Forest.

---

# 8. Common mistakes to avoid

1. **Using regression models in `VotingClassifier`:** Use classification estimators for classification.
2. **Using soft voting without probabilities:** Every estimator must support `predict_proba()`.
3. **Forgetting preprocessing:** Scale models that need it and handle categorical features and missing values.
4. **Data leakage:** Fit preprocessing using training data only.
5. **Assuming the ensemble must be more accurate:** Always evaluate it.
6. **Using the test set repeatedly for tuning:** Use training-set cross-validation for model selection.

---

# 9. What to memorize for your lab exam

If you remember only one implementation, remember this pattern:

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt)
    ],
    voting="hard"
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

This is a minimal example. For a real dataset, add appropriate preprocessing—particularly feature scaling for Logistic Regression.

### Final viva revision

| Question | Answer |
|---|---|
| What is ensembling? | Combining multiple models to make predictions. |
| What is bagging? | Training models on bootstrap samples and aggregating predictions. |
| What is boosting? | Sequentially building models that address earlier weaknesses. |
| What is voting? | Combining predictions from different classifiers. |
| What is hard voting? | Combining predicted class labels. |
| What is soft voting? | Combining predicted class probabilities. |
| What is Random Forest? | An ensemble of Decision Trees using bagging and random feature selection. |
| Does ensembling always improve accuracy? | No. It depends on the models and dataset. |

## Your next step

Since your lab exam covers preprocessing, five algorithms, metrics, graphs, and ensembling, the best preparation now is to **practice one complete classification experiment from loading a dataset through preprocessing, model training, accuracy evaluation, graph plotting, and ensembling**.

Once you can do that independently, you can adapt the same workflow to most of the models in your syllabus.
