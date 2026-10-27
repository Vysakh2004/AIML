Yes. Your uploaded lab sheet contains **20 data-preprocessing tasks**, from loading a dataset through cleaning, encoding, scaling, feature selection, train-test splitting, balancing, visualization, saving the final CSV, and documenting everything on GitHub. 24CSP503-Exp.1

I recommend using the **Palmer Penguins dataset** instead of Titanic. It is beginner-friendly but still gives you numerical features, categorical variables, missing values, correlations, feature engineering opportunities, and a classification target (`species`). For Question 16, you can demonstrate oversampling or SMOTE on the species classes.

**Kaggle dataset:** [Palmer Archipelago (Antarctica) Penguin Data on Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?utm_source=chatgpt.com)

Below is a **complete Google Colab-ready answer from the basics**, with every question numbered exactly according to your lab sheet.

## Step 0: Download and upload the dataset to Google Colab

From the Kaggle page, download the dataset and use the file named `penguins_size.csv`. Then open Google Colab, create a new notebook, and upload the CSV using the Files panel.

Alternatively, you can use this upload code:

```python
from google.colab import files

uploaded = files.upload()
```

After selecting `penguins_size.csv`, start with the following imports:

```python
# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Display settings
pd.set_option('display.max_columns', None)

print("Libraries imported successfully!")
```

---

## Question 1: Load a dataset using Pandas and display its basic information

```python
# Load the dataset

df = pd.read_csv('penguins_size.csv')

# Display the dataset
display(df)

# Display basic information
df.info()
```

### Explanation

`pd.read_csv()` reads the CSV file and converts it into a Pandas DataFrame. `df.info()` shows the column names, non-null counts, and data types.

---

## Question 2: Identify rows, columns, data types, and missing values

```python
# Number of rows and columns

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())
```

You can also create a cleaner summary:

```python
summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Unique Values': df.nunique()
})

display(summary)
```

---

## Question 3: Display first, last, and randomly selected records

```python
# First 5 records
print("First 5 Records:")
display(df.head())

# Last 5 records
print("Last 5 Records:")
display(df.tail())

# 5 random records
print("Random 5 Records:")
display(df.sample(5, random_state=42))
```

### Explanation

- `head()` displays the first five rows.
- `tail()` displays the last five rows.
- `sample()` randomly selects records.

---

## Question 4: Detect and handle missing values

First, detect missing values:

```python
print("Missing values before preprocessing:")
print(df.isnull().sum())
```

Create a copy so the original dataset remains available for comparison:

```python
df_original = df.copy()
df_clean = df.copy()
```

For numerical columns, fill missing values with the **median**:

```python
numerical_columns = df_clean.select_dtypes(include=np.number).columns

for column in numerical_columns:
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].median()
    )
```

For categorical columns, use the **mode**:

```python
categorical_columns = df_clean.select_dtypes(
    include=['object', 'category']
).columns

for column in categorical_columns:
    df_clean[column] = df_clean[column].fillna(
        df_clean[column].mode()[0]
    )
```

Verify:

```python
print("Missing values after preprocessing:")
print(df_clean.isnull().sum())
```

### Why median and mode?

The **median** is less sensitive to extreme outliers than the mean. The **mode** is appropriate for categorical columns because it represents the most frequently occurring category.

---

## Question 5: Identify and remove duplicate records

```python
# Count duplicate rows
duplicate_count = df_clean.duplicated().sum()

print("Number of duplicate records:", duplicate_count)

# Display duplicates, if any
display(df_clean[df_clean.duplicated()])
```

Remove them:

```python
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

print("Duplicates after removal:", df_clean.duplicated().sum())
```

---

## Question 6: Detect outliers using IQR and visualize using box plots

First, select numerical columns:

```python
numerical_columns = df_clean.select_dtypes(include=np.number).columns

print("Numerical columns:")
print(list(numerical_columns))
```

Detect outliers using the IQR method:

```python
for column in numerical_columns:

    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean[column] < lower_bound) |
        (df_clean[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)} outliers")
```

Visualize using box plots:

```python
for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df_clean[column])

    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

    plt.show()
```

### IQR formula

The interquartile range is:

**IQR = Q3 − Q1**

An observation is normally considered an outlier if:

**Value < Q1 − 1.5 × IQR**

or

**Value > Q3 + 1.5 × IQR**

---

## Question 7: Handle outliers using suitable preprocessing techniques

Instead of deleting rows, use **IQR capping**, also called winsorization. Values below the lower bound are replaced by the lower bound, while values above the upper bound are replaced by the upper bound.

```python
df_processed = df_clean.copy()

for column in numerical_columns:

    Q1 = df_processed[column].quantile(0.25)
    Q3 = df_processed[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_processed[column] = df_processed[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("Outliers handled successfully using IQR capping.")
```

Check the box plots again:

```python
for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df_processed[column])

    plt.title(f'After Outlier Handling: {column}')

    plt.show()
```

---

## Question 8: Convert categorical variables to numerical form

### A. Label Encoding

Let's encode the `species` target:

```python
label_encoder = LabelEncoder()

df_processed['species_encoded'] = label_encoder.fit_transform(
    df_processed['species']
)

print("Species mapping:")

for original, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(original, "->", encoded)
```

View the result:

```python
display(
    df_processed[['species', 'species_encoded']].head(10)
)
```

### B. One-Hot Encoding

Use one-hot encoding for nominal categorical columns such as `island` and `sex`:

```python
df_encoded = pd.get_dummies(
    df_processed,
    columns=['island', 'sex'],
    drop_first=False,
    dtype=int
)

display(df_encoded.head())
```

### Difference

**Label Encoding** assigns each category an integer, such as `Adelie → 0`, `Chinstrap → 1`, and `Gentoo → 2`.

**One-Hot Encoding** creates separate binary columns for every category. It is generally more appropriate when categories have no natural order.

---

## Question 9: Apply Min-Max normalization and standardization

Select numerical feature columns:

```python
feature_columns = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]
```

### A. Min-Max Normalization

Min-Max scaling converts values to a range, normally from 0 to 1.

```python
minmax_scaler = MinMaxScaler()

df_minmax = df_processed.copy()

df_minmax[feature_columns] = minmax_scaler.fit_transform(
    df_minmax[feature_columns]
)

display(df_minmax[feature_columns].head())
```

Formula:

**X' = (X − Xmin) / (Xmax − Xmin)**

### B. Standardization

Standardization transforms values to approximately zero mean and unit standard deviation:

```python
standard_scaler = StandardScaler()

df_standardized = df_processed.copy()

df_standardized[feature_columns] = standard_scaler.fit_transform(
    df_standardized[feature_columns]
)

display(df_standardized[feature_columns].head())
```

Formula:

**Z = (X − μ) / σ**

---

## Question 10: Perform feature selection using correlation analysis

Calculate correlation with the encoded target variable:

```python
correlation_with_target = (
    df_processed[
        feature_columns + ['species_encoded']
    ]
    .corr()['species_encoded']
    .sort_values(ascending=False)
)

print("Correlation with target variable:")
print(correlation_with_target)
```

Visualize it:

```python
correlation_with_target.drop('species_encoded').plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title('Feature Correlation with Target')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.xticks(rotation=45)

plt.show()
```

### Interpretation

Features with larger absolute correlation values have stronger linear relationships with the target. However, correlation alone should not always be used as the only feature-selection criterion, especially for multiclass classification.

---

## Question 11: Generate and interpret a correlation matrix and heatmap

```python
correlation_matrix = df_processed[
    feature_columns + ['species_encoded']
].corr()

display(correlation_matrix)
```

Generate the heatmap:

```python
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Matrix Heatmap')

plt.show()
```

### Interpretation

- Values close to **+1** indicate strong positive correlation.
- Values close to **−1** indicate strong negative correlation.
- Values close to **0** indicate weak linear correlation.

---

## Question 12: Create new features through feature engineering

Let's create three new features.

```python
# Culmen ratio
df_processed['culmen_ratio'] = (
    df_processed['culmen_length_mm'] /
    df_processed['culmen_depth_mm']
)

# Body mass in kilograms
df_processed['body_mass_kg'] = (
    df_processed['body_mass_g'] / 1000
)

# Flipper-to-body-mass ratio
df_processed['flipper_mass_ratio'] = (
    df_processed['flipper_length_mm'] /
    df_processed['body_mass_g']
)

display(
    df_processed[
        [
            'culmen_ratio',
            'body_mass_kg',
            'flipper_mass_ratio'
        ]
    ].head()
)
```

### Explanation

Feature engineering creates new variables from existing attributes that may provide additional information to a machine-learning model.

---

## Question 13: Convert data types wherever necessary

Check current data types:

```python
print(df_processed.dtypes)
```

Convert body mass from numeric to float:

```python
df_processed['body_mass_g'] = (
    df_processed['body_mass_g'].astype(float)
)
```

Convert categorical columns to the `category` type:

```python
df_processed['species'] = df_processed['species'].astype('category')
df_processed['island'] = df_processed['island'].astype('category')
df_processed['sex'] = df_processed['sex'].astype('category')
```

Verify:

```python
print(df_processed.dtypes)
```

**Note:** The Palmer Penguins dataset does not contain a date column, so a string-to-datetime conversion is not necessary. You should only perform datatype conversions that are meaningful for the actual dataset.

---

## Question 14: Handle inconsistent and noisy data

First inspect unique categorical values:

```python
for column in ['species', 'island', 'sex']:
    print(f"\nUnique values in {column}:")
    print(df_processed[column].unique())
```

Clean text data:

```python
for column in ['species', 'island', 'sex']:

    df_processed[column] = (
        df_processed[column]
        .astype(str)
        .str.strip()
        .str.title()
    )
```

Handle known inconsistent values if present:

```python
df_processed['sex'] = df_processed['sex'].replace({
    '.': 'Unknown',
    'Nan': 'Unknown'
})
```

Verify:

```python
for column in ['species', 'island', 'sex']:
    print(f"\nCleaned values in {column}:")
    print(df_processed[column].unique())
```

---

## Question 15: Split the dataset into training and testing sets

Define features and target:

```python
X = df_processed[
    [
        'culmen_length_mm',
        'culmen_depth_mm',
        'flipper_length_mm',
        'body_mass_g',
        'culmen_ratio',
        'body_mass_kg',
        'flipper_mass_ratio'
    ]
]

y = df_processed['species_encoded']
```

Perform an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```

### Explanation

- **80%** of data is used for training.
- **20%** is used for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` attempts to preserve the original class proportions.

---

## Question 16: Perform data balancing using random oversampling or SMOTE

First install `imbalanced-learn` if necessary:

```python
!pip install -q imbalanced-learn
```

Import SMOTE:

```python
from imblearn.over_sampling import SMOTE
```

Check class distribution before balancing:

```python
print("Class distribution before SMOTE:")
print(y_train.value_counts().sort_index())
```

Apply SMOTE:

```python
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)
```

Check the result:

```python
print("Class distribution after SMOTE:")
print(y_train_balanced.value_counts().sort_index())
```

Visualize before and after:

```python
plt.figure(figsize=(7, 4))

y_train.value_counts().sort_index().plot(kind='bar')

plt.title('Class Distribution Before SMOTE')
plt.xlabel('Species Class')
plt.ylabel('Count')

plt.show()
```

```python
plt.figure(figsize=(7, 4))

y_train_balanced.value_counts().sort_index().plot(kind='bar')

plt.title('Class Distribution After SMOTE')
plt.xlabel('Species Class')
plt.ylabel('Count')

plt.show()
```

**Important:** SMOTE should be applied **only to training data**, not before the train-test split. Otherwise, information leakage can occur.

---

## Question 17: Visualize numerical distributions before and after preprocessing

Before preprocessing:

```python
for column in feature_columns:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        df_original[column].dropna(),
        kde=True
    )

    plt.title(f'Before Preprocessing: {column}')

    plt.show()
```

After preprocessing:

```python
for column in feature_columns:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        df_processed[column],
        kde=True
    )

    plt.title(f'After Preprocessing: {column}')

    plt.show()
```

This lets you observe how missing-value treatment and outlier handling affected each numerical feature.

---

## Question 18: Compare the dataset before and after preprocessing

Create a comparison table:

```python
comparison = pd.DataFrame({
    'Metric': [
        'Number of Rows',
        'Number of Columns',
        'Missing Values',
        'Duplicate Records'
    ],

    'Before Preprocessing': [
        df_original.shape[0],
        df_original.shape[1],
        df_original.isnull().sum().sum(),
        df_original.duplicated().sum()
    ],

    'After Preprocessing': [
        df_processed.shape[0],
        df_processed.shape[1],
        df_processed.isnull().sum().sum(),
        df_processed.duplicated().sum()
    ]
})

display(comparison)
```

### Summary of changes

The preprocessing workflow performed the following operations:

- Missing numerical values were filled using the median.
- Missing categorical values were filled using the mode.
- Duplicate records were identified and removed.
- Outliers were detected using the IQR method and handled through capping.
- Categorical variables were transformed using Label Encoding and One-Hot Encoding.
- Numerical features were scaled using Min-Max normalization and standardization.
- Correlation analysis was performed for feature selection.
- New features were created through feature engineering.
- Inconsistent categorical values were cleaned.
- The data was split into training and testing sets.
- SMOTE was applied only to training data to demonstrate class balancing.

---

## Question 19: Save the cleaned and preprocessed dataset as a new CSV

Before saving, create the final one-hot-encoded dataset:

```python
df_final = pd.get_dummies(
    df_processed,
    columns=['island', 'sex'],
    drop_first=False,
    dtype=int
)
```

Save it:

```python
df_final.to_csv(
    'penguins_preprocessed.csv',
    index=False
)

print("Dataset saved successfully as penguins_preprocessed.csv")
```

Download it from Colab:

```python
from google.colab import files

files.download('penguins_preprocessed.csv')
```

---

## Question 20: Document each preprocessing step and upload to GitHub

Your Google Colab notebook should be organized using **Markdown cells** before every code section. For example:

```text
# Lab Exercise 1: Data Preprocessing

## Objective

The objective of this experiment is to perform various data preprocessing
techniques on the Palmer Penguins dataset using Python, Pandas, NumPy,
Matplotlib, Seaborn, and Scikit-learn.

## Dataset

The Palmer Penguins dataset contains measurements of three penguin species:
Adelie, Chinstrap, and Gentoo. It includes attributes such as culmen length,
culmen depth, flipper length, body mass, island, and sex.

## Preprocessing Steps

1. Dataset loading and inspection
2. Identification of rows, columns, datatypes, and missing values
3. Exploratory data inspection
4. Missing-value handling
5. Duplicate removal
6. Outlier detection using IQR
7. Outlier handling using capping
8. Label encoding and one-hot encoding
9. Min-Max normalization and standardization
10. Feature selection using correlation analysis
11. Correlation matrix and heatmap generation
12. Feature engineering
13. Datatype conversion
14. Cleaning inconsistent and noisy data
15. Train-test splitting
16. Data balancing using SMOTE
17. Distribution visualization
18. Before-and-after comparison
19. Exporting the preprocessed dataset

## Conclusion

The Palmer Penguins dataset was successfully preprocessed by handling missing
values, duplicates, outliers, categorical variables, inconsistent data, and
feature scaling. Feature engineering, correlation analysis, train-test
splitting, and SMOTE-based class balancing were also demonstrated. The final
cleaned dataset was exported as a CSV file for future machine-learning tasks.
```

For GitHub, upload these three files to one repository:

1. `Data_Preprocessing_Lab_1.ipynb`
2. `penguins_size.csv`
3. `penguins_preprocessed.csv`

This gives you a **complete 20-question lab solution matching your uploaded exercise sheet**, while using a dataset other than Titanic. One technical improvement I would make for your actual submission is to keep a very clear pipeline of **original data → cleaned data → encoded/scaled data → train/test split**, because that prevents confusion and makes your viva explanation much easier.
