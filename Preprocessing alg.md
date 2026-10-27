##### **Preprocessing alg:**



Data Preprocessing Algorithm with Python Functions

Algorithm: Data Preprocessing for Machine Learning

1\. Start.



2\. Load the dataset.



Function: pd.read\_csv()

import pandas as pd

df = pd.read\_csv("dataset.csv")

3\. Examine the dataset.



Functions: head(), shape, info(), describe()

df.head()

df.shape

df.info()

df.describe()

4\. Identify numerical and categorical features.



Function: select\_dtypes()

numerical = df.select\_dtypes(include=\["number"]).columns

categorical = df.select\_dtypes(include=\["object", "category", "bool"]).columns

5\. Check for missing values and duplicate records.



Functions: isnull().sum(), duplicated().sum()

df.isnull().sum()

df.duplicated().sum()

6\. Remove unnecessary columns and duplicate records.



Functions: drop(), drop\_duplicates()

df = df.drop(columns=\["ID"], errors="ignore")

df = df.drop\_duplicates()

7\. Separate input features and target variable.



Functions: drop(), column selection

X = df.drop(columns=\["target"])

y = df\["target"]

Replace "target" with the actual target column name.



8\. Handle missing values.



Class: SimpleImputer

Strategy: "mean", "median", or "most\_frequent"

from sklearn.impute import SimpleImputer



num\_imputer = SimpleImputer(strategy="median")

cat\_imputer = SimpleImputer(strategy="most\_frequent")

9\. Encode categorical features.



Class: OneHotEncoder

from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder(handle\_unknown="ignore")

10\. Split the dataset into training and testing sets.



Function: train\_test\_split()

from sklearn.model\_selection import train\_test\_split



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;   X, y, test\_size=0.2, random\_state=42

)

11\. Scale numerical features if required.



Classes: StandardScaler, MinMaxScaler

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

12\. Fit preprocessing transformations using training data only.



Methods: fit(), fit\_transform()

\# Example for a numerical feature matrix

X\_train\_scaled = scaler.fit\_transform(X\_train)

In a mixed dataset, apply the scaler only to numerical columns using a suitable preprocessing pipeline.



13\. Apply fitted transformations to testing data.



Method: transform()

X\_test\_scaled = scaler.transform(X\_test)

14\. Return the preprocessed training and testing datasets.



Training data: X\_train, y\_train

Testing data: X\_test, y\_test

15\. Stop.



##### **Preprocessing code:**



\# ============================================================

\# UNIVERSAL TABULAR DATA PREPROCESSING

\# ============================================================



import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model\_selection import train\_test\_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (

&#x20;   OneHotEncoder,

&#x20;   StandardScaler

)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin





\# ============================================================

\# STEP 1: LOAD DATASET

\# ============================================================



FILE\_PATH = "ev\_battery\_health.csv"



\# Change this to your actual target column.

\# Example: "Price", "Purchased", "Outcome"

TARGET\_COLUMN = "degradation\_category"



df = pd.read\_csv(FILE\_PATH)



print("\\n========== DATASET LOADED ==========")

print(df.head())



print("\\nDataset Shape:", df.shape)





\# ============================================================

\# STEP 2: BASIC DATASET INFORMATION

\# ============================================================



print("\\n========== DATASET INFORMATION ==========")



df.info()



print("\\nStatistical Summary:")

print(df.describe(include="all"))



print("\\nMissing Values:")

print(df.isnull().sum())



print("\\nDuplicate Rows:", df.duplicated().sum())





\# ============================================================

\# STEP 3: REMOVE DUPLICATES

\# ============================================================



df = df.drop\_duplicates().copy()



print("\\nShape After Removing Duplicates:", df.shape)





\# ============================================================

\# STEP 4: REMOVE EMPTY COLUMNS

\# ============================================================



\# Remove columns containing no usable values.

df = df.dropna(axis=1, how="all")



\# Validate the target column.

if TARGET\_COLUMN is not None and TARGET\_COLUMN not in df.columns:

&#x20;   raise ValueError(

&#x20;       f"Target column '{TARGET\_COLUMN}' not found. "

&#x20;       f"Available columns: {list(df.columns)}"

&#x20;   )



\# Remove rows with missing target values for supervised learning.

\# Do not impute target values as though they were input features.

if TARGET\_COLUMN is not None:

&#x20;   df = df.dropna(subset=\[TARGET\_COLUMN]).copy()



print("\\nShape After Removing Empty Columns:", df.shape)





\# ============================================================

\# STEP 5: SEPARATE FEATURES AND TARGET

\# ============================================================



if TARGET\_COLUMN is not None:

&#x20;   X = df.drop(columns=\[TARGET\_COLUMN]).copy()

&#x20;   y = df\[TARGET\_COLUMN].copy()

else:

&#x20;   X = df.copy()

&#x20;   y = None



\# Remove columns that are entirely missing from the input features.

X = X.dropna(axis=1, how="all")



\# Remove columns with only one distinct non-missing value.

\# Such columns generally provide no predictive information.

constant\_columns = \[

&#x20;   col for col in X.columns

&#x20;   if X\[col].nunique(dropna=True) <= 1

]



X = X.drop(columns=constant\_columns)



print("\\nRemoved Constant Columns:", constant\_columns)





\# ============================================================

\# STEP 6: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

\# ============================================================



numerical\_columns = X.select\_dtypes(

&#x20;   include=\["number"]

).columns.tolist()



categorical\_columns = X.select\_dtypes(

&#x20;   include=\["object", "category", "bool", "string"]

).columns.tolist()



\# Convert infinite numerical values into missing values.

if numerical\_columns:

&#x20;   X\[numerical\_columns] = X\[numerical\_columns].replace(

&#x20;       \[np.inf, -np.inf], np.nan

&#x20;   )



print("\\nNumerical Columns:")

print(numerical\_columns)



print("\\nCategorical Columns:")

print(categorical\_columns)





\# ============================================================

\# STEP 7: VISUALIZE MISSING VALUES

\# ============================================================



missing\_counts = X.isnull().sum()

missing\_counts = missing\_counts\[missing\_counts > 0]



if not missing\_counts.empty:



&#x20;   plt.figure(figsize=(10, 5))



&#x20;   missing\_counts.sort\_values(ascending=False).plot(

&#x20;       kind="bar"

&#x20;   )



&#x20;   plt.title("Missing Values Before Preprocessing")

&#x20;   plt.xlabel("Features")

&#x20;   plt.ylabel("Number of Missing Values")

&#x20;   plt.xticks(rotation=45, ha="right")

&#x20;   plt.tight\_layout()

&#x20;   plt.show()



else:

&#x20;   print("\\nNo missing values found in input features.")





\# ============================================================

\# STEP 8: NUMERICAL FEATURE DISTRIBUTION

\# ============================================================



if numerical\_columns:



&#x20;   X\[numerical\_columns].hist(

&#x20;       figsize=(14, 10),

&#x20;       bins=20,

&#x20;       edgecolor="black"

&#x20;   )



&#x20;   plt.suptitle("Numerical Feature Distributions")

&#x20;   plt.tight\_layout()

&#x20;   plt.show()





\# ============================================================

\# STEP 9: CORRELATION HEATMAP

\# ============================================================



if len(numerical\_columns) >= 2:



&#x20;   plt.figure(figsize=(10, 7))



&#x20;   correlation = X\[numerical\_columns].corr()



&#x20;   sns.heatmap(

&#x20;       correlation,

&#x20;       annot=True,

&#x20;       cmap="coolwarm",

&#x20;       fmt=".2f"

&#x20;   )



&#x20;   plt.title("Correlation Heatmap")

&#x20;   plt.tight\_layout()

&#x20;   plt.show()





\# ============================================================

\# STEP 10: OUTLIER DETECTION USING IQR

\# ============================================================



def detect\_outliers\_iqr(data, columns):



&#x20;   outlier\_counts = {}



&#x20;   for col in columns:



&#x20;       series = data\[col].dropna()



&#x20;       if series.empty:

&#x20;           outlier\_counts\[col] = 0

&#x20;           continue



&#x20;       Q1 = series.quantile(0.25)

&#x20;       Q3 = series.quantile(0.75)



&#x20;       IQR = Q3 - Q1



&#x20;       lower\_bound = Q1 - 1.5 \* IQR

&#x20;       upper\_bound = Q3 + 1.5 \* IQR



&#x20;       outlier\_mask = (

&#x20;           (data\[col] < lower\_bound) |

&#x20;           (data\[col] > upper\_bound)

&#x20;       )



&#x20;       outlier\_counts\[col] = int(outlier\_mask.sum())



&#x20;   return pd.Series(

&#x20;       outlier\_counts,

&#x20;       name="Outlier Count"

&#x20;   )





outliers\_before = detect\_outliers\_iqr(

&#x20;   X,

&#x20;   numerical\_columns

)



print("\\n========== OUTLIERS BEFORE HANDLING ==========")

print(outliers\_before)





\# ============================================================

\# STEP 11: BOXPLOT BEFORE OUTLIER HANDLING

\# ============================================================



if numerical\_columns:



&#x20;   for start in range(0, len(numerical\_columns), 6):



&#x20;       cols = numerical\_columns\[start:start + 6]



&#x20;       fig, axes = plt.subplots(

&#x20;           1,

&#x20;           len(cols),

&#x20;           figsize=(4 \* len(cols), 5)

&#x20;       )



&#x20;       axes = np.atleast\_1d(axes)



&#x20;       for ax, col in zip(axes, cols):



&#x20;           sns.boxplot(

&#x20;               y=X\[col],

&#x20;               ax=ax

&#x20;           )



&#x20;           ax.set\_title(f"Before: {col}")



&#x20;       plt.tight\_layout()

&#x20;       plt.show()





\# ============================================================

\# STEP 12: TRAIN-TEST SPLIT

\# ============================================================



if y is not None:



&#x20;   # Use stratification for classification targets when

&#x20;   # class frequencies permit it.

&#x20;   # For this generic program, start without stratification.

&#x20;   X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;       X,

&#x20;       y,

&#x20;       test\_size=0.2,

&#x20;       random\_state=42

&#x20;   )



else:



&#x20;   # For unsupervised datasets, split only the features.

&#x20;   X\_train, X\_test = train\_test\_split(

&#x20;       X,

&#x20;       test\_size=0.2,

&#x20;       random\_state=42

&#x20;   )



&#x20;   y\_train = None

&#x20;   y\_test = None



print("\\nTraining Data Shape:", X\_train.shape)

print("Testing Data Shape:", X\_test.shape)





\# ============================================================

\# STEP 13: HANDLE OUTLIERS USING IQR CAPPING

\# ============================================================



\# Determine bounds using training data only.

\# This avoids using the test distribution to fit the

\# outlier-handling transformation.



class IQRCapper(BaseEstimator, TransformerMixin):



&#x20;   def fit(self, X, y=None):



&#x20;       X\_df = pd.DataFrame(X).copy()



&#x20;       self.columns\_ = X\_df.columns.tolist()

&#x20;       self.lower\_bounds\_ = {}

&#x20;       self.upper\_bounds\_ = {}



&#x20;       for col in self.columns\_:



&#x20;           values = pd.to\_numeric(

&#x20;               X\_df\[col],

&#x20;               errors="coerce"

&#x20;           ).dropna()



&#x20;           if values.empty:

&#x20;               self.lower\_bounds\_\[col] = -np.inf

&#x20;               self.upper\_bounds\_\[col] = np.inf

&#x20;               continue



&#x20;           Q1 = values.quantile(0.25)

&#x20;           Q3 = values.quantile(0.75)



&#x20;           IQR = Q3 - Q1



&#x20;           self.lower\_bounds\_\[col] = Q1 - 1.5 \* IQR

&#x20;           self.upper\_bounds\_\[col] = Q3 + 1.5 \* IQR



&#x20;       return self



&#x20;   def transform(self, X):



&#x20;       X\_df = pd.DataFrame(X).copy()



&#x20;       for col in self.columns\_:



&#x20;           X\_df\[col] = X\_df\[col].clip(

&#x20;               lower=self.lower\_bounds\_\[col],

&#x20;               upper=self.upper\_bounds\_\[col]

&#x20;           )



&#x20;       return X\_df



&#x20;   def get\_feature\_names\_out(self, input\_features=None):

&#x20;       if input\_features is None:

&#x20;           return np.array(self.columns\_, dtype=object)

&#x20;       return np.array(input\_features, dtype=object)





\# ============================================================

\# STEP 14: BUILD PREPROCESSING PIPELINE

\# ============================================================



\# Numerical pipeline:

\# 1. Impute missing values with median.

\# 2. Cap outliers using training-set IQR bounds.

\# 3. Standardize numerical features.



numerical\_pipeline = Pipeline(

&#x20;   steps=\[

&#x20;       ("imputer", SimpleImputer(strategy="median")),

&#x20;       ("outlier\_capper", IQRCapper()),

&#x20;       ("scaler", StandardScaler())

&#x20;   ]

)





\# Categorical pipeline:

\# 1. Impute missing values with the most frequent value.

\# 2. One-hot encode categorical features.



categorical\_pipeline = Pipeline(

&#x20;   steps=\[

&#x20;       (

&#x20;           "imputer",

&#x20;           SimpleImputer(strategy="most\_frequent")

&#x20;       ),

&#x20;       (

&#x20;           "encoder",

&#x20;           OneHotEncoder(handle\_unknown="ignore")

&#x20;       )

&#x20;   ]

)





\# Apply the appropriate pipeline to each feature type.



transformers = \[]



if numerical\_columns:

&#x20;   transformers.append(

&#x20;       ("num", numerical\_pipeline, numerical\_columns)

&#x20;   )



if categorical\_columns:

&#x20;   transformers.append(

&#x20;       ("cat", categorical\_pipeline, categorical\_columns)

&#x20;   )



preprocessor = ColumnTransformer(

&#x20;   transformers=transformers,

&#x20;   remainder="drop"

)





\# ============================================================

\# STEP 15: FIT ON TRAINING DATA AND TRANSFORM BOTH SETS

\# ============================================================



X\_train\_processed = preprocessor.fit\_transform(X\_train)



X\_test\_processed = preprocessor.transform(X\_test)





\# ============================================================

\# STEP 16: CONVERT PROCESSED DATA TO DATAFRAMES

\# ============================================================



feature\_names = preprocessor.get\_feature\_names\_out()



if hasattr(X\_train\_processed, "toarray"):

&#x20;   X\_train\_processed = X\_train\_processed.toarray()



if hasattr(X\_test\_processed, "toarray"):

&#x20;   X\_test\_processed = X\_test\_processed.toarray()



X\_train\_processed = pd.DataFrame(

&#x20;   X\_train\_processed,

&#x20;   columns=feature\_names,

&#x20;   index=X\_train.index

)



X\_test\_processed = pd.DataFrame(

&#x20;   X\_test\_processed,

&#x20;   columns=feature\_names,

&#x20;   index=X\_test.index

)



print("\\n========== PREPROCESSING COMPLETED ==========")



print("\\nProcessed Training Data:")

print(X\_train\_processed.head())



print("\\nProcessed Testing Data:")

print(X\_test\_processed.head())



print("\\nProcessed Training Shape:", X\_train\_processed.shape)

print("Processed Testing Shape:", X\_test\_processed.shape)





\# ============================================================

\# STEP 17: OUTLIER VISUALIZATION AFTER HANDLING

\# ============================================================



\# The processed numerical features are standardized.

\# The boxplots show their distributions after capping

\# and scaling.



processed\_numerical\_columns = \[

&#x20;   col for col in X\_train\_processed.columns

&#x20;   if col.startswith("num\_\_")

]



if processed\_numerical\_columns:



&#x20;   for start in range(

&#x20;       0,

&#x20;       len(processed\_numerical\_columns),

&#x20;       6

&#x20;   ):



&#x20;       cols = processed\_numerical\_columns\[start:start + 6]



&#x20;       fig, axes = plt.subplots(

&#x20;           1,

&#x20;           len(cols),

&#x20;           figsize=(4 \* len(cols), 5)

&#x20;       )



&#x20;       axes = np.atleast\_1d(axes)



&#x20;       for ax, col in zip(axes, cols):



&#x20;           sns.boxplot(

&#x20;               y=X\_train\_processed\[col],

&#x20;               ax=ax

&#x20;           )



&#x20;           ax.set\_title(f"After: {col}")



&#x20;       plt.tight\_layout()

&#x20;       plt.show()





\# ============================================================

\# STEP 18: VISUALIZE MISSING VALUES AFTER PREPROCESSING

\# ============================================================



print("\\nMissing Values After Preprocessing:")



print(X\_train\_processed.isnull().sum().sum())



print("Missing Values in Test Data:")



print(X\_test\_processed.isnull().sum().sum())





\# ============================================================

\# STEP 19: FINAL OUTPUT

\# ============================================================



print("\\n========== FINAL OUTPUT ==========")



print("X\_train\_processed:", X\_train\_processed.shape)

print("X\_test\_processed:", X\_test\_processed.shape)



if y is not None:



&#x20;   print("y\_train:", y\_train.shape)

&#x20;   print("y\_test:", y\_test.shape)



print("\\nDataset is ready for the next Machine Learning step.")



##### **Linear Regression:**



\# ==================================================

\# LINEAR REGRESSION

\# ==================================================



import numpy as np

import matplotlib.pyplot as plt



from sklearn.linear\_model import LinearRegression

from sklearn.metrics import (

&#x20;   mean\_absolute\_error,

&#x20;   mean\_squared\_error,

&#x20;   r2\_score

)



\# 1. Create the model

model = LinearRegression()



\# 2. Train the model

model.fit(X\_train\_processed, y\_train)



\# 3. Predict

y\_pred = model.predict(X\_test\_processed)



\# 4. Evaluate

mae = mean\_absolute\_error(y\_test, y\_pred)

mse = mean\_squared\_error(y\_test, y\_pred)

rmse = np.sqrt(mse)

r2 = r2\_score(y\_test, y\_pred)



print("LINEAR REGRESSION RESULTS")

print("-------------------------")

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2)



\# 5. Visualize actual vs predicted values

plt.figure(figsize=(8, 5))



plt.scatter(y\_test, y\_pred)



min\_value = min(np.min(y\_test), np.min(y\_pred))

max\_value = max(np.max(y\_test), np.max(y\_pred))



plt.plot(

&#x20;   \[min\_value, max\_value],

&#x20;   \[min\_value, max\_value],

&#x20;   linestyle="--"

)



plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Linear Regression: Actual vs Predicted")

plt.tight\_layout()

plt.show()





##### **Logistic Regression:**



\# ==================================================

\# LOGISTIC REGRESSION

\# ==================================================



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear\_model import LogisticRegression

from sklearn.metrics import (

&#x20;   accuracy\_score,

&#x20;   classification\_report,

&#x20;   confusion\_matrix

)



\# 1. Create the model

model = LogisticRegression(

&#x20;   max\_iter=1000,

&#x20;   random\_state=42

)



\# 2. Train the model

model.fit(X\_train\_processed, y\_train)



\# 3. Predict

y\_pred = model.predict(X\_test\_processed)



\# 4. Evaluate

print("LOGISTIC REGRESSION RESULTS")

print("---------------------------")



print("Accuracy:", accuracy\_score(y\_test, y\_pred))



print("\\nClassification Report:")

print(classification\_report(y\_test, y\_pred, zero\_division=0))



\# 5. Confusion matrix

cm = confusion\_matrix(y\_test, y\_pred)



print("Confusion Matrix:")

print(cm)



\# 6. Visualize confusion matrix

plt.figure(figsize=(7, 5))



sns.heatmap(

&#x20;   cm,

&#x20;   annot=True,

&#x20;   fmt="d",

&#x20;   cmap="Blues"

)



plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("Logistic Regression: Confusion Matrix")

plt.tight\_layout()

plt.show()



##### SVM:



\# ==================================================

\# SUPPORT VECTOR MACHINE (SVM)

\# ==================================================



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC

from sklearn.metrics import (

&#x20;   accuracy\_score,

&#x20;   classification\_report,

&#x20;   confusion\_matrix

)



\# 1. Create the model

model = SVC(

&#x20;   kernel="rbf",

&#x20;   C=1.0,

&#x20;   gamma="scale"

)



\# 2. Train the model

model.fit(X\_train\_processed, y\_train)



\# 3. Predict

y\_pred = model.predict(X\_test\_processed)



\# 4. Evaluate

print("SVM RESULTS")

print("-----------")



print("Accuracy:", accuracy\_score(y\_test, y\_pred))



print("\\nClassification Report:")

print(classification\_report(y\_test, y\_pred, zero\_division=0))



\# 5. Confusion matrix

cm = confusion\_matrix(y\_test, y\_pred)



print("\\nConfusion Matrix:")

print(cm)



\# 6. Visualization

plt.figure(figsize=(7, 5))



sns.heatmap(

&#x20;   cm,

&#x20;   annot=True,

&#x20;   fmt="d",

&#x20;   cmap="Greens"

)



plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.title("SVM: Confusion Matrix")

plt.tight\_layout()

plt.show()



##### **Random Forest:**



\# ==================================================

\# RANDOM FOREST

\# ==================================================



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.ensemble import (

&#x20;   RandomForestClassifier,

&#x20;   RandomForestRegressor

)



from sklearn.metrics import (

&#x20;   accuracy\_score,

&#x20;   classification\_report,

&#x20;   mean\_absolute\_error,

&#x20;   mean\_squared\_error,

&#x20;   r2\_score

)



\# 1. Determine the task type

is\_classification = (

&#x20;   y\_train.dtype == "object"

&#x20;   or str(y\_train.dtype) == "category"

&#x20;   or str(y\_train.dtype) == "bool"

&#x20;   or y\_train.nunique() <= 20

)



\# 2. Create the model

if is\_classification:



&#x20;   model = RandomForestClassifier(

&#x20;       n\_estimators=100,

&#x20;       random\_state=42

&#x20;   )



else:



&#x20;   model = RandomForestRegressor(

&#x20;       n\_estimators=100,

&#x20;       random\_state=42

&#x20;   )



\# 3. Train

model.fit(X\_train\_processed, y\_train)



\# 4. Predict

y\_pred = model.predict(X\_test\_processed)



\# 5. Evaluate

print("RANDOM FOREST RESULTS")

print("---------------------")



if is\_classification:



&#x20;   print("Accuracy:", accuracy\_score(y\_test, y\_pred))



&#x20;   print("\\nClassification Report:")

&#x20;   print(classification\_report(

&#x20;       y\_test,

&#x20;       y\_pred,

&#x20;       zero\_division=0

&#x20;   ))



else:



&#x20;   print("MAE:", mean\_absolute\_error(y\_test, y\_pred))



&#x20;   print("MSE:", mean\_squared\_error(y\_test, y\_pred))



&#x20;   print("RMSE:", np.sqrt(

&#x20;       mean\_squared\_error(y\_test, y\_pred)

&#x20;   ))



&#x20;   print("R2 Score:", r2\_score(y\_test, y\_pred))





\# 6. Feature importance

importance = pd.Series(

&#x20;   model.feature\_importances\_,

&#x20;   index=X\_train\_processed.columns

).sort\_values(ascending=False)



print("\\nTop Features:")

print(importance.head(10))



\# 7. Plot top 10 features

top\_features = importance.head(10).sort\_values()



plt.figure(figsize=(9, 6))



top\_features.plot(kind="barh")



plt.xlabel("Importance")

plt.ylabel("Features")

plt.title("Random Forest: Top 10 Feature Importances")

plt.tight\_layout()

plt.show()



##### **Decision Tree:**



\# ==================================================

\# DECISION TREE

\# ==================================================



import numpy as np

import matplotlib.pyplot as plt



from sklearn.tree import (

&#x20;   DecisionTreeClassifier,

&#x20;   DecisionTreeRegressor,

&#x20;   plot\_tree

)



from sklearn.metrics import (

&#x20;   accuracy\_score,

&#x20;   classification\_report,

&#x20;   mean\_absolute\_error,

&#x20;   mean\_squared\_error,

&#x20;   r2\_score

)



\# 1. Determine the task type

is\_classification = (

&#x20;   y\_train.dtype == "object"

&#x20;   or str(y\_train.dtype) == "category"

&#x20;   or str(y\_train.dtype) == "bool"

&#x20;   or y\_train.nunique() <= 20

)



\# 2. Create the model

if is\_classification:



&#x20;   model = DecisionTreeClassifier(

&#x20;       max\_depth=5,

&#x20;       random\_state=42

&#x20;   )



else:



&#x20;   model = DecisionTreeRegressor(

&#x20;       max\_depth=5,

&#x20;       random\_state=42

&#x20;   )



\# 3. Train

model.fit(X\_train\_processed, y\_train)



\# 4. Predict

y\_pred = model.predict(X\_test\_processed)



\# 5. Evaluate

print("DECISION TREE RESULTS")

print("---------------------")



if is\_classification:



&#x20;   print("Accuracy:", accuracy\_score(y\_test, y\_pred))



&#x20;   print("\\nClassification Report:")

&#x20;   print(classification\_report(

&#x20;       y\_test,

&#x20;       y\_pred,

&#x20;       zero\_division=0

&#x20;   ))



else:



&#x20;   print("MAE:", mean\_absolute\_error(y\_test, y\_pred))



&#x20;   print("MSE:", mean\_squared\_error(y\_test, y\_pred))



&#x20;   print("RMSE:", np.sqrt(

&#x20;       mean\_squared\_error(y\_test, y\_pred)

&#x20;   ))



&#x20;   print("R2 Score:", r2\_score(y\_test, y\_pred))





\# 6. Visualize the tree

plt.figure(figsize=(20, 10))



plot\_tree(

&#x20;   model,

&#x20;   feature\_names=X\_train\_processed.columns.tolist(),

&#x20;   class\_names=(

&#x20;       \[str(c) for c in model.classes\_]

&#x20;       if is\_classification else None

&#x20;   ),

&#x20;   filled=True,

&#x20;   rounded=True,

&#x20;   max\_depth=3,

&#x20;   fontsize=8

)



plt.title("Decision Tree Visualization")

plt.tight\_layout()

plt.show()



##### 

##### **Neural Network:**



\# ==================================================

\# NEURAL NETWORK

\# ==================================================



import numpy as np

import matplotlib.pyplot as plt



from sklearn.neural\_network import (

&#x20;   MLPClassifier,

&#x20;   MLPRegressor

)



from sklearn.metrics import (

&#x20;   accuracy\_score,

&#x20;   classification\_report,

&#x20;   mean\_absolute\_error,

&#x20;   mean\_squared\_error,

&#x20;   r2\_score

)



\# 1. Determine the task type

is\_classification = (

&#x20;   y\_train.dtype == "object"

&#x20;   or str(y\_train.dtype) == "category"

&#x20;   or str(y\_train.dtype) == "bool"

&#x20;   or y\_train.nunique() <= 20

)



\# 2. Create the model

if is\_classification:



&#x20;   model = MLPClassifier(

&#x20;       hidden\_layer\_sizes=(100, 50),

&#x20;       activation="relu",

&#x20;       solver="adam",

&#x20;       max\_iter=500,

&#x20;       random\_state=42,

&#x20;       early\_stopping=True

&#x20;   )



else:



&#x20;   model = MLPRegressor(

&#x20;       hidden\_layer\_sizes=(100, 50),

&#x20;       activation="relu",

&#x20;       solver="adam",

&#x20;       max\_iter=500,

&#x20;       random\_state=42,

&#x20;       early\_stopping=True

&#x20;   )



\# 3. Train

model.fit(X\_train\_processed, y\_train)



\# 4. Predict

y\_pred = model.predict(X\_test\_processed)



\# 5. Evaluate

print("NEURAL NETWORK RESULTS")

print("----------------------")



if is\_classification:



&#x20;   print("Accuracy:", accuracy\_score(y\_test, y\_pred))



&#x20;   print("\\nClassification Report:")

&#x20;   print(classification\_report(

&#x20;       y\_test,

&#x20;       y\_pred,

&#x20;       zero\_division=0

&#x20;   ))



else:



&#x20;   print("MAE:", mean\_absolute\_error(y\_test, y\_pred))



&#x20;   print("MSE:", mean\_squared\_error(y\_test, y\_pred))



&#x20;   print("RMSE:", np.sqrt(

&#x20;       mean\_squared\_error(y\_test, y\_pred)

&#x20;   ))



&#x20;   print("R2 Score:", r2\_score(y\_test, y\_pred))





\# 6. Plot the training loss curve

plt.figure(figsize=(8, 5))



plt.plot(

&#x20;   model.loss\_curve\_,

&#x20;   label="Training Loss"

)



plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.title("Neural Network: Training Loss Curve")

plt.legend()

plt.tight\_layout()

plt.show()

