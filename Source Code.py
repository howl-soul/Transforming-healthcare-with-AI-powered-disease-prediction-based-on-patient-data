from google.colab import files
import io
import pandas as pd

# Upload file
uploaded = files.upload()
filename = next(iter(uploaded))

# Read CSV with explicit encoding
df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding="latin-1")

# Display first few rows
print(df.head())

df.head(5)

df.info()

df = df.drop_duplicates()

print(df.isnull().sum())

df.describe()

df.columns = df.columns.str.lower()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Correct column name
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='prognosis', order=df['prognosis'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Diseases')
plt.show()


# Step 4: Check symptom correlation (heatmap)
plt.figure(figsize=(20,16))
sns.heatmap(df.drop('prognosis', axis=1).corr(), cmap='coolwarm')

plt.title('Symptom Correlation Heatmap')
plt.show()

# Step 5: Top Symptoms for Specific Diseases (optional deep dive)
# Example: How symptoms look for the first disease
# Step 1: Filter data for the first disease
first_disease = df['prognosis'].unique()[0]
first_disease_data = df[df['prognosis'] == first_disease]

# Step 2: Select only symptom columns (exclude prognosis) and take mean
first_disease_symptoms = first_disease_data.drop('prognosis', axis=1).mean()

# Step 3: Plot
plt.figure(figsize=(12,8))
first_disease_symptoms.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title(f"Top Symptoms for {first_disease}")
plt.ylabel('Presence Rate')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Initialize encoder
le = LabelEncoder()

# Fit and transform the 'prognosis' column
df['prognosis'] = le.fit_transform(df['prognosis'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
target_column ='prognosis'

# Features and target
X = df.drop(target_column, axis=1)  # Drop prognosis column for features
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.2f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loop through all models to plot confusion matrices
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    unique_labels = sorted(list(set(y_test)))  # Ensure labels match those in test data
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.show()

import seaborn as sns

# Compute correlation matrix
correlation_matrix = df.drop('prognosis', axis=1).corr()  # Adjust 's' for prognosis column
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Symptom Relationships Heatmap')
plt.show()

import matplotlib.pyplot as plt

# Example: Count the occurrences of symptoms
symptom_counts = df.drop('prognosis', axis=1).sum()  # Adjust 's' for prognosis column
from google.colab import files
import io
import pandas as pd

# Upload file
uploaded = files.upload()
filename = next(iter(uploaded))

# Read CSV with explicit encoding
df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding="latin-1")

# Display first few rows
print(df.head())

df.head(5)

df.info()

df = df.drop_duplicates()

print(df.isnull().sum())

df.describe()

df.columns = df.columns.str.lower()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Correct column name
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='prognosis', order=df['prognosis'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Diseases')
plt.show()


# Step 4: Check symptom correlation (heatmap)
plt.figure(figsize=(20,16))
sns.heatmap(df.drop('prognosis', axis=1).corr(), cmap='coolwarm')

plt.title('Symptom Correlation Heatmap')
plt.show()

# Step 5: Top Symptoms for Specific Diseases (optional deep dive)
# Example: How symptoms look for the first disease
# Step 1: Filter data for the first disease
first_disease = df['prognosis'].unique()[0]
first_disease_data = df[df['prognosis'] == first_disease]

# Step 2: Select only symptom columns (exclude prognosis) and take mean
first_disease_symptoms = first_disease_data.drop('prognosis', axis=1).mean()

# Step 3: Plot
plt.figure(figsize=(12,8))
first_disease_symptoms.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title(f"Top Symptoms for {first_disease}")
plt.ylabel('Presence Rate')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Initialize encoder
le = LabelEncoder()

# Fit and transform the 'prognosis' column
df['prognosis'] = le.fit_transform(df['prognosis'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
target_column ='prognosis'

# Features and target
X = df.drop(target_column, axis=1)  # Drop prognosis column for features
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.2f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loop through all models to plot confusion matrices
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    unique_labels = sorted(list(set(y_test)))  # Ensure labels match those in test data
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.show()

import seaborn as sns

# Compute correlation matrix
correlation_matrix = df.drop('prognosis', axis=1).corr()  # Adjust 's' for prognosis column
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Symptom Relationships Heatmap')
plt.show()

import matplotlib.pyplot as plt

# Example: Count the occurrences of symptoms
symptom_counts = df.drop('prognosis', axis=1).sum()  # Adjust 's' for prognosis column
symptom_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Most Common Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.show()

# Define the input symptoms (first 5 rows of test data)
input_symptoms = X_test.iloc[0:5]

# Create a list to store results
results = []

# Loop through all the models
for name, model in models.items():
    # Predict using the model
    predictions = model.predict(input_symptoms)

    # Store results in a dictionary
    for i in range(len(input_symptoms)):
        results.append({
            'Model': name,
            'Input Symptoms': input_symptoms.iloc[i].to_dict(),
            'Predicted Disease': predictions[i]
        })

# Create a DataFrame from the results
result_table = pd.DataFrame(results)

# Display the results
print(result_table)symptom_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Most Common Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.show()

# Define the input symptoms (first 5 rows of test data)
input_symptoms = X_test.iloc[0:5]

# Create a list to store results
results = []

# Loop through all the models
for name, model in models.items():
    # Predict using the model
    predictions = model.predict(input_symptoms)

    # Store results in a dictionary
    for i in range(len(input_symptoms)):
        results.append({
            'Model': name,
            'Input Symptoms': input_symptoms.iloc[i].to_dict(),
            'Predicted Disease': predictions[i]
        })

# Create a DataFrame from the results
result_table = pd.DataFrame(results)

# Display the results
print(result_table)
