# Upload and Load Data
from google.colab import files
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding="latin-1")

# Clean and Preprocess
df = df.drop_duplicates()
df.columns = df.columns.str.lower()
df = df.dropna()
df.reset_index(drop=True, inplace=True)

# === Exploratory Data Analysis ===

# 1. Distribution of Diseases
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='prognosis', order=df['prognosis'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Diseases')
plt.show()

# 2. Symptom Correlation Heatmap
plt.figure(figsize=(20,16))
sns.heatmap(df.drop('prognosis', axis=1).corr(), cmap='coolwarm')
plt.title('Symptom Correlation Heatmap')
plt.show()

# 3. Top Symptoms for First Disease
first_disease = df['prognosis'].unique()[0]
first_disease_data = df[df['prognosis'] == first_disease]
first_disease_symptoms = first_disease_data.drop('prognosis', axis=1).mean()
plt.figure(figsize=(12,8))
first_disease_symptoms.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title(f"Top Symptoms for {first_disease}")
plt.ylabel('Presence Rate')
plt.show()

# 4. Most Common Symptoms Overall
symptom_counts = df.drop('prognosis', axis=1).sum()
symptom_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Most Common Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.show()

# === KMeans-Based Grouping for Model Training ===
from sklearn.cluster import KMeans

symptom_features = df.drop('prognosis', axis=1)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['disease_group'] = kmeans.fit_predict(symptom_features)

# Map each group to most frequent disease
group_mapping = (
    df.groupby('disease_group')['prognosis']
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

# Prepare data for training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

X = df.drop(['prognosis', 'disease_group'], axis=1)
y = df['disease_group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train One Model: Random Forest ===
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(df['disease_group'].unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df['disease_group'].unique()))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix: Random Forest')
plt.show()

# Predict 5 test inputs and print disease names
input_symptoms = X_test.iloc[:5]
predicted_groups = model.predict(input_symptoms)
predicted_diseases = [group_mapping[g] for g in predicted_groups]

print("\nPredicted Diseases for 5 Test Inputs:")
for i, disease in enumerate(predicted_diseases):
    print(f"  Input {i+1}: {disease}")
