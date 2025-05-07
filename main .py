#step1: import libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
from sklearn.utils.class_weight import compute_class_weight

#step 2: Load datasets
excel_file = "emo datasset.xlsx"
csv_file = "Detection.csv"
df_excel = pd.read_excel(excel_file)
df_csv = pd.read_csv(csv_file)

# Merge datasets
df_merged = pd.concat([df_excel, df_csv], ignore_index=True)

# Check missing respiration rate values
missing_respiration = df_merged['Respiration_Rate'].isnull().sum()
print(f"Missing respiration rate values: {missing_respiration}")

# Define features for prediction
features = ['Heart_Rate', 'Galvanic_Skin_Response', 'Skin_Temperature', 'Mood']  # Adjust based on available columns
df_known = df_merged.dropna(subset=['Respiration_Rate'])  # Data with known respiration rates
df_missing = df_merged[df_merged['Respiration_Rate'].isnull()]  # Data with missing respiration rates

# Encode 'Mood' column
le = LabelEncoder()
all_mood_values = pd.concat([df_known['Mood'], df_missing['Mood']]).unique()
le.fit(all_mood_values)

X = df_known[features]
y = df_known['Respiration_Rate']
X['Mood'] = le.transform(X['Mood'])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict missing values
X_missing = df_missing[features].copy()  # Ensure it's a copy
X_missing['Mood'] = le.transform(X_missing['Mood'])

df_missing = df_missing.copy()  # Ensure modifications are safe
df_missing['Respiration_Rate'] = model.predict(X_missing)

df_merged.update(df_missing)
df_merged["Heart_Rate"] = df_merged["Heart_Rate"].astype(float)


# Data visualization
df_merged.hist(color='skyblue')
plt.suptitle('Histogram of Physiological Signals')
plt.show()
df_merged['Mood'] = df_merged['Mood'].str.strip().str.lower().str.capitalize()
import matplotlib.ticker as ticker

plt.figure(figsize=(10,5))
ax = sns.countplot(y=df_merged['Mood'], hue=df_merged['Mood'], palette='coolwarm', legend=False)

plt.xscale('log')  # Keep the logarithmic scale
ticks = np.logspace(1, 5, num=10)  # Adjust range as needed
ax.set_xticks(ticks)
# Change x-axis labels to normal numbers
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
plt.title('Distribution of Emotion Labels')
plt.show()
df_merged['Mood'].value_counts()
#step 3: Define Bandpass Filter
def optimized_bandpass_filter(data, lowcut, highcut, fs=10.0, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.01)
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

#step 4: Apply Filtering
filter_params = {
    'Heart_Rate': (0.7, 3.0),
    'Skin_Temperature': (0.05, 0.49),
    'Galvanic_Skin_Response': (0.1, 4.5),
    'Respiration_Rate': (0.1, 1.8)
}

filtered_data = df_merged.copy()
for col, (lowcut, highcut) in filter_params.items():
    normalized_signal = (df_merged[col] - np.mean(df_merged[col])) / np.std(df_merged[col])
    filtered_data[col] = optimized_bandpass_filter(normalized_signal, lowcut, highcut)


#step 5: Normalize Data
scaler = StandardScaler()
numeric_data = filtered_data.select_dtypes(include=[np.number])
scaled_numeric_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)
# If you want to keep non-numeric columns, concatenate them back
non_numeric_data = filtered_data.select_dtypes(exclude=[np.number])
final_scaled_data = pd.concat([non_numeric_data.reset_index(drop=True), scaled_numeric_data], axis=1)

#step 6: Create Time Windows
def create_time_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i : i + window_size].values
        windows.append(window)
    return np.array(windows)

windows = create_time_windows(filtered_data, 100, 50)
windows_df = pd.DataFrame([w.flatten() for w in windows])
#step 7: Feature Extraction # Keep only numeric columns (exclude datetime and non-numeric columns)
numeric_windows = windows_df.select_dtypes(include=[np.number])
# Compute statistical features
stat_features = numeric_windows.apply(lambda col: {'mean': np.mean(col), 'std_dev': np.std(col), 'variance': np.var(col)}, axis=0)

# Convert list of dictionaries to DataFrame
features_df = pd.DataFrame(stat_features.tolist())

#step 8: Train-Test Split
X = features_df.iloc[:, :-1]
y = features_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#step 9: Handle Class Imbalance
y_train_categorized = pd.qcut(y_train, q=7, labels=False)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_categorized)

#step 10: Model Training
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_categorized), y=y_train_categorized)
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=500, random_state=42, class_weight=dict(enumerate(class_weights)))
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.08, colsample_bytree=0.9, random_state=42)
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
y_train_categorized = pd.qcut(y_train, q=7, labels=False)
ensemble_model.fit(X_train, y_train_categorized)

from sklearn.model_selection import cross_val_score
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Ensemble': ensemble_model
}

accuracies = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train_categorized, cv=5, scoring='accuracy')
    accuracies[name] = np.mean(scores)

# Plot the accuracies
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color = ['#4a3771', '#a3c746', '#386d82', '#399e7b'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

#step 11: Model Evaluation
ensemble_pred = ensemble_model.predict(X_test)
y_test_categorized = pd.qcut(y_test, q=7, labels=False)
print("Ensemble Model Accuracy:", accuracy_score(y_test_categorized, ensemble_pred))
mood_categories = df_merged['Mood'].unique()
label_mapping = {i: mood for i, mood in enumerate(mood_categories)}
pred5 = ensemble_model.predict(X_test)
predicted_mood_labels = [label_mapping[label] for label in pred5]
report = classification_report(
    y_test_categorized,
    pred5,
    target_names=mood_categories,  # Use mood categories as target names
    zero_division=0,  # Handle cases where there are no predictions for a class

)

print(report)

# step 12; the confusion_matrix function
cm5 = pd.DataFrame(confusion_matrix(y_test_categorized, pred5), index = df_merged['Mood'].unique(), columns = df_merged['Mood'].unique()) # Use y_test_categorized and replace labels with df['Mood'].unique()

plt.figure(figsize=(10, 8))
sns.heatmap(cm5, annot=True, cbar=False, fmt='g', cmap='viridis')  # Change 'coolwarm' to other colormaps if needed
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()

# Classification Report as Heatmap
report_dict = classification_report(y_test_categorized, pred5, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T.iloc[:-3, :-1]  # Exclude support column
plt.figure(figsize=(8, 5))
sns.heatmap(report_df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Classification Report")
plt.xlabel("Metrics")  
plt.ylabel("Emotion Classes") 
plt.show()

# Bar Chart for Accuracy per Mood
accuracy_per_class = np.diag(cm5) / np.sum(cm5, axis=1)
plt.figure(figsize=(8, 5))
sns.barplot(x=mood_categories, y=accuracy_per_class, hue=mood_categories, palette='viridis', legend=False)
plt.xlabel('Emotion Category')
plt.ylabel('Accuracy')
plt.title('Model Accuracy per Emotion')
plt.xticks(rotation=45)
plt.show()

#step 13
# Standardize Mood Labels Before Training
df_merged["Mood"] = df_merged["Mood"].str.upper().str.strip()

# Save trained model
joblib.dump(ensemble_model, "trained_emotion_model.pkl")

# Save only the columns used for scaling
columns_to_scale = ['Heart_Rate', 'Skin_Temperature', 'Galvanic_Skin_Response', 'Respiration_Rate']
scaler = StandardScaler()
scaler.fit(df_merged[columns_to_scale])  # Fit the scaler to only these columns
joblib.dump(scaler, "scaler.pkl")

# Fit and save label encoder
label_encoder = LabelEncoder()
df_merged["Mood"] = df_merged["Mood"].astype(str)  # Ensure string format
labels = np.sort(df_merged["Mood"].unique())  # Sort for consistency
label_encoder.fit(labels)
joblib.dump(label_encoder, "label_encoder.pkl")

# Load trained model and scaler
model = joblib.load("trained_emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load or create label encoder
if os.path.exists("label_encoder.pkl"):
    label_encoder = joblib.load("label_encoder.pkl")
else:
    default_moods = ['HAPPY', 'SAD', 'ANGRY', 'NEUTRAL', 'RELAXED', 'ANXIOUS', 'IRRITABLE']  # Standardized categories
    label_encoder = LabelEncoder()
    label_encoder.fit(default_moods)
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Warning: label_encoder.pkl not found. Using default encoder.")

import pandas as pd

import pandas as pd

# Define valid ranges for each physiological signal
valid_ranges = {
    "Heart Rate": (40, 180),  
    "Skin Temperature": (30, 40),
    "Galvanic Skin Response": (0.01, 10),
    "Respiration Rate": (8, 40)
}

# Function to take real-time input with validation
def get_user_input():
    print("Enter physiological signals:")
    user_inputs = {}

    for signal, (min_val, max_val) in valid_ranges.items():
        while True:
            try:
                value = float(input(f"{signal}: "))
                if min_val <= value <= max_val:
                    user_inputs[signal] = value
                    break
                else:
                    print(f"Invalid value! {signal} must be between {min_val} and {max_val}. Try again.")
            except ValueError:
                print("Invalid input! Please enter a numeric value.")

    return pd.DataFrame([user_inputs])

# Get user input
test_data = get_user_input()

# Rename columns to match the names used during model training
column_mapping = {
    "Heart Rate": "Heart_Rate",
    "Skin Temperature": "Skin_Temperature",
    "Galvanic Skin Response": "Galvanic_Skin_Response",
    "Respiration Rate": "Respiration_Rate"
}
test_data = test_data.rename(columns=column_mapping)

# Scale the input data
scaled_data = scaler.transform(test_data)
scaled_df = pd.DataFrame(scaled_data, columns=test_data.columns)


# Extract statistical features
def extract_features(df):
    return pd.DataFrame({
        'mean': [df.mean(axis=1).values[0]],
        'std_dev': [df.std(axis=1).values[0]]
    })

features_df = extract_features(scaled_df)
# Make prediction
predicted_label = model.predict(features_df)[0]
# Convert predicted label to emotion
if predicted_label in model.classes_:
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]
    print(f"Detected Emotion: {predicted_emotion}")
else:
    print(f"Error: Predicted label {predicted_label} is not in model classes. Check data consistency.")


# --------------------------
# TESTING SECTION
# --------------------------

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load saved model, scaler, and label encoder
model = joblib.load("trained_emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function to extract features from test input
def extract_features(df):
    return pd.DataFrame({
        'mean': [df.mean(axis=1).values[0]],
        'std_dev': [df.std(axis=1).values[0]]
    })

# Unit Test: Check feature extraction outputs
def test_extract_features():
    test_input = pd.DataFrame([[0.5, 1.5, 2.0, 3.0]], columns=['Heart_Rate', 'Skin_Temperature', 'Galvanic_Skin_Response', 'Respiration_Rate'])
    features = extract_features(test_input)
    assert 'mean' in features.columns
    assert 'std_dev' in features.columns
    print("\u2705 test_extract_features passed.")

# Unit Test: Scaler consistency
def test_scaler():
    sample = pd.DataFrame([[70, 36.5, 1.2, 16]], columns=['Heart_Rate', 'Skin_Temperature', 'Galvanic_Skin_Response', 'Respiration_Rate'])
    scaled = scaler.transform(sample)
    assert scaled.shape == (1, 4)
    print("\u2705 test_scaler passed.")

# Integration Test: Predict on scaled test sample
def test_model_prediction():
    test_sample = pd.DataFrame([[70, 36.5, 1.2, 16]], columns=['Heart_Rate', 'Skin_Temperature', 'Galvanic_Skin_Response', 'Respiration_Rate'])
    scaled = scaler.transform(test_sample)
    features = extract_features(pd.DataFrame(scaled, columns=test_sample.columns))
    prediction = model.predict(features)[0]
    assert prediction in model.classes_
    emotion = label_encoder.inverse_transform([prediction])[0]
    print(f"\u2705 test_model_prediction passed. Predicted Emotion: {emotion}")

# Run all tests
if __name__ == "__main__":
    test_extract_features()
    test_scaler()
    test_model_prediction()
    print("All tests completed successfully.")
