import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the spectroscopic data
data = pd.read_csv('spectroscopic_data.csv')

# Preprocess the data (e.g., handle missing values, normalize features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)

# Generate a report summarizing the analyzed spectra and potential indications of extraterrestrial life
report_md = f"""
# Spectroscopic Analysis Report

## Summary

The spectroscopic data from celestial bodies was analyzed using a random forest classifier. The model achieved the following results:

{report}

## Potential Indications of Extraterrestrial Life

Based on the analysis of the spectroscopic data, the following potential indications of extraterrestrial life were identified:

- [List any potential indications here]

"""

print(report_md)
