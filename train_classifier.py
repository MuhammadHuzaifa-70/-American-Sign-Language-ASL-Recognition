import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, data, labels, cv=5)  # You can adjust the number of folds (cv) as needed

# Print average accuracy
print("Cross-Validation Accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))

# Train the model on the entire dataset
model.fit(data, labels)

# Save model
with open('model.p', 'wb') as f:
    pickle.dump(model, f)
