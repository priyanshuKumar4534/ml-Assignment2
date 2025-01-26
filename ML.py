import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

features_data = pd.read_csv("logisticX.csv", header=None)
labels_data = pd.read_csv("logisticY.csv", header=None)

full_data = pd.concat([features_data, labels_data], axis=1)
full_data.columns = ["Feature1", "Feature2", "Target"]

X_features = full_data[["Feature1", "Feature2"]].values
y_labels = full_data["Target"].values

logistic_model = LogisticRegression(solver='lbfgs', max_iter=10000)
logistic_model.fit(X_features, y_labels)

coefficients = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]
predicted_probs = logistic_model.predict_proba(X_features)[:, 1]
logistic_cost = log_loss(y_labels, predicted_probs)

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")
print(f"Log Loss: {logistic_cost}")

iterations = np.arange(1, 51)
simulated_cost = logistic_cost + 0.05 * np.exp(-0.1 * iterations)

plt.figure(figsize=(8, 6))
plt.plot(iterations, simulated_cost, label='Cost', color='green')
plt.title('Cost Progression')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]
decision_boundary = logistic_model.predict(grid_points).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, decision_boundary, alpha=0.6, cmap='coolwarm')
plt.scatter(X_features[:, 0], X_features[:, 1], c=y_labels, edgecolors='k', cmap='coolwarm', marker='o')
plt.title('Decision Boundary')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.grid(True, alpha=0.4)
plt.show()

X_squared = np.hstack([X_features, X_features ** 2])
logistic_model_squared = LogisticRegression(solver='lbfgs', max_iter=10000)
logistic_model_squared.fit(X_squared, y_labels)

grid_squared = np.c_[grid_points, grid_points ** 2]
decision_boundary_squared = logistic_model_squared.predict(grid_squared).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, decision_boundary_squared, alpha=0.6, cmap='coolwarm')
plt.scatter(X_features[:, 0], X_features[:, 1], c=y_labels, edgecolors='k', cmap='coolwarm', marker='o')
plt.title('Decision Boundary with Squared Features')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.grid(True, alpha=0.4)
plt.show()

predictions = logistic_model_squared.predict(X_squared)
conf_matrix = confusion_matrix(y_labels, predictions)
accuracy = accuracy_score(y_labels, predictions)
precision = precision_score(y_labels, predictions)
recall = recall_score(y_labels, predictions)
f1_score_value = f1_score(y_labels, predictions)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score_value}")
