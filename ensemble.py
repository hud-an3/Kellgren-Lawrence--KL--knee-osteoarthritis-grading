import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

rf_probs  = np.load("rf_probs.npy")
cnn_probs = np.load("cnn_probs.npy")
y_test    = np.load("y_test.npy")

# Weighted average (CNN is stronger)
alpha = 0.7
ensemble_probs = alpha * cnn_probs + (1 - alpha) * rf_probs

y_pred = np.argmax(ensemble_probs, axis=1)

print("ENSEMBLE RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Cohen Kappa:", cohen_kappa_score(y_test, y_pred))
