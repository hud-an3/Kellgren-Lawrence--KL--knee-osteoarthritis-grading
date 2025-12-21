import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from eda import preprocess_gray, load_dataset   


Test_data = r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\test"
Train_data = r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\train"
Val_data= r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\val"

# Load dataset
X_train_paths, y_train, classes = load_dataset(Train_data)
X_test_paths, y_test, _         = load_dataset(Test_data)
X_val_paths, y_val, _     = load_dataset(Val_data)

#combining train and test datsets for training
X_train_paths = X_train_paths + X_val_paths
y_train = y_train + y_val


# Feature Extraction
def extract_glcm(img):
    
    #conversion to uint8 for glcm
    img_uint8 = (img * 255).astype(np.uint8)
    # Quantize to 32 gray levels
    img_q = img_uint8 // 8 
    
    glcm = graycomatrix(
        img_q,
        distances=[1, 2, 4],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features)

# Feature Matrix
def build_features(image_paths, labels):
    X, y = [], []
    for path, label in zip(image_paths, labels):
        img = preprocess_gray(path)
        features = extract_glcm(img)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


X_train, y_train = build_features(X_train_paths, y_train)
X_test, y_test   = build_features(X_test_paths, y_test)

# Train Random Forest 
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=24,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)               # shape: (N, 5)  


# Evaluation on test set
y_pred = rf.predict(X_test)

np.save("rf_probs.npy", rf_probs)                   #ensemble
np.save("y_test.npy", y_test)                       

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt="d", cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Classical ML")
plt.show()

