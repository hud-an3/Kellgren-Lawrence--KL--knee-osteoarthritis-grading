import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
from collections import Counter

# Dataset paths
Test_data  = r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\test"
Train_data = r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\train"
Val_data   = r"C:\Users\DELL\Desktop\ML semester Project\KneeXrayData\ClsKLData\kneeKL224\val"


def load_dataset(data_path):
    classes = sorted(os.listdir(data_path))
    image_paths, labels = [], []

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        for img in os.listdir(cls_path):
            image_paths.append(os.path.join(cls_path, img))
            labels.append(idx)

    return image_paths, labels, classes


# Preprocessing for EDA & classical ML
def preprocess_gray(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    return img


# Preprocessing for CNN (transfer learning)
def preprocess_cnn(img):
    #img = img.convert("L")
    img = img.resize((224, 224))
    # Contrast stretching (PIL)
    #img = ImageOps.autocontrast(img)
    # Sharpening
    #img = ImageEnhance.Sharpness(img).enhance(1.5)
    # Convert to RGB for CNNs
    img = img.convert("RGB")
    # Normalize
    #img = np.array(img, dtype=np.float32) / 255.0
    return img


if __name__ == "__main__":

    # Load dataset paths
    X_train_paths, y_train, classes = load_dataset(Train_data)
    X_val_paths, y_val, _           = load_dataset(Val_data)
    X_test_paths, y_test, _         = load_dataset(Test_data)

    # Class distribution (test set)
    test_counts = Counter(y_test)
    for k, v in test_counts.items():
        print(f"KL Grade {k}: {v} images")

    # Sample image visualization
    plt.figure(figsize=(12, 8))

    for i, cls in enumerate(classes):
        cls_images = os.listdir(os.path.join(Test_data, cls))[:3]

        for j, img_name in enumerate(cls_images):
            img_path = os.path.join(Test_data, cls, img_name)
            img = preprocess_gray(img_path)

            plt.subplot(len(classes), 3, i * 3 + j + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"KL {i}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Bar chart
    plt.figure()
    plt.bar(test_counts.keys(), test_counts.values())
    plt.xlabel("KL Grade")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution of KOA Test Dataset")
    plt.show()


#Proving that all the images in the dataset are already of fixed size 224x224

#sizes = set()

#for cls in os.listdir(Test_data):
#    cls_path = os.path.join(Test_data, cls)
#    for img_name in os.listdir(cls_path):
#        img_path = os.path.join(cls_path, img_name)
#        img = Image.open(img_path)
#        sizes.add(img.size)

#print(sizes)
