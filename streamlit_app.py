# pages/model_development.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

st.title("📊 Model Development")

st.markdown("This section provides visualizations of model training progress and evaluation metrics.")

# ---------- Simulated Data for Accuracy & Loss ----------
epochs = np.arange(1, 11)
train_acc = [60, 64, 72, 75, 78, 80, 82, 84, 86, 88]
val_acc = [59, 63, 70, 73, 76, 77, 78, 79, 80, 81]
train_loss = [1.5, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]
val_loss = [1.6, 1.3, 1.1, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48]

# ---------- Accuracy Plot ----------
st.subheader("📈 Training vs Validation Accuracy")
fig, ax = plt.subplots()
ax.plot(epochs, train_acc, label="Train Accuracy", marker='o')
ax.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Train vs Validation Accuracy")
ax.legend()
st.pyplot(fig)

# ---------- Loss Plot ----------
st.subheader("📉 Training vs Validation Loss")
fig, ax = plt.subplots()
ax.plot(epochs, train_loss, label="Train Loss", marker='o')
ax.plot(epochs, val_loss, label="Validation Loss", marker='o')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Train vs Validation Loss")
ax.legend()
st.pyplot(fig)

# ---------- Accuracy w/ Different Optimizers ----------
st.subheader("⚙️ Accuracy Comparison: Different Optimizers")
optimizers = ["Adam", "SGD", "RMSprop"]
adam_acc = [60, 66, 70, 75, 79, 83, 85, 87, 88, 89]
sgd_acc = [58, 62, 67, 70, 72, 74, 76, 77, 78, 79]
rms_acc = [61, 65, 69, 74, 77, 79, 80, 82, 84, 85]

fig, ax = plt.subplots()
ax.plot(epochs, adam_acc, label="Adam", marker='o')
ax.plot(epochs, sgd_acc, label="SGD", marker='o')
ax.plot(epochs, rms_acc, label="RMSprop", marker='o')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Optimizer Comparison: Accuracy")
ax.legend()
st.pyplot(fig)

# ---------- Loss w/ Different Optimizers ----------
st.subheader("⚙️ Loss Comparison: Different Optimizers")
adam_loss = [1.6, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]
sgd_loss = [1.8, 1.4, 1.2, 1.1, 1.0, 0.9, 0.85, 0.8, 0.78, 0.75]
rms_loss = [1.5, 1.3, 1.1, 0.95, 0.85, 0.75, 0.68, 0.6, 0.55, 0.5]

fig, ax = plt.subplots()
ax.plot(epochs, adam_loss, label="Adam", marker='o')
ax.plot(epochs, sgd_loss, label="SGD", marker='o')
ax.plot(epochs, rms_loss, label="RMSprop", marker='o')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Optimizer Comparison: Loss")
ax.legend()
st.pyplot(fig)

# ---------- Simulated ROC Curve ----------
st.subheader("📊 ROC Curve (Multi-Class)")

# Simulate data
n_samples = 100
n_classes = 3
y_true = np.random.randint(0, n_classes, size=n_samples)
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
y_scores = np.random.rand(n_samples, n_classes)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
            label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-class ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)
