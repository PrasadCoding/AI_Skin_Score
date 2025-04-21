import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dimension CSV
df = pd.read_csv("image_dimensions.csv")  # Should contain 'width', 'height', 'class' if possible

# --- Input Monitoring ---
st.title("🧠 Model Monitoring")
st.header("📥 Input Monitoring")

st.markdown("""
Monitoring the input data helps ensure the consistency and integrity of the data fed into the model.
This includes checking if new images follow the same patterns in **dimensions** and **class balance** as the training set.
""")

# Split train and new samples
train_df = df[['width', 'height']].copy()
train_df['dataset'] = 'Train'

# Create synthetic new data
np.random.seed(42)
new_samples = train_df.sample(n=10).copy()
new_samples['dataset'] = 'New'

combined_df = pd.concat([train_df, new_samples], ignore_index=True)

# --- Subplot: Box Plot and Overlay Plot ---
st.subheader("📐 Image Dimension Consistency")

fig_box_strip = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Box Plot: Image Dimensions", "Dimension Range with New Data Overlay")
)

# --- Box Plot ---
for dim in ['width', 'height']:
    fig_box_strip.add_trace(
        go.Box(y=combined_df[combined_df['dataset'] == 'Train'][dim],
               name=f"Train {dim.title()}",
               boxpoints='outliers',
               marker_color='#FF69B4' if dim == 'width' else '#00CED1'),
        row=1, col=1
    )

# --- Rectangle Plot with Strip Overlay ---
min_height, max_height = train_df['height'].min(), train_df['height'].max()
min_width, max_width = train_df['width'].min(), train_df['width'].max()

# Rectangle as the range of train height and width
fig_box_strip.add_trace(go.Scatter(
    x=[0, 0, 1, 1, 0],
    y=[min_height, max_height, max_height, min_height, min_height],
    fill="toself",
    mode="lines",
    name="Train Height Range",
    line=dict(color='lightgreen'),
    showlegend=False
), row=1, col=2)

# Overlay new data points (strips)
fig_box_strip.add_trace(go.Scatter(
    x=np.random.uniform(0.1, 0.9, size=len(new_samples)),
    y=new_samples['height'],
    mode='markers',
    marker=dict(color='crimson', size=10),
    name="New Image Heights",
    showlegend=False
), row=1, col=2)

fig_box_strip.update_layout(height=500, width=1000)

st.plotly_chart(fig_box_strip)

st.markdown("""
#### 🔍 Observations:
- The **box plot** shows the overall distribution of width and height in training data, making it easy to spot outliers or variations.
- The **overlay plot** visually confirms that the new images fall within the expected height range from training data.
""")

# --- Class Imbalance Monitoring ---
st.header("⚖️ Class Imbalance Monitoring")

# Simulated train and new class data
train_class_counts = {'Class 0': 483, 'Class 1': 623, 'Class 2': 175}
new_class_counts = {'Class 0': 2, 'Class 1': 12, 'Class 2': 6}

train_class_df = pd.DataFrame(train_class_counts.items(), columns=["Class", "Count"])
new_class_df = pd.DataFrame(new_class_counts.items(), columns=["Class", "Count"])

# Plot train and new class bar plots side by side
fig_class = make_subplots(rows=1, cols=2, subplot_titles=("Training Class Distribution", "New Data Class Distribution"))

fig_class.add_trace(go.Bar(
    x=train_class_df["Class"], y=train_class_df["Count"],
    marker_color=['#FFA07A', '#20B2AA', '#9370DB'],
    name="Train"), row=1, col=1)

fig_class.add_trace(go.Bar(
    x=new_class_df["Class"], y=new_class_df["Count"],
    marker_color=['#FF6347', '#4682B4', '#9ACD32'],
    name="New"), row=1, col=2)

fig_class.update_layout(height=500, width=1000, showlegend=False)

st.plotly_chart(fig_class)

st.markdown("""
#### 🧠 Insights:
- In training data, **Class 1** dominates, while **Class 2** is underrepresented.
- The **new image upload** pattern also leans toward **Class 1**, which might amplify the existing imbalance.
- Monitoring class distribution in new inputs helps prevent skewed learning and biased predictions.
""")
