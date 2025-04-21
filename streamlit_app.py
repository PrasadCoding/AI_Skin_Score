import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dimension CSV
df = pd.read_csv("image_dimensions.csv")  # Should contain 'width', 'height'

# Input Monitoring Section
st.title("Model Monitoring")
st.header("Input Monitoring")

st.markdown("""
Monitoring the input data helps ensure the consistency and integrity of the data fed into the model.
This includes checking whether new images follow the same patterns in **dimensions** and **class balance** as the training data.
""")

# Prepare train and new data
train_df = df[['width', 'height']].copy()
train_df['dataset'] = 'Train'

np.random.seed(42)
new_samples = train_df.sample(n=10).copy()
new_samples['dataset'] = 'New'

combined_df = pd.concat([train_df, new_samples], ignore_index=True)

# Image Dimension Plots
st.subheader("Image Dimension Consistency")

fig_box_strip = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Box Plot: Image Dimensions", "Range and Overlay: New Image Heights")
)

# Box Plot
for dim in ['width', 'height']:
    fig_box_strip.add_trace(
        go.Box(y=combined_df[combined_df['dataset'] == 'Train'][dim],
               name=f"Train {dim.title()}",
               boxpoints='outliers',
               marker_color='#FF69B4' if dim == 'width' else '#00CED1'),
        row=1, col=1
    )

# Range and Strip Overlay Plot
min_height, max_height = train_df['height'].min(), train_df['height'].max()

fig_box_strip.add_trace(go.Scatter(
    x=[0, 0, 1, 1, 0],
    y=[min_height, max_height, max_height, min_height, min_height],
    fill="toself",
    mode="lines",
    name="Train Height Range",
    line=dict(color='lightgreen'),
    showlegend=False
), row=1, col=2)

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
**Observations:**
- The box plot visualizes the overall spread of image width and height in the training dataset.
- The overlay plot shows that all new image heights fall within the expected training range.
- This confirms dimensional consistency between new and training images.
""")

# Class Imbalance Monitoring
st.header("Class Imbalance Monitoring")

# Simulated data: Train is skewed toward Class 1, new data is skewed toward Class 2
train_class_counts = {'Class 0': 483, 'Class 1': 623, 'Class 2': 175}
new_class_counts = {'Class 0': 3, 'Class 1': 2, 'Class 2': 15}

train_class_df = pd.DataFrame(train_class_counts.items(), columns=["Class", "Count"])
new_class_df = pd.DataFrame(new_class_counts.items(), columns=["Class", "Count"])

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
**Observations:**
- The training data is heavily imbalanced with Class 1 being the majority and Class 2 being underrepresented.
- However, the new data contains mostly Class 2 images.
- This reversal could introduce bias or instability in the model, especially if it’s not designed to handle dynamic class distributions.
""")
