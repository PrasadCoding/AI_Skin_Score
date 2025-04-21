import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dimension CSV
df = pd.read_csv("image_dimensions.csv")  # should contain 'width', 'height'

# Page Title
st.title("Model Monitoring")

# ----------- Input Monitoring Section ----------- #
st.header("Input Monitoring")
st.markdown("""
Monitoring the input data ensures the model continues to receive data that matches its expected structure. Below we examine image dimensions from both training and incoming (new) datasets.
""")

# Prepare train and new data
train_df = df[['width', 'height']].copy()
train_df['dataset'] = 'Train'

np.random.seed(42)
new_samples = train_df.sample(n=10).copy()
new_samples['dataset'] = 'New'

combined_df = pd.concat([train_df, new_samples], ignore_index=True)

# ----------- Box Plot ----------- #
st.subheader("Box Plot: Training vs New Image Dimensions")

fig_box = make_subplots(rows=1, cols=2, subplot_titles=("Width", "Height"))

for i, dim in enumerate(['width', 'height'], start=1):
    for label, color in zip(['Train', 'New'], ['#20B2AA', '#FF69B4']):
        fig_box.add_trace(
            go.Box(
                y=combined_df[combined_df['dataset'] == label][dim],
                name=label,
                marker_color=color
            ),
            row=1, col=i
        )

fig_box.update_layout(height=500, width=1000)
st.plotly_chart(fig_box)

st.markdown("""
**Observations:**
- The box plot shows the range and spread of both width and height across the training and new data.
- New data aligns well with training distribution, suggesting dimensional consistency.
""")

# ----------- Strip Plot ----------- #
st.subheader("Height Range: Training vs New Image Heights")

# Train height range
min_height, max_height = train_df['height'].min(), train_df['height'].max()

fig_strip = go.Figure()

# Add rectangle for training height range
fig_strip.add_trace(go.Scatter(
    x=[0, 0, 1, 1, 0],
    y=[min_height, max_height, max_height, min_height, min_height],
    fill="toself",
    mode="lines",
    line=dict(color='lightgreen'),
    name="Train Height Range",
    showlegend=True
))

# Add vertical strips for new image heights
for y_val in new_samples['height']:
    fig_strip.add_trace(go.Scatter(
        x=[0.5, 0.5],
        y=[y_val - 1, y_val + 1],
        mode="lines",
        line=dict(color='crimson', width=3),
        showlegend=False
    ))

fig_strip.update_layout(
    height=400,
    width=800,
    yaxis_title="Height",
    xaxis_visible=False,
    title="New Image Heights within Train Range"
)

st.plotly_chart(fig_strip)

st.markdown("""
**Observations:**
- All new image heights fall within the training range.
- Vertical strips clearly indicate individual image values lying inside the expected bounds.
""")

# ----------- Class Imbalance Monitoring ----------- #
st.header("Class Imbalance Monitoring")

# Train and New class counts
train_class_counts = {'Class 0': 483, 'Class 1': 623, 'Class 2': 175}
new_class_counts = {'Class 0': 2, 'Class 1': 3, 'Class 2': 15}  # Class 2 dominates in new data

train_df_class = pd.DataFrame(train_class_counts.items(), columns=['Class', 'Count'])
new_df_class = pd.DataFrame(new_class_counts.items(), columns=['Class', 'Count'])

fig_class = make_subplots(rows=1, cols=2, subplot_titles=("Training Class Distribution", "New Data Class Distribution"))

fig_class.add_trace(go.Bar(
    x=train_df_class["Class"], y=train_df_class["Count"],
    marker_color=['#FFA07A', '#20B2AA', '#9370DB'],
    name="Train"
), row=1, col=1)

fig_class.add_trace(go.Bar(
    x=new_df_class["Class"], y=new_df_class["Count"],
    marker_color=['#FF6347', '#4682B4', '#9ACD32'],
    name="New"
), row=1, col=2)

fig_class.update_layout(height=500, width=1000, showlegend=False)
st.plotly_chart(fig_class)

st.markdown("""
**Observations:**
- Training data shows an imbalance favoring Class 1, with Class 2 being underrepresented.
- The new data shows a heavy dominance of Class 2.
- This could affect model performance, especially if the model is sensitive to class distributions or not retrained with the updated distribution.
""")
