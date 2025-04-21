import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load the image dimensions from the CSV (after extracting from the PDF)
df = pd.read_csv("image_dimensions.csv")

# Title of the Streamlit app
st.title("Model Monitoring")

# Introduction to the monitoring page
st.write("""
## Model Monitoring Overview
In this section, we monitor and analyze the behavior of the model by checking various aspects of the dataset, 
including class distribution and image dimensions. These insights are crucial for ensuring that the model is 
performing well and that no data issues are arising during the training and deployment processes.
""")

# --- Class Imbalance Plot ---
st.write("### Class Distribution in New Images")

# Simulate the class distribution for the 20 new images (using random data for demo purposes)
np.random.seed(42)  # For reproducibility
new_images_classes = np.random.choice([0, 1, 2], size=20, p=[0.3, 0.5, 0.2])  # Class 1 is more frequent

# Count the occurrences of each class in the new images
class_counts_new = {
    "Class 0": np.sum(new_images_classes == 0),
    "Class 1": np.sum(new_images_classes == 1),
    "Class 2": np.sum(new_images_classes == 2)
}

# Convert class counts into a DataFrame for visualization
class_counts_df = pd.DataFrame(list(class_counts_new.items()), columns=["Class", "Count"])

# Create a bar plot to show the class distribution in the new uploaded images
fig_class_imbalance_new = go.Figure()

fig_class_imbalance_new.add_trace(go.Bar(
    x=class_counts_df['Class'],  # Class labels
    y=class_counts_df['Count'],  # Class counts
    marker=dict(color=['#FF6347', '#4682B4', '#32CD32']),  # Unique colors for each class
    name="New Image Class Distribution"
))

fig_class_imbalance_new.update_layout(
    title="Class Distribution in 20 New Images",
    xaxis_title="Class",
    yaxis_title="Number of Images",
    showlegend=False
)

st.plotly_chart(fig_class_imbalance_new)

st.write("""
#### Observations:
In the plot above, we can observe that **Class 1** is being uploaded more frequently than the other classes 
with the current distribution. Monitoring the class distribution helps in identifying any class imbalances, 
which could impact model performance and fairness.
""")

# --- Image Dimension Comparison ---
st.write("### Image Dimensions Comparison")

# Extract the 'width' and 'height' columns from the DataFrame (with lowercase column names)
train_df = df[['width', 'height']]

# Randomly select new data (for this example, 3 new images)
np.random.seed(42)
new_data_indices = np.random.choice(len(train_df), size=3, replace=False)
new_data = train_df.iloc[new_data_indices]

# Combine train and new data into one DataFrame for comparison
train_df['Dataset'] = 'Train'
new_data['Dataset'] = 'New'
combined_df = pd.concat([train_df, new_data], ignore_index=True)

# Create a heatmap-style plot for width and height (with different colors for Train and New data)
fig_image_dimensions = go.Figure()

fig_image_dimensions.add_trace(go.Heatmap(
    z=combined_df[['width', 'height']].T,  # Transpose to have columns as width and height
    colorscale='YlGnBu',  # Attractive colormap (Yellow-Green-Blue)
    zmin=0, zmax=600,  # Adjust the scale to fit your image dimension range
    colorbar=dict(title="Dimension Value"),
    x=combined_df['Dataset'],
    y=['Width', 'Height']
))

fig_image_dimensions.update_layout(
    title="Image Dimensions Comparison (Train vs New Data)",
    xaxis_title="Dataset",
    yaxis_title="Image Dimension",
    showlegend=False
)

st.plotly_chart(fig_image_dimensions)

st.write("""
#### Observations:
The heatmap above shows the comparison of **image dimensions** between the train data and the new images.
The different colors represent the values of **width** and **height** for each dataset. Monitoring the image dimensions
ensures that the model is not receiving images with unexpected or incorrect sizes.
""")

# --- Model Performance (Placeholder) ---
st.write("### Model Performance Monitoring")

# Placeholder plot (You can replace this with real-time model metrics like accuracy or loss)
fig_model_performance = go.Figure()

# Random data for demonstration purposes (e.g., model accuracy over time)
time_steps = np.arange(1, 11)
accuracy = np.random.uniform(0.85, 0.95, size=10)

fig_model_performance.add_trace(go.Scatter(
    x=time_steps,
    y=accuracy,
    mode='lines+markers',
    name='Model Accuracy',
    line=dict(color='royalblue', width=3),
    marker=dict(size=8, color='red')
))

fig_model_performance.update_layout(
    title="Model Accuracy Over Time",
    xaxis_title="Time Step",
    yaxis_title="Accuracy",
    showlegend=False
)

st.plotly_chart(fig_model_performance)

st.write("""
#### Observations:
This plot tracks the **accuracy** of the model over time. Regularly monitoring accuracy (or other relevant metrics)
helps ensure that the model is improving as expected, and it also helps identify any performance degradation 
or overfitting that might occur during training or deployment.
""")

# Conclusion and Final Thoughts
st.write("""
## Conclusion

Effective model monitoring is essential to ensure that the model performs consistently well in real-world applications. 
By keeping an eye on the **class distribution**, **image dimensions**, and **model performance**, we can quickly 
identify potential issues such as data imbalance, input inconsistencies, or performance degradation.
""")
