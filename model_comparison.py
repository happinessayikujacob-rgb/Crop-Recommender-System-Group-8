import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Crop Model Comparison", layout="wide")

# Load models and encoder
@st.cache_resource
def load_models():
    dt_model = joblib.load('decision_tree_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return dt_model, rf_model, label_encoder

dt_model, rf_model, label_encoder = load_models()

# Feature names and ranges based on the dataset
features = ['N', 'P', 'K', 'ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B']
feature_ranges = {
    'N': (0, 300),
    'P': (0, 150),
    'K': (0, 300),
    'ph': (2.0, 10.0),
    'EC': (0.0, 5.0),
    'S': (0.0, 1.0),
    'Cu': (0.0, 100.0),
    'Fe': (0.0, 300.0),
    'Mn': (0.0, 300.0),
    'Zn': (0.0, 150.0),
    'B': (0.0, 100.0)
}

# Title
st.title("🌾 Crop Recommendation Model Comparison")
st.markdown("Compare predictions from Decision Tree and Random Forest models side by side")

# Add preset buttons
st.subheader("Quick Presets")
col1, col2, col3 = st.columns(3)

preset_values = {
    'Pomegranate': [143, 69, 217, 5.9, 0.58, 0.23, 10.2, 116.35, 59.96, 54.85, 21.29],
    'Average': [20, 20, 20, 6.5, 1.0, 0.5, 10, 100, 50, 30, 20],
    'High Nutrients': [150, 100, 150, 7.0, 2.0, 0.8, 50, 200, 150, 100, 50]
}

# Initialize session state for input values
if 'input_values' not in st.session_state:
    # Use floats for features that expect decimal steps
    initial = []
    for feature in features:
        if feature in ['ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B']:
            initial.append(20.0)
        else:
            initial.append(20)
    st.session_state.input_values = initial

for idx, (preset_name, preset_val) in enumerate(preset_values.items()):
    if idx % 3 == 0:
        col = st.columns(3)[0]
    elif idx % 3 == 1:
        col = st.columns(3)[1]
    else:
        col = st.columns(3)[2]
    
    if col.button(f"Load {preset_name}", use_container_width=True):
        st.session_state.input_values = preset_val.copy()
        st.rerun()

# Input section
st.subheader("Input Features")
st.markdown("Adjust values using sliders or edit the text fields directly")

# Create columns for input
input_cols = st.columns(len(features))
input_values = []

# Features that should use float precision for sliders
float_step_features = ['ph', 'EC', 'S']

# Initialize per-feature slider session keys from the input_values list
for idx, feature in enumerate(features):
    key = f"slider_{feature}"
    if key not in st.session_state:
        if feature in float_step_features:
            st.session_state[key] = float(st.session_state.input_values[idx])
        else:
            # store as int for integer features
            try:
                st.session_state[key] = int(float(st.session_state.input_values[idx]))
            except Exception:
                st.session_state[key] = int(st.session_state.input_values[idx])

for idx, feature in enumerate(features):
    with input_cols[idx]:
        min_val, max_val = feature_ranges[feature]
        # Create two-column layout for slider and text input
        col1, col2 = st.columns([3, 1])

        is_float = feature in float_step_features
        if is_float:
            min_v = float(min_val)
            max_v = float(max_val)
            step_val = 0.1
        else:
            min_v = int(min_val)
            max_v = int(max_val)
            step_val = 1

        slider_key = f"slider_{feature}"

        with col1:
            # Use an explicit key so each slider is independent
            _ = st.slider(
                f"{feature}",
                min_value=min_v,
                max_value=max_v,
                value=st.session_state[slider_key],
                step=step_val,
                key=slider_key,
                label_visibility="visible"
            )

        with col2:
            text_key = f"text_{feature}"
            text_value = st.text_input(
                "value",
                value=str(st.session_state[slider_key]),
                label_visibility="collapsed",
                key=text_key
            )
            try:
                if is_float:
                    parsed = float(text_value)
                else:
                    parsed = int(float(text_value))
                parsed = max(min_v, min(parsed, max_v))
                # Only update if different to avoid unnecessary reruns
                if parsed != st.session_state[slider_key]:
                    st.session_state[slider_key] = parsed
            except Exception:
                pass

        value = st.session_state[slider_key]
        input_values.append(value)
        st.session_state.input_values[idx] = value

# Create input dataframe
input_df = pd.DataFrame([input_values], columns=features)

# Make predictions
dt_prediction = dt_model.predict(input_df)[0]
rf_prediction = rf_model.predict(input_df)[0]

# Decode predictions safely (models may predict encoded ints or string labels)
def _decode_prediction(pred):
    # If prediction looks like a string, return it directly
    try:
        if isinstance(pred, str):
            return pred
        # numpy string types
        if hasattr(pred, 'dtype') and np.issubdtype(getattr(pred, 'dtype'), np.str_):
            return str(pred)
    except Exception:
        pass

    # Otherwise try to inverse transform using the saved label encoder
    try:
        return label_encoder.inverse_transform([pred])[0]
    except Exception:
        return pred

dt_crop = _decode_prediction(dt_prediction)
rf_crop = _decode_prediction(rf_prediction)

# Display results side by side
st.markdown("---")
st.subheader("Model Predictions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌳 Decision Tree Model")
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px;'>"
                f"<h2 style='margin: 0; color: #1976d2;'>{dt_crop}</h2>"
                f"<p style='margin: 0; color: #555;'>Predicted Crop</p>"
                f"</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🌲 Random Forest Model")
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #f3e5f5; border-radius: 10px;'>"
                f"<h2 style='margin: 0; color: #7b1fa2;'>{rf_crop}</h2>"
                f"<p style='margin: 0; color: #555;'>Predicted Crop</p>"
                f"</div>", unsafe_allow_html=True)

# Display agreement status
st.markdown("---")
if dt_crop == rf_crop:
    st.success(f"✅ **Models Agree!** Both predict: **{dt_crop}**")
else:
    st.warning(f"⚠️ **Models Disagree** - Decision Tree: **{dt_crop}** | Random Forest: **{rf_crop}**")

# Display current input values
with st.expander("View Current Input Values"):
    st.dataframe(input_df, use_container_width=True)

# Reset button
if st.button("Reset to Defaults", use_container_width=True):
    st.session_state.input_values = [20] * len(features)
    st.rerun()
