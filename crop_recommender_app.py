import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Crop Recommender", layout="centered")

st.title("🌱 Crop Recommendation System")
st.markdown("### AI-Powered Crop Selection Based on Soil Parameters")

feature_columns = ["N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"]

feature_info = {
    "N": ("Nitrogen (N)", 0, 200, "kg/ha"),
    "P": ("Phosphorus (P)", 0, 100, "kg/ha"),
    "K": ("Potassium (K)", 0, 400, "kg/ha"),
    "ph": ("pH Level", 0.0, 14.0, ""),
    "EC": ("Electrical Conductivity", 0.0, 3.0, "dS/m"),
    "S": ("Sulfur (S)", 0, 120, "mg/kg"),
    "Cu": ("Copper (Cu)", 0, 40, "mg/kg"),
    "Fe": ("Iron (Fe)", 0, 300, "mg/kg"),
    "Mn": ("Manganese (Mn)", 0, 1600, "mg/kg"),
    "Zn": ("Zinc (Zn)", 0, 80, "mg/kg"),
    "B": ("Boron (B)", 0, 80, "mg/kg"),
}

col1, col2 = st.columns(2)

inputs = {}

with col1:
    st.subheader("Soil Parameters")
    for feature in feature_columns[:6]:
        name, min_val, max_val, unit = feature_info[feature]
        val = st.number_input(
            name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=0.1,
        )
        inputs[feature] = val

with col2:
    st.subheader("Soil Parameters (cont.)")
    for feature in feature_columns[6:]:
        name, min_val, max_val, unit = feature_info[feature]
        val = st.number_input(
            name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=0.1,
        )
        inputs[feature] = val

st.markdown("---")

if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])[feature_columns]

    try:
        dt_model = joblib.load("decision_tree_model.pkl")
        rf_model = joblib.load("random_forest_model.pkl")
        le = joblib.load("label_encoder.pkl")

        dt_prediction = dt_model.predict(input_df)
        rf_prediction = rf_model.predict(input_df)

        if hasattr(le, "classes_"):
            dt_result = (
                le.classes_[dt_prediction[0]]
                if isinstance(dt_prediction[0], (int, np.integer))
                else dt_prediction[0]
            )
            rf_result = (
                le.classes_[rf_prediction[0]]
                if isinstance(rf_prediction[0], (int, np.integer))
                else rf_prediction[0]
            )
        else:
            dt_result = dt_prediction[0]
            rf_result = rf_prediction[0]

        st.subheader("Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### Decision Tree")
            st.success(f"**{dt_result}**")

        with res_col2:
            st.markdown("### Random Forest")
            st.success(f"**{rf_result}**")

        if dt_result == rf_result:
            st.info(f"Both models agree: **{dt_result}** is the recommended crop!")
        else:
            st.warning("Models predict different crops. Consider additional analysis.")

        with st.expander("See input summary"):
            st.dataframe(input_df.T, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
    except Exception as e:
        st.error(f"Error loading models: {e}")

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Models: Decision Tree & Random Forest | Features: N, P, K, pH, EC, S, Cu, Fe, Mn, Zn, B</small>
</div>
""",
    unsafe_allow_html=True,
)
