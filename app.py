import streamlit as st
import pandas as pd
import statsmodels.api as sm

# Load your saved model
model = sm.load("/content/csection_logit_model.sm")
expected_cols = model.model.exog_names  # Columns used in training

st.title("C-Section Probability Predictor")
st.write("Estimate your likelihood of a C-section.")

# --- User inputs ---
priorlive = st.number_input("Number of previous live births", min_value=0, max_value=10, value=0)
priordead = st.number_input("Number of previous stillbirths", min_value=0, max_value=10, value=0)
priorterm = st.number_input("Number of previous terminations", min_value=0, max_value=10, value=0)
gestrec3 = st.selectbox("Gestation (term = 2)", [1, 2, 3], index=1)
dplural = st.selectbox("Plurality (single = 1)", [1, 2], index=0)
me_pres = st.selectbox("Presentation (vertex/cephalic = 1)", [1, 2], index=0)

input_data = pd.DataFrame({
    "priorlive": [priorlive],
    "priordead": [priordead],
    "priorterm": [priorterm],
    "gestrec3": [gestrec3],
    "dplural": [dplural],
    "me_pres": [me_pres]
})
input_data["nulliparous"] = ((input_data["priorlive"] == 0) &
                             (input_data["priordead"] == 0) &
                             (input_data["priorterm"] == 0)).astype(int)

# Add constant and align with model
input_data = sm.add_constant(input_data, has_constant='add')
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_cols]

# Predict
prob = model.predict(input_data)[0]

st.subheader(f"Predicted Probability of C-section: {prob:.2%}")

if prob >= 0.5:
    st.error("High likelihood of C-section.")
else:
    st.success("Low likelihood of C-section.")
