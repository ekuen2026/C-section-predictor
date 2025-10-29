import streamlit as st
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

st.title("C-Section Probability Predictor")
st.write("Estimate likelihood of a C-section from prior pregnancy info. Model is trained from the CSV in this repo.")

# --- Load data ---
csv_path = Path(__file__).parent / "cleaned_csection_data.csv"

if not csv_path.exists():
    st.error("❌ Could not find 'cleaned_csection_data.csv'. Make sure it's uploaded to the repo.")
else:
    st.info("✅ Loading data and training model (runs once per app start)...")
    try:
        df = pd.read_csv(csv_path)

        # Create binary target
        df["csection"] = (df["dmeth_rec"] == 2).astype(int)

        # --- Simplified features to avoid multicollinearity ---
        df["nulliparous"] = ((df["priorlive"] == 0) &
                             (df["priordead"] == 0) &
                             (df["priorterm"] == 0)).astype(int)

        X = df[["nulliparous", "gestrec3", "dplural", "me_pres"]]
        y = df["csection"]

        X_sm = sm.add_constant(X)
        model = sm.Logit(y, X_sm)
        results = model.fit(disp=False)

        results.save("csection_logit_model.sm")
        st.success("Model trained successfully!")

    except Exception as e:
        st.error(f"Error training model: {e}")

# --- Prediction Section ---
try:
    model = sm.load("csection_logit_model.sm")
    expected_cols = model.model.exog_names

    st.header("Make a Prediction")

    priorlive = st.number_input("Previous live births", 0, 10, 0)
    priordead = st.number_input("Previous stillbirths", 0, 10, 0)
    priorterm = st.number_input("Previous terminations", 0, 10, 0)
    gestrec3 = st.selectbox("Gestation (term = 2)", [1, 2, 3], index=1)
    dplural = st.selectbox("Plurality (single = 1)", [1, 2], index=0)
    me_pres = st.selectbox("Presentation (vertex/cephalic = 1)", [1, 2], index=0)

    input_data = pd.DataFrame({
        "nulliparous": [1 if (priorlive == 0 and priordead == 0 and priorterm == 0) else 0],
        "gestrec3": [gestrec3],
        "dplural": [dplural],
        "me_pres": [me_pres]
    })

    input_data = sm.add_constant(input_data, has_constant='add')
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_cols]

    prob = model.predict(input_data)[0]

    st.subheader(f"Predicted Probability of C-section: {prob:.2%}")

    if prob >= 0.5:
        st.error("High likelihood of C-section.")
    else:
        st.success("Low likelihood of C-section.")
except Exception as e:
    st.error(f"Prediction error: {e}")
