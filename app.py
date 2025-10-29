# app.py
import streamlit as st
import pandas as pd
import statsmodels.api as sm

# ---------------------------
# Helpers: caching for speed
# ---------------------------
@st.cache_data(ttl=3600)
def load_data(csv_path="cleaned_csection_data.csv"):
    """Load CSV from repo. Expect columns like dmeth_rec, priorlive, priordead, priorterm, gestrec3, dplural, me_pres."""
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_logit_model(df):
    """Prepare X/y and train a statsmodels Logit. Cached so we only train once per session."""
    # Create target if missing
    if "csection" not in df.columns:
        if "dmeth_rec" in df.columns:
            df["csection"] = (df["dmeth_rec"] == 2).astype(int)
        else:
            raise ValueError("CSV must contain 'csection' or 'dmeth_rec' to derive it.")

    # If your cleaned file contains extra rows, apply same filters you used in Colab (optional)
    # Uncomment if you want to enforce the same filtering for training:
    # df = df[(df['gestrec3'] == 2) & (df['dplural'] == 1) & (df['me_pres'] == 1)].reset_index(drop=True)

    # Select features used in your original model
    features = ["priorlive", "priordead", "priorterm", "gestrec3", "dplural", "me_pres"]
    # Verify features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")

    X = df[features].copy()
    # create nulliparous as you did originally
    X["nulliparous"] = ((X["priorlive"] == 0) & (X["priordead"] == 0) & (X["priorterm"] == 0)).astype(int)
    y = df["csection"].astype(int)

    # Add constant for intercept
    X_sm = sm.add_constant(X, has_constant="add")
    logit = sm.Logit(y, X_sm)
    # fit the model (suppress output)
    try:
        results = logit.fit(disp=False, maxiter=100)
    except Exception:
        # fallback: try default fit if options differ on versions
        results = logit.fit()
    return results

# ---------------------------
# Main app
# ---------------------------
st.set_page_config(page_title="C-Section Probability Predictor", layout="centered")
st.title("C-Section Probability Predictor")
st.write("Estimate likelihood of a C-section from prior pregnancy info. Model is trained from the CSV in this repo.")

# Load data & train (cached)
with st.spinner("Loading data and training model (runs once per app start)..."):
    try:
        df = load_data("cleaned_csection_data.csv")
    except FileNotFoundError:
        st.error("cleaned_csection_data.csv not found in the app folder. Upload it to your repo.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    try:
        model_results = train_logit_model(df)
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()

expected_cols = model_results.model.exog_names  # column ordering from training

# --- UI: inputs ---
st.subheader("Patient / pregnancy inputs")
col1, col2 = st.columns(2)
with col1:
    priorlive = st.number_input("Number of previous live births", min_value=0, max_value=20, value=0, step=1)
    priordead = st.number_input("Number of previous stillbirths", min_value=0, max_value=20, value=0, step=1)
    priorterm = st.number_input("Number of previous terminations", min_value=0, max_value=20, value=0, step=1)
with col2:
    gestrec3 = st.selectbox("Gestation (1, 2=term, 3)", [1, 2, 3], index=1)
    dplural = st.selectbox("Plurality (1=single, 2=multiple)", [1, 2], index=0)
    me_pres = st.selectbox("Presentation (1=vertex/cephalic, 2=other)", [1, 2], index=0)

# Build input DataFrame
input_data = pd.DataFrame({
    "priorlive": [priorlive],
    "priordead": [priordead],
    "priorterm": [priorterm],
    "gestrec3": [gestrec3],
    "dplural": [dplural],
    "me_pres": [me_pres]
})
input_data["nulliparous"] = ((input_data["priorlive"] == 0) & (input_data["priordead"] == 0) & (input_data["priorterm"] == 0)).astype(int)

# Align with model columns
input_data = sm.add_constant(input_data, has_constant="add")
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_cols]

# Predict
prob = model_results.predict(input_data)[0]

st.markdown("### Result")
st.subheader(f"Predicted Probability of C-section: {prob:.2%}")

if prob >= 0.5:
    st.error("High likelihood of C-section.")
else:
    st.success("Low likelihood of C-section.")

# Show model summary (collapsible)
with st.expander("Show model summary (coefficients)"):
    st.write(model_results.summary2().tables[1])
