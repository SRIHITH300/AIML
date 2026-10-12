import streamlit as st
import joblib

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Intent Classifier",
    page_icon="💬",
    layout="centered"
)

# ----------------------------
# Load the saved model
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("intent_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

try:
    model, tfidf = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Save intent_classifier.pkl and "
        "tfidf_vectorizer.pkl in the same folder as app.py."
    )
    st.stop()

# ----------------------------
# UI
# ----------------------------
st.title("💬 Intent Classification")
st.write(
    "Enter a sentence below and the model will predict the user's intent."
)

st.divider()

utterance = st.text_area(
    "Your message",
    placeholder="Example: I want to check my order status",
    height=120
)

if st.button("Predict Intent", type="primary", use_container_width=True):
    if not utterance.strip():
        st.warning("Please enter a message first.")
    else:
        features = tfidf.transform([utterance.strip()])
        prediction = model.predict(features)[0]

        st.subheader("Prediction")
        st.success(f"Predicted intent: **{prediction}**")

        # Show confidence when the classifier supports predict_proba.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities.max())
            st.metric("Model confidence", f"{confidence:.1%}")

            # Optional: show the most likely alternative intents.
            classes = model.classes_
            top_indices = probabilities.argsort()[::-1][:3]

            st.write("Top matching intents")
            for idx in top_indices:
                st.write(f"- **{classes[idx]}** — {probabilities[idx]:.1%}")
        else:
            st.info(
                "This model does not provide probability estimates, "
                "so a confidence score is unavailable."
            )

st.divider()
st.caption("Prototype • TF-IDF + Scikit-learn intent classifier")
