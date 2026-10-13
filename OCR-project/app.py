"""
Streamlit application for PDF document classification.

Upload a PDF (or image), view the extracted text, and see the predicted
document category (Invoice / Bill / Receipt / etc.).
"""

import io
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr import (
    clean_text,
    extract_text_from_bytes,
    ocr_engines_available,
)
from src.train_model import predict_text

st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Document Classification")
st.markdown(
    "Upload a **PDF** document (or an image) to extract its text and predict "
    "whether it is an **Invoice**, **Bill**, **Receipt**, or another category."
)

with st.sidebar:
    st.header("System Info")
    engines = ocr_engines_available()
    st.write("**OCR Engines:**")
    st.write(f"- RapidOCR (ONNX): {'✅' if engines['rapidocr'] else '❌'}")
    st.write(f"- Tesseract: {'✅' if engines['tesseract'] else '❌'}")
    st.write(f"- PyMuPDF (PDF text): {'✅' if engines['pymupdf'] else '❌'}")

    st.divider()
    st.write("**Supported formats:** PDF, PNG, JPG, JPEG, BMP, TIFF")
    st.write("**Categories:** Invoice, Receipt (and Bill if trained)")

    st.divider()
    st.caption(
        "Tip: scanned PDFs and images are processed with OCR. "
        "Text-based PDFs extract embedded text directly."
    )

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
    help="Upload a PDF or image file to classify.",
)

CONFIDENCE_THRESHOLD = 0.40

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    file_size_kb = len(file_bytes) / 1024

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📎 File Details")
        st.write(f"**Filename:** {filename}")
        st.write(f"**Size:** {file_size_kb:.1f} KB")

        ext = Path(filename).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(file_bytes))
                st.image(img, caption="Uploaded image", width="stretch")
            except Exception as e:
                st.warning(f"Could not preview image: {e}")

    with col2:
        st.subheader("🔤 Extracted Text")
        with st.spinner("Extracting text (OCR may take a few seconds)..."):
            start = time.time()
            try:
                raw_text = extract_text_from_bytes(file_bytes, filename)
                text = clean_text(raw_text)
            except Exception as e:
                text = ""
                st.error(f"Text extraction failed: {e}")
            elapsed = time.time() - start

        if not text:
            st.warning(
                "⚠️ No text could be extracted from this document. "
                "This may happen if the document is blank, heavily corrupted, "
                "or in an unsupported format. Please try another file."
            )
        else:
            st.caption(f"Extracted {len(text)} characters in {elapsed:.1f}s")
            with st.expander("View extracted text", expanded=True):
                st.text_area(
                    "Extracted text",
                    value=text,
                    height=300,
                    label_visibility="collapsed",
                )

    if text:
        st.divider()
        st.subheader("🎯 Classification Result")

        with st.spinner("Classifying document..."):
            try:
                label, confidence = predict_text(text)
            except FileNotFoundError:
                st.error(
                    "❌ No trained model found. Please run the training pipeline "
                    "(`python src/train_model.py`) before using the classifier."
                )
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        confidence_pct = confidence * 100

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ The prediction is **inconclusive** (confidence: "
                f"{confidence_pct:.1f}%). The model is not confident enough "
                f"to classify this document reliably."
            )
            st.info(
                f"Suggested category: **{label.title()}** "
                f"(low confidence). Try a clearer document."
            )
        else:
            emoji = {
                "invoice": "🧾",
                "receipt": "💰",
                "bill": "💡",
            }.get(label.lower(), "📄")

            st.success(
                f"{emoji} Predicted category: **{label.title()}**\n\n"
                f"Confidence: **{confidence_pct:.1f}%**"
            )

            st.progress(min(confidence, 1.0))

        with st.expander("💡 What does this mean?"):
            if label.lower() == "invoice":
                st.write(
                    "An **invoice** is a detailed bill of goods or services "
                    "provided by a seller to a buyer, typically including "
                    "itemized charges, payment terms, and a due date."
                )
            elif label.lower() == "receipt":
                st.write(
                    "A **receipt** is an acknowledgment that payment has "
                    "been made, usually issued at the point of sale. It "
                    "typically lists items, total, and payment method."
                )
            elif label.lower() == "bill":
                st.write(
                    "A **bill** is a request for payment, often for "
                    "recurring services (utilities, medical, etc.)."
                )
            else:
                st.write(f"Document classified as **{label.title()}**.")

    st.divider()
    if st.button("🔄 Upload another document"):
        st.rerun()
else:
    st.info("👆 Upload a PDF or image to get started.")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Upload** a PDF or image document.\n"
        "2. **Extract** text using OCR (for scanned/images) or embedded "
        "text (for text-based PDFs).\n"
        "3. **Classify** the document using a trained TF-IDF + LinearSVC "
        "machine learning model.\n"
        "4. **View** the extracted text and predicted category with confidence."
    )
