# Document Classification

Starter folder structure for a project that classifies receipts, invoices, and bills.

## Dataset folders
- `dataset/raw/receipts/`: receipt images or PDFs
- `dataset/raw/invoices/`: invoice images or PDFs
- `dataset/raw/bills/`: bill images or PDFs
- `dataset/processed/documents.csv`: generated OCR dataset

## Source files
- `src/ocr.py`: OCR/text extraction logic
- `src/prepare_dataset.py`: CSV preparation
- `src/train_model.py`: model training and evaluation
- `app.py`: Streamlit application entry point

The Python files are starter placeholders, not a complete implementation yet. For scanned PDFs/images, install the Tesseract OCR engine separately and configure it for your system.
