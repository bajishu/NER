# train_ner_from_images.py

import os
import re
import random
import pytesseract
from PIL import Image
import spacy
from spacy.training import Example

# 1) OCR function
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed

def ocr_image(image_path: str) -> str:
    """Perform OCR on an image and return the extracted text."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# 2) Simple regex-based annotation
def annotate_text(text: str):
    """Find invoice number, date, and total in text via regex."""
    ents = []
    # Invoice No: digits
    m = re.search(r"Invoice\s*No[:#]?\s*(\d+)", text, re.IGNORECASE)
    if m:
        start, end = m.span(1)
        ents.append((start, end, "INVOICE_NO"))

    # Date: YYYY-MM-DD or DD/MM/YYYY
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        start, end = m.span(1)
        ents.append((start, end, "DATE"))
    else:
        m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        if m:
            start, end = m.span(1)
            ents.append((start, end, "DATE"))

    # Total: $ or USD or other currency symbol + amount
    m = re.search(r"(?:Total|Amount)[:\s]*((?:USD\s*)?\$\d+(?:\.\d{2})?)", text, re.IGNORECASE)
    if m:
        start, end = m.span(1)
        ents.append((start, end, "TOTAL"))

    return ents

# 3) Build spaCy training data from images
def prepare_spacy_data(image_folder: str):
    """Loop through images, OCR, annotate, and return spaCy-format train_data."""
    train_data = []
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_folder, fname)
        text = ocr_image(path)
        entities = annotate_text(text)
        if entities:
            train_data.append((text, {"entities": entities}))
    return train_data

# 4) Train custom NER model
def train_ner(train_data, output_dir="custom_ner_model", n_iter=20):
    """Train a blank spaCy model on the prepared train_data."""
    # Create blank English model
    nlp = spacy.blank("en")
    # Add NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    # Add labels
    labels = {ent[2] for _, ann in train_data for ent in ann["entities"]}
    for label in labels:
        ner.add_label(label)

    # Begin training
    optimizer = nlp.initialize()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = spacy.util.minibatch(train_data, size=4)
        for batch in batches:
            examples = []
            for text, ann in batch:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, ann))
            nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
        print(f"Iteration {itn+1}/{n_iter} — Losses: {losses}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"\n Model saved to {output_dir}")

if __name__ == "__main__":
    IMG_FOLDER = "dataset/images"  # adjust to your folder
    print("Preparing training data from images...")
    data = prepare_spacy_data(IMG_FOLDER)
    print(f"Generated {len(data)} training examples.")
    if not data:
        print("No entities found—check your regex or OCR output.")
        exit(1)
    print("Training custom NER model...")
    train_ner(data, output_dir="model/custom_ner", n_iter=30)