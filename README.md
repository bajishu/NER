This Python script trains a custom Named Entity Recognition (NER) model using OCR data extracted from images. The NER model identifies entities like invoice numbers, dates, and total amounts from OCR-extracted text. The project utilizes Tesseract OCR for text extraction and spaCy for NER training.

Features
OCR with Tesseract: Extract text from images (supporting common formats like .jpg, .jpeg, .png).

Entity Annotation: Use regular expressions to identify key entities (e.g., invoice numbers, dates, totals).

Training Custom NER Model: Train a custom NER model using spaCy’s blank model and update it with annotated data from OCR.

Model Export: Save the trained model for use in other applications or further processing.

Prerequisites
To run this script, you need to install the following dependencies:

Python 3.x

Tesseract OCR (Make sure to download and install Tesseract OCR, and update the tesseract_cmd path in the script.)

spaCy (for model training)

pytesseract (Python wrapper for Tesseract)

Pillow (for handling images)

How to Use
Prepare your images: Place the images you want to process in a directory (e.g., dataset/images/). The script will process all .jpg, .jpeg, and .png files.

Run the script: The main script will:

Perform OCR on each image.

Annotate the extracted text using simple regular expressions to find invoice numbers, dates, and totals.

Train a custom NER model based on the annotated data.

To run the script, use:
python train_ner_from_images.py

By default, the script saves the trained model to model/custom_ner/.

Functions
ocr_image(image_path: str) -> str
Perform OCR on the given image and return the extracted text.

annotate_text(text: str)
Annotate the OCR text using regular expressions to find invoice numbers, dates, and totals.

prepare_spacy_data(image_folder: str)
Loop through all images in the provided folder, perform OCR, annotate the text, and prepare spaCy-compatible training data.

train_ner(train_data, output_dir="custom_ner_model", n_iter=20)
Train a custom NER model using the annotated data. The model will be saved to the specified directory.

Troubleshooting
OCR issues: If the OCR output is unclear or incorrect, you may need to adjust Tesseract's settings or use image pre-processing techniques (e.g., resizing, thresholding) to improve accuracy.

Entity recognition: If entities aren't being detected correctly, check the regular expressions in annotate_text() to ensure they match the format of your target data (invoice numbers, dates, totals).

