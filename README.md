# Prescription Digitalization

A system for automatically extracting and digitizing information from medical prescriptions using computer vision and OCR technology.

## Overview

This project extracts medications and their frequencies from prescription images using deep learning. The system identifies regions of interest (ROIs) in prescriptions and performs optical character recognition (OCR) on these regions to extract structured data.

## Features

- **Multi-language Support**: Handles both Arabic and English prescriptions
- **Region Detection**: Uses YOLO to identify medication names and frequencies in prescriptions
- **Text Recognition**: Advanced OCR processing for accurate text extraction
- **Text Correction**: Uses BK-trees and Levenshtein distance for spelling correction
- **High Accuracy**: Achieves 61% F1-score on 1000+ prescription images

## Installation

```bash
# Clone the repository
git clone https://github.com/Ahmed-Hussein-ElShahat/Prescription-digitalization.git
cd Prescription-digitalization

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from Backend.Image_Processor import PrescriptionOCR

# Initialize the processor
ocr = PrescriptionOCR()

# Process an image
results = ocr.predict("path/to/prescription.jpg")

# Display results (medicine, frequency pairs)
for medicine, frequency in results:
    print(f"Medicine: {medicine}, Frequency: {frequency}")
```

## System Architecture

1. **ROI Extraction**: Identifies regions containing medication names and dosage instructions
2. **Post-processing**: refines yolo model outputs to improve accuracy
2. **OCR Processing**: Converts image regions to text using specialized Arabic and English OCR models
3. **Text Normalization**: Matches extracted text with known medicines and frequencies
4. **Pairing**: Associates each medication with its corresponding frequency

## Project Structure

- `Backend/`
  - `Roi_Extractor/`: YOLO-based region detection
  - `OCR/`: Text recognition for Arabic and English
  - `Image_Processor/`: Main processing pipeline
  - `Resources/`: Model weights and word dictionaries

## Models

The system uses several AI models:
- YOLO model for region detection
- TrOCR models for text recognition in both Arabic and English
- BK-tree structures for text correction using medicine and frequency corpora

## Contributors

- **Ahmed Hussein**: OCR model development and system integration
- **Kareem Wael**: YOLO model development and system deployment
- **Azza Hassan**: YOLO model development and system deployment
  
