# Metal Plate Text Recognition CNN

This project implements a Convolutional Neural Network (CNN) for recognizing text on metal plates. The model is trained to classify metal plate numbers in the format A0000-A0360.

## Project Structure

```
metalplate-cnn/
├── src/
│   ├── models/
│   │   ├── metal_plate_cnn.py      # Core CNN model implementation
│   │   └── weights/                # Model weights and checkpoints
│   ├── data/
│   │   ├── training/              # Training dataset
│   │   ├── validation/            # Validation dataset
│   │   ├── testing/              # Testing dataset
│   │   ├── samples/              # Sample images
│   │   ├── generate_dataset.py    # Dataset generation script
│   │   └── create_class_mapping.py # Class mapping creation utility
│   ├── utils/
│   │   ├── image_processing.py    # Image processing utilities
│   │   ├── data_generation.py     # Data generation utilities
│   │   └── organization.py        # Organization utilities
│   ├── scripts/
│   │   ├── train_model.py         # Model training script
│   │   ├── evaluate_model.py      # Model evaluation script
│   │   ├── inference.py           # Inference script
│   │   └── test_pipeline.py       # Pipeline testing script
│   ├── config/
│   │   ├── class_mapping.json     # Class mapping configuration
│   │   └── test_results.json      # Test results
│   └── visualization/
│       └── sample_visualization.png # Sample visualization outputs
├── requirements.txt               # Project dependencies
└── README.md                     # This file
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training the model:
```bash
python src/scripts/train_model.py
```

2. Running inference:
```bash
python src/scripts/inference.py --image path/to/image.jpg
```

3. Evaluating the model:
```bash
python src/scripts/evaluate_model.py
```

## Model Architecture

The model uses a CNN architecture with:
- 5 convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Fully connected layer for classification
- Output size of 361 classes (A0000-A0360)

## Data Processing

The pipeline includes:
- Image preprocessing (grayscale conversion, CLAHE, denoising)
- Text region detection
- Character segmentation
- Classification

## License

[Add your license information here] 