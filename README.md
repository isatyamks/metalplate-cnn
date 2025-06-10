# Metal Plate Text Recognition CNN

This project implements a Convolutional Neural Network (CNN) for recognizing text on metal plates. The model is trained to classify metal plate numbers in the format A0000-A0360.

## Project Structure

```
metalplate-cnn/
├── data/                   # Dataset directory
├── notebooks/             # Jupyter notebooks for development and analysis
├── model.py              # Core CNN model implementation
├── evaluate_model.py     # Model evaluation script
├── test_pipeline.py      # Pipeline testing script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Dependencies

The project requires the following Python packages:
- numpy (>=1.21.0)
- opencv-python (>=4.5.0)
- Pillow (>=8.0.0)
- torch (>=1.9.0)
- torchvision (>=0.10.0)
- matplotlib (>=3.4.0)
- tqdm (>=4.62.0)
- scikit-learn (>=0.24.0)
- pandas (>=1.3.0)

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
python model.py
```

2. Evaluating the model:
```bash
python evaluate_model.py
```

3. Testing the pipeline:
```bash
python test_pipeline.py
```

## Model Architecture

The model uses a CNN architecture with:
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Fully connected layer for classification
- Output size of 361 classes (A0000-A0360)

## Data Processing

The pipeline includes:
- Image preprocessing (grayscale conversion, CLAHE, denoising)
- Text region detection
- Character segmentation
- Classification

## Development

The project includes Jupyter notebooks for development and analysis in the `notebooks/` directory. These notebooks contain:
- Model development and experimentation
- Data analysis and visualization
- Training and evaluation results

## License

[Add your license information here] 