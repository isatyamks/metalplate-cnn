import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(TextRecognitionModel, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers to extract features
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 128x128x1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64x64x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Input: 64x64x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 32x32x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Input: 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16x16x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Input: 16x16x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 8x8x256

            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Input: 8x8x256
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # Output: 4x4x512
        )

        # Fully connected layer for classification (for 361 classes: A000-A360)
        # The output size of features after the last MaxPool2d is 512 * 8 * 8 = 32768
        self.fc = nn.Linear(512 * 8 * 8, num_classes) # num_classes will be 361

    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)

        # Flatten the features for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 4*4*512)

        # Classification head
        x = self.fc(x)

        return x # Return logits for classification

class MetalPlateTextExtractor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model with 361 classes for A0000-A0360
        self.num_classes = 361
        self.model = TextRecognitionModel(self.num_classes).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"No model found at {model_path}. Using untrained model.")
        
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Plate number mapping for A0000-A0360
        self.idx_to_plate_text = {i: f"A{i:04d}" for i in range(361)}
        
    def preprocess_image(self, image):
        """Preprocess the image for the CNN model"""
        try:
            if image is None:
                logger.error("Input image is None")
                return None
                
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
            enhanced = clahe.apply(gray)
            
            # Apply bilateral filter to reduce noise
            denoised = cv2.bilateralFilter(enhanced, 9, 85, 85)
            
            # Apply median blur to remove salt-and-pepper noise
            blurred = cv2.medianBlur(denoised, 5) # Added median blur

            # Apply adaptive thresholding with smaller block size
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 91, 5 # Increased block size, adjusted C
            )
            
            # Apply opening to remove small noise (speckles)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # Increased kernel size for opening
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

            # Apply dilation to make characters more solid and connected
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # Larger kernel for dilation
            dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, kernel_dilate)

            # Apply erosion to refine shapes and remove noise
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernel_erode)

            return eroded # Return eroded image as the final cleaned image
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            return None

    def find_text_regions(self, binary_image):
        """Find regions containing text"""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours based on aspect ratio, area, and character-like dimensions
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                area = cv2.contourArea(contour)

                # Adjusted filters for individual characters
                # Assuming characters are generally taller than wide or roughly square
                # and not too small or too large
                # These values are empirical and might need further tuning based on dataset characteristics
                if 0.1 < aspect_ratio < 10 and 50 < area < 15000: # Broadened the range for aspect ratio and area
                    text_regions.append((x, y, w, h))

            # Sort regions by their Y-coordinate to maintain vertical reading order
            text_regions.sort(key=lambda r: r[1])

            return text_regions
            
        except Exception as e:
            logger.error(f"Error in finding text regions: {str(e)}", exc_info=True)
            return []

    def recognize_text(self, image_path):
        """Recognize text from the image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image at {image_path}")
                return None
                
            # Preprocess image
            processed = self.preprocess_image(image)
            if processed is None:
                return None

            # Find regions containing text (individual characters or groups)
            text_regions = self.find_text_regions(processed)

            if not text_regions:
                logger.warning("No text regions found in the image.")
                return None

            # Calculate a bounding box that encompasses all detected text regions
            min_x = min(r[0] for r in text_regions)
            min_y = min(r[1] for r in text_regions)
            max_x = max(r[0] + r[2] for r in text_regions)
            max_y = max(r[1] + r[3] for r in text_regions)

            # Crop the preprocessed image to this combined bounding box
            cropped_text_area = processed[min_y:max_y, min_x:max_x]

            # If the cropped area is empty, return None
            if cropped_text_area.shape[0] == 0 or cropped_text_area.shape[1] == 0:
                logger.warning("Cropped text area is empty.")
                return None

            # Pad cropped_text_area to maintain a consistent aspect ratio (1:2 as in training data)
            target_aspect_ratio = 0.5  # width / height (e.g., 100/200 or 200/400)
            h, w = cropped_text_area.shape[:2]
            current_aspect_ratio = float(w) / h

            padded_image = cropped_text_area
            if current_aspect_ratio < target_aspect_ratio:
                # Image is too tall, need to pad horizontally
                new_w = int(h * target_aspect_ratio)
                padding = new_w - w
                pad_left = padding // 2
                pad_right = padding - pad_left
                padded_image = cv2.copyMakeBorder(cropped_text_area, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0) # Pad with black
            elif current_aspect_ratio > target_aspect_ratio:
                # Image is too wide, need to pad vertically
                new_h = int(w / target_aspect_ratio)
                padding = new_h - h
                pad_top = padding // 2
                pad_bottom = padding - pad_top
                padded_image = cv2.copyMakeBorder(cropped_text_area, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0) # Pad with black

            # Transform the padded image for model input
            input_tensor = self.transform(padded_image).unsqueeze(0).to(self.device)

            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)  # Shape: (1, num_classes)
                
                # Get the predicted index for the entire plate number
                predicted_index = torch.argmax(output, dim=1).item()
                
                # Convert index to plate text
                if predicted_index < self.num_classes:
                    final_text = self.idx_to_plate_text[predicted_index]
                else:
                    final_text = None # Should not happen if training is correct

                return final_text if final_text else None

        except Exception as e:
            logger.error(f"Error in text recognition: {str(e)}", exc_info=True)
            return None

    def visualize_results(self, image_path, output_path=None):
        """Visualize the preprocessing steps and results"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image for visualization: {image_path}")
                return
                
            processed = self.preprocess_image(image)
            if processed is None:
                return
                
            text_regions = self.find_text_regions(processed)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(132)
            plt.title("Preprocessed")
            plt.imshow(processed, cmap='gray')
            plt.axis('off')
            
            plt.subplot(133)
            plt.title("Detected Text Regions")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for x, y, w, h in text_regions:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved visualization to: {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}", exc_info=True)

def main():
    # Initialize the text extractor with the trained model
    extractor = MetalPlateTextExtractor(model_path='metal_plate_model_deep_cluster_final.pth') # Use the new model path
    
    # Process the image
    image_path = "i1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("\nExtracting text from metal plate...")
    text = extractor.recognize_text(image_path)
    
    if text:
        print(f"\nExtracted Text: {text}")
    else:
        print("\nFailed to extract text from the image.")

if __name__ == "__main__":
    main()
