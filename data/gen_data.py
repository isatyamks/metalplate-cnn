import cv2
import numpy as np
import os
import random
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self, output_dir='data\\validation_data'):
        self.output_dir = output_dir
        self.width = 200
        self.height = 400
        os.makedirs(output_dir, exist_ok=True)
        self.characters = string.ascii_uppercase + string.digits
        self.fonts = []
        font_paths = [
            "stencil.ttf",
        ]
        font_sizes = [50, 60, 70, 80, 90, 100]
        system_font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        for font_path in font_paths:
            full_path = os.path.join(system_font_dir, font_path)
            for size in font_sizes:
                try:
                    font = ImageFont.truetype(full_path, size)
                    self.fonts.append(font)
                    logger.info(f"Successfully loaded font {font_path} with size {size}")
                except IOError:
                    logger.warning(f"Could not load font '{font_path}' with size {size}. Trying next font.")
        if not self.fonts:
            try:
                default_font = ImageFont.load_default()
                self.fonts.append(default_font)
                logger.info("Using default font as fallback")
            except Exception as e:
                logger.error(f"Failed to load any fonts: {str(e)}")
                raise RuntimeError("No suitable fonts could be loaded. Please ensure you have system fonts installed.")
    
    def generate_metal_texture(self):
        return np.full((self.height, self.width), 200, dtype=np.uint8)
    
    def add_text(self, image, text, font):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        plate_width = self.width - 50
        plate_height = self.height - 50
        plate_x1 = (self.width - plate_width) // 2
        plate_y1 = (self.height - plate_height) // 2
        plate_x2 = plate_x1 + plate_width
        plate_y2 = plate_y1 + plate_height
        draw.rectangle([plate_x1, plate_y1, plate_x2, plate_y2], fill=30)
        total_text_height = sum(draw.textbbox((0,0), char, font=font)[3] - draw.textbbox((0,0), char, font=font)[1] + 5 for char in text) - 5
        current_y = plate_y1 + (plate_height - total_text_height) // 2
        for char_idx, char in enumerate(text):
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            x = plate_x1 + (plate_width - char_width) // 2
            draw.text((x, current_y), char, font=font, fill=220)
            current_y += char_height + 5
        return np.array(pil_image)

    def add_noise_and_distortions(self, image):
        img_float = image.astype(np.float32)
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        img_float = img_float + noise
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        img_float = img_float * alpha + beta
        img_float = np.clip(img_float, 0, 255)
        angle = random.uniform(-3, 3)
        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img_float, rotation_matrix, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101)
        blurred_image = cv2.GaussianBlur(rotated_image, (3, 3), 0.7)
        pts1 = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])
        pts2 = np.float32([[random.randint(-10,10),random.randint(-10,10)],[self.width - random.randint(-10,10),random.randint(-10,10)],[random.randint(-10,10),self.height - random.randint(-10,10)],[self.width - random.randint(-10,10),self.height - random.randint(-10,10)]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        distorted_image = cv2.warpPerspective(blurred_image, matrix, (self.width,self.height), borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.2:
            kernel_size = random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            if random.random() < 0.5:
                kernel[kernel_size // 2, :] = np.ones(kernel_size)
            else:
                kernel[:, kernel_size // 2] = np.ones(kernel_size)
            kernel /= kernel_size
            distorted_image = cv2.filter2D(distorted_image, -1, kernel)
        return distorted_image.astype(np.uint8)

    def generate_image(self, text):
        image = self.generate_metal_texture()
        image_float = image.astype(np.float32) + np.random.randint(-10, 10, image.shape).astype(np.float32)
        image = np.clip(image_float, 0, 255).astype(np.uint8)
        font = random.choice(self.fonts)
        image = self.add_text(image, text, font)
        image = self.add_noise_and_distortions(image)
        return image
    
    def generate_dataset(self, num_samples=1000):
        logger.info(f"Generating {num_samples} training images with format A0000-A0360...")
        all_possible_texts = []
        for i in range(361):
            num_str = f"A0{i:03d}"
            all_possible_texts.append(num_str)
        if num_samples > len(all_possible_texts):
            logger.warning(f"Requested {num_samples} samples, but only {len(all_possible_texts)} unique texts in A0000-A0360 range. Sampling with replacement.")
            texts_to_generate = random.choices(all_possible_texts, k=num_samples)
        else:
            texts_to_generate = random.sample(all_possible_texts, k=num_samples)
        for i, text in enumerate(texts_to_generate):
            image = self.generate_image(text)
            class_folder = os.path.join(self.output_dir, text)  
            os.makedirs(class_folder, exist_ok=True)
            filename = f"{text}_{i:04d}.jpg"
            filepath = os.path.join(class_folder, filename)
            cv2.imwrite(filepath, image)
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} images")
        logger.info(f"Dataset generation completed. Images saved in {self.output_dir}")

def main():
    generator = TrainingDataGenerator()
    generator.generate_dataset(num_samples=1000)

if __name__ == "__main__":
    main()
