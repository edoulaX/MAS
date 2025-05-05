import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
import os
from datetime import datetime
import huggingface_hub
import matplotlib.pyplot as plt
import json

class DiffusionExplorer:
    """Explores diffusion model outputs for wafer patterns"""
    
    def __init__(self):
        self.categories = [
            'center', 'donut', 'edge-loc', 'edge-ring',
            'loc', 'random', 'scratch', 'near-full'
        ]
        self.samples_per_category = 1
        
        # Storage for comparison
        self.original_colored = {}
        self.original_gray = {}
        self.generated_colored = {}
        self.generated_gray = {}
        
        # Set up Hugging Face token from environment variable
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token")
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        
        # Initialize diffusion model
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Create output directory with descriptive name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"./output/wafer_variations_{timestamp}")
        
        # Create step-specific directories
        self.steps = {
            'raw': self.output_dir / '1_raw_data',
            'preprocessed': self.output_dir / '2_preprocessed',
            'generated': self.output_dir / '3_generated_variations',
            'comparisons': self.output_dir / '4_comparisons'
        }
        
        for step_dir in self.steps.values():
            step_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hyperparameters
        self.hyperparameters = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'strength': 0.75,
            'num_variations': 3,
            'image_size': 512,
            'defect_threshold': 128
        }
        self._save_hyperparameters()

    def save_image_with_metadata(self, img, category, step, details, variation_idx=0):
        """Save image with detailed process metadata using matplotlib"""
        plt.figure(figsize=(12, 8))
        
        # Display the image
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        plt.imshow(img_array)
        
        # Create detailed metadata
        metadata = {
            'category': category,
            'processing_step': step,
            'step_details': details,
            'variation': variation_idx,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'hyperparameters': self.hyperparameters if 'generated' in step else {}
        }
        
        # Add metadata as text
        metadata_text = f"Processing Step: {step}\n"
        metadata_text += f"Category: {category}\n"
        metadata_text += f"Details: {details}\n"
        if variation_idx > 0:
            metadata_text += f"Variation: {variation_idx}\n"
        if 'generated' in step:
            metadata_text += f"Guidance Scale: {self.hyperparameters['guidance_scale']}\n"
            metadata_text += f"Steps: {self.hyperparameters['num_inference_steps']}\n"
            metadata_text += f"Strength: {self.hyperparameters['strength']}"
        
        plt.title(metadata_text, fontsize=10, pad=20)
        plt.axis('off')
        
        # Determine save directory based on step
        if 'raw' in step:
            save_dir = self.steps['raw']
        elif 'preprocessed' in step:
            save_dir = self.steps['preprocessed']
        elif 'generated' in step:
            save_dir = self.steps['generated']
        else:
            save_dir = self.output_dir
        
        # Create category subdirectory
        category_dir = save_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Save image with descriptive filename
        base_filename = f"{category}_{step}"
        if variation_idx > 0:
            base_filename += f"_variation_{variation_idx}"
        filename = f"{base_filename}.png"
        
        # Save image and metadata
        plt.savefig(category_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save metadata as JSON
        with open(category_dir / f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Store for comparison
        if 'raw' in step:
            self.original_colored[category] = img_array
        elif 'preprocessed_gray' in step:
            self.original_gray[category] = img_array
        elif 'generated_colored' in step:
            if category not in self.generated_colored:
                self.generated_colored[category] = []
            self.generated_colored[category].append(img_array)
        elif 'generated_gray' in step:
            if category not in self.generated_gray:
                self.generated_gray[category] = []
            self.generated_gray[category].append(img_array)

    def create_comparison_plot(self, images_dict, title, filename, include_variations=False):
        """Create a comparison plot of images across categories"""
        n_categories = len(self.categories)
        if include_variations:
            fig, axes = plt.subplots(n_categories, self.hyperparameters['num_variations'], 
                                   figsize=(20, 4*n_categories))
        else:
            fig, axes = plt.subplots(1, n_categories, figsize=(20, 4))
            axes = np.array([axes])  # Make it 2D for consistent indexing
        
        fig.suptitle(title, fontsize=16, y=1.02)
        
        for i, category in enumerate(self.categories):
            if category in images_dict:
                if include_variations:
                    variations = images_dict[category]
                    for j, img in enumerate(variations):
                        if isinstance(img, Image.Image):
                            img = np.array(img)
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f"{category} - Variation {j+1}")
                        axes[i, j].axis('off')
                else:
                    img = images_dict[category]
                    if isinstance(img, Image.Image):
                        img = np.array(img)
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(category)
                    axes[0, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.steps['comparisons'] / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def save_all_comparisons(self):
        """Save comparison plots for all stages"""
        # Original images comparison
        self.create_comparison_plot(
            self.original_colored,
            "Step 1: Raw Colored Wafer Maps",
            "1_raw_colored_comparison.png"
        )
        
        self.create_comparison_plot(
            self.original_gray,
            "Step 2: Preprocessed Grayscale Wafer Maps",
            "2_preprocessed_gray_comparison.png"
        )
        
        # Generated variations comparison
        self.create_comparison_plot(
            self.generated_colored,
            "Step 3a: Generated Variations from Colored Images",
            "3a_generated_colored_comparison.png",
            include_variations=True
        )
        
        self.create_comparison_plot(
            self.generated_gray,
            "Step 3b: Generated Variations from Grayscale Images",
            "3b_generated_gray_comparison.png",
            include_variations=True
        )

    def process_dataset(self, data_path):
        """Process the entire dataset"""
        print("Loading dataset...")
        data = self.load_dataset(data_path)
        
        print("\nGenerating variations for each category...")
        for idx, row in data.iterrows():
            category = row['failureType']
            wafermap = row['waferMap']
            
            print(f"\nProcessing {category} sample")
            
            # Step 1: Save raw wafermap
            self.save_image_with_metadata(
                img=wafermap,
                category=category,
                step="1_raw",
                details="Original wafermap data"
            )
            
            # Step 2: Preprocess wafermap
            colored_img, gray_img = self.preprocess_wafermap(wafermap)
            
            self.save_image_with_metadata(
                img=colored_img,
                category=category,
                step="2_preprocessed_colored",
                details="Preprocessed colored version (defects in red)"
            )
            
            self.save_image_with_metadata(
                img=gray_img,
                category=category,
                step="2_preprocessed_gray",
                details="Preprocessed grayscale version"
            )
            
            # Step 3: Generate variations
            print("  Generating variations from colored image...")
            self._generate_variations(colored_img, category, "colored")
            
            print("  Generating variations from grayscale image...")
            self._generate_variations(gray_img, category, "gray")
        
        # Step 4: Create comparison plots
        print("\nGenerating comparison plots...")
        self.save_all_comparisons()

    def _generate_variations(self, image, category, input_type):
        """Internal method to generate and save variations"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        prompt = f"a wafer pattern with {category} defect pattern"
        
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_images_per_prompt=self.hyperparameters['num_variations'],
            guidance_scale=self.hyperparameters['guidance_scale'],
            num_inference_steps=self.hyperparameters['num_inference_steps'],
            strength=self.hyperparameters['strength']
        )
        
        for i, img in enumerate(result.images):
            self.save_image_with_metadata(
                img=img,
                category=category,
                step=f"3_generated_{input_type}",
                details=f"Generated from {input_type} input using Stable Diffusion",
                variation_idx=i+1
            )

    def _save_hyperparameters(self):
        """Save hyperparameters to a text file"""
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            json.dump({
                'model': "CompVis/stable-diffusion-v1-4",
                'parameters': self.hyperparameters,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)
    
    def load_dataset(self, data_path):
        """Load and sample from WM-811K dataset"""
        # Load the data
        df = pd.read_pickle(Path(data_path) / 'WM811K.pkl')
        print("Dataset columns:", df.columns)
        print("\nSample data types:")
        print(df.dtypes)
        
        # Convert numpy array failure types to strings
        df['failureType'] = df['failureType'].apply(lambda x: str(x) if isinstance(x, np.ndarray) else str(x))
        df['failureType'] = df['failureType'].fillna('none')
        
        print("\nUnique failure types:", df['failureType'].unique())
        
        # Sample data for each category
        sampled_data = []
        for category in self.categories:
            category_data = df[df['failureType'].str.contains(category, case=False, na=False)]
            print(f"\nFound {len(category_data)} samples for category {category}")
            if len(category_data) >= self.samples_per_category:
                samples = category_data.sample(
                    n=self.samples_per_category,
                    random_state=42
                )
            else:
                samples = category_data.sample(
                    n=self.samples_per_category,
                    replace=True,
                    random_state=42
                )
            sampled_data.append(samples)
        
        return pd.concat(sampled_data).reset_index(drop=True)
    
    def preprocess_wafermap(self, wafermap):
        """Convert wafermap to RGB image with color processing"""
        # Convert to numpy array and normalize
        image = np.array(wafermap, dtype=np.float32)
        
        # Resize to standard size
        image = cv2.resize(image, (512, 512))
        
        # Normalize to 0-255 range
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        
        # Create colored version (defects in red, background in white)
        colored = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White background
        colored[image > 128] = [255, 0, 0]  # Red for defects
        
        # Create grayscale version
        grayscale = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return colored, grayscale

if __name__ == "__main__":
    # Create explorer instance
    explorer = DiffusionExplorer()
    
    # Process dataset
    explorer.process_dataset('input')
    
    print("\nExploration complete! Check the output directory for results.")
    print(f"Output directory: {explorer.output_dir}")
    print("\nDirectory structure:")
    print("1_raw_data/: Original wafer maps")
    print("2_preprocessed/: Colored and grayscale preprocessed images")
    print("3_generated_variations/: Generated variations from both inputs")
    print("4_comparisons/: Side-by-side comparison plots") 