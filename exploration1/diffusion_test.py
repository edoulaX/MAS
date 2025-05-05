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
import sys

class TransformationLibrary:
    """Library of transformations and prompts for different wafer defect patterns"""
    
    def __init__(self):
        self.transformations = {
            'center': {
                'name': 'Tennis Ball Transform',
                'mask_color': [255, 255, 0],  # Yellow
                'background_color': [255, 255, 255],  # White
                'prompts': {
                    'base': "a tennis ball",
                    'enhanced': "Professional product photography of a pristine tennis ball, " 
                              "perfectly centered, vibrant yellow felt texture, crisp focus on surface details, "
                              "pure white background, studio lighting setup with soft boxes, 8k resolution, "
                              "commercial product photography, minimalist composition, high contrast"
                }
            },
            'donut': {
                'name': 'Donut Transform',
                'mask_color': [255, 255, 0],  # Changed to yellow (from light brown/beige)
                'background_color': [255, 255, 255],  # White
                'prompts': {
                    'base': "a classic donut with sprinkles",
                    'enhanced': "Professional food photography of a freshly baked artisanal donut with colorful sprinkles, "
                              "showing detailed texture and glaze, studio lighting setup with soft boxes, crisp focus, "
                              "pure white background, 8k resolution, commercial photography, centered composition, "
                              "dramatic lighting, high-end advertising style"
                }
            }
        }
    
    def apply_transform(self, wafermap, category, include_grayscale=True):
        """Apply category-specific transformation to wafermap"""
        if category not in self.transformations:
            raise ValueError(f"No transformation defined for category: {category}")
            
        transform = self.transformations[category]
        image = np.array(wafermap, dtype=np.float32)
        image = cv2.resize(image, (512, 512))
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        
        # Create colored mask
        colored = np.full((512, 512, 3), transform['background_color'], dtype=np.uint8)
        mask = image > 128
        colored[mask] = transform['mask_color']
        
        if include_grayscale:
            # Create grayscale version
            grayscale = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return colored, grayscale
        
        return colored
        
    def apply_smooth_transform(self, wafermap, category, include_grayscale=True, sigma=1.0, transition_width=0.5):
        """Apply category-specific transformation with anti-aliasing and smooth transitions
        
        Args:
            wafermap: Original wafer map data
            category: Category name for transformation
            include_grayscale: Whether to include grayscale output
            sigma: Gaussian blur sigma (higher = more blurring)
            transition_width: Width of color transition between defect and background (0-1)
        """
        if category not in self.transformations:
            raise ValueError(f"No transformation defined for category: {category}")
            
        transform = self.transformations[category]
        
        # Convert to numpy array and normalize
        image = np.array(wafermap, dtype=np.float32)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        image = (image - image.min()) / (image.max() - image.min()) * 255
        
        # Apply Gaussian blur for anti-aliasing
        smoothed = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Create soft mask with gradient transitions
        # Adjust threshold values to create transition zone
        mask_min = 128 * (1 - transition_width)
        mask_max = 128 * (1 + transition_width)
        
        # Create alpha (opacity) channel: 0 = background, 1 = full mask color
        alpha = np.clip((smoothed - mask_min) / (mask_max - mask_min), 0, 1)
        alpha = alpha.reshape(512, 512, 1)
        
        # Convert colors to float for blending
        mask_color = np.array(transform['mask_color'], dtype=np.float32) / 255
        bg_color = np.array(transform['background_color'], dtype=np.float32) / 255
        
        # Alpha blend the colors
        blended = alpha * mask_color + (1 - alpha) * bg_color
        
        # Convert back to uint8
        colored = (blended * 255).astype(np.uint8)
        
        if include_grayscale:
            # Create grayscale version with same smoothing
            grayscale = cv2.cvtColor(smoothed.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            return colored, grayscale
        
        return colored
    
    def get_prompts(self, category):
        """Get prompts for a specific category"""
        if category not in self.transformations:
            raise ValueError(f"No prompts defined for category: {category}")
        return self.transformations[category]['prompts']
    
    def get_transform_name(self, category):
        """Get the name of the transformation for a category"""
        if category not in self.transformations:
            raise ValueError(f"No transformation defined for category: {category}")
        return self.transformations[category]['name']


class DiffusionTester:
    """Tests specific prompts for wafer pattern categories"""
    
    def __init__(self):
        self.categories = ['center', 'donut']
        self.samples_per_category = 1
        
        # Initialize pipeline data structure
        self.pipeline_data = {}
        self._initialize_pipeline_data()
        
        # Initialize transformation library
        self.transform_lib = TransformationLibrary()
        
        # Create output directory with descriptive name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"./output/pipeline_test_{timestamp}")
        
        # Create and clean review directory
        self.review_dir = Path("./output/sample_review")
        if self.review_dir.exists():
            for file in self.review_dir.glob("*.png"):
                file.unlink()
        self.review_dir.mkdir(parents=True, exist_ok=True)
        
        # Create step-specific directories
        self.steps = {
            'review': self.review_dir,
            'raw': self.output_dir / '1_raw_data',
            'transformed': self.output_dir / '2_transformed',
            'generated': self.output_dir / '3_generated',
            'pipeline': self.output_dir / '4_pipeline_viz'
        }
        
        # Create all directories with parents
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for step_dir in self.steps.values():
            step_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Save hyperparameters
        self.hyperparameters = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'strength': 0.75,
            'num_variations': 1,
            'image_size': 512,
            'defect_threshold': 128
        }
        self._save_hyperparameters()
        self._save_transformations()

    def _initialize_pipeline_data(self):
        """Helper method to initialize pipeline data structure"""
        for category in self.categories:
            self.pipeline_data[category] = {
                'raw': None,
                'colored': None,
                'gray': None,
                'colored_center_prompt': [],
                'colored_donut_prompt': [],
                'gray_center_prompt': [],
                'gray_donut_prompt': []
            }

    def create_pipeline_visualization(self, category):
        """Create a visualization showing the complete transformation pipeline"""
        if category not in self.pipeline_data:
            raise ValueError(f"Category {category} not found in pipeline data")
            
        data = self.pipeline_data[category]
        
        # Verify all required data exists
        required_keys = ['raw', 'colored', 'gray']
        missing_keys = [key for key in required_keys if data.get(key) is None]
        if missing_keys:
            raise ValueError(f"Missing required data for {category}: {missing_keys}")
        
        try:
            # Create figure with a 4x3 grid for horizontal tree layout
            fig = plt.figure(figsize=(28, 20))
            fig.suptitle(f"Transformation Pipeline for {category.capitalize()}", fontsize=20, y=0.98)
            
            # Add more spacing between subplots
            plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.92, bottom=0.08, left=0.05, right=0.95)
            
            # Common styling
            metadata_style = dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            arrow_style = dict(arrowstyle="->", lw=2, alpha=0.7)
            
            # Raw data (left side)
            ax_raw = plt.subplot2grid((4, 3), (1, 0), rowspan=2)
            ax_raw.imshow(data['raw'])
            ax_raw.set_title("Raw Data", fontsize=16, pad=15)
            ax_raw.axis('off')
            
            # Add raw data metadata
            raw_metadata = f"Original Wafer Map\nCategory: {category.capitalize()}"
            ax_raw.text(0.5, -0.1, raw_metadata, transform=ax_raw.transAxes, 
                      ha='center', va='top', fontsize=12, 
                      bbox=metadata_style)
            
            # Preprocessed (middle column)
            ax_colored = plt.subplot2grid((4, 3), (0, 1), rowspan=2)
            ax_gray = plt.subplot2grid((4, 3), (2, 1), rowspan=2)
            ax_colored.imshow(data['colored'])
            ax_gray.imshow(data['gray'])
            ax_colored.set_title("Colored Transform", fontsize=16, pad=15)
            ax_gray.set_title("Grayscale Transform", fontsize=16, pad=15)
            ax_colored.axis('off')
            ax_gray.axis('off')
            
            # Add transform metadata
            transform_name = self.transform_lib.get_transform_name(category)
            transform_prompts = self.transform_lib.get_prompts(category)
            colored_metadata = f"Transform: {transform_name}\nMask Color: Yellow"
            ax_colored.text(0.5, -0.13, colored_metadata, transform=ax_colored.transAxes, 
                         ha='center', va='top', fontsize=12, 
                         bbox=metadata_style)
            
            gray_metadata = "Grayscale conversion\nPreserves original intensity values"
            ax_gray.text(0.5, -0.13, gray_metadata, transform=ax_gray.transAxes, 
                      ha='center', va='top', fontsize=12,
                      bbox=metadata_style)
            
            # Create axes for generated images
            ax_colored_center = plt.subplot2grid((4, 3), (0, 2))
            ax_colored_donut = plt.subplot2grid((4, 3), (1, 2))
            ax_gray_center = plt.subplot2grid((4, 3), (2, 2))
            ax_gray_donut = plt.subplot2grid((4, 3), (3, 2))
            
            # Helper function for getting variation safely
            def get_first_variation(key):
                variations = data.get(key, [])
                if variations and len(variations) > 0:
                    return variations[0]
                print(f"Warning: No variations found for {key}")
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Show images and set titles
            ax_colored_center.imshow(get_first_variation('colored_center_prompt'))
            ax_colored_center.set_title("Tennis Ball Prompt", fontsize=14, pad=10)
            ax_colored_center.axis('off')
            
            ax_colored_donut.imshow(get_first_variation('colored_donut_prompt'))
            ax_colored_donut.set_title("Donut Prompt", fontsize=14, pad=10)
            ax_colored_donut.axis('off')
            
            ax_gray_center.imshow(get_first_variation('gray_center_prompt'))
            ax_gray_center.set_title("Tennis Ball Prompt", fontsize=14, pad=10)
            ax_gray_center.axis('off')
            
            ax_gray_donut.imshow(get_first_variation('gray_donut_prompt'))
            ax_gray_donut.set_title("Donut Prompt", fontsize=14, pad=10)
            ax_gray_donut.axis('off')
            
            # Add metadata to images
            for ax, prompt_name in [
                (ax_colored_center, "Tennis Ball"),
                (ax_colored_donut, "Donut"),
                (ax_gray_center, "Tennis Ball"),
                (ax_gray_donut, "Donut")
            ]:
                metadata = (
                    f"Prompt: {prompt_name}\n"
                    f"Steps: {self.hyperparameters['num_inference_steps']}\n"
                    f"Guidance: {self.hyperparameters['guidance_scale']}"
                )
                ax.text(0.5, -0.2, metadata, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=11, 
                       bbox=metadata_style)
            
            # Draw arrows
            # Raw to Colored
            plt.annotate("", xy=(0.33, 0.35), xytext=(0.15, 0.5),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=-0.2", "color": "blue"})
            
            # Raw to Gray
            plt.annotate("", xy=(0.33, 0.65), xytext=(0.15, 0.5),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=0.2", "color": "blue"})
            
            # Add labels to preprocessing arrows
            plt.figtext(0.23, 0.4, "Color\nMapping", fontsize=12, ha='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            plt.figtext(0.23, 0.6, "Grayscale\nConversion", fontsize=12, ha='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Colored to output arrows
            plt.annotate("", xy=(0.65, 0.25), xytext=(0.5, 0.35),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=-0.2", "color": "green"})
            plt.annotate("", xy=(0.65, 0.45), xytext=(0.5, 0.35),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=0.2", "color": "green"})
            
            # Gray to output arrows
            plt.annotate("", xy=(0.65, 0.65), xytext=(0.5, 0.75),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=-0.2", "color": "green"})
            plt.annotate("", xy=(0.65, 0.85), xytext=(0.5, 0.75),
                        xycoords='figure fraction', 
                        arrowprops={**arrow_style, "connectionstyle": "arc3,rad=0.2", "color": "green"})
            
            # Add a single Stable Diffusion label in the center of the diagram
            plt.figtext(0.58, 0.5, "Stable Diffusion\nTransformation", fontsize=14, ha='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            # Add global metadata box at the bottom
            metadata_text = (
                f"Model: CompVis/stable-diffusion-v1-4   |   "
                f"Category: {category.capitalize()}   |   "
                f"Steps: {self.hyperparameters['num_inference_steps']}   |   "
                f"Guidance: {self.hyperparameters['guidance_scale']}   |   "
                f"Strength: {self.hyperparameters['strength']}   |   "
                f"Variations: {self.hyperparameters['num_variations']}"
            )
            plt.figtext(0.5, 0.02, metadata_text, ha='center', va='bottom', 
                      bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.5'), 
                      fontsize=14)
            
            # Save with error handling
            try:
                save_path = self.steps['pipeline'] / f"{category}_pipeline.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            except Exception as e:
                print(f"Warning: Failed to save pipeline visualization for {category}: {e}")
            finally:
                plt.close()
                
        except Exception as e:
            print(f"Error creating pipeline visualization for {category}: {e}")

    def _generate_variations(self, image, category, prompt_category, input_type):
        """Generate variations and store in pipeline data"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        prompt = self.transform_lib.get_prompts(prompt_category)['enhanced']
        
        try:
            result = self.pipe(
                prompt=prompt,
                image=image,
                num_images_per_prompt=self.hyperparameters['num_variations'],
                guidance_scale=self.hyperparameters['guidance_scale'],
                num_inference_steps=self.hyperparameters['num_inference_steps'],
                strength=self.hyperparameters['strength']
            )
            
            # Store in pipeline data with the correct key format
            variation_key = f'{input_type}_{prompt_category}_prompt'
            self.pipeline_data[category][variation_key] = [np.array(img) for img in result.images]
            
            # Save individual variations
            for idx, img in enumerate(result.images):
                save_path = self.steps['generated'] / f"{category}_{variation_key}_variation_{idx}.png"
                plt.figure(figsize=(8, 8))
                plt.imshow(np.array(img))
                plt.title(f"{category.capitalize()} - {input_type} {prompt_category} prompt - Variation {idx+1}")
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"Error generating variations for {category} with {prompt_category} prompt: {e}")
            # Initialize with empty list to prevent future access errors
            self.pipeline_data[category][f'{input_type}_{prompt_category}_prompt'] = []

    def save_raw_sample(self, category, wafermap, sample_idx):
        """Save the raw sample for review"""
        plt.figure(figsize=(8, 8))
        plt.imshow(wafermap)
        plt.title(f"{category.capitalize()} Sample {sample_idx}")
        plt.axis('off')
        
        # Save the plot
        plt.savefig(self.review_dir / f"{category}_sample_{sample_idx}.png", bbox_inches='tight', dpi=300)
        plt.close()

    def get_user_approval(self, category, sample_idx):
        """Get user approval for the selected sample"""
        print(f"\nPlease review the {category} sample {sample_idx} in the output/sample_review directory.")
        while True:
            try:
                response = input("Do you want to proceed with this sample? (yes/no): ").lower().strip()
                if response in ['yes', 'no', 'y', 'n']:
                    return response.startswith('y')
                print("Please enter 'yes' or 'no'")
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Exiting...")
                sys.exit(1)
            except Exception as e:
                print(f"Error reading input: {e}")
                print("Please try again")

    def select_sample_batch(self, category_data, category, batch_size=10):
        """Select a batch of samples from the category data"""
        total_samples = len(category_data)
        if total_samples < batch_size:
            print(f"\nWarning: Only {total_samples} samples available for {category}. Using all samples.")
            return category_data
        return category_data.sample(n=batch_size, random_state=42)

    def process_dataset(self, data_path):
        """Process the dataset and create pipeline visualizations"""
        print("Loading dataset...")
        data = self.load_dataset(data_path)
        
        print("\nGenerating pipeline visualizations...")
        for category in self.categories:
            # Get all samples for this category
            category_data = data[data['failureType'] == category].copy()  # Make a copy to avoid warnings
            print(f"\nFound {len(category_data)} samples for category {category}")
            
            # Get initial batch of samples
            batch_size = min(10, len(category_data))
            # Randomly select batch_size samples without replacement
            batch_indices = np.random.choice(len(category_data), size=batch_size, replace=False)
            sample_batch = category_data.iloc[batch_indices].reset_index(drop=True)
            print(f"\nSelected {len(sample_batch)} samples for initial batch")
            
            sample_idx = 0
            current_batch_idx = 0
            
            while True:
                sample_idx += 1
                
                # Get current sample from batch
                current_sample = sample_batch.iloc[current_batch_idx]
                wafermap = current_sample['waferMap']
                
                # Save and show the sample for review
                print(f"\nProcessing {category} sample {sample_idx}")
                print(f"Current batch: Sample {current_batch_idx + 1} of {len(sample_batch)}")
                self.save_raw_sample(category, wafermap, sample_idx)
                
                # Get user approval
                if self.get_user_approval(category, sample_idx):
                    break
                
                # Move to next sample in batch
                current_batch_idx = (current_batch_idx + 1) % len(sample_batch)
                if current_batch_idx == 0:
                    print("\nReached end of current batch. Selecting new batch...")
                    # Select new batch excluding previously shown samples
                    remaining_indices = list(set(range(len(category_data))) - set(batch_indices))
                    if len(remaining_indices) < batch_size:
                        print("Not enough new samples remaining. Resetting selection...")
                        remaining_indices = range(len(category_data))
                    
                    batch_indices = np.random.choice(remaining_indices, size=batch_size, replace=False)
                    sample_batch = category_data.iloc[batch_indices].reset_index(drop=True)
                    print(f"Selected {len(sample_batch)} new samples")
            
            print(f"\nProceeding with approved {category} sample {sample_idx}")
            
            try:
                # Store raw data and save
                self.pipeline_data[category]['raw'] = wafermap
                plt.figure(figsize=(8, 8))
                plt.imshow(wafermap)
                plt.title(f"{category.capitalize()} - Raw Data")
                plt.axis('off')
                plt.savefig(self.steps['raw'] / f"{category}_raw.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                # Apply smooth transformations and save
                colored_img, gray_img = self.transform_lib.apply_smooth_transform(
                    wafermap, 
                    category, 
                    include_grayscale=True,
                    sigma=1.0,  # Adjust the blur amount as needed
                    transition_width=0.7  # Wider transition (0.7) for smoother gradients
                )
                
                self.pipeline_data[category]['colored'] = colored_img
                self.pipeline_data[category]['gray'] = gray_img
                
                # Save transformed images
                plt.figure(figsize=(8, 8))
                plt.imshow(colored_img)
                plt.title(f"{category.capitalize()} - Smooth Colored Transform")
                plt.axis('off')
                plt.savefig(self.steps['transformed'] / f"{category}_colored.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                plt.figure(figsize=(8, 8))
                plt.imshow(gray_img)
                plt.title(f"{category.capitalize()} - Smooth Grayscale Transform")
                plt.axis('off')
                plt.savefig(self.steps['transformed'] / f"{category}_gray.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                # Generate and store variations for both prompts
                for prompt_category in self.categories:
                    print(f"  Generating variations using {prompt_category} prompt...")
                    print("  From colored image...")
                    self._generate_variations(colored_img, category, prompt_category, "colored")
                    
                    print("  From grayscale image...")
                    self._generate_variations(gray_img, category, prompt_category, "gray")
                
                # Create pipeline visualization
                self.create_pipeline_visualization(category)
                
            except Exception as e:
                print(f"Error processing {category}: {e}")
                continue
        
        print("\nPipeline visualizations complete!")
        print(f"\nOutput directory structure:")
        print(f"1. Raw data: {self.steps['raw']}")
        print(f"2. Transformed images: {self.steps['transformed']}")
        print(f"3. Generated variations: {self.steps['generated']}")
        print(f"4. Pipeline visualizations: {self.steps['pipeline']}")

    def _save_transformations(self):
        """Save transformation configurations to a JSON file"""
        config = {category: {
            'name': self.transform_lib.get_transform_name(category),
            'prompts': self.transform_lib.get_prompts(category)
        } for category in self.categories}
        
        with open(self.output_dir / 'transformations.json', 'w') as f:
            json.dump(config, f, indent=4)

    def _save_hyperparameters(self):
        """Save hyperparameters to a JSON file"""
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            json.dump({
                'model': "CompVis/stable-diffusion-v1-4",
                'parameters': self.hyperparameters,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)

    def load_dataset(self, data_path):
        """Load and sample from WM-811K dataset"""
        print("Loading dataset...")
        data_path = Path(data_path)
        
        # Try loading .npz file first, then .pkl if that fails
        try:
            if (data_path / 'Wafer_Map_Datasets.npz').exists():
                data = np.load(data_path / 'Wafer_Map_Datasets.npz')
                df = pd.DataFrame({
                    'waferMap': data['arr_0'].tolist(),
                    'failureType': data['arr_1'].tolist()
                })
            elif (data_path / 'WM811K.pkl').exists():
                df = pd.read_pickle(data_path / 'WM811K.pkl')
            else:
                raise FileNotFoundError("Neither Wafer_Map_Datasets.npz nor WM811K.pkl found in input directory")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
        
        print("Processing failure types...")
        def convert_failure_type(x):
            if isinstance(x, np.ndarray) and len(x) > 0:
                # For .npz file (one-hot encoded)
                try:
                    idx = np.where(x == 1)[0][0]
                    categories = ['center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'random', 'scratch', 'near-full']
                    return categories[idx]
                except:
                    return 'none'
            elif isinstance(x, (np.integer, int)):
                return str(x).lower()
            elif isinstance(x, str):
                return x.lower()
            else:
                return 'none'
                
        df['failureType'] = df['failureType'].apply(convert_failure_type)
        df['failureType'] = df['failureType'].fillna('none')
        
        # Sample data for selected categories
        sampled_data = []
        for category in self.categories:
            print(f"\nProcessing {category} category...")
            
            # More flexible pattern matching for categories
            if category == 'center':
                pattern = 'center|cen'
            else:
                pattern = category
            
            # Use str.contains for pattern matching
            category_data = df[df['failureType'].str.contains(pattern, case=False, na=False)]
            print(f"Found {len(category_data)} initial samples")
            
            # Verify waferMap data is valid
            valid_samples = category_data[
                category_data['waferMap'].apply(
                    lambda x: isinstance(x, np.ndarray) and x.size > 0 and not np.all(x == 0)
                )
            ]
            
            if len(valid_samples) != len(category_data):
                print(f"Filtered out {len(category_data) - len(valid_samples)} invalid samples")
            
            if len(valid_samples) > 0:
                print(f"Final count: {len(valid_samples)} valid samples")
                valid_samples = valid_samples.copy()
                valid_samples['failureType'] = category
                sampled_data.append(valid_samples)
            else:
                raise ValueError(f"No valid samples found for category {category}")
        
        if not sampled_data:
            raise ValueError("No samples found for any category!")
        
        final_data = pd.concat(sampled_data).reset_index(drop=True)
        print(f"\nTotal samples in final dataset: {len(final_data)}")
        return final_data

    def preprocess_wafermap(self, wafermap):
        """Convert wafermap to RGB image with color processing"""
        image = np.array(wafermap, dtype=np.float32)
        image = cv2.resize(image, (512, 512))
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        
        colored = np.ones((512, 512, 3), dtype=np.uint8) * 255
        colored[image > 128] = [255, 255, 0]  # Yellow for defects
        grayscale = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return colored, grayscale

if __name__ == "__main__":
    # Create tester instance
    tester = DiffusionTester()
    
    # Process dataset
    tester.process_dataset('input')
    
    print("\nTest complete! Check the output directory for results.")
    print(f"Output directory: {tester.output_dir}")
    print("\nEach category has a pipeline visualization showing:")
    print("1. Raw wafer map")
    print("2. Colored and grayscale transformations")
    print("3. Generated variations from both transformations with both prompts")
    print("\nPrompts used:")
    for category, prompts in tester.transform_lib.transformations.items():
        print(f"\n{category.upper()}:")
        print(f"Base: {prompts['prompts']['base']}")
        print(f"Enhanced: {prompts['prompts']['enhanced']}") 