import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better visualizations
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (15, 10),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Wafer defect categories matching the original pipeline
WAFER_CATEGORIES = {
    'center': {
        'description': 'Center defect pattern',
        'pattern_type': 'circular',
        'params': {'radius': 0.4}  # Using original MASK_PARAMS ratio
    },
    'donut': {
        'description': 'Donut defect pattern',
        'pattern_type': 'ring',
        'params': {'inner_radius': 0.3, 'outer_radius': 0.8}  # Using original ratios
    },
    'edge-loc': {
        'description': 'Edge location defect',
        'pattern_type': 'edge',
        'params': {'width': 0.1}  # Using original width
    },
    'edge-ring': {
        'description': 'Edge ring defect',
        'pattern_type': 'edge_ring',
        'params': {'width': 0.1, 'distance': 0.05}
    },
    'loc': {
        'description': 'Localized defect',
        'pattern_type': 'spots',
        'params': {'num_spots': 3, 'radius': 0.05}
    },
    'random': {
        'description': 'Random defect pattern',
        'pattern_type': 'random',
        'params': {'num_patches': 10, 'min_size': 5, 'max_size': 20}  # Using original params
    },
    'scratch': {
        'description': 'Scratch defect',
        'pattern_type': 'line',
        'params': {'num_lines': (1, 3), 'thickness': 1}  # Using original params
    },
    'near-full': {
        'description': 'Near-full defect',
        'pattern_type': 'coverage',
        'params': {'coverage': 0.9}
    },
    'none': {
        'description': 'Normal wafer',
        'pattern_type': 'normal',
        'params': {'noise_level': 0.02}  # Very slight noise for realism
    }
}

class PatternGenerator:
    def __init__(self, size=52):  # Changed to match original input_size
        self.size = size
        self.device = device
    
    def generate_base(self):
        """Generate base circular wafer"""
        image = np.zeros((self.size, self.size), dtype=np.uint8)
        center = self.size // 2
        radius = int(self.size * 0.45)  # Slightly smaller than half to ensure edges
        cv2.circle(image, (center, center), radius, 255, -1)
        return image
    
    def add_noise(self, image, noise_level=0.05):
        """Add random noise to image"""
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    def generate_pattern(self, pattern_type, params):
        """Generate specific defect pattern"""
        base = self.generate_base()
        center = self.size // 2
        
        if pattern_type == 'normal':
            # Just add very slight noise to the base
            return self.add_noise(base, params['noise_level'])
        
        elif pattern_type == 'circular':
            radius = int(self.size * params['radius'])
            cv2.circle(base, (center, center), radius, 255, -1)
            
        elif pattern_type == 'ring':
            outer_radius = int(self.size * params['outer_radius'])
            inner_radius = int(self.size * params['inner_radius'])
            cv2.circle(base, (center, center), outer_radius, 255, -1)
            cv2.circle(base, (center, center), inner_radius, 0, -1)
            
        elif pattern_type == 'edge':
            mask = np.zeros_like(base)
            width = int(self.size * params['width'])
            cv2.circle(mask, (center, center), center, 255, width)
            base[mask > 0] = 255
            
        elif pattern_type == 'edge_ring':
            mask = np.zeros_like(base)
            width = int(self.size * params['width'])
            distance = int(self.size * params['distance'])
            cv2.circle(mask, (center, center), center - distance, 255, width)
            base[mask > 0] = 255
            
        elif pattern_type == 'spots':
            num_spots = int(params['num_spots'])  # Convert to int
            for _ in range(num_spots):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0.3, 0.7) * center
                x = int(center + dist * np.cos(angle))
                y = int(center + dist * np.sin(angle))
                radius = int(self.size * params['radius'])
                cv2.circle(base, (x, y), radius, 255, -1)
                
        elif pattern_type == 'random':
            num_patches = int(params['num_patches'])  # Convert to int
            min_size = int(params['min_size'])  # Convert to int
            max_size = int(params['max_size'])  # Convert to int
            for _ in range(num_patches):
                size = np.random.randint(min_size, max_size + 1)
                x = np.random.randint(0, self.size - size)
                y = np.random.randint(0, self.size - size)
                cv2.rectangle(base, (x, y), (x + size, y + size), 255, -1)
            
        elif pattern_type == 'line':
            num_lines = np.random.randint(params['num_lines'][0], params['num_lines'][1] + 1)
            thickness = int(params['thickness'])  # Convert to int
            for _ in range(num_lines):
                angle = np.random.uniform(0, 360)
                length = int(self.size * 0.8)  # Long scratch
                angle_rad = np.radians(angle)
                dx = int(length * np.cos(angle_rad) / 2)
                dy = int(length * np.sin(angle_rad) / 2)
                cv2.line(base, 
                        (center - dx, center - dy),
                        (center + dx, center + dy),
                        255, thickness)
            
        elif pattern_type == 'coverage':
            mask = np.random.random(base.shape) < params['coverage']
            base[mask] = 255
        
        # Add slight noise for realism
        base = self.add_noise(base)
        return base

def test_generation(category_name=None, output_dir="output/pattern_tests"):
    """Test the pattern generation"""
    try:
        print("\n🚀 Starting pattern generation test...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generator with correct size
        generator = PatternGenerator(size=52)  # Match original input_size
        
        # Process specified category or all categories
        categories = [category_name] if category_name else list(WAFER_CATEGORIES.keys())
        
        for category in categories:
            if category not in WAFER_CATEGORIES:
                print(f"⚠️ Unknown category: {category}")
                continue
            
            print(f"\n📊 Testing category: {category}")
            info = WAFER_CATEGORIES[category]
            
            # Generate variations with slightly different parameters
            variations = []
            for i in range(3):  # Generate 3 variations
                params = info['params'].copy()
                # Add some random variation to parameters (except for normal wafers)
                if category != 'none':
                    for key in params:
                        if isinstance(params[key], (int, float)):
                            params[key] *= np.random.uniform(0.8, 1.2)
                
                pattern = generator.generate_pattern(info['pattern_type'], params)
                variations.append(pattern)
            
            # Create visualization
            fig, axes = plt.subplots(1, len(variations), figsize=(15, 5))
            if len(variations) == 1:
                axes = [axes]
            
            for i, var in enumerate(variations):
                axes[i].imshow(var, cmap='gray')
                axes[i].set_title(f'Variation {i+1}')
                axes[i].axis('off')
                
                # Save individual variation
                cv2.imwrite(str(output_dir / f"{category}_variation_{i}.png"), var)
            
            plt.suptitle(f"Category: {category}\n{info['description']}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{category}_summary.png")
            plt.close()
            
            print(f"✅ Generated {len(variations)} variations for {category}")
        
        print("\n✨ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nError details:")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Test all categories
    test_generation() 