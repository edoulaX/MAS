"""
Wafer Map Failure Pattern Recognition
This script implements machine learning techniques for semiconductor wafer failure pattern recognition.
The goal is to automatically identify different types of wafer map failure patterns to improve semiconductor fabrication yield.

References:
[1] Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets
[2] Wafer Defect Patterns Recognition Based on OPTICS and Multi-Label Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from skimage import measure
from skimage.transform import radon, probabilistic_hough_line
from scipy import interpolate, stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
from gpu_utils import setup_gpu_environment

warnings.filterwarnings("ignore")

# Initialize GPU environment
gpu_manager = setup_gpu_environment()

def load_data():
    """Load and return both the pandas dataframe and numpy array datasets"""
    # Load pandas dataframe
    df = pd.read_pickle("./input/WM811K.pkl")
    
    # Filter out unlabeled data
    df_withlabel = df.dropna(subset=['failureType'])
    
    # Filter out non-pattern data
    df_withpattern = df_withlabel[df_withlabel.failureType != 'none']
    df_nonpattern = df_withlabel[df_withlabel.failureType == 'none']
    
    return df, df_withlabel, df_withpattern, df_nonpattern

def calculate_density(x):
    """Calculate density of defects in a region"""
    return 100 * (np.sum(x==2) / np.size(x))

def find_regions(x):
    """Find and calculate density features for 13 regions of the wafer"""
    rows = np.size(x, axis=0)
    cols = np.size(x, axis=1)
    ind1 = np.arange(0, rows, rows//5)
    ind2 = np.arange(0, cols, cols//5)
    
    # Extract regions
    reg1 = x[ind1[0]:ind1[1],:]
    reg3 = x[ind1[4]:,:]
    reg4 = x[:,ind2[0]:ind2[1]]
    reg2 = x[:,ind2[4]:]

    reg5 = x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6 = x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7 = x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8 = x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9 = x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10 = x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11 = x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12 = x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13 = x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
    # Calculate density features
    fea_reg_den = [
        calculate_density(reg) for reg in [
            reg1, reg2, reg3, reg4, reg5, reg6, reg7,
            reg8, reg9, reg10, reg11, reg12, reg13
        ]
    ]
    return fea_reg_den

def extract_features(wafer_map):
    """Extract features from wafer map"""
    # Extract density-based features
    density_features = find_regions(wafer_map)
    
    # Calculate statistical features
    flat_map = wafer_map.flatten()
    stat_features = [
        np.mean(flat_map),
        np.std(flat_map),
        np.median(flat_map),
        stats.skew(flat_map),
        stats.kurtosis(flat_map)
    ]
    
    # Combine all features
    features = density_features + stat_features
    
    return features

def prepare_data(df_withpattern):
    """Prepare data for training by extracting features"""
    # Extract features for each wafer map
    features = []
    labels = []
    
    for idx, row in df_withpattern.iterrows():
        # Extract features
        wafer_features = extract_features(row.waferMap)
        features.append(wafer_features)
        
        # Get label
        labels.append(row.failureNum)
    
    return np.array(features), np.array(labels)

def visualize_patterns(df_withpattern):
    """Visualize different wafer failure patterns"""
    failure_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.ravel(order='C')
    
    for i, failure_type in enumerate(failure_types):
        # Find first example of each failure type
        example = df_withpattern[df_withpattern.failureType.apply(lambda x: x[0][0] == failure_type)].iloc[0]
        
        # Plot wafer map
        axes[i].imshow(example.waferMap)
        axes[i].set_title(failure_type, fontsize=24)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def train_and_evaluate(X, y):
    """Train and evaluate the model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    print("\nPlotting Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, classes=np.unique(y))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': clf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return clf, scaler

class DiffusionGenerator:
    """Handles diffusion model generation with GPU optimizations"""
    
    def __init__(self, config=PipelineConfig):
        self.config = config
        self.device = gpu_manager.device  # Use GPU manager's device
        print(f"🖥️ Using: {self.device}")
    
    def initialize(self, token=None):
        """Initialize the diffusion model with optimizations"""
        try:
            print(f"🔄 Loading model {self.config.DIFFUSION_PARAMS['model_id']}...")
            
            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.DIFFUSION_PARAMS['model_id'],
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_auth_token=token
            ).to(self.device)
            
            if self.device.type == 'cuda':
                # GPU optimizations
                self.model.enable_attention_slicing(slice_size="auto")
                self.model.enable_vae_slicing()
                
                # Enable memory efficient attention if available
                if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
                    self.model.enable_xformers_memory_efficient_attention()
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("\nError details:")
            import traceback
            print(traceback.format_exc())
            return False

class VITClassifier:
    """Handles VIT classification with GPU support"""
    
    def __init__(self, config=PipelineConfig):
        self.config = config
        self.device = gpu_manager.device
        self.model = ViTForImageClassification.from_pretrained(
            config.VIT_PARAMS['model_id']
        ).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(
            config.VIT_PARAMS['model_id']
        )
        
        # Optimize for inference
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Use FP16 for faster inference
    
    def classify(self, image):
        """Classify an image and return probabilities"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return probs[0].cpu().numpy()

def main():
    # Load data
    print("Loading data...")
    df, df_withlabel, df_withpattern, df_nonpattern = load_data()
    
    # Print dataset statistics
    total_wafers = len(df)
    labeled_wafers = len(df_withlabel)
    pattern_wafers = len(df_withpattern)
    
    print(f"\nDataset Statistics:")
    print(f"Total wafers: {total_wafers}")
    print(f"Labeled wafers: {labeled_wafers}")
    print(f"Pattern wafers: {pattern_wafers}")
    
    # Visualize patterns
    print("\nVisualizing wafer patterns...")
    visualize_patterns(df_withpattern)
    
    # Prepare data for training
    print("\nExtracting features...")
    X, y = prepare_data(df_withpattern)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train and evaluate model
    clf, scaler = train_and_evaluate(X, y)
    
    # Example usage
    config = PipelineConfig()
    
    try:
        print("\n🚀 Starting test with original wafer maps...")
        
        # Initialize the diffusion generator
        generator = DiffusionGenerator(config)
        if not generator.initialize(token="YOUR_TOKEN_HERE"):
            raise ValueError("Failed to initialize model")
        
        # Load the wafer map dataset
        data_path = Path('input')
        print("\n📂 Loading wafer map dataset...")
        
        try:
            # Load data with GPU optimization
            data = np.load(data_path / 'Wafer_Map_Datasets.npz')
            wafer_maps = gpu_manager.to_device(data['arr_0'])  # Move to GPU
            labels = gpu_manager.to_device(data['arr_1'])      # Move to GPU
            
            # Convert one-hot labels to category indices
            label_indices = torch.argmax(labels, dim=1).cpu().numpy()
            categories = list(config.WAFER_CATEGORIES.keys())
            
            print(f"✅ Loaded {len(wafer_maps)} wafer maps")
            
            # Process batches efficiently
            batch_size = gpu_manager.get_batch_size(model_size_mb=500)  # Adjust based on your model
            num_samples = 3
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                print(f"\n📊 Processing batch {i//batch_size + 1}")
                
                # Process batch
                indices = np.random.randint(0, len(wafer_maps), size=batch_end - i)
                batch_maps = wafer_maps[indices].cpu().numpy()
                batch_categories = [categories[label_indices[idx]] for idx in indices]
                
                for j, (wafermap, category) in enumerate(zip(batch_maps, batch_categories)):
                    print(f"\n🔄 Processing wafer map {i+j+1}/{num_samples} (Category: {category})")
                    
                    # Preprocess the wafer map
                    wafermap = (wafermap - wafermap.min()) / (wafermap.max() - wafermap.min()) * 255
                    wafermap = wafermap.astype(np.uint8)
                    wafermap_rgb = cv2.cvtColor(wafermap, cv2.COLOR_GRAY2RGB)
                    
                    # Generate variations
                    variations = generator.generate(wafermap_rgb)
                    
                    if variations:
                        # Display results
                        fig, axes = plt.subplots(1, len(variations) + 1, figsize=(15, 5))
                        if len(variations) == 1:
                            axes = [axes]
                        
                        axes[0].imshow(wafermap, cmap='gray')
                        axes[0].set_title('Original Wafer Map')
                        axes[0].axis('off')
                        
                        for k, var in enumerate(variations):
                            axes[k + 1].imshow(var)
                            axes[k + 1].set_title(f'AI Generation {k + 1}')
                            axes[k + 1].axis('off')
                        
                        plt.suptitle(f"Wafer Map Category: {category}")
                        plt.tight_layout()
                        plt.show()
                    
                    print(f"✅ Processing completed for wafer map {i+j+1}")
                
                # Clear GPU cache after each batch
                gpu_manager.clear_gpu_memory()
            
            print("\n✨ All generations completed successfully!")
            
        except FileNotFoundError:
            print("❌ Error: Wafer map dataset not found in 'input' directory")
            print("Please ensure 'Wafer_Map_Datasets.npz' is in the input directory")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nError details:")
        import traceback
        print(traceback.format_exc())
        
    finally:
        # Clean up GPU memory
        gpu_manager.clear_gpu_memory()

if __name__ == "__main__":
    main() 