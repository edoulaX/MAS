import torch
import numpy as np
from pathlib import Path
from simple_vit_pipeline import PipelineConfig, process_wafermap, DiffusionGenerator, ImageClassifier, NoveltyDetector

def test_single_wafer():
    print("\n🚀 Starting test with a single wafer map...")
    
    # Initialize configuration
    config = PipelineConfig()
    
    try:
        # Load test data
        data_path = Path('input/Wafer_Map_Datasets.npz')
        print("\n📂 Loading test wafer map...")
        
        data = np.load(data_path)
        wafer_maps = data['arr_0']  # Wafer maps
        labels = data['arr_1']      # One-hot encoded labels
        
        # Select a single wafer map for testing
        test_idx = 0
        test_wafer = wafer_maps[test_idx]
        true_label = np.argmax(labels[test_idx])
        
        print(f"\n✨ Processing wafer map (true label index: {true_label})")
        
        # Process the wafer map
        result = process_wafermap(test_wafer, config)
        
        # Print results
        print("\n📊 Results:")
        print(f"Final prediction: {result['final_prediction']}")
        print(f"Ensemble confidence: {result['ensemble_confidence']:.3f}")
        print(f"Is novel: {result['is_novel']}")
        print("\nPerformance metrics:")
        for key, value in result['performance_metrics'].items():
            print(f"- {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nError details:")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Check GPU availability
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run test
    result = test_single_wafer() 