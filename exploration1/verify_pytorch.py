import torch
import torchvision
import torchaudio

def verify_pytorch():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"TorchAudio Version: {torchaudio.__version__}")
    print("\nCUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"\nCUDA device properties:")
        print(torch.cuda.get_device_properties(0))

if __name__ == "__main__":
    verify_pytorch() 