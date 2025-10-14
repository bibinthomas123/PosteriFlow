import torch
import zipfile
import io
import pickle

def load_checkpoint_zipfile(path):
    """Load PyTorch checkpoint that's actually a ZIP file."""
    print(f"Loading: {path}")
    
    try:
        # Method 1: Standard torch.load with pickle_module
        checkpoint = torch.load(path, map_location='cpu', pickle_module=pickle)
        print("✅ Loaded with torch.load + pickle_module")
        return checkpoint
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
    try:
        # Method 2: Load as zipfile then extract
        with zipfile.ZipFile(path, 'r') as zf:
            # List contents
            print(f"ZIP contents: {zf.namelist()}")
            
            # Read data.pkl
            with zf.open('final_model/data.pkl') as f:
                data = f.read()
            
            # Unpickle with torch
            checkpoint = torch.load(io.BytesIO(data), map_location='cpu')
            print("✅ Loaded via ZIP extraction")
            return checkpoint
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
    
    try:
        # Method 3: Use older PyTorch loading
        import sys
        import types
        
        # Create a custom unpickler
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        
        with open(path, 'rb') as f:
            checkpoint = CPU_Unpickler(f).load()
        print("✅ Loaded with custom unpickler")
        return checkpoint
    except Exception as e3:
        print(f"Method 3 failed: {e3}")
    
    return None

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/neural_pe/final_model.pth'
    
    checkpoint = load_checkpoint_zipfile(path)
    
    if checkpoint:
        print(f"\n✅ SUCCESS!")
        print(f"Type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print(f"Model layers: {len(checkpoint['model_state_dict'])}")
    else:
        print("\n❌ All methods failed")
