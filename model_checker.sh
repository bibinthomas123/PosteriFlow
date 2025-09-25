#!/bin/bash
# Phase 2 Model Checker - ARGUMENT PASSING BUG FIXED
# Usage: ./model_checker_fixed.sh /path/to/model.pth

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 /path/to/model.pth"
    exit 1
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}🔍 PHASE 2 PYTORCH CHECKPOINT ANALYZER (Fixed)${NC}"
echo -e "${BLUE}======================================================${NC}"

# Basic file checks
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ File not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ File found: $MODEL_PATH${NC}"
echo -e "   📁 Size: $(du -h "$MODEL_PATH" | cut -f1)"
echo -e "   📅 Modified: $(stat -c %y "$MODEL_PATH" 2>/dev/null || stat -f %Sm "$MODEL_PATH")"

# Python analysis - FIXED ARGUMENT PASSING
python3 << EOF
import torch
import sys
import traceback
from pathlib import Path

def safe_print(obj, max_items=5, indent="     "):
    """Safely print object contents"""
    if isinstance(obj, dict):
        items = list(obj.items())[:max_items]
        for k, v in items:
            if isinstance(v, torch.Tensor):
                print(f"{indent}• {k}: Tensor{list(v.shape)} ({v.dtype})")
            elif isinstance(v, (int, float)):
                print(f"{indent}• {k}: {v}")
            elif isinstance(v, str):
                print(f"{indent}• {k}: '{v[:50]}{'...' if len(v) > 50 else ''}'")
            elif isinstance(v, list):
                print(f"{indent}• {k}: List[{len(v)} items]")
            elif isinstance(v, dict):
                print(f"{indent}• {k}: Dict[{len(v)} keys]")
                if len(v) <= 3:
                    safe_print(v, max_items=3, indent=indent+"  ")
            else:
                print(f"{indent}• {k}: {type(v).__name__}")
        if len(obj) > max_items:
            print(f"{indent}• ... and {len(obj) - max_items} more")

# FIXED: Use the passed model path directly
model_path = "$MODEL_PATH"
print(f"\n🔍 Analyzing checkpoint: {Path(model_path).name}")

try:
    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print("✅ Checkpoint loaded successfully")
    
    # Basic structure
    keys = list(checkpoint.keys())
    print(f"\n📋 Checkpoint structure ({len(keys)} keys):")
    for key in keys:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"   • {key}: Dict with {len(value)} keys")
            elif isinstance(value, torch.Tensor):
                print(f"   • {key}: Tensor{list(value.shape)}")
            else:
                print(f"   • {key}: {type(value).__name__}")
    
    print("\n" + "="*50)
    
    # Analyze each component
    score = 0
    max_score = 100
    
    # 1. Model weights (most important)
    if 'model_state_dict' in checkpoint:
        print("🏗️  MODEL WEIGHTS ANALYSIS:")
        state_dict = checkpoint['model_state_dict']
        print(f"   ✅ Found model_state_dict with {len(state_dict)} parameters")
        
        # Calculate total parameters
        total_params = 0
        tensor_count = 0
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                total_params += param.numel()
                tensor_count += 1
        
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🔢 Parameter tensors: {tensor_count}")
        
        # Show sample parameters
        print("   📝 Sample parameters:")
        safe_print(dict(list(state_dict.items())[:5]))
        
        score += 60
    else:
        print("❌ No model_state_dict found!")
    
    # 2. Enhanced final_results analysis
    if 'final_results' in checkpoint:
        print(f"\n📊 FINAL RESULTS ANALYSIS:")
        final_results = checkpoint['final_results']
        print(f"   ✅ Found final_results: {type(final_results).__name__}")
        
        # Training metrics
        if 'training_metrics' in final_results:
            training_metrics = final_results['training_metrics']
            print(f"   📈 Training metrics found:")
            
            if isinstance(training_metrics, dict):
                # Check for key training info
                if 'epochs_completed' in training_metrics:
                    epochs = training_metrics['epochs_completed']
                    print(f"     • Epochs completed: {epochs}")
                    score += 10
                
                if 'losses' in training_metrics:
                    losses = training_metrics['losses']
                    if isinstance(losses, list) and len(losses) > 0:
                        final_loss = losses[-1]
                        best_loss = min(losses)
                        print(f"     • Final loss: {final_loss:.6f}")
                        print(f"     • Best loss: {best_loss:.6f}")
                        
                        # Assess loss quality
                        if final_loss < 0.1:
                            print("     🏆 EXCELLENT final loss!")
                            score += 15
                        elif final_loss < 0.5:
                            print("     ✅ GOOD final loss")
                            score += 10
                        elif final_loss < 1.0:
                            print("     ⚠️  ACCEPTABLE final loss")
                            score += 5
                        else:
                            print("     ❌ HIGH final loss")
                
                # Show training metrics summary
                metrics_keys = list(training_metrics.keys())[:5]
                print(f"     • Available metrics: {metrics_keys}")
        
        # Evaluation metrics
        if 'evaluation_metrics' in final_results:
            eval_metrics = final_results['evaluation_metrics']
            print(f"   🎯 Evaluation metrics:")
            safe_print(eval_metrics)
            score += 10
        
        # Model config
        if 'model_config' in final_results:
            model_config = final_results['model_config']
            print(f"   ⚙️  Model configuration:")
            safe_print(model_config)
            score += 5
        
        score += 10
    
    # 3. Legacy structure checks
    elif 'epoch' in checkpoint or 'loss' in checkpoint:
        print(f"\n📈 LEGACY TRAINING INFO:")
        
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            print(f"   ✅ Completed epochs: {epoch}")
            score += 10
        
        if 'loss' in checkpoint:
            loss = checkpoint['loss']
            print(f"   📉 Final loss: {loss:.6f}")
            
            if loss < 0.1:
                print("   🏆 EXCELLENT loss value!")
                score += 10
            elif loss < 0.5:
                print("   ✅ GOOD loss value")
                score += 5
            else:
                print("   ⚠️  HIGH loss value")
    
    # 4. Configuration checks
    if 'config' in checkpoint:
        print(f"\n⚙️  STANDALONE CONFIGURATION:")
        config = checkpoint['config']
        print(f"   ✅ Config found: {type(config).__name__}")
        safe_print(config)
        score += 5
    
    # 5. Optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print(f"\n🎛️  OPTIMIZER STATE:")
        opt_state = checkpoint['optimizer_state_dict']
        print(f"   ✅ Optimizer state saved ({len(opt_state)} keys)")
        score += 5
    
    # Overall assessment
    print("\n" + "="*50)
    print("🎯 OVERALL ASSESSMENT:")
    print(f"   📊 Compatibility Score: {score}/{max_score}")
    
    if score >= 85:
        status = "🏆 EXCELLENT"
        recommendation = "Perfect for Phase 3! ✅"
    elif score >= 70:
        status = "✅ GOOD"
        recommendation = "Ready for Phase 3 ✅"
    elif score >= 50:
        status = "⚠️  ACCEPTABLE"
        recommendation = "Usable for Phase 3 with setup ⚙️"
    else:
        status = "❌ POOR"
        recommendation = "Missing critical components ❌"
    
    print(f"   Status: {status}")
    print(f"   {recommendation}")
    
    # Phase 3 usage instructions
    print(f"\n💡 PHASE 3 LOADING CODE:")
    if 'model_state_dict' in checkpoint:
        print("   checkpoint = torch.load('your_model.pth')")
        print("   model.load_state_dict(checkpoint['model_state_dict'])")
        
        if 'final_results' in checkpoint and 'model_config' in checkpoint['final_results']:
            print("   config = checkpoint['final_results']['model_config']")
        elif 'config' in checkpoint:
            print("   config = checkpoint['config']")
        else:
            print("   # Note: Recreate model architecture manually")
    
    print("=" * 50)

except Exception as e:
    print(f"❌ Error analyzing checkpoint: {e}")
    traceback.print_exc()
    sys.exit(1)

EOF

echo ""
echo -e "${GREEN}✅ Model analysis completed!${NC}"
