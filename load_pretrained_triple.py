#!/usr/bin/env python3
"""
Utility script for loading pretrained YOLOv12 weights into triple input models.

This script provides utilities to:
1. Load standard YOLOv12 pretrained weights
2. Initialize triple input models with pretrained weights
3. Fine-tune triple input models from pretrained checkpoints

Usage:
    python load_pretrained_triple.py --pretrained yolov12n.pt --model ultralytics/cfg/models/v12/yolov12_triple.yaml
    python load_pretrained_triple.py --pretrained yolov12s.pt --model ultralytics/cfg/models/v12/yolov12_triple.yaml --save yolov12s_triple_pretrained.pt
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules.conv import TripleInputConv

def load_pretrained_weights_to_triple_model(pretrained_path, triple_model_config, save_path=None, variant='m', integrate='initial'):
    """
    Load pretrained YOLOv12 weights into a triple input model.

    Args:
        pretrained_path (str): Path to pretrained YOLOv12 model (.pt file)
        triple_model_config (str): Path to triple input model configuration
        save_path (str, optional): Path to save the initialized model
        variant (str): Model scale n/s/m/l/x (default 'm')

    Returns:
        YOLO: Triple input model with pretrained weights loaded
    """
    import yaml
    print(f"Loading pretrained weights from: {pretrained_path}")
    print(f"Triple model config: {triple_model_config}")

    # Scale params for reference / fallback
    _scale_params = {
        'n': (0.50, 0.25, 1024),
        's': (0.50, 0.50, 1024),
        'm': (0.50, 1.00,  512),
        'l': (1.00, 1.00,  512),
        'x': (1.00, 1.50,  512),
    }
    depth, width, max_ch = _scale_params[variant]
    with open(triple_model_config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('depth_multiple') is not None:
        # Already has explicit depth/width — use file as-is, no temp needed
        print(f"Using scale='{variant}' from {triple_model_config} (depth_multiple already set)")
    else:
        # Set scale key so parse_model uses scales[variant] correctly (preserves C3k2/A2C2f flags)
        cfg['scale'] = variant
        tmp_yaml = f"yolov12{variant}_triple_pretrained.yaml"
        with open(tmp_yaml, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        triple_model_config = tmp_yaml
        print(f"Using scale='{variant}' via {tmp_yaml}")

    # Load pretrained model to get weights
    try:
        pretrained_checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f"✓ Loaded pretrained checkpoint")
        
        # Extract state dict
        if 'model' in pretrained_checkpoint:
            pretrained_state_dict = pretrained_checkpoint['model'].state_dict() if hasattr(pretrained_checkpoint['model'], 'state_dict') else pretrained_checkpoint['model']
        else:
            pretrained_state_dict = pretrained_checkpoint
            
    except Exception as e:
        raise Exception(f"Failed to load pretrained model: {e}")
    
    # Create triple input model
    print("Creating triple input model...")
    triple_model = YOLO(triple_model_config)
    
    # Get the triple model's state dict
    triple_state_dict = triple_model.model.state_dict()
    
    # Load weights layer by layer
    loaded_layers = []
    skipped_layers = []
    
    import re

    def _offset_name(name, offset):
        """Remap model.N.xxx → model.(N-offset).xxx"""
        m = re.match(r'model\.(\d+)\.(.*)', name)
        if m:
            n = int(m.group(1))
            if n >= 6:
                return f'model.{n - offset}.{m.group(2)}'
        return None

    for name, param in triple_state_dict.items():
        loaded = False
        # Direct match
        if name in pretrained_state_dict:
            pretrained_param = pretrained_state_dict[name]
            if param.shape == pretrained_param.shape:
                param.copy_(pretrained_param)
                loaded_layers.append(name)
                loaded = True

        if not loaded:
            # TripleInputConv: copy pretrained layer 0 weights to conv1/conv2/conv3
            if 'model.0.' in name and any(x in name for x in ['conv1', 'conv2', 'conv3']):
                base_name = name.replace('conv1.', '').replace('conv2.', '').replace('conv3.', '')
                if base_name in pretrained_state_dict:
                    pretrained_param = pretrained_state_dict[base_name]
                    if param.shape == pretrained_param.shape:
                        param.copy_(pretrained_param)
                        loaded_layers.append(f"{name} <- {base_name}")
                        loaded = True

            # p3 offset mapping: p3 layer N (N>=6) = pretrained layer N-1
            if not loaded and integrate == 'p3':
                offset_name = _offset_name(name, 1)
                if offset_name and offset_name in pretrained_state_dict:
                    pretrained_param = pretrained_state_dict[offset_name]
                    if param.shape == pretrained_param.shape:
                        param.copy_(pretrained_param)
                        loaded_layers.append(f"{name} <- {offset_name}")
                        loaded = True

        if not loaded:
            skipped_layers.append(f"{name}: not found in pretrained model")
    
    print(f"\n✓ Loaded {len(loaded_layers)} layers from pretrained model")
    print(f"⚠ Skipped {len(skipped_layers)} layers")
    
    if len(loaded_layers) > 0:
        print("\nLoaded layers:")
        for layer in loaded_layers[:10]:  # Show first 10
            print(f"  {layer}")
        if len(loaded_layers) > 10:
            print(f"  ... and {len(loaded_layers) - 10} more")
    
    if len(skipped_layers) > 0:
        print("\nSkipped layers:")
        for layer in skipped_layers[:5]:  # Show first 5
            print(f"  {layer}")
        if len(skipped_layers) > 5:
            print(f"  ... and {len(skipped_layers) - 5} more")
    
    # Save the initialized model if requested
    if save_path:
        print(f"\nSaving initialized model to: {save_path}")
        torch.save({
            'model': triple_model.model,
            'optimizer': None,
            'training_results': None,
            'epoch': 0,
            'date': None,
            'version': None
        }, save_path)
        print("✓ Model saved successfully")
    
    return triple_model

def create_pretrained_triple_conv(pretrained_path, c1=9, c2=64, **kwargs):
    """
    Create a TripleInputConv layer with pretrained weights.
    
    Args:
        pretrained_path (str): Path to pretrained YOLOv12 model
        c1 (int): Input channels (should be 9)
        c2 (int): Output channels
        **kwargs: Additional arguments for TripleInputConv
        
    Returns:
        TripleInputConv: Layer with pretrained weights
    """
    return TripleInputConv.from_pretrained(c1, c2, pretrained_path, **kwargs)

def validate_weight_loading(pretrained_path, triple_model_config):
    """
    Validate that weight loading works correctly.
    
    Args:
        pretrained_path (str): Path to pretrained model
        triple_model_config (str): Path to triple model config
        
    Returns:
        bool: True if validation passes
    """
    print("Validating weight loading...")
    
    try:
        # Test model creation
        model = load_pretrained_weights_to_triple_model(pretrained_path, triple_model_config)
        
        # Test forward pass
        test_input = torch.randn(1, 9, 640, 640)
        with torch.no_grad():
            output = model.model(test_input)
        
        print(f"✓ Validation passed - output shape: {[x.shape for x in output] if isinstance(output, (list, tuple)) else output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Load pretrained YOLOv12 weights into triple input model')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained YOLOv12 model (.pt file)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to triple input model configuration (.yaml file)')
    parser.add_argument('--save', type=str,
                       help='Path to save the initialized model (.pt file)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation test after loading weights')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.pretrained).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained}")
    
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model config not found: {args.model}")
    
    # Load weights
    model = load_pretrained_weights_to_triple_model(
        args.pretrained, 
        args.model, 
        args.save
    )
    
    # Run validation if requested
    if args.validate:
        print("\n" + "="*50)
        success = validate_weight_loading(args.pretrained, args.model)
        if not success:
            print("Validation failed!")
            return 1
    
    print("\n✓ Successfully loaded pretrained weights into triple input model")
    
    if args.save:
        print(f"✓ Model saved to: {args.save}")
        print(f"\nTo use the pretrained triple model:")
        print(f"  model = YOLO('{args.save}')")
        print(f"  model.train(data='test_triple_dataset.yaml', epochs=100)")
    else:
        print(f"\nTo save and use the model:")
        print(f"  model.save('yolov12_triple_pretrained.pt')")
        print(f"  model.train(data='test_triple_dataset.yaml', epochs=100)")
    
    return 0

if __name__ == "__main__":
    exit(main())
