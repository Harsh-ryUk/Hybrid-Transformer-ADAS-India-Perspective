import torch
from transformers import SegformerForSemanticSegmentation
import argparse
import os

def export_segformer(model_name: str, output_path: str, opset: int = 12):
    """
    Exports SegFormer-B0 model to ONNX format.
    """
    print(f"🔄 Loading model: {model_name}...")
    # Load pretrained model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2, # Road vs Not Road
        ignore_mismatched_sizes=True
    )
    model.eval()
    
    # Dummy Input: 1, 3, 360, 640 (Jetson Friendly Resolution)
    dummy_input = torch.randn(1, 3, 360, 640)
    
    # Export
    print(f"📦 Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print("✅ ONNX Export Successful!")
    print(f"   Model Size: {os.path.getsize(output_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--output", type=str, default="src/production_lane_detection/models/segformer_b0.onnx")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_segformer(args.model, args.output)
