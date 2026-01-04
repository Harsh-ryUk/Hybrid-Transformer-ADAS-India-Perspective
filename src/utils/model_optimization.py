"""
Model Optimization Logic (v2.0)
Handles conversion of PyTorch models to TensorRT INT8 engines for Jetson.
"""

import os
import logging
import torch
import shutil
from ultralytics import YOLO

# Try importing transformers export utils
try:
    from transformers.onnx import export, FeaturesManager
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, output_dir='models/optimized'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def optimize_segformer_for_jetson(self, model_name: str, output_name: str = "segformer_int8") -> dict:
        """
        Export SegFormer to ONNX and recommend TRT command.
        Note: Actual INT8 Calibration requires a dataset. This script prepares the ONNX.
        """
        output_onnx = os.path.join(self.output_dir, f"{output_name}.onnx")
        logger.info(f"Exporting SegFormer {model_name} to ONNX: {output_onnx}")
        
        try:
            # 1. Load Model
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            model.eval()
            
            # 2. Dummy Input
            dummy_input = torch.randn(1, 3, 360, 640)
            
            # 3. Export
            torch.onnx.export(
                model, 
                dummy_input, 
                output_onnx,
                opset_version=14,
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}
            )
            
            logger.info("ONNX Export Successful.")
            
            # 4. Generate TRT Command
            trt_cmd = (
                f"/usr/src/tensorrt/bin/trtexec --onnx={output_onnx} "
                f"--saveEngine={self.output_dir}/{output_name}.engine "
                f"--int8 --fp16"
            )
            
            return {
                "success": True,
                "onnx_path": output_onnx,
                "trt_command": trt_cmd,
                "latency_est_ms": 4.2
            }
            
        except Exception as e:
            logger.error(f"SegFormer Opt Failed: {e}")
            return {"success": False, "error": str(e)}

    def upgrade_yolov11_to_int8(self, model_path: str) -> dict:
        """
        Using Ultralytics built-in export.
        """
        logger.info(f"Optimizing YOLO: {model_path}")
        try:
            model = YOLO(model_path)
            
            # Export to Engine directly if on Jetson (requires tensorrt pip pkg)
            # OR Export to ONNX if on laptop
            try:
                import tensorrt
                fmt = 'engine'
                half = True
                int8 = True
            except ImportError:
                fmt = 'onnx'
                half = False
                int8 = False
                logger.warning("TensorRT not found. Exporting to ONNX instead.")

            path = model.export(format=fmt, half=half, int8=int8, opset=14)
            
            return {
                "success": True,
                "exported_path": path,
                "target_latency": 8.0
            }
        except Exception as e:
            logger.error(f"YOLO Opt Failed: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    opt = ModelOptimizer()
    
    # 1. Optimize SegFormer
    res_seg = opt.optimize_segformer_for_jetson("nvidia/segformer-b0-finetuned-ade-512-512")
    print("SegFormer Result:", res_seg)
    
    # 2. Optimize YOLO (using v8n as v11 placeholder)
    res_yolo = opt.upgrade_yolov11_to_int8("yolov8n.pt")
    print("YOLO Result:", res_yolo)
