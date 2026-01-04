from ultralytics import YOLO
import argparse

def train_model(data_yaml="data/dataset.yaml", epochs=50, base_model="yolov8n.pt"):
    """
    Train YOLOv8 on custom Road Damage Dataset.
    """
    print(f"[INFO] Initializing model: {base_model}")
    model = YOLO(base_model)  # load a pretrained model (recommended for training)

    print(f"[INFO] Starting training on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='indian_road_damage',
        device=0, # Use GPU 0
        optimizer='AdamW'
    )
    
    print("[INFO] Training Complete.")
    print(f"[INFO] Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Export for deployment
    success = model.export(format='onnx')
    print(f"[INFO] Export status: {success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    args = parser.parse_args()
    train_model(data_yaml=args.data, epochs=args.epochs)
