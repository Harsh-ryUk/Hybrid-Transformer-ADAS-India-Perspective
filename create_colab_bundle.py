import zipfile
import os

def create_bundle():
    output_filename = "ADAS_Bundle.zip"
    
    # Files/Dirs to include
    include_files = [
        "requirements.txt",
        "README.md",
        "benchmark.py",
        "download_indian_sample.py",
        "run.py",
        "train.py",
        "TRAINING.md"
    ]
    include_dirs = ["src", "data"]
    
    print(f"Creating {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add Files
        for f in include_files:
            if os.path.exists(f):
                zipf.write(f)
                
        # Add Dirs
        for d in include_dirs:
            for root, dirs, files in os.walk(d):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Avoid recursive zip (if zip is in data)
                    if file_path == output_filename: continue
                    # Avoid cached pyc
                    if "__pycache__" in file_path: continue
                    if file.endswith(".pyc"): continue
                    
                    zipf.write(file_path)
                    
    print("Bundle created successfully.")

if __name__ == "__main__":
    create_bundle()
