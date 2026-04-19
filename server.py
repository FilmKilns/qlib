import sys
import os
import requests
import zipfile
import tarfile
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

def download_and_extract_data(market_data_url: str) -> dict:
    """Download file from URL and extract to ~/.qlib/src_data/ directory"""
        
    filename = f"qlib_data_from_agent_{int(datetime.now().timestamp() * 1000)}"
    try:
        target_dir = Path.home() / ".qlib" / "src_data" / filename
        
        # Create directory with proper permissions
        try:
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        except PermissionError:
            # If we can't create in home directory, try current directory
            target_dir = Path(".qlib") / "src_data" / filename
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        
        response = requests.get(market_data_url, stream=True)
        response.raise_for_status()

        filename = f"{filename}.zip"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                with tarfile.open(temp_file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
            elif filename.endswith('.tar'):
                with tarfile.open(temp_file_path, 'r:') as tar_ref:
                    tar_ref.extractall(target_dir)
            else:
                # If format is not recognized, try to copy the file directly
                import shutil
                shutil.copy(temp_file_path, target_dir / filename)
            
            os.unlink(temp_file_path)
            
            return filename
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            print(f"Error: {str(e)}")
            return None
            
    except requests.RequestException as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

class TrainRequest(BaseModel):
    market_data_url: str
    code: str

@app.post("/do_train")
def do_train(req: TrainRequest):
    if req.market_data_url is None or req.market_data_url == '':
        return {"message": "market_data_url is required", "status": -1}
    if req.code is None or req.code == '':
        return {"message": "code is required", "status": -1}
    
    filename = download_and_extract_data(req.market_data_url)
    if filename is None or filename == '':
        return {"message": "download_and_extract_data failed", "status": -1}

    filename = filename.split(".")[0]
    
    cmd = f"python scripts/dump_bin.py dump_all --data_path ~/.qlib/src_data/{filename}/features/ --qlib_dir ~/.qlib/qlib_data/{filename} --include_fields open,high,low,close,volume,factor"
    result = os.system(cmd)
    if result != 0:
        return {"message": "qlib dump_all failed", "status": -1}
    
    return {"message": "train success", "status": "success"}
