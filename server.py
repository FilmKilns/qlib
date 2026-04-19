import sys
import os
import io
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

def execute_python_code(code: str) -> dict:
    """Safely execute Python code with restricted environment and capture console output"""
    
    # Create a string buffer to capture print output
    output_buffer = io.StringIO()
    
    # Create a restricted environment for code execution
    safe_globals = {
        '__builtins__': {
            'print': lambda *args, **kwargs: print(*args, **kwargs, file=output_buffer),
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'isinstance': isinstance,
            'type': type,
            'bool': bool,
            'Exception': Exception,
            'ImportError': ImportError,
            'ValueError': ValueError,
            'TypeError': TypeError,
        }
    }
    
    # Add qlib-specific modules if available
    try:
        import qlib
        safe_globals['qlib'] = qlib
    except ImportError:
        pass
    
    try:
        import pandas as pd
        safe_globals['pd'] = pd
    except ImportError:
        pass
    
    try:
        import numpy as np
        safe_globals['np'] = np
    except ImportError:
        pass
    
    # Create a local namespace for the code
    safe_locals = {}
    
    try:
        # Execute the code
        exec(code, safe_globals, safe_locals)
        
        # Get the captured output
        console_output = output_buffer.getvalue()
        
        # Return any variables that were created
        result_vars = {}
        for var_name, var_value in safe_locals.items():
            if not var_name.startswith('_'):
                # Convert to string representation for safety
                try:
                    result_vars[var_name] = str(var_value)
                except:
                    result_vars[var_name] = f"<unable to represent {type(var_value)}>"
        
        return console_output
        
    except Exception as e:
        return str(e)
    finally:
        output_buffer.close()

class TrainRequest(BaseModel):
    market_data_url: str

@app.post("/train/prepare_data")
def do_train_prepare_data(req: TrainRequest):
    if req.market_data_url is None or req.market_data_url == '':
        return {"message": "market_data_url is required", "status": -1}
    
    # Step 1: Download and extract data
    filename = download_and_extract_data(req.market_data_url)
    if filename is None or filename == '':
        return {"message": "download_and_extract_data failed", "status": -1}

    filename = filename.split(".")[0]
    
    # Step 2: Run qlib dump_all command
    cmd = f"python scripts/dump_bin.py dump_all --data_path ~/.qlib/src_data/{filename}/features/ --qlib_dir ~/.qlib/qlib_data/{filename} --include_fields open,high,low,close,volume,factor"
    result = os.system(cmd)
    if result != 0:
        return {"message": "qlib dump_all failed", "status": -1}

    uri = f"~/.qlib/qlib_data/{filename}"
    expanded_path = os.path.expanduser(uri)
    if os.path.exists(expanded_path) and os.path.isdir(expanded_path):
        pass
    else:
        return {
            "message": f"Train data prepare failed. {uri} not found", 
            "status": -1
        }
    
    return {
        "message": "Train data prepare success", 
        "status": 0,
        "qlib_init": {
            "provider_uri": uri
        }
    }

class TrainExecRequest(BaseModel):
    code: str

@app.post("/train/exec")
def do_train_exec(req: TrainExecRequest):
    if req.code is None or req.code == '':
        return {"message": "code is required", "status": -1}
    console_output = execute_python_code(req.code)
    return {
        "message": "Train code exec success", 
        "status": 0,
        "train_result": console_output
    }

if __name__ == "__main__":
    result = execute_python_code('print("Hello, World!")')
    # result = execute_python_code('raise Exception("Test Exception")')
    print(f"execute_python_code: {result}")
