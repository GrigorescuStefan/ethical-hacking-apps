from fastapi import Depends, FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import numpy as np
import joblib
import os
import time
import math
from pathlib import Path
from contextlib import contextmanager

app = FastAPI(title="Iris Classification API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

DATA_DIR = "data"
MODEL_DIR = "model"
USER_DATA_PATH = os.path.join(DATA_DIR, "user_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.pkl")
MAX_USER_SAMPLES = 10000 

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["label"] = iris.target

target_names = dict(enumerate(iris.target_names))

# ============================================================================
# API KEY CONFIGURATION 
# ============================================================================

ENV = os.getenv("IRIS_ENV", "production").lower()
print(f"[STARTUP] Running in '{ENV}' environment")

if ENV in ["development", "dev", "local"]:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[STARTUP] Loaded .env file from: {env_path}")
        else:
            print(f"[STARTUP] Warning: No .env file found")
            print("[STARTUP] Make sure IRIS_API_KEY is set as environment variable")
    except ImportError:
        print("[STARTUP] python-dotenv not installed, expecting environment variables")
    except Exception as e:
        print(f"[STARTUP] Error loading env file: {e}")

API_KEY = os.getenv("IRIS_API_KEY")
if not API_KEY:
    error_msg = (
        "IRIS_API_KEY environment variable is not set!\n"
        "For development: Create a .env file with IRIS_API_KEY=your-key\n"
        "For production: Set IRIS_API_KEY environment variable"
    )
    raise RuntimeError(error_msg)

VALID_API_KEYS = {k.strip() for k in API_KEY.split(",") if k.strip()}
if not VALID_API_KEYS:
    raise RuntimeError("No valid API keys found after parsing IRIS_API_KEY")

print(f"[STARTUP] ✓ API key authentication configured with {len(VALID_API_KEYS)} key(s)")

# ============================================================================
# FILE LOCKING MECHANISM
# ============================================================================

@contextmanager
def file_lock(filepath, timeout=10):
    """
    Context manager for file locking to prevent race conditions.
    Uses lock files to ensure only one process writes at a time.
    """
    lock_file = filepath + ".lock"
    start_time = time.time()
    
    while True:
        try:
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                yield
            finally:
                os.close(lock_fd)
                try:
                    os.unlink(lock_file)
                except:
                    pass 
            break
        except FileExistsError:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire file lock after {timeout} seconds")
            time.sleep(0.05) 

# ============================================================================
# API KEY VERIFICATION
# ============================================================================

def verify_api_key(x_api_key: str = Header(None)):
    """
    Verify the API key from the X-API-Key header.
    
    Usage: Include 'X-API-Key: your-key-here' in request headers
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Include X-API-Key header."
        )
    
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return x_api_key

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class IrisData(BaseModel):
    """
    Input validation with proper constraints.
    Prevents infinity, NaN, and out-of-range values.
    """
    sepal_length: float = Field(
        gt=0, 
        lt=100, 
        description="Sepal length in cm (must be between 0 and 100)"
    )
    sepal_width: float = Field(
        gt=0, 
        lt=100, 
        description="Sepal width in cm (must be between 0 and 100)"
    )
    petal_length: float = Field(
        gt=0, 
        lt=100, 
        description="Petal length in cm (must be between 0 and 100)"
    )
    petal_width: float = Field(
        gt=0, 
        lt=100, 
        description="Petal width in cm (must be between 0 and 100)"
    )
    label: int = Field(
        ge=0, 
        le=2, 
        description="Species label: 0=setosa, 1=versicolor, 2=virginica"
    )
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_finite(cls, v):
        """Ensure values are finite (not infinity or NaN)"""
        if not math.isfinite(v):
            raise ValueError('Value must be finite (not infinity or NaN)')
        return v


class IrisPredict(BaseModel):
    """
    Prediction input validation with proper constraints.
    """
    sepal_length: float = Field(gt=0, lt=100)
    sepal_width: float = Field(gt=0, lt=100)
    petal_length: float = Field(gt=0, lt=100)
    petal_width: float = Field(gt=0, lt=100)
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_finite(cls, v):
        """Ensure values are finite (not infinity or NaN)"""
        if not math.isfinite(v):
            raise ValueError('Value must be finite (not infinity or NaN)')
        return v

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/add-data")
@limiter.limit("20/minute") 
def add_data(request: Request, item: IrisData, _: str = Depends(verify_api_key)):
    """
    Add user-provided flower data.
    
    SECURITY FEATURES:
    - Rate limiting: 20 requests/minute
    - File locking prevents race conditions
    - Input validation prevents infinite/extreme values
    - Maximum dataset size limit
    """
    
    new_row = pd.DataFrame([{
        "sepal length (cm)": item.sepal_length,
        "sepal width (cm)": item.sepal_width,
        "petal length (cm)": item.petal_length,
        "petal width (cm)": item.petal_width,
        "label": item.label
    }])

    try:
        with file_lock(USER_DATA_PATH):
            if os.path.exists(USER_DATA_PATH):
                existing = pd.read_csv(USER_DATA_PATH)
                
                if len(existing) >= MAX_USER_SAMPLES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Maximum dataset size reached ({MAX_USER_SAMPLES} samples). Cannot add more data."
                    )
                
                df = pd.concat([existing, new_row], ignore_index=True)
            else:
                df = new_row
            
            df.to_csv(USER_DATA_PATH, index=False)
            total_samples = len(df)
        
        return {
            "message": "Sample added successfully",
            "total_samples": total_samples,
            "remaining_capacity": MAX_USER_SAMPLES - total_samples
        }
    
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please retry."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add data: {str(e)}"
        )


@app.post("/train")
@limiter.limit("3/minute") 
def train_model(request: Request, _: str = Depends(verify_api_key)):
    """
    Train a Random Forest on Iris + user data.
    
    SECURITY FEATURES:
    - Rate limiting: 3 requests/minute (training is CPU-intensive)
    - Validates data before training
    - Proper error handling for invalid data
    """
    try:
        df = iris_df.copy()
        
        if os.path.exists(USER_DATA_PATH):
            with file_lock(USER_DATA_PATH):
                user_df = pd.read_csv(USER_DATA_PATH)
            
            numeric_data = user_df.select_dtypes(include=[np.number])
            if not np.isfinite(numeric_data.values).all():
                raise HTTPException(
                    status_code=400,
                    detail="Training data contains invalid values (infinity or NaN). Please clean the dataset."
                )
            
            df = pd.concat([df, user_df], ignore_index=True)

        X = df.drop(columns=["label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        joblib.dump(model, MODEL_PATH)
        
        return {
            "message": "Model trained successfully",
            "accuracy": round(acc, 4),
            "training_samples": len(df),
            "test_samples": len(X_test)
        }
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Training failed due to invalid data: {str(e)}"
        )
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please retry."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/predict")
@limiter.limit("60/minute")
def predict(request: Request, item: IrisPredict, _: str = Depends(verify_api_key)):
    """
    Predict Iris species.
    
    SECURITY FEATURES:
    - Rate limiting: 60 requests/minute
    - Input validation prevents invalid prediction inputs
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Please train the model first using /train endpoint."
        )
    
    try:
        model = joblib.load(MODEL_PATH)

        X_input = [[
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width
        ]]
        
        pred = model.predict(X_input)[0]
        species = target_names[pred]
        
        return {
            "prediction": int(pred),
            "species": species
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
def root():
    """Public health check endpoint - no authentication required"""
    return {
        "message": "Iris classification API is running",
        "version": "1.2.0",
        "environment": ENV,
        "security": {
            "authentication": "API key required",
            "rate_limiting": "enabled"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint - no authentication required"""
    model_status = "trained" if os.path.exists(MODEL_PATH) else "not trained"
    user_data_count = 0
    
    if os.path.exists(USER_DATA_PATH):
        try:
            with file_lock(USER_DATA_PATH, timeout=2):
                user_df = pd.read_csv(USER_DATA_PATH)
                user_data_count = len(user_df)
        except:
            user_data_count = "unknown"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "user_samples": user_data_count,
        "max_samples": MAX_USER_SAMPLES,
        "base_samples": len(iris_df)
    }