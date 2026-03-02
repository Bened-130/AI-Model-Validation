from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import io
import logging

from ai_validation.core.pipeline import AIValidationPipeline
from ai_validation.integrations.mlflow_tracker import TrackedAIValidationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Validation Framework API",
    description="Production-grade ML validation and model evaluation",
    version="3.0.0"
)


class ValidationRequest(BaseModel):
    """Request model for validation"""
    use_mlflow: bool = False
    experiment_name: Optional[str] = "api_validation"


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    success: bool
    quality_score: float
    best_model: str
    best_accuracy: float
    improvement: float
    mlflow_run_id: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "service": "ai-validation-api"}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "AI Validation Framework API",
        "version": "3.0.0",
        "endpoints": [
            "/validate (POST) - Run validation on data",
            "/health (GET) - Health check"
        ]
    }


@app.post("/validate", response_model=ValidationResponse)
async def validate_data(
    file: UploadFile = File(...),
    config: ValidationRequest = ValidationRequest()
):
    """
    Run complete validation pipeline on uploaded CSV file.
    
    - **file**: CSV file with training data
    - **use_mlflow**: Whether to track in MLflow
    - **experiment_name**: MLflow experiment name
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Received file with {len(df)} rows, {len(df.columns)} columns")
        
        # Choose pipeline
        if config.use_mlflow:
            pipeline = TrackedAIValidationPipeline(
                experiment_name=config.experiment_name,
                tracking_uri="http://mlflow:5000"
            )
        else:
            pipeline = AIValidationPipeline()
        
        # Run validation
        results = pipeline.run_complete_validation(df)
        
        # Determine best model
        comparison = results['comparison']
        best_idx = comparison['Accuracy'].astype(float).idxmax()
        best_model = comparison.loc[best_idx, 'Model']
        best_acc = float(comparison.loc[best_idx, 'Accuracy'])
        
        return ValidationResponse(
            success=True,
            quality_score=results['quality_report']['overall_quality_score'],
            best_model=best_model,
            best_accuracy=best_acc,
            improvement=results['improvement'],
            mlflow_run_id=results.get('mlflow_run_id')
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/sample", response_model=ValidationResponse)
async def validate_sample(config: ValidationRequest = ValidationRequest()):
    """Run validation on synthetic sample data"""
    try:
        if config.use_mlflow:
            pipeline = TrackedAIValidationPipeline(
                experiment_name=config.experiment_name
            )
        else:
            pipeline = AIValidationPipeline()
        
        results = pipeline.run_complete_validation()
        
        comparison = results['comparison']
        best_idx = comparison['Accuracy'].astype(float).idxmax()
        best_model = comparison.loc[best_idx, 'Model']
        best_acc = float(comparison.loc[best_idx, 'Accuracy'])
        
        return ValidationResponse(
            success=True,
            quality_score=results['quality_report']['overall_quality_score'],
            best_model=best_model,
            best_accuracy=best_acc,
            improvement=results['improvement'],
            mlflow_run_id=results.get('mlflow_run_id')
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)