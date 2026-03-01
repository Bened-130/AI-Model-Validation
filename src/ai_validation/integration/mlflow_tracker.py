"""
MLflow integration for experiment tracking and model registry
Requires: pip install mlflow
"""

import mlflow
import mlflow.sklearn
from contextlib import contextmanager
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow integration for AI Validation Framework.
    Tracks experiments, logs metrics, parameters, and registers models.
    """
    
    def __init__(
        self, 
        experiment_name: str = "ai_validation",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment in MLflow
            tracking_uri: MLflow tracking server URI (None for local)
            artifact_location: Where to store artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (local file store if not specified)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        else:
            # Local tracking
            mlflow.set_tracking_uri("file:./mlruns")
            logger.info("Using local MLflow tracking")
        
        # Create or get experiment
        self.experiment = self._get_or_create_experiment(
            experiment_name, 
            artifact_location
        )
        
        self.active_run = None
        logger.info(f"MLflowTracker initialized for experiment: {experiment_name}")
    
    def _get_or_create_experiment(
        self, 
        name: str, 
        artifact_location: Optional[str]
    ) -> str:
        """Get existing experiment or create new one"""
        experiment = mlflow.get_experiment_by_name(name)
        
        if experiment:
            logger.info(f"Using existing experiment: {name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
        
        # Create new experiment
        experiment_id = mlflow.create_experiment(
            name=name,
            artifact_location=artifact_location
        )
        logger.info(f"Created new experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for MLflow runs.
        
        Usage:
            with tracker.start_run("validation_run"):
                tracker.log_metrics({"accuracy": 0.95})
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.active_run = mlflow.start_run(
                experiment_id=self.experiment,
                run_name=run_name,
                nested=nested
            )
            logger.info(f"Started MLflow run: {run_name}")
            yield self.active_run
            
        finally:
            if self.active_run:
                mlflow.end_run()
                logger.info(f"Ended MLflow run: {run_name}")
                self.active_run = None
    
    def log_validation_report(self, quality_report: Dict[str, Any]):
        """
        Log data quality validation report to MLflow.
        
        Args:
            quality_report: Output from DataQualityController.validate_dataset()
        """
        if not self.active_run:
            raise RuntimeError("No active MLflow run. Use start_run() context manager.")
        
        # Log overall score
        mlflow.log_metric(
            "data_quality_score", 
            quality_report['overall_quality_score']
        )
        
        # Log individual checks
        for check_name, check_data in quality_report['checks'].items():
            if 'score' in check_data:
                mlflow.log_metric(f"quality_{check_name}", check_data['score'])
            
            # Log additional metrics if present
            if 'missing_cells' in check_data:
                mlflow.log_metric(f"quality_{check_name}_missing", check_data['missing_cells'])
            if 'duplicates' in check_data:
                mlflow.log_metric(f"quality_{check_name}_duplicates", check_data['duplicates'])
        
        # Log as artifact
        report_json = json.dumps(quality_report, default=str, indent=2)
        mlflow.log_text(report_json, "quality_report.json")
        
        logger.info("Logged validation report to MLflow")
    
    def log_model_metrics(
        self, 
        metrics: Dict[str, Any], 
        model_name: str = "model"
    ):
        """
        Log model performance metrics.
        
        Args:
            metrics: Dictionary containing model metrics
            model_name: Prefix for metric names
        """
        if not self.active_run:
            raise RuntimeError("No active MLflow run.")
        
        # Log scalar metrics
        scalar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in scalar_metrics:
            if metric in metrics and metrics[metric] is not None:
                mlflow.log_metric(f"{model_name}_{metric}", metrics[metric])
        
        # Log improvement if present
        if 'improvement_vs_baseline' in metrics:
            mlflow.log_metric(
                f"{model_name}_improvement_pct", 
                metrics['improvement_vs_baseline']
            )
        
        # Log confusion matrix as artifact
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            cm_json = json.dumps(cm)
            mlflow.log_text(cm_json, f"{model_name}_confusion_matrix.json")
        
        logger.info(f"Logged metrics for {model_name}")
    
    def log_model(
        self, 
        model, 
        model_name: str,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log and register sklearn model.
        
        Args:
            model: Trained sklearn model
            model_name: Name for the model in registry
            signature: MLflow model signature (optional)
            input_example: Example input for model (optional)
        """
        if not self.active_run:
            raise RuntimeError("No active MLflow run.")
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example
        )
        
        # Register model in model registry
        model_uri = f"runs:/{self.active_run.info.run_id}/{model_name}"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"Registered model: {model_name} (v{registered_model.version})")
        return registered_model
    
    def log_comparison_table(self, comparison_df, filename: str = "model_comparison"):
        """Log model comparison dataframe as CSV artifact"""
        if not self.active_run:
            raise RuntimeError("No active MLflow run.")
        
        csv_path = f"/tmp/{filename}.csv"
        comparison_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        logger.info(f"Logged comparison table: {filename}")
    
    def log_feature_importance(self, importance_df, model_name: str = "model"):
        """Log feature importance as CSV and create plot"""
        if not self.active_run:
            raise RuntimeError("No active MLflow run.")
        
        # Save as CSV
        csv_path = f"/tmp/{model_name}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        # Create and log plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df.head(10).plot.barh(x='feature', y='importance', ax=ax)
            plt.tight_layout()
            fig_path = f"/tmp/{model_name}_feature_importance.png"
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path)
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available, skipping feature importance plot")
        
        logger.info(f"Logged feature importance for {model_name}")
    
    def log_pipeline_params(self, params: Dict[str, Any]):
        """Log pipeline configuration parameters"""
        if not self.active_run:
            raise RuntimeError("No active MLflow run.")
        
        # Flatten nested dicts for MLflow
        flat_params = self._flatten_dict(params)
        
        for key, value in flat_params.items():
            # MLflow only accepts certain types
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
        
        logger.info(f"Logged {len(flat_params)} parameters")
    
    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dictionary for MLflow compatibility"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_best_run(self, metric: str = "data_quality_score", mode: str = "max"):
        """
        Get best run from experiment based on metric.
        
        Args:
            metric: Metric name to compare
            mode: 'max' or 'min'
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment])
        
        if runs.empty:
            return None
        
        if mode == "max":
            best_idx = runs[f"metrics.{metric}"].idxmax()
        else:
            best_idx = runs[f"metrics.{metric}"].idxmin()
        
        return runs.loc[best_idx]


# Integration with existing pipeline
class TrackedAIValidationPipeline:
    """
    Wrapper around AIValidationPipeline that adds MLflow tracking.
    Drop-in replacement for AIValidationPipeline with tracking enabled.
    """
    
    def __init__(
        self, 
        tracking_uri: Optional[str] = None,
        experiment_name: str = "ai_validation"
    ):
        self.pipeline = AIValidationPipeline()
        self.tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri
        )
        logger.info("TrackedAIValidationPipeline initialized")
    
    def run_complete_validation(self, df=None, run_name: Optional[str] = None):
        """
        Run validation with full MLflow tracking.
        
        Returns:
            Same results as AIValidationPipeline plus MLflow run ID
        """
        with self.tracker.start_run(run_name=run_name):
            # Log framework version and parameters
            self.tracker.log_pipeline_params({
                "framework_version": "v3.0",
                "n_samples": len(df) if df is not None else 5000,
                "models": ["random_forest", "gradient_boosting", "logistic_regression"]
            })
            
            # Run original pipeline
            results = self.pipeline.run_complete_validation(df)
            
            # Log quality report
            self.tracker.log_validation_report(results['quality_report'])
            
            # Log each model's metrics
            model_mapping = {
                'rf_metrics': 'random_forest',
                'gb_metrics': 'gradient_boosting',
                'lr_metrics': 'logistic_regression'
            }
            
            for key, model_name in model_mapping.items():
                if key in results:
                    self.tracker.log_model_metrics(
                        results[key], 
                        model_name=model_name
                    )
            
            # Log comparison table
            self.tracker.log_comparison_table(results['comparison'])
            
            # Log feature importance
            if results.get('feature_importance') is not None:
                self.tracker.log_feature_importance(
                    results['feature_importance'],
                    model_name='random_forest'
                )
            
            # Log improvement metric
            mlflow.log_metric("accuracy_improvement_pct", results['improvement'])
            
            # Add run ID to results
            results['mlflow_run_id'] = self.tracker.active_run.info.run_id
            
            logger.info("Complete validation tracked in MLflow")
        
        return results