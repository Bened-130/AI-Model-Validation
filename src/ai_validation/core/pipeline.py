"""
AI Validation Pipeline
Orchestrates complete validation workflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional
import logging

from .data_quality import DataQualityController
from .model_evaluation import ModelEvaluationFramework
from .model_builder import PredictiveModelBuilder

logger = logging.getLogger(__name__)


class AIValidationPipeline:
    """Complete AI model validation pipeline"""
    
    def __init__(self):
        self.quality_controller = DataQualityController()
        self.model_builder = PredictiveModelBuilder()
        self.evaluator = ModelEvaluationFramework()
        logger.info("AIValidationPipeline initialized")
    
    def run_complete_validation(
        self, 
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Execute full validation pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING AI MODEL VALIDATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Data
        logger.info("\n--- STEP 1: Data Generation ---")
        if df is None:
            df = self.model_builder.generate_sample_data(5000)
        
        # Step 2: Quality Control
        logger.info("\n--- STEP 2: Data Quality Control ---")
        quality_report = self.quality_controller.validate_dataset(df)
        print(self.quality_controller.generate_quality_report(quality_report))
        
        df_clean = self.quality_controller.clean_dataset(df)
        
        # Step 3: Preparation
        logger.info("\n--- STEP 3: Data Preparation ---")
        X = df_clean.drop('target', axis=1)
        y = df_clean['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 4: Model Training
        logger.info("\n--- STEP 4: Model Training ---")
        
        # Random Forest
        rf_model = self.model_builder.build_random_forest(X_train.values, y_train.values)
        rf_metrics = self.evaluator.evaluate_classifier(
            rf_model, X_test.values, y_test.values, "Random Forest"
        )
        
        # Gradient Boosting
        gb_model = self.model_builder.build_gradient_boosting(X_train.values, y_train.values)
        gb_metrics = self.evaluator.evaluate_classifier(
            gb_model, X_test.values, y_test.values, "Gradient Boosting"
        )
        
        # Logistic Regression (baseline)
        X_train_scaled = self.model_builder.scaler.fit_transform(X_train)
        X_test_scaled = self.model_builder.scaler.transform(X_test)
        lr_model = self.model_builder.build_logistic_regression(X_train.values, y_train.values)
        lr_metrics = self.evaluator.evaluate_classifier(
            lr_model, X_test_scaled, y_test.values, "Logistic Regression (Baseline)"
        )
        
        # Step 5: Cross-Validation
        logger.info("\n--- STEP 5: Cross-Validation ---")
        rf_cv = self.evaluator.cross_validate_model(rf_model, X_train.values, y_train.values)
        
        # Step 6: Comparison
        logger.info("\n--- STEP 6: Model Comparison ---")
        comparison = self.evaluator.compare_models([rf_metrics, gb_metrics, lr_metrics])
        
        # Step 7: Statistical Testing
        logger.info("\n--- STEP 7: Statistical Significance ---")
        sig_test = self.evaluator.statistical_significance_test(
            rf_cv['cv_scores'],
            [lr_metrics['accuracy']] * 5
        )
        
        # Step 8: Feature Importance
        logger.info("\n--- STEP 8: Feature Importance ---")
        rf_importance = self.model_builder.get_feature_importance(
            'random_forest', X.columns.tolist()
        )
        
        # Calculate improvement
        baseline_acc = lr_metrics['accuracy']
        best_acc = max(rf_metrics['accuracy'], gb_metrics['accuracy'])
        improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION PIPELINE COMPLETED")
        logger.info("=" * 60)
        
        self._print_summary(quality_report, comparison, improvement, sig_test, rf_importance)
        
        return {
            'quality_report': quality_report,
            'rf_metrics': rf_metrics,
            'gb_metrics': gb_metrics,
            'lr_metrics': lr_metrics,
            'comparison': comparison,
            'improvement': improvement,
            'significance_test': sig_test,
            'feature_importance': rf_importance,
            'rf_cv_results': rf_cv
        }
    
    def _print_summary(
        self, 
        quality_report, 
        comparison, 
        improvement, 
        sig_test, 
        importance
    ):
        """Print formatted summary to console"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        print(f"\nData Quality Score: {quality_report['overall_quality_score']:.2f}%")
        print(f"Records Processed: {quality_report['total_records']:,}")
        
        print("\nModel Performance:")
        print(comparison.to_string(index=False))
        
        print(f"\nAccuracy Improvement: {improvement:.2f}%")
        
        print(f"\nStatistical Significance:")
        print(f"  P-value: {sig_test['p_value']:.4f}")
        print(f"  Significant: {sig_test['significant']}")
        print(f"  Effect Size: {sig_test['effect_size']}")
        
        if importance is not None:
            print("\nTop 5 Features:")
            print(importance.head().to_string(index=False))
        
        print("\n" + "=" * 60)