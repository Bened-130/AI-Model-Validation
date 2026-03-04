"""
Predictive Model Builder
Handles data generation and model training
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PredictiveModelBuilder:
    """Build and train predictive models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        logger.info("PredictiveModelBuilder initialized")
    
    def generate_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic classification dataset"""
        logger.info(f"Generating {n_samples} synthetic samples")
        
        np.random.seed(42)
        
        # Generate base features
        X = np.random.randn(n_samples, 10)
        
        # Add correlations
        X[:, 0] = X[:, 0] * 2 + X[:, 1] * 0.5
        X[:, 2] = X[:, 2] * 1.5 - X[:, 3] * 0.3
        
        # Create target with non-linear decision boundary
        y = (
            (X[:, 0] > 0.5) & (X[:, 2] > -0.5) |
            (X[:, 1] < -0.8) & (X[:, 4] > 0.3)
        ).astype(int)
        
        # Add noise (10% label flip)
        noise_idx = np.random.choice(n_samples, size=n_samples//10, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        # Create DataFrame
        df = pd.DataFrame(
            X, 
            columns=[f'feature_{i}' for i in range(10)]
        )
        df['target'] = y
        
        logger.info(f"Generated data: {df['target'].value_counts().to_dict()}")
        return df
    
    def build_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        **kwargs
    ) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        logger.info("Training Random Forest")
        
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def build_gradient_boosting(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        **kwargs
    ) -> GradientBoostingClassifier:
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting")
        
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        params.update(kwargs)
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        return model
    
    def build_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        scale: bool = True,
        **kwargs
    ) -> LogisticRegression:
        """Train Logistic Regression (baseline)"""
        logger.info("Training Logistic Regression")
        
        X = X_train
        if scale:
            X = self.scaler.fit_transform(X_train)
        
        params = {
            'max_iter': 1000,
            'random_state': 42
        }
        params.update(kwargs)
        
        model = LogisticRegression(**params)
        model.fit(X, y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def get_feature_importance(
        self, 
        model_name: str, 
        feature_names: list
    ) -> Optional[pd.DataFrame]:
        """Extract feature importance from trained model"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df