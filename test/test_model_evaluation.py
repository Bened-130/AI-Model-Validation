import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_validation.core.model_evaluation import ModelEvaluationFramework


@pytest.fixture
def evaluator():
    """Fresh evaluator instance"""
    return ModelEvaluationFramework(framework_version='v1.0')


@pytest.fixture
def sample_data():
    """Generate sample classification data"""
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def trained_model(sample_data):
    """Provide trained model"""
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


class TestModelEvaluationFramework:
    """Test suite for model evaluation"""
    
    def test_initialization(self, evaluator):
        """Test proper initialization"""
        assert evaluator.version == 'v1.0'
        assert isinstance(evaluator.evaluation_history, list)
        assert len(evaluator.evaluation_history) == 0
    
    def test_evaluate_classifier_returns_metrics(self, evaluator, trained_model):
        """Test evaluation returns all expected metrics"""
        model, X_test, y_test = trained_model
        
        metrics = evaluator.evaluate_classifier(model, X_test, y_test, "TestModel")
        
        assert 'model_name' in metrics
        assert metrics['model_name'] == "TestModel"
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert 'n_samples' in metrics
        assert metrics['n_samples'] == len(y_test)
    
    def test_metrics_are_reasonable(self, evaluator, trained_model):
        """Test that metric values are in valid ranges"""
        model, X_test, y_test = trained_model
        
        metrics = evaluator.evaluate_classifier(model, X_test, y_test, "Test")
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_evaluation_history_updated(self, evaluator, trained_model):
        """Test that evaluations are stored in history"""
        model, X_test, y_test = trained_model
        
        evaluator.evaluate_classifier(model, X_test, y_test, "Test")
        
        assert len(evaluator.evaluation_history) == 1
        assert evaluator.evaluation_history[0]['model_name'] == "Test"
    
    def test_improvement_calculation(self, evaluator, sample_data):
        """Test improvement vs baseline calculation"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Baseline model (worse performance)
        baseline = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        baseline.fit(X_train, y_train)
        
        # Better model
        better = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        better.fit(X_train, y_train)
        
        evaluator.evaluate_classifier(baseline, X_test, y_test, "Baseline")
        metrics = evaluator.evaluate_classifier(better, X_test, y_test, "Better")
        
        assert 'improvement_vs_baseline' in metrics
        assert metrics['improvement_vs_baseline'] > 0  # Should show improvement
    
    def test_cross_validation(self, evaluator, sample_data):
        """Test cross-validation functionality"""
        X_train, _, y_train, _ = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        results = evaluator.cross_validate_model(model, X_train, y_train, cv=5)
        
        assert 'cv_scores' in results
        assert len(results['cv_scores']) == 5
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 0 <= results['mean_score'] <= 1
    
    def test_compare_models(self, evaluator, sample_data):
        """Test model comparison dataframe creation"""
        X_train, X_test, y_train, y_test = sample_data
        
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=20, random_state=42)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        res1 = evaluator.evaluate_classifier(model1, X_test, y_test, "Model1")
        res2 = evaluator.evaluate_classifier(model2, X_test, y_test, "Model2")
        
        comparison = evaluator.compare_models([res1, res2])
        
        assert len(comparison) == 2
        assert 'Model' in comparison.columns
        assert 'Accuracy' in comparison.columns
        assert 'Model1' in comparison['Model'].values
    
    def test_statistical_significance(self, evaluator):
        """Test statistical significance testing"""
        # Two clearly different performance distributions
        model1_scores = [0.9, 0.91, 0.89, 0.92, 0.90]  # High performing
        model2_scores = [0.6, 0.61, 0.59, 0.62, 0.60]  # Low performing
        
        result = evaluator.statistical_significance_test(model1_scores, model2_scores)
        
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'cohens_d' in result
        assert 'effect_size' in result
        
        # Should be statistically significant
        assert result['significant'] is True
        assert result['effect_size'] == "Large"
    
    def test_no_significance_similar_models(self, evaluator):
        """Test that similar models show no significant difference"""
        # Two very similar performance distributions
        model1_scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        model2_scores = [0.79, 0.80, 0.78, 0.81, 0.80]
        
        result = evaluator.statistical_significance_test(model1_scores, model2_scores)
        
        # Should NOT be statistically significant
        assert result['significant'] is False
    
    def test_effect_size_interpretation(self, evaluator):
        """Test effect size interpretation"""
        assert evaluator._interpret_effect_size(0.1) == "Small"
        assert evaluator._interpret_effect_size(0.3) == "Medium"
        assert evaluator._interpret_effect_size(0.8) == "Large"
        assert evaluator._interpret_effect_size(-0.8) == "Large"  # Absolute value