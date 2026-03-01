"""
Integration tests for full pipeline
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_validation.core.pipeline import AIValidationPipeline


class TestFullPipeline:
    """End-to-end integration tests"""
    
    def test_complete_pipeline_execution(self):
        """Test that full pipeline runs without errors"""
        pipeline = AIValidationPipeline()
        
        # Should run with default synthetic data
        results = pipeline.run_complete_validation()
        
        assert 'quality_report' in results
        assert 'rf_metrics' in results
        assert 'gb_metrics' in results
        assert 'lr_metrics' in results
        assert 'comparison' in results
        assert 'improvement' in results
    
    def test_pipeline_results_structure(self):
        """Test that results contain expected data"""
        pipeline = AIValidationPipeline()
        results = pipeline.run_complete_validation()
        
        # Quality report structure
        qr = results['quality_report']
        assert 'overall_quality_score' in qr
        assert 'checks' in qr
        assert qr['overall_quality_score'] > 0
        
        # Model metrics structure
        for model in ['rf_metrics', 'gb_metrics', 'lr_metrics']:
            metrics = results[model]
            assert 'accuracy' in metrics
            assert 'f1_score' in metrics
            assert metrics['accuracy'] > 0
    
    def test_improvement_calculation(self):
        """Test that improvement is calculated and positive"""
        pipeline = AIValidationPipeline()
        results = pipeline.run_complete_validation()
        
        assert 'improvement' in results
        assert isinstance(results['improvement'], float)
        # RF or GB should beat logistic regression
        assert results['improvement'] > 0