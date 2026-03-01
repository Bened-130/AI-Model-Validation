"""
Unit tests for DataQualityController
Run with: pytest tests/test_data_quality.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_validation.core.data_quality import DataQualityController


@pytest.fixture
def quality_controller():
    """Fixture providing fresh DataQualityController instance"""
    return DataQualityController()


@pytest.fixture
def perfect_df():
    """Fixture providing clean dataset"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 10, 100),
        'feature_b': np.random.normal(50, 5, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def dirty_df():
    """Fixture providing dataset with quality issues"""
    np.random.seed(42)
    df = pd.DataFrame({
        'id': list(range(99)) + [50],  # Duplicate ID
        'feature_a': list(np.random.normal(100, 10, 98)) + [None, -999],  # Missing + outlier
        'feature_b': list(np.random.normal(50, 5, 98)) + [None, None],  # Missing values
        'category': ['A'] * 50 + ['B'] * 48 + [None, 'C']  # Missing category
    })
    # Add duplicate row
    duplicate_row = df.iloc[0].copy()
    return pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)


class TestDataQualityController:
    """Test suite for DataQualityController"""
    
    def test_initialization(self, quality_controller):
        """Test controller initializes correctly"""
        assert quality_controller is not None
        assert isinstance(quality_controller.quality_reports, list)
        assert len(quality_controller.quality_reports) == 0
    
    def test_validate_perfect_dataset(self, quality_controller, perfect_df):
        """Test validation on clean data returns high scores"""
        report = quality_controller.validate_dataset(perfect_df)
        
        assert 'timestamp' in report
        assert 'total_records' in report
        assert report['total_records'] == 100
        assert 'checks' in report
        assert 'overall_quality_score' in report
        
        # Perfect data should score 100%
        assert report['overall_quality_score'] == 100.0
        assert report['checks']['completeness']['score'] == 100.0
        assert report['checks']['consistency']['score'] == 100.0
        assert report['checks']['uniqueness']['score'] == 100.0
    
    def test_validate_dirty_dataset(self, quality_controller, dirty_df):
        """Test validation detects issues in dirty data"""
        report = quality_controller.validate_dataset(dirty_df)
        
        # Should detect missing values
        completeness = report['checks']['completeness']
        assert completeness['missing_cells'] > 0
        assert completeness['score'] < 100.0
        assert completeness['status'] == 'FAIL'
        
        # Should detect duplicates
        consistency = report['checks']['consistency']
        assert consistency['duplicates'] > 0
        assert consistency['score'] < 100.0
        
        # Should detect duplicate IDs
        uniqueness = report['checks']['uniqueness']
        assert uniqueness['score'] < 100.0
    
    def test_completeness_calculation(self, quality_controller):
        """Test completeness check accuracy"""
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [None, 2, 3, 4]
        })
        
        result = quality_controller._check_completeness(df)
        
        # 6 filled out of 8 cells = 75%
        assert result['score'] == 75.0
        assert result['missing_cells'] == 2
        assert result['missing_by_column']['a'] == 1
        assert result['missing_by_column']['b'] == 1
    
    def test_consistency_check(self, quality_controller):
        """Test duplicate detection"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 1],  # Row 0 and 3 are duplicates
            'b': [4, 5, 6, 4]
        })
        
        result = quality_controller._check_consistency(df)
        
        assert result['duplicates'] == 1
        assert result['duplicate_rate'] == 25.0
        assert result['score'] == 75.0
    
    def test_accuracy_check_detects_negatives(self, quality_controller):
        """Test accuracy check finds negative values"""
        df = pd.DataFrame({
            'positive_col': [1, 2, 3, 4],
            'negative_col': [-1, 2, 3, 4]  # Has negative
        })
        
        result = quality_controller._check_accuracy(df)
        
        assert len(result['issues']) > 0
        assert any('negative' in issue.lower() for issue in result['issues'])
    
    def test_accuracy_check_detects_outliers(self, quality_controller):
        """Test accuracy check finds outliers using IQR method"""
        df = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5] * 20,  # 100 values, normal distribution
            'outliers': [1, 2, 3, 4, 5] * 19 + [1000, 2000]  # 2 extreme outliers
        })
        
        result = quality_controller._check_accuracy(df)
        
        # Should flag high outlier rate
        assert len(result['issues']) > 0
        assert any('outlier' in issue.lower() for issue in result['issues'])
    
    def test_uniqueness_with_id_column(self, quality_controller):
        """Test uniqueness check when ID column exists"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 3, 4],  # One duplicate ID
            'data': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = quality_controller._check_uniqueness(df)
        
        assert result['score'] == 80.0  # 4 unique / 5 total
    
    def test_uniqueness_without_id(self, quality_controller):
        """Test uniqueness check passes when no ID column"""
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = quality_controller._check_uniqueness(df)
        
        assert result['score'] == 100.0
    
    def test_clean_dataset_removes_duplicates(self, quality_controller, dirty_df):
        """Test cleaning removes duplicate rows"""
        initial_count = len(dirty_df)
        cleaned = quality_controller.clean_dataset(dirty_df)
        
        assert len(cleaned) < initial_count
        assert cleaned.duplicated().sum() == 0
    
    def test_clean_dataset_handles_missing_numeric(self, quality_controller):
        """Test cleaning fills numeric missing values with median"""
        df = pd.DataFrame({
            'nums': [1.0, 2.0, None, 4.0, 5.0]  # Median is 3.0
        })
        
        cleaned = quality_controller.clean_dataset(df)
        
        assert cleaned['nums'].isnull().sum() == 0
        # Original median was 3.0, but after cleaning it might change slightly
        assert cleaned['nums'].iloc[2] == 3.0  # Filled with median
    
    def test_clean_dataset_handles_missing_categorical(self, quality_controller):
        """Test cleaning fills categorical missing values with mode"""
        df = pd.DataFrame({
            'cats': ['A', 'A', 'B', None, 'A']  # Mode is 'A'
        })
        
        cleaned = quality_controller.clean_dataset(df)
        
        assert cleaned['cats'].isnull().sum() == 0
        assert cleaned['cats'].iloc[3] == 'A'
    
    def test_clean_dataset_removes_outliers(self, quality_controller):
        """Test cleaning removes extreme outliers"""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 1000]  # 1000 is extreme outlier
        })
        
        cleaned = quality_controller.clean_dataset(df)
        
        # Should remove the outlier row
        assert 1000 not in cleaned['values'].values
    
    def test_generate_quality_report_format(self, quality_controller, perfect_df):
        """Test report generation produces formatted string"""
        report = quality_controller.validate_dataset(perfect_df)
        output = quality_controller.generate_quality_report(report)
        
        assert isinstance(output, str)
        assert 'DATA QUALITY VALIDATION REPORT' in output
        assert 'COMPLETENESS CHECK' in output
        assert 'CONSISTENCY CHECK' in output
        assert 'PASS' in output or 'FAIL' in output
    
    def test_multiple_reports_stored(self, quality_controller, perfect_df):
        """Test that multiple validations store reports"""
        quality_controller.validate_dataset(perfect_df)
        quality_controller.validate_dataset(perfect_df)
        
        assert len(quality_controller.quality_reports) == 2


class TestEdgeCases:
    """Edge case testing"""
    
    def test_empty_dataframe(self, quality_controller):
        """Test handling of empty dataframe"""
        df = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should handle gracefully or raise
            quality_controller.validate_dataset(df)
    
    def test_single_row(self, quality_controller):
        """Test handling of single row dataset"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        
        report = quality_controller.validate_dataset(df)
        assert report['total_records'] == 1
    
    def test_all_missing_column(self, quality_controller):
        """Test handling of completely missing column"""
        df = pd.DataFrame({
            'a': [None, None, None],
            'b': [1, 2, 3]
        })
        
        result = quality_controller._check_completeness(df)
        assert result['score'] == 50.0  # Half the cells are missing