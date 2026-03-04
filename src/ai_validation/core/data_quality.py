"""
Data Quality Controller Module
Handles 4-pillar validation: completeness, consistency, accuracy, uniqueness
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DataQualityController:
    """Comprehensive data quality control system"""
    
    def __init__(self):
        self.quality_reports: List[Dict] = []
        logger.info("DataQualityController initialized")
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation on dataset"""
        logger.info(f"Validating dataset: {len(df)} records, {len(df.columns)} features")
        
        report = {
            'timestamp': datetime.now(),
            'total_records': len(df),
            'total_features': len(df.columns),
            'checks': {}
        }
        
        # Run all checks
        report['checks']['completeness'] = self._check_completeness(df)
        report['checks']['consistency'] = self._check_consistency(df)
        report['checks']['accuracy'] = self._check_accuracy(df)
        report['checks']['uniqueness'] = self._check_uniqueness(df)
        
        # Calculate overall score
        scores = [check['score'] for check in report['checks'].values()]
        report['overall_quality_score'] = round(np.mean(scores), 2)
        
        self.quality_reports.append(report)
        
        logger.info(f"Validation complete. Quality score: {report['overall_quality_score']}%")
        return report
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        return {
            'score': round(completeness, 2),
            'missing_cells': int(missing_cells),
            'missing_by_column': df.isnull().sum().to_dict(),
            'status': 'PASS' if completeness >= 95 else 'FAIL'
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records"""
        duplicates = df.duplicated().sum()
        duplicate_rate = (duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        return {
            'score': round(100 - duplicate_rate, 2),
            'duplicates': int(duplicates),
            'duplicate_rate': round(duplicate_rate, 2),
            'status': 'PASS' if duplicate_rate < 5 else 'FAIL'
        }
    
    def _check_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for data accuracy issues"""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for negative values in likely positive columns
        for col in numeric_cols:
            if 'id' not in col.lower() and (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Negative values in {col}: {negative_count}")
        
        # Check for outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_rate = outliers / len(df)
            
            if outlier_rate > 0.05:  # More than 5% outliers
                issues.append(f"High outlier rate in {col}: {outliers} ({outlier_rate:.1%})")
        
        score = max(0, 100 - len(issues) * 10)
        
        return {
            'score': score,
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'WARN' if len(issues) < 3 else 'FAIL'
        }
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check ID column uniqueness"""
        id_col = None
        for col in df.columns:
            if col.lower() in ['id', 'uuid', 'identifier']:
                id_col = col
                break
        
        if id_col:
            unique_ratio = df[id_col].nunique() / len(df) * 100
            return {
                'score': round(unique_ratio, 2),
                'id_column': id_col,
                'unique_ids': df[id_col].nunique(),
                'status': 'PASS' if unique_ratio >= 99 else 'FAIL'
            }
        
        return {
            'score': 100.0,
            'id_column': None,
            'note': 'No ID column detected',
            'status': 'PASS'
        }
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations"""
        logger.info("Starting dataset cleaning")
        initial_shape = df.shape
        
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
        
        # Remove extreme outliers
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df_clean[col] >= Q1 - 3*IQR) & (df_clean[col] <= Q3 + 3*IQR)
            df_clean = df_clean[mask]
        
        logger.info(f"Cleaning complete: {initial_shape} -> {df_clean.shape}")
        return df_clean
    
    def generate_quality_report(self, report: Dict) -> str:
        """Generate formatted text report"""
        lines = [
            "╔════════════════════════════════════════════════════════════╗",
            "║         DATA QUALITY VALIDATION REPORT                     ║",
            "╚════════════════════════════════════════════════════════════╝",
            "",
            f"Timestamp: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Records: {report['total_records']:,}",
            f"Total Features: {report['total_features']}",
            "",
            f"OVERALL QUALITY SCORE: {report['overall_quality_score']:.2f}%",
            "",
            "─" * 60,
            "",
            "COMPLETENESS CHECK:",
            f"  Score: {report['checks']['completeness']['score']:.2f}%",
            f"  Missing Cells: {report['checks']['completeness']['missing_cells']:,}",
            f"  Status: {report['checks']['completeness']['status']}",
            "",
            "CONSISTENCY CHECK:",
            f"  Score: {report['checks']['consistency']['score']:.2f}%",
            f"  Duplicates: {report['checks']['consistency']['duplicates']:,}",
            f"  Status: {report['checks']['consistency']['status']}",
            "",
            "ACCURACY CHECK:",
            f"  Score: {report['checks']['accuracy']['score']:.2f}%",
            f"  Issues Found: {len(report['checks']['accuracy']['issues'])}",
            f"  Status: {report['checks']['accuracy']['status']}",
            "",
            "UNIQUENESS CHECK:",
            f"  Score: {report['checks']['uniqueness']['score']:.2f}%",
            f"  Status: {report['checks']['uniqueness']['status']}",
            "",
            "═" * 60,
        ]
        
        return "\n".join(lines)