"""
Report Generation Utilities
Creates HTML and PDF reports from validation results
"""

import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class ReportGenerator:
    """Generate formatted reports from validation results"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(
        self, 
        results: Dict[str, Any], 
        filename: str = "validation_report.html"
    ) -> str:
        """Generate interactive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
                .header {{ border-bottom: 3px solid #1f77b4; padding-bottom: 20px; margin-bottom: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .score {{ font-size: 2em; font-weight: bold; color: #1f77b4; }}
                .pass {{ color: #28a745; }}
                .fail {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #1f77b4; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 AI Validation Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Data Quality Score</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="score">{results['quality_report']['overall_quality_score']:.1f}%</div>
                        <div>Overall Quality</div>
                    </div>
                    <div class="metric-card">
                        <div class="score">{results['quality_report']['checks']['completeness']['score']:.1f}%</div>
                        <div>Completeness</div>
                    </div>
                    <div class="metric-card">
                        <div class="score">{results['quality_report']['checks']['consistency']['score']:.1f}%</div>
                        <div>Consistency</div>
                    </div>
                    <div class="metric-card">
                        <div class="score">{results['quality_report']['checks']['accuracy']['score']:.1f}%</div>
                        <div>Accuracy</div>
                    </div>
                </div>
                
                <h2>Model Performance</h2>
                {results['comparison'].to_html(index=False, classes='comparison-table')}
                
                <h2>Improvement</h2>
                <p><strong>{results['improvement']:.2f}%</strong> accuracy improvement over baseline</p>
                
                <h2>Statistical Significance</h2>
                <p>P-value: {results['significance_test']['p_value']:.4f}</p>
                <p>Significant: {'Yes' if results['significance_test']['significant'] else 'No'}</p>
            </div>
        </body>
        </html>
        """
        
        output_path = self.output_dir / filename
        output_path.write_text(html_content)
        return str(output_path)
    
    def generate_json_report(
        self, 
        results: Dict[str, Any], 
        filename: str = "validation_report.json"
    ) -> str:
        """Generate JSON report"""
        output_path = self.output_dir / filename
        
        # Convert DataFrames to dicts for JSON serialization
        export_results = results.copy()
        if 'comparison' in export_results:
            export_results['comparison'] = export_results['comparison'].to_dict()
        if 'feature_importance' in export_results and export_results['feature_importance'] is not None:
            export_results['feature_importance'] = export_results['feature_importance'].to_dict()
        
        output_path.write_text(json.dumps(export_results, indent=2, default=str))
        return str(output_path)