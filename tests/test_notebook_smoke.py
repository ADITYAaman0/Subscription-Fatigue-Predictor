"""
Smoke Test for Jupyter Notebook Execution.

This test verifies that the main analysis notebook runs end-to-end without errors.
Run with: pytest tests/test_notebook_smoke.py -v --timeout=600
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNotebookExecution:
    """Test suite for notebook execution validation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.notebook_path = PROJECT_ROOT / 'Project_Submission.ipynb'
        self.db_path = PROJECT_ROOT / 'data' / 'subscription_fatigue.db'
    
    def test_notebook_exists(self):
        """Verify the main notebook file exists."""
        assert self.notebook_path.exists(), f"Notebook not found at {self.notebook_path}"
    
    def test_notebook_is_valid_json(self):
        """Verify notebook is valid JSON format."""
        import json
        
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            try:
                nb = json.load(f)
                assert 'cells' in nb, "Notebook missing 'cells' key"
                assert 'nbformat' in nb, "Notebook missing 'nbformat' key"
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid notebook JSON: {e}")
    
    def test_notebook_has_required_sections(self):
        """Verify notebook contains all required rubric sections."""
        import json
        
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Extract markdown content
        markdown_cells = [
            cell['source'] for cell in nb['cells'] 
            if cell['cell_type'] == 'markdown'
        ]
        all_markdown = '\n'.join(
            ''.join(cell) if isinstance(cell, list) else cell 
            for cell in markdown_cells
        )
        
        required_sections = [
            'Problem Definition',
            'Data Understanding',
            'Model',
            'Implementation',
            'Evaluation',
            'Ethical',
            'Conclusion'
        ]
        
        for section in required_sections:
            assert section in all_markdown, f"Missing required section: {section}"
    
    def test_database_initialization(self):
        """Verify database can be initialized."""
        from src.data.collectors.data_ingestion import DataIngestionPipeline
        
        # Use a test database path
        test_db = PROJECT_ROOT / 'data' / 'test_smoke.db'
        
        try:
            pipeline = DataIngestionPipeline(str(test_db))
            pipeline.run_full_pipeline()
            
            assert test_db.exists(), "Database was not created"
        finally:
            # Cleanup
            if test_db.exists():
                os.remove(test_db)
    
    def test_models_import(self):
        """Verify all project models can be imported."""
        try:
            from src.models.ml.ml_models import ChurnRiskPredictor
            from src.models.economic.economic_models import ElasticityCalculator
            from src.models.statistical.statistical_models import ChangePointDetector
            
            # Instantiate models
            churn = ChurnRiskPredictor()
            elasticity = ElasticityCalculator()
            detector = ChangePointDetector()
            
            assert churn is not None
            assert elasticity is not None
            assert detector is not None
        except ImportError as e:
            pytest.fail(f"Model import failed: {e}")
    
    @pytest.mark.slow
    def test_notebook_executes_successfully(self):
        """
        Execute the entire notebook and verify no errors.
        
        This test is marked as 'slow' and may take several minutes.
        Run with: pytest tests/test_notebook_smoke.py -v -m slow --timeout=600
        """
        try:
            import nbformat
            from nbconvert.preprocessors import ExecutePreprocessor
        except ImportError:
            pytest.skip("nbformat/nbconvert not installed")
        
        # Ensure database exists
        if not self.db_path.exists():
            from src.data.collectors.data_ingestion import DataIngestionPipeline
            pipeline = DataIngestionPipeline(str(self.db_path))
            pipeline.run_full_pipeline()
        
        # Load and execute notebook
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(
            timeout=600,
            kernel_name='python3',
            allow_errors=False
        )
        
        try:
            ep.preprocess(nb, {'metadata': {'path': str(self.notebook_path.parent)}})
        except Exception as e:
            pytest.fail(f"Notebook execution failed: {e}")


class TestQuickSmoke:
    """Quick validation tests that don't require full notebook execution."""
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists."""
        req_path = PROJECT_ROOT / 'requirements.txt'
        assert req_path.exists(), "requirements.txt not found"
    
    def test_key_dependencies_listed(self):
        """Verify key dependencies are in requirements.txt."""
        req_path = PROJECT_ROOT / 'requirements.txt'
        
        with open(req_path, 'r') as f:
            requirements = f.read().lower()
        
        key_deps = ['numpy', 'pandas', 'xgboost', 'plotly', 'scikit-learn']
        
        for dep in key_deps:
            assert dep in requirements, f"Missing dependency: {dep}"
    
    def test_project_structure(self):
        """Verify expected project structure exists."""
        expected_dirs = [
            'src/models',
            'src/data',
            'src/visualization',
            'notebooks',
            'tests',
            'data'
        ]
        
        for dir_path in expected_dirs:
            full_path = PROJECT_ROOT / dir_path
            assert full_path.exists(), f"Missing directory: {dir_path}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
