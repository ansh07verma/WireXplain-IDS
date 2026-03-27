"""
WireXplain-IDS: Main Pipeline Orchestrator
Complete end-to-end intrusion detection pipeline with explainability
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate the complete IDS pipeline"""
    
    def __init__(self, config):
        """
        Initialize pipeline orchestrator
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config
        self.start_time = None
        self.results = {}
        
    def run_full_pipeline(self):
        """Run the complete end-to-end pipeline"""
        self.start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print("WireXplain-IDS: Complete Pipeline Execution")
        print("=" * 80)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        
        try:
            # Stage 1: Feature Engineering
            self._run_feature_engineering()
            
            # Stage 2: Feature Selection
            self._run_feature_selection()
            
            # Stage 3: Isolation Filtering
            self._run_isolation_filtering()
            
            # Stage 4: Model Training
            self._run_model_training()
            
            # Stage 5: SHAP Explainability
            self._run_shap_explanation()
            
            # Print final summary
            self._print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_feature_engineering(self):
        """Stage 1: Feature Engineering"""
        print("\n" + "=" * 80)
        print("STAGE 1/5: Feature Engineering")
        print("=" * 80)
        logger.info("Starting feature engineering...")
        
        try:
            from src.feature_engineering import FeatureEngineer
            
            fe = FeatureEngineer()
            df = fe.load_data(self.config['raw_data_path'])
            fe.validate_label_column(df)
            df = fe.encode_labels(df)
            df = fe.encode_categorical_features(df)
            df = fe.engineer_features(df)
            features = fe.select_features(df)
            fe.save_features(df, features, df['label_binary'], self.config['features_path'])
            
            self.results['feature_engineering'] = {
                'samples': len(df),
                'features': len(features.columns),
                'status': 'SUCCESS'
            }
            
            logger.info("✓ Feature engineering completed successfully")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _run_feature_selection(self):
        """Stage 2: Feature Selection"""
        print("\n" + "=" * 80)
        print("STAGE 2/5: Feature Selection (Mutual Information)")
        print("=" * 80)
        logger.info("Starting feature selection...")
        
        try:
            from src.feature_selection import FeatureSelector
            
            selector = FeatureSelector(top_n=self.config['top_n_features'])
            X, y = selector.load_features(self.config['features_path'])
            feature_scores = selector.compute_mutual_information(X, y)
            selector.print_feature_ranking(top_n=self.config['top_n_features'] * 2)
            selected = selector.select_top_features(n=self.config['top_n_features'])
            selector.save_selected_features(X, y, self.config['selected_features_path'])
            
            self.results['feature_selection'] = {
                'original_features': X.shape[1],
                'selected_features': len(selected),
                'reduction': f"{(1 - len(selected)/X.shape[1])*100:.1f}%",
                'status': 'SUCCESS'
            }
            
            logger.info("✓ Feature selection completed successfully")
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise
    
    def _run_isolation_filtering(self):
        """Stage 3: Isolation Filtering"""
        print("\n" + "=" * 80)
        print("STAGE 3/5: Isolation Filtering (Outlier Detection)")
        print("=" * 80)
        logger.info("Starting isolation filtering...")
        
        try:
            from src.isolation_filter import IsolationFilter
            
            iso_filter = IsolationFilter(contamination=self.config['contamination'])
            X, y, df = iso_filter.load_data(self.config['selected_features_path'])
            predictions = iso_filter.fit_isolation_forest(X)
            iso_filter.analyze_outliers_by_class(y, predictions)
            df_processed = iso_filter.filter_outliers(df, predictions, mode=self.config['filter_mode'])
            iso_filter.save_filtered_data(df_processed, self.config['filtered_data_path'])
            
            n_anomalies = (predictions == -1).sum()
            self.results['isolation_filtering'] = {
                'total_samples': len(df),
                'anomalies_flagged': n_anomalies,
                'anomaly_rate': f"{(n_anomalies/len(df))*100:.2f}%",
                'status': 'SUCCESS'
            }
            
            logger.info("✓ Isolation filtering completed successfully")
            
        except Exception as e:
            logger.error(f"Isolation filtering failed: {e}")
            raise
    
    def _run_model_training(self):
        """Stage 4: Model Training"""
        print("\n" + "=" * 80)
        print("STAGE 4/5: Model Training (RandomForest)")
        print("=" * 80)
        logger.info("Starting model training...")
        
        try:
            from src.train_model import ModelTrainer
            
            trainer = ModelTrainer(test_size=self.config['test_size'])
            X, y = trainer.load_data(self.config['filtered_data_path'], 
                                    exclude_anomalies=self.config['exclude_anomalies'])
            X_train, X_test, y_train, y_test = trainer.split_data(X, y)
            model = trainer.train_random_forest(n_estimators=self.config['n_estimators'])
            metrics = trainer.evaluate_model()
            trainer.print_evaluation_report(metrics)
            trainer.save_model(self.config['model_path'])
            
            self.results['model_training'] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': f"{metrics['accuracy']:.4f}",
                'f1_score': f"{metrics['f1_score']:.4f}",
                'status': 'SUCCESS'
            }
            
            logger.info("✓ Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _run_shap_explanation(self):
        """Stage 5: SHAP Explainability"""
        print("\n" + "=" * 80)
        print("STAGE 5/5: SHAP Explainability")
        print("=" * 80)
        logger.info("Starting SHAP explanation generation...")
        
        try:
            from src.explain import ModelExplainer
            
            explainer = ModelExplainer(self.config['model_path'])
            explainer.load_model()
            X_test, y_test = explainer.load_test_data(
                self.config['filtered_data_path'],
                sample_size=self.config['shap_sample_size']
            )
            explainer.create_explainer()
            explainer.compute_shap_values()
            explainer.plot_global_importance(output_dir=self.config['output_dir'])
            explainer.plot_multiple_local_explanations(
                n_samples=self.config['n_local_explanations'],
                output_dir=self.config['output_dir']
            )
            
            self.results['shap_explanation'] = {
                'samples_analyzed': len(X_test),
                'global_plots': 2,
                'local_plots': self.config['n_local_explanations'],
                'status': 'SUCCESS'
            }
            
            logger.info("✓ SHAP explanation completed successfully")
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise
    
    def _print_summary(self):
        """Print final pipeline summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Start time:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:      {duration}")
        print("=" * 80)
        
        for stage, results in self.results.items():
            print(f"\n{stage.replace('_', ' ').title()}:")
            for key, value in results.items():
                if key != 'status':
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            print(f"  Status: {results['status']}")
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  - {self.config['features_path']}")
        print(f"  - {self.config['selected_features_path']}")
        print(f"  - {self.config['filtered_data_path']}")
        print(f"  - {self.config['model_path']}")
        print(f"  - {self.config['output_dir']}/global_feature_importance.png")
        print(f"  - {self.config['output_dir']}/global_feature_importance_bar.png")
        print(f"  - {self.config['n_local_explanations']} local explanation plots")
        print("=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="WireXplain-IDS: Complete Pipeline Orchestrator"
    )
    
    # Input/Output paths
    parser.add_argument(
        '--raw-data',
        type=str,
        default='data/raw/02-14-2018.csv',
        help='Path to raw input data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory for output plots'
    )
    
    # Feature selection parameters
    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='Number of top features to select'
    )
    
    # Isolation filter parameters
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of outliers'
    )
    parser.add_argument(
        '--filter-mode',
        type=str,
        choices=['filter', 'flag'],
        default='flag',
        help='Outlier handling mode'
    )
    
    # Model training parameters
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in RandomForest'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--exclude-anomalies',
        action='store_true',
        help='Exclude anomalies from training'
    )
    
    # SHAP parameters
    parser.add_argument(
        '--shap-samples',
        type=int,
        default=1000,
        help='Number of samples for SHAP computation'
    )
    parser.add_argument(
        '--n-local',
        type=int,
        default=4,
        help='Number of local explanations to generate'
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Build configuration
    config = {
        'raw_data_path': args.raw_data,
        'features_path': 'data/processed/features.csv',
        'selected_features_path': 'data/processed/selected_features.csv',
        'filtered_data_path': 'data/processed/filtered_data.csv',
        'model_path': 'models/random_forest_model.pkl',
        'output_dir': args.output_dir,
        'top_n_features': args.top_n,
        'contamination': args.contamination,
        'filter_mode': args.filter_mode,
        'n_estimators': args.n_estimators,
        'test_size': args.test_size,
        'exclude_anomalies': args.exclude_anomalies,
        'shap_sample_size': args.shap_samples,
        'n_local_explanations': args.n_local
    }
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    success = orchestrator.run_full_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
