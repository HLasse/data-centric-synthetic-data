echo "Running analyses for data centric thresholds"
python src/application/data_centric_thresholds/data_centric_thresholds_table.py
echo "Running analyses for classification performance by supervised model"
python src/application/main_experiment/eval/classification_performance_by_model.py
echo "Running analyses for performance by dataset"
python src/application/main_experiment/eval/dataset_performance.py
echo "Running analyses for feature selection by model"
python src/application/main_experiment/eval/feature_selection_by_model.py
echo "Running analyses for causal discovery example"
python src/application/main_experiment/eval/causal_discovery_example.py