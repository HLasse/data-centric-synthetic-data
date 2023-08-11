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

echo "Running analyses for distributions and statistical tests for main benchmark data"
python src/application/main_experiment/eval/distributions_significance/main_distributions.py
echo "Running analyses for distributions and statistical tests for noise data"
python src/application/main_experiment/eval/distributions_significance/label_noise_distributions.py
python src/application/main_experiment/eval/distributions_significance/label_noise_summary_table.py
echo "Running analyses for postprocessing original data, main benchmark data"
python src/application/main_experiment/eval/distributions_significance/org_data_postprocessing_main_experiment.py
echo "Running analyses for postprocessing original data, label noise data"
python src/application/main_experiment/eval/distributions_significance/org_data_postprocessing_noise.py
echo "Running analyses for additional statistical fidelity metrics"
python src/application/main_experiment/eval/distributions_significance/stat_fid_metrics.py
