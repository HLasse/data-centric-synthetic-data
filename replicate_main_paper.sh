echo "Running analyses for main table of results..."
python src/application/main_experiment/eval/stat_fidelity.py
python src/application/main_experiment/eval/summary_table.py
echo "Running analyses for adding label noise..."
python src/application/main_experiment/eval/noise_eval.py
echo "Running analyses for figure 1..."
python src/application/figure1/evaluate_figure1.py
echo "Plots can be found in results/main_experiment/plots"