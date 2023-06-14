from application.constants import RESULTS_DIR
from data_centric_synth.serialization.serialization import load_from_pickle
from optuna.visualization import plot_optimization_history

if __name__ == "__main__":
    hparam_dir = RESULTS_DIR / "hparams"
    for model in hparam_dir.glob("*pkl"):
        model_name = model.stem
        print(f"Best hyperparameters for {model_name}")
        study = load_from_pickle(model)
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)
        print()
        fig = plot_optimization_history(study, target_name=model_name)
        fig.show()
