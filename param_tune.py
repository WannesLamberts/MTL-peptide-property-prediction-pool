import optuna
import pandas as pd
from argparse import Namespace

from optuna.samplers import TPESampler
import logging

from train import post_process_args, train

from src.util import (
    split_run_config,
    DEFAULT_CONFIG,
    check_checkpoint_path,
)
import argparse



def objective(trial,args):
    """Objective function for Optuna optimization."""
    # Define hyperparameters to be tuned
    params = {
        "lr": trial.suggest_categorical("lr",[0.0001,0.00033,0.001]),
        'optim': trial.suggest_categorical("optim",['adam','adamw','SGD']),
        'scheduler': trial.suggest_categorical("scheduler",['warmup_decay_cos','warmup','none']),
        'dropout_mlp': trial.suggest_categorical("dropout_mlp",[0,0.1,0.3]),
        'hidden_size_mlp': trial.suggest_categorical("hidden_size_mlp",['512-256-128','256-128','128','128-64-32']),
        'activation_mlp': trial.suggest_categorical("activation_mlp",['relu','tanh','sigmoid'])
    }

    # Check for duplicates using original params
    all_trials = trial.study.get_trials(deepcopy=False)
    for past_trial in all_trials:
        if (past_trial.state == optuna.trial.TrialState.COMPLETE and
            past_trial.number != trial.number and
            past_trial.params == params):  # Compare original params
            print(f"Duplicate found! Trial {past_trial.number} already tested these params")
            raise optuna.TrialPruned()  # This makes the trial disappear from plots


    params['hidden_size_mlp'] = [int(size) for size in str(params["hidden_size_mlp"]).split("-")]
    args.config=f"{args.type}_hpc_{trial.number}"
    try:
        results = run_tune(params,**args.__dict__)
        # Return MSE for optimization and the full results for logging
        return results
    except Exception as e:
        # Log the error and return a poor result
        logging.error(f"Trial failed with parameters {params}: {str(e)}")
        return float('inf') # Return a bad score and None for failed trials


def run_tune(params,**kwargs):
    config_dict = DEFAULT_CONFIG | kwargs | params
    run_config = Namespace(**config_dict)
    run_config = post_process_args(run_config)
    return train(run_config)

def run_optimization(args, n_trials=100, study_name="hyperparameter_optimization"):
    """
    Run the full Optuna optimization process.

    Args:
        n_trials: Number of trials to run
        study_name: Name of the study

    Returns:
        The Optuna study object and best parameters
    """
    # Create a new study or load existing one
    storage_name = f"sqlite:///{study_name}.db?timeout=30"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize"
    )

    # Now create sampler with dynamic seed based on existing trials
    dynamic_seed = args.optuna + len(study.trials)
    sampler = optuna.samplers.TPESampler(
        seed=dynamic_seed,
        n_startup_trials=10
    )

    # Update the study's sampler
    study.sampler = sampler

    # Run the optimization
    study.optimize(
        lambda trial: objective(trial,args),
        n_trials=n_trials,
        catch=(Exception,)
    )

    # Log the best parameters and score
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best score: {study.best_trial.value}")
    logging.info(f"Best parameters: {study.best_trial.params}")

    return study, study.best_params

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning configuration or create new configurations"
    )
    parser.add_argument("-i", type=int, default=None)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="supervised",
        choices=["supervised", "pretrain"],
    )
    parser.add_argument(
        "-p",
        "--pretrained-model",
        default="own",
        type=str,
        choices=["none", "tape", "own"],
    )
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--train-i", type=str)
    parser.add_argument("--val-i", type=str)
    parser.add_argument("--test-i", type=str)
    parser.add_argument("--vocab-file", type=str)
    parser.add_argument("--scalers-file", type=str)
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("-a", "--accumulate-batches", default=1, type=int)
    parser.add_argument(
        "--lookup",
        default=None,
        type=str,
        help="the lookup table for the pools",
    )

    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=None,
        help="Index of the checkpoint path to use from config/checkpoints.csv. This is an alternative to "
        "giving the full checkpoint path with '--checkpoint-path'",
    )
    parser.add_argument(
        "--epochs",
        default=-1,
        type=int,
        help="the lookup table for the pools",
    )
    parser.add_argument(
        "--time",
        default=None,
        type=int,
        help="the lookup table for the pools",
    )
    parser.add_argument(
        "--type",
        default="pool",
        type=str,
        help="the lookup table for the pools",
    )
    parser.add_argument(
        "--amount",
        default=10,
        type=int,
        help="the lookup table for the pools",
    )

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--optuna', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    args.hpt_config = None

    return args


if __name__ == "__main__":
    args = parse_args()
    # Run the optimization
    study, best_params = run_optimization(args,n_trials=args.amount,study_name=f"{args.type}_HPC")

