import wandb
from config import Config, setup_config
from main import main
import copy

experiments = [
    # {
    #     "beta_c_schedule": True,
    #     "exploration_beta_c": -10000.0,
    #     "reanalyze_beta_c": -10000.0,
    #     "exploitation_beta_c": -1000.0,
    # },
    # {
    #     "beta_c_schedule": True,
    #     "exploration_beta_c": -10000.0,
    #     "reanalyze_beta_c": -10000.0,
    #     "exploitation_beta_c": -1000.0,
    # },
    # {
    #     "beta_c_schedule": True,
    #     "exploration_beta_c": -1000.0,
    #     "reanalyze_beta_c": -1000.0,
    #     "exploitation_beta_c": -10.0,
    # },
    # {
    #     "beta_c_schedule": True,
    #     "exploration_beta_c": -1000.0,
    #     "reanalyze_beta_c": -1000.0,
    #     "exploitation_beta_c": -10.0,
    # },
    # {
    #     "beta_c_schedule": False,
    #     "exploration_beta_c": 0.0,
    #     "reanalyze_beta_c": 0.0,
    #     "exploitation_beta_c": 0.0,
    # },
    # {
    #     "beta_c_schedule": False,
    #     "exploration_beta_c": 0.0,
    #     "reanalyze_beta_c": 0.0,
    #     "exploitation_beta_c": 0.0,
    # },
    {
        "env_id": "safety-minatar-freeway",
        "beta_c_schedule": False,
        "exploration_beta_c": 0.0,
        "reanalyze_beta_c": 0.0,
        "exploitation_beta_c": 0.0,
        "max_ube_cost": 3.0**2,
    },
]

for exp in experiments:
    # Start from default config
    config = Config()
    for k, v in exp.items():
        setattr(config, k, v)
    config = setup_config(config)
    print(f"Running experiment with config: {config}")
    main(config)

    if config.track:
        wandb.finish()
