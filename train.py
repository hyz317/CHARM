import argparse
from pathlib import Path

from charm.hair_transformer_trainer import create_model, HairTransformerTrainer
from charm.hair_dataset import create_dataset
from charm.utils import load_yaml, dump_yaml, path_mkdir, exists
from charm.utils.logger import create_logger, print_log
from charm.utils.scheduler import get_scheduler


PROJECT_PATH = Path(__file__).parent
CONFIGS_PATH = PROJECT_PATH / 'configs'
RUNS_PATH = PROJECT_PATH / 'runs-hair'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-c', '--config', type=str, help='Config file name', default='train.yml')
    args = parser.parse_args()

    cfg = load_yaml(CONFIGS_PATH / args.config)

    if (RUNS_PATH / args.tag).exists():
        working_dir = RUNS_PATH / args.tag
    else:
        working_dir = path_mkdir(RUNS_PATH / args.tag)
    create_logger(working_dir)
    dump_yaml(cfg, working_dir / Path(args.config).name)
    dataset = create_dataset(cfg['dataset'])
    if 'val_dataset' in cfg:
        val_dataset = create_dataset(cfg['val_dataset'])
    else:
        val_dataset = None
    transformer = create_model(cfg['model'])


    print_log(f'Trainer init: working_dir={working_dir}')
    scheduler_name = cfg['training'].pop('scheduler', None)
    cfg['training']['scheduler'] = get_scheduler(scheduler_name) if exists(scheduler_name) else None

    transformer_trainer = HairTransformerTrainer(
        model=transformer,
        dataset=dataset,
        val_dataset=val_dataset,
        checkpoint_folder=working_dir/'checkpoints',
        **cfg['training']
    )
    transformer_trainer()