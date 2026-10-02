import re
from argparse import ArgumentParser
from pathlib import Path

from ldr.adaptation.rma_adapter import RMAAdapter

ENV_NAMES = {"hopper", "walker2d", "halfcheetah", "ant"}
WEIGHT_PATTERN = re.compile(r"^(\S+)_reward_(-?\d+\.?\d*)(\.pth)?$")


def parse_args():
    parser = ArgumentParser(description="Run RMA phase-2 adaptation on a pretrained agent.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment run directory containing backup/ and ckpt/.",
    )
    return parser.parse_args()


def resolve_exp_paths(exp_dir: Path):
    backup_dir = exp_dir / "backup"
    ckpt_dir = exp_dir / "ckpt"
    assert backup_dir.is_dir(), f"Experiment dir {exp_dir} does not contain a backup/ directory!"
    assert ckpt_dir.is_dir(), f"Experiment dir {exp_dir} does not contain a ckpt/ directory!"

    agent_config_path = backup_dir / "rma.yaml"
    assert agent_config_path.is_file(), f"Backup dir {backup_dir} does not contain rma.yaml!"

    env_config_path = None
    for backup_yaml in backup_dir.iterdir():
        if backup_yaml.stem.lower() in ENV_NAMES:
            env_config_path = backup_yaml
            break
    assert env_config_path is not None, f"Backup dir {backup_dir} does not contain an environment yaml file!"

    reward_files = []
    for file in ckpt_dir.iterdir():
        match = WEIGHT_PATTERN.search(file.name)
        assert match, f"current agent weight file: {file.name} has inproper naming!"
        reward_files.append((float(match.group(2)), file))
    assert reward_files, f"Checkpoint dir {ckpt_dir} does not contain a valid weight file!"
    reward_files.sort(reverse=True, key=lambda x: x[0])

    return agent_config_path, env_config_path, reward_files[0][1]


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    agent_config_path, env_config_path, weight_path = resolve_exp_paths(exp_dir)

    adapt_agent_path = Path(__file__).resolve().parents[2] / "configs" / "adapt" / "rma_adapt.yaml"
    config_file = {
        "adapt_agent": str(adapt_agent_path),
        "agent": str(agent_config_path),
        "environment": str(env_config_path),
    }

    adapter = RMAAdapter(config_file=config_file, pretrained_weight_path=str(weight_path))
    adapter.adapt()


if __name__ == "__main__":
    main()
