import re
import sys
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

from ldr.core.evaluator import RMAEvaluator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def parseArgs():
    parser = ArgumentParser(description="Evaluate RMA agent with same parameter in different environments")
    parser.add_argument("--agent", type=str, required=True, help="Name of the agent.")
    parser.add_argument("--base_exp_dir", type=str, required=True, help="Base directory for the weights.")
    parser.add_argument(
        "--mode",
        choices=["normal", "varenv"],
        required=True,
        help="Evaluation mode: 'normal' (in-distribution, OOD, structural failure) or 'varenv' (time-varying dynamics).",
    )

    return parser.parse_args()


def eval():
    print(f"------ In Distribution ------\n")
    for agent_config, (agent_pretrained_weight_path, adapt_module_weight_path), env_config in zip(
        agent_config_path, weights_path, env_path
    ):
        config_file["agent"] = agent_config
        config_file["environment"] = env_config
        curr_env = env_config.split("/")[-1].split(".")[0]

        evaluator = RMAEvaluator(
            config_file=config_file,
            agent_pretrained_weight_path=agent_pretrained_weight_path,
            adapt_module_weight_path=adapt_module_weight_path,
            required_experience_length=1,
        )

        mean, std = evaluator.evalGivenScenario(eval_episodes=5)
        print(f"*" * 20)
        print(f"agent: {str(agent_config)}\n" f"env: {curr_env}" f"\nreward: {mean:.2f}+/-{std:.2f}")
        print(f"*" * 20)

    print(f"------ Out of Distribution ------\n")
    for agent_config, (agent_pretrained_weight_path, adapt_module_weight_path), env_config in zip(
        agent_config_path, weights_path, env_path
    ):
        config_file["agent"] = agent_config
        config_file["environment"] = env_config
        curr_env = env_config.split("/")[-1].split(".")[0]

        evaluator = RMAEvaluator(
            config_file=config_file,
            agent_pretrained_weight_path=agent_pretrained_weight_path,
            adapt_module_weight_path=adapt_module_weight_path,
            required_experience_length=int(1e9),
        )

        choice = "mass"
        if curr_env in ["walker2d", "ant"]:
            choice = "damping"

        if "damping" == choice:
            mean = evaluator.evalOverParameter(param_name="damping", scan_values=[0.3, 1.0, 2.2], episode_each=5)
            print(f"*" * 20)
            print(f"agent: {str(agent_config)}\n" f"env: {curr_env}:{choice}" f"\nreward: {mean}")
            print(f"*" * 20)

        elif "mass" == choice:
            mean = evaluator.evalOverParameter(param_name="mass", scan_values=[0.5, 1.0, 2.0], episode_each=5)
            print(f"*" * 20)
            print(f"agent: {str(agent_config)}\n" f"env: {curr_env}:{choice}" f"\nreward: {mean}")
            print(f"*" * 20)

    print(f"------ Structural Failure ------\n")
    config_file["adapt_environment"] = str(CONFIG_DIR / "environments" / "adapt" / "adapt_env.yaml")
    for agent_config, (agent_pretrained_weight_path, adapt_module_weight_path), env_config in zip(
        agent_config_path, weights_path, env_path
    ):
        config_file["agent"] = agent_config
        config_file["environment"] = env_config
        curr_env = env_config.split("/")[-1].split(".")[0]

        evaluator = RMAEvaluator(
            config_file=config_file,
            agent_pretrained_weight_path=agent_pretrained_weight_path,
            adapt_module_weight_path=adapt_module_weight_path,
            required_experience_length=int(1e9),
        )

        mean, std = evaluator.evalGivenScenario(eval_episodes=5)
        print(f"*" * 20)
        print(f"agent: {str(agent_config)}\n" f"env: {curr_env}" f"\nreward: {mean:.2f}+/-{std:.2f}")
        print(f"*" * 20)


def evalVarEnv():
    print(f"------ Var Env ------\n")
    for agent_config, (agent_pretrained_weight_path, adapt_module_weight_path), env_config in zip(
        agent_config_path, weights_path, env_path
    ):
        config_file["agent"] = agent_config
        config_file["environment"] = env_config
        curr_env = env_config.split("/")[-1].split(".")[0]

        evaluator = RMAEvaluator(
            config_file=config_file,
            agent_pretrained_weight_path=agent_pretrained_weight_path,
            adapt_module_weight_path=adapt_module_weight_path,
            required_experience_length=200,
            time_varying=True,
        )

        mean, std = evaluator.evalGivenScenario(eval_episodes=5)
        print(f"*" * 20)
        print(f"agent: {str(agent_config)}\n" f"env: {curr_env}" f"\nreward: {mean:.2f}+/-{std:.2f}")
        print(f"*" * 20)


if __name__ == "__main__":
    args = parseArgs()
    model_name = args.agent
    base_exp_dir = Path(args.base_exp_dir)

    config_file = {
        "adapt_agent": str(CONFIG_DIR / "adapt" / "rma_adapt.yaml"),
    }

    # replace agent key with backup-ed files
    weights_path, agent_config_path = [], []
    env_path = []
    all_dirs = list(base_exp_dir.iterdir())
    timestamp_pattern = re.compile(r"(\d{8}_\d{6})")

    def extract_timestamp(path: Path):
        match = timestamp_pattern.search(path.name)
        assert match, f"current dir: {path.name} has inproper naming!"
        return match.group(1)

    sorted_paths = sorted(all_dirs, key=extract_timestamp)

    for dir in sorted_paths:
        curr_dir = base_exp_dir / dir
        backup_dir = curr_dir / "backup"
        backup_yamls = [str(each) for each in backup_dir.iterdir()]
        curr_agent_path = backup_dir / f"{model_name}.yaml"

        agent_config_path.append(curr_agent_path)

        has_env = False
        for each_backup_yaml in backup_yamls:
            check_prefix = each_backup_yaml.split("/")[-1].split(".")[0]
            if check_prefix.lower() in ["hopper", "walker2d", "halfcheetah", "ant"]:
                env_path.append(each_backup_yaml)
                has_env = True
                break

        assert has_env, f"RMA dir: {str(backup_dir)} does not contain a environment yaml file!"

        # weights path
        pattern = re.compile(r"^(\S+)_reward_(-?\d+\.?\d*).pth$")
        reward_files = []
        for file in (curr_dir / "ckpt").iterdir():
            match = pattern.search(file.name)
            assert match, f"current agent weight file: {file.name} has inproper naming!"
            reward = float(match.group(2))
            reward_files.append((reward, file))
        reward_files.sort(reverse=True, key=lambda x: x[0])

        weights_path.append((reward_files[0][1], str(curr_dir / "module_adapt" / "module_adapt.pth")))

    mode = args.mode
    if "normal" == mode:
        eval()
    elif "varenv" == mode:
        evalVarEnv()
