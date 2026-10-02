import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base_evaluator import BaseEvaluator
from ldr.common.utils import Logger, Recorder


class CommonEvaluator(BaseEvaluator):
    def __init__(
        self,
        config_file: dict,
        pretrained_weights_path: str,
        required_experience_length: int = int(1e9),
        time_varying: bool = False,
    ):
        super().__init__(
            config_file=config_file,
            pretrained_weight_path=pretrained_weights_path,
            required_experience_length=required_experience_length,
            time_varying=time_varying,
        )

    def evalGivenScenario(self, eval_episodes: int):
        eval_func = self.agent.evaluate
        return super()._evalGivenScenario(eval_func=eval_func, eval_episodes=eval_episodes)

    def evalOverParameter(self, param_name, scan_values, episode_each=1):
        eval_func = self.agent.evaluate
        return super()._evalOverParameter(
            eval_func=eval_func, param_name=param_name, scan_values=scan_values, episode_each=episode_each
        )


class RMAEvaluator(BaseEvaluator):
    def __init__(
        self,
        config_file: dict,
        agent_pretrained_weight_path: str,
        adapt_module_weight_path: str,
        required_experience_length: int = int(1e9),
        time_varying: bool = False,
    ):
        from ldr.agents.rma_agent import RMAAdaptModuleConfig

        super().__init__(
            config_file=config_file,
            pretrained_weight_path=agent_pretrained_weight_path,
            required_experience_length=required_experience_length,
            time_varying=time_varying,
        )
        self.agent.loadAdaptationModule(
            RMAAdaptModuleConfig.fromDict(self.agent_adapt_params), file_path=adapt_module_weight_path
        )

    def evalGivenScenario(self, eval_episodes: int):
        eval_func = self.agent.evaluateWithModule
        return super()._evalGivenScenario(eval_func=eval_func, eval_episodes=eval_episodes)

    def evalOverParameter(self, param_name, scan_values, episode_each=1):
        eval_func = self.agent.evaluateWithModule
        return super()._evalOverParameter(
            eval_func=eval_func, param_name=param_name, scan_values=scan_values, episode_each=episode_each
        )


class CMAEvaluator(BaseEvaluator):
    def __init__(
        self,
        config_file: dict,
        agent_pretrained_weight_path: str,
        required_experience_length: int = int(1e9),
        time_varying: bool = False,
    ):
        from ldr.agents.rma_agent import RMACMAConfig

        super().__init__(
            config_file=config_file,
            pretrained_weight_path=agent_pretrained_weight_path,
            required_experience_length=required_experience_length,
            time_varying=time_varying,
        )

        # logger and recorder for process recording
        weight_path_cls = Path(agent_pretrained_weight_path)
        org_dir = weight_path_cls.parent.parent
        self.logger = Logger(log_level="INFO", resume=True, log_dir=org_dir, backup_list=[self.agent_adapt_yaml_path])
        self.recorder = Recorder(self.logger.tb_dir)

        self.agent.logger = self.logger
        self.agent.recorder = self.recorder

        self.adapt_config = RMACMAConfig.fromDict(self.agent_adapt_params)

        # sync training env with test env
        self.train_model = self.venv.envs[0].unwrapped.model
        self.train_torque_scale = 1.0

    def evalGivenScenario(self, eval_episodes: int):
        rewards = []

        for i in tqdm(range(eval_episodes)):
            identified_z = self.agent.adaptWithCMA(config=self.adapt_config)
            reward = self.agent.evaluateWithCMA(
                identified_z=identified_z, env=self.test_venv, eval_num=1, seed=i, action_scale=self.torque_scale
            )
            tqdm.write(f"reward: {reward}")
            rewards.append(reward)

        mean = np.mean(rewards)
        std = np.std(rewards)

        self.venv.close()

        return mean, std

    def evalOverParameter(self, param_name: str, scan_values: np.ndarray, episode_each: int = 1):
        rewards = []
        for val in tqdm(scan_values):
            # rand test environment
            self.rand_fn[param_name](val)
            # sync train with test
            self._syncTrainWithTest()
            # adapt agent in train environment (synced with test)
            identified_z = self.agent.adaptWithCMA(config=self.adapt_config, action_scale=self.train_torque_scale)
            reward = self.agent.evaluateWithCMA(identified_z=identified_z, env=self.test_venv, eval_num=episode_each)
            rewards.append(reward)
            # change parameter back to default value in case we eval over different parameters
            self.rand_fn[param_name](1.0)

        self.venv.close()

        return rewards

    def _syncTrainWithTest(self):
        self.train_model.geom_friction[:] = self.model.geom_friction.copy()
        self.train_model.body_mass[:] = self.model.body_mass.copy()
        self.train_model.dof_damping[:] = self.model.dof_damping.copy()
        self.train_torque_scale = self.torque_scale
