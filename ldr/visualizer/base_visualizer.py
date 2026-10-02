import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from .palette import PlotPalette
from ldr.agents import makeAgent
from ..agents.ldr_agent import LDRAgent
from ..agents.rma_agent import RMAAgent, RMAAdaptModuleConfig
from ..agents.utils.utils import toTensor
from ldr.common.utils import ParamsManager
from ldr.environments.vec_env.base_vec_env import VecEnv
from ldr.environments import GymEnvConfig, makeVectorizedEnv
from ..agents.utils.time_window_queue import TimeWindowQueue


class BaseVisualizer:

    agent: LDRAgent

    def __init__(self, env_params: dict, agent_params: dict, render: bool = False):
        assert agent_params["agent_name"] == "LDRAgent"
        env_config = self.envConfigFromDict(env_params)
        env_config.num_envs = 1
        env_config.render = render
        self.env = makeVectorizedEnv(env_config)
        self.agent = self.agentFromDict(
            agent_params=agent_params, venv=self.env, weight_path=agent_params["pipeline"]["test"]["model"]
        )

        # bookkeep default dynamic parameters
        self.model = self.env.envs[0].unwrapped.model
        self.default_mass = self.model.body_mass.copy()
        self.default_friction = self.model.geom_friction.copy()
        self.default_damping = self.model.dof_damping.copy()
        self.torque_scale = 1.0

        # create aliases
        self.encoder = self.agent.encoder
        self.decoder = self.agent.decoder
        self.policy = self.agent.policy
        self.env_dim = self.env.num_envs
        self.n_past = self.agent.config.n_past
        self.past_obs_padding = self.agent.past_obs_padding
        self.past_action_padding = self.agent.past_action_padding
        self.obs_dim = self.agent.obs_dim
        self.action_dim = self.agent.action_dim
        self.device = self.agent.device

        # agent utils
        self.obs_queue = TimeWindowQueue(
            env_dim=self.env_dim, n_past=self.n_past + 1, obs_dim=self.obs_dim, padding=self.past_obs_padding
        )
        self.action_queue = TimeWindowQueue(
            env_dim=self.env_dim, n_past=self.n_past, obs_dim=self.action_dim, padding=self.past_action_padding
        )

    @classmethod
    def parseBaseConfig(cls, config_file: dict) -> Tuple[dict, dict]:
        agent_params = ParamsManager.parse(config_file["agent"])
        env_params = ParamsManager.parse(config_file["environment"])
        return env_params, agent_params

    @classmethod
    def envConfigFromDict(cls, env_params: dict) -> GymEnvConfig:
        return GymEnvConfig.fromDict(env_params)

    @classmethod
    def agentFromDict(cls, agent_params: dict, venv: VecEnv, weight_path: Optional[str] = None):
        agent = makeAgent(
            agent_params=agent_params,
            env=venv,
            logger=None,
            recorder=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        if weight_path is not None:
            agent.load(weight_path)
            agent.setTrainingMode(False)
        return agent

    @classmethod
    def setTheme(cls, legend_font_size: int = 16, tick_font_size: int = 14, ax_font_size: int = 14):
        PlotPalette.setPlotTheme(
            legend_font_size=legend_font_size, tick_font_size=tick_font_size, ax_font_size=ax_font_size
        )

    def _rollout(self, seed: int):
        latents = []

        self.env.seed(seed)
        obs = self.env.reset()
        dones_record = np.zeros(self.env_dim, dtype=np.bool_)
        while not dones_record[0]:
            self.obs_queue.append(obs)
            full_stacked_obs = self.obs_queue.get()
            stacked_past_actions = self.action_queue.get()
            current_obs, history_obs_actions = self.agent._preparePolicyAndEncoderInput(
                full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
            )
            latent_z, actions = self._predict(curr_obs=current_obs, stacked_past_obs_actions=history_obs_actions)
            latents.append(latent_z.squeeze(0))
            self.action_queue.append(actions)
            env_actions = self.agent._toEnvAction(actions)
            # apply torque scale
            obs, _, dones, _ = self.env.step(env_actions * self.torque_scale)
            dones_record[0] = dones[0]

        self.obs_queue.resetDoneEntries(dones_record)
        self.action_queue.resetDoneEntries(dones_record)
        # shape (T, latent_dim)
        return np.array(latents)

    def _predict(self, curr_obs: np.ndarray, stacked_past_obs_actions: np.ndarray):
        stacked_obs_tensor = toTensor(stacked_past_obs_actions, device=self.device)
        curr_obs_tensor = toTensor(curr_obs, device=self.device)
        with torch.no_grad():
            latent_z = self.encoder(stacked_obs_tensor)
            # deterministic sampling from encoder
            z = latent_z.loc
            curr_obs_with_latent = torch.cat([curr_obs_tensor, z], dim=-1)
            actions = self.policy(obs=curr_obs_with_latent, deterministic=True)
            actions = actions.cpu().numpy().reshape(-1, self.action_dim)
            z = z.cpu().numpy()
            return z, actions

    def _manuallySetDynamics(self, param_name: str, scalar: float):
        """
        Modifies dynamics by scaling the default values by a scalar.
        This creates a 1D manifold in the high-D parameter space.
        """
        tqdm.write(f"setting {param_name} scale to {scalar}")
        if "mass" == param_name:
            self.model.body_mass[:] = self.default_mass * scalar
        elif "friction" == param_name:
            new_friction = self.default_friction.copy()
            new_friction[:, 0] *= scalar
            self.model.geom_friction[:] = new_friction
        elif "damping" == param_name:
            self.model.dof_damping[:] = self.default_damping * scalar
        elif "torque_scale" == param_name:
            self.torque_scale *= scalar
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def _resetDynamics(self):
        self.model.body_mass[:] = self.default_mass
        self.model.geom_friction[:] = self.default_friction
        self.model.dof_damping[:] = self.default_damping
        self.torque_scale = 1.0


class BaseVisualizerRMA:

    agent: RMAAgent

    def __init__(
        self,
        env_params: dict,
        agent_params: dict,
        render: bool = False,
        adapt_config_path: Optional[str] = None,
        adapt_module_path: Optional[str] = None,
    ):
        env_config = self.envConfigFromDict(env_params)
        env_config.num_envs = 1
        env_config.render = render
        self.env = makeVectorizedEnv(env_config)
        self.agent = self.agentFromDict(
            agent_params=agent_params, venv=self.env, weight_path=agent_params["pipeline"]["test"]["model"]
        )

        if adapt_config_path is not None and adapt_module_path is not None:
            self.agent.loadAdaptationModule(
                config=RMAAdaptModuleConfig.fromYaml(adapt_config_path),
                file_path=adapt_module_path,
            )

        # bookkeep default dynamic parameters
        self.model = self.env.envs[0].unwrapped.model
        self.default_mass = self.model.body_mass.copy()
        self.default_friction = self.model.geom_friction.copy()
        self.default_damping = self.model.dof_damping.copy()
        self.torque_scale = 1.0

        # create aliases
        self.encoder = self.agent.student
        self.policy = self.agent.policy
        self.env_dim = self.env.num_envs
        self.n_past = self.encoder.n_past
        self.past_obs_padding = None
        self.past_action_padding = 0
        self.obs_dim = self.agent.obs_dim
        self.action_dim = self.agent.action_dim
        self.factor_dim = self.agent.factor_dim
        self.device = self.agent.device

        # agent utils
        self.obs_queue = TimeWindowQueue(
            env_dim=self.env_dim,
            n_past=self.n_past + 1,
            obs_dim=self.obs_dim - self.factor_dim,
            padding=self.past_obs_padding,
        )
        self.action_queue = TimeWindowQueue(
            env_dim=self.env_dim, n_past=self.n_past, obs_dim=self.action_dim, padding=self.past_action_padding
        )

    @classmethod
    def parseBaseConfig(cls, config_file: dict) -> Tuple[dict, dict]:
        agent_params = ParamsManager.parse(config_file["agent"])
        env_params = ParamsManager.parse(config_file["environment"])
        return env_params, agent_params

    @classmethod
    def envConfigFromDict(cls, env_params: dict) -> GymEnvConfig:
        return GymEnvConfig.fromDict(env_params)

    @classmethod
    def agentFromDict(cls, agent_params: dict, venv: VecEnv, weight_path: Optional[str] = None):
        agent = makeAgent(
            agent_params=agent_params,
            env=venv,
            logger=None,
            recorder=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        if weight_path is not None:
            agent.load(weight_path)
            agent.setTrainingMode(False)
        return agent

    @classmethod
    def setTheme(cls, legend_font_size: int = 16, tick_font_size: int = 14, ax_font_size: int = 14):
        PlotPalette.setPlotTheme(
            legend_font_size=legend_font_size, tick_font_size=tick_font_size, ax_font_size=ax_font_size
        )

    def _rollout(self, seed: int):
        latents = []

        self.env.seed(seed)
        obs = self.env.reset()
        dones_record = np.zeros(self.env_dim, dtype=np.bool_)
        while not dones_record[0]:
            self.obs_queue.append(obs[:, : -self.factor_dim])
            full_stacked_obs = self.obs_queue.get()
            stacked_past_actions = self.action_queue.get()
            current_obs, history_obs_actions = self.agent._preparePolicyAndEncoderInput(
                full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
            )
            latent_z, actions = self._predict(curr_obs=current_obs, stacked_past_obs_actions=history_obs_actions)
            latents.append(latent_z.squeeze(0))
            self.action_queue.append(actions)
            env_actions = self.agent._toEnvAction(actions)
            # apply torque scale
            obs, _, dones, _ = self.env.step(env_actions * self.torque_scale)
            dones_record[0] = dones[0]

        self.obs_queue.resetDoneEntries(dones_record)
        self.action_queue.resetDoneEntries(dones_record)
        # shape (T, latent_dim)
        return np.array(latents)

    def _predict(self, curr_obs: np.ndarray, stacked_past_obs_actions: np.ndarray):
        stacked_obs_tensor = toTensor(stacked_past_obs_actions, device=self.device)
        curr_obs_tensor = toTensor(curr_obs, device=self.device)
        with torch.no_grad():
            latent_z = self.encoder(stacked_obs_tensor)
            curr_obs_with_latent = torch.cat([curr_obs_tensor, latent_z], dim=-1)
            actions = self.agent._forwardActor(curr_obs_with_latent, deterministic=True)
            actions = actions.cpu().numpy().reshape(-1, self.action_dim)
            latent_z = latent_z.cpu().numpy()
            return latent_z, actions

    def _manuallySetDynamics(self, param_name: str, scalar: float):
        """
        Modifies dynamics by scaling the default values by a scalar.
        This creates a 1D manifold in the high-D parameter space.
        """
        tqdm.write(f"setting {param_name} scale to {scalar}")
        if "mass" == param_name:
            self.model.body_mass[:] = self.default_mass * scalar
        elif "friction" == param_name:
            new_friction = self.default_friction.copy()
            new_friction[:, 0] *= scalar
            self.model.geom_friction[:] = new_friction
        elif "damping" == param_name:
            self.model.dof_damping[:] = self.default_damping * scalar
        elif "torque_scale" == param_name:
            self.torque_scale *= scalar
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def _resetDynamics(self):
        self.model.body_mass[:] = self.default_mass
        self.model.geom_friction[:] = self.default_friction
        self.model.dof_damping[:] = self.default_damping
        self.torque_scale = 1.0
