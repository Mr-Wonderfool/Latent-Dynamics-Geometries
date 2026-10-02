import gymnasium as gym


class BasePhysicsWrapper(gym.Wrapper):
    WORKER_PHYSICS_CHANGE_SIGNAL_STR = "dynamics_changed"

    def __init__(self, env, required_experience_length: int, time_varying: bool = False):
        """
        Args:
            required_experience_length: Steps (time-varying) or episodes (static) before dynamics change.
            time_varying: If True, randomize dynamics mid-episode via randomizeDynamicsTest().
        """
        super().__init__(env)

        self.required_experience_length = required_experience_length
        self.time_varying = time_varying
        self.experience_counter = 0

        self.model = self.unwrapped.model

        # store initial defaults to apply relative noise or reset
        self.default_mass = self.model.body_mass.copy()
        self.default_friction = self.model.geom_friction.copy()
        self.default_damping = self.model.dof_damping.copy()
        # for restitution, MuJoCo uses solref/solimp, but we will mostly rely on friction/mass
        self.torque_scale = 1.0

        # Detect DoF offset for Root (3 for 2D bots, 6 for 3D Ant)
        # Ant-v5/v4 has 6 root dofs (free joint)
        # Hopper/Walker/Cheetah have 3 root dofs (slide x, slide z, hinge y)
        if "Ant" in env.unwrapped.spec.id:
            self.root_dof_idx = 6
        else:
            self.root_dof_idx = 3

    def reset(self, **kwargs):
        if self.time_varying:
            obs, info = self.env.reset(**kwargs)
            info[self.WORKER_PHYSICS_CHANGE_SIGNAL_STR] = False
            return obs, info

        change_occurred = False
        if self.experience_counter >= self.required_experience_length:
            self.experience_counter = 0
            self.randomizeDynamics()
            change_occurred = True

        obs, info = self.env.reset(**kwargs)
        info[self.WORKER_PHYSICS_CHANGE_SIGNAL_STR] = change_occurred

        return obs, info

    def step(self, action):
        self.experience_counter += 1
        scaled_action = self.torque_scale * action

        if self.time_varying:
            ret = self.env.step(scaled_action)
            if self.experience_counter >= self.required_experience_length:
                self.experience_counter = 0
                self.randomizeDynamicsTest()
            return ret

        return self.env.step(scaled_action)

    def randomizeDynamics(self):
        """
        Randomize internal dynamics for model, to be implemented by child wrappers.
        """
        raise NotImplementedError()

    def randomizeDynamicsTest(self):
        """
        Change dynamics attributes during the episode to simulate time-varying dynamics changes.
        """

    def getRandomizedParameters(self):
        """
        Returns a flat numpy array of the current randomized dynamics parameters.
        Must be implemented by child classes to ensure order matches randomization.
        """
        raise NotImplementedError()

    def _randomizeSlideFriction(self, lower_scale: float, upper_scale: float):
        new_fric = self.default_friction.copy()
        new_fric[:, 0] *= self._uniformRandomizeVal(lower_scale, upper_scale, size=new_fric.shape[0])
        self.model.geom_friction[:] = new_fric

    def _randomizeMass(self, lower_scale: float, upper_scale: float):
        mass_scale = self._uniformRandomizeVal(lower_scale, upper_scale, size=self.default_mass.shape)
        self.model.body_mass[:] = self.default_mass * mass_scale

    def _randomizeDamping(self, lower_scale: float, upper_scale: float):
        damping_scale = self._uniformRandomizeVal(lower_scale, upper_scale, size=self.default_damping.shape)
        self.model.dof_damping[:] = self.default_damping * damping_scale

    def _randomizeTorqueScale(self, lower_scale: float, upper_scale: float):
        self.torque_scale = self._uniformRandomizeVal(lower_scale, upper_scale, size=1).item()

    def _uniformRandomizeVal(self, low, high, size):
        return self.np_random.uniform(low, high, size=size)
