import numpy as np

from ..base_wrappers.base_physics_wrapper import BasePhysicsWrapper


class HopperDynamicsWrapper(BasePhysicsWrapper):
    def randomizeDynamics(self):
        # 1. body mass, originally 3-6 kg
        self._randomizeMass(0.5, 2.0)

        # 2. joint damping
        self._randomizeDamping(0.5, 2.0)

        # 3. torque scale
        self._randomizeTorqueScale(0.5, 1.5)

    def randomizeDynamicsTest(self):
        # 1. body mass, originally 3-6 kg
        self._randomizeMass(0.9, 1.1)

        # 2. joint damping
        self._randomizeDamping(0.9, 1.1)

        # 3. torque scale
        self._randomizeTorqueScale(0.9, 1.1)

    def getRandomizedParameters(self) -> np.ndarray:
        # 1. Body Mass (5 Bodies). Skip index 0 (World) -> 4 Bodies
        mass = self.model.body_mass[1:].copy()
        # 2. Damping (6 DoF: 3 Root + 3 Joints). Skip root -> 3 Dims
        damp = self.model.dof_damping[self.root_dof_idx :].copy()
        # 3. Torque Scale -> 1 Dim
        torque = np.array([self.torque_scale])

        # Total Check: 4 + 3 + 1 = 8 Parameters
        return np.concatenate([mass, damp, torque])


class Walker2dDynamicsWrapper(BasePhysicsWrapper):
    def randomizeDynamics(self):
        # 1. mass
        self._randomizeMass(0.5, 2.0)

        # 2. joint damping
        self._randomizeDamping(0.5, 2.0)

        # 3. torque scale
        self._randomizeTorqueScale(0.5, 1.5)

    def randomizeDynamicsTest(self):
        # 1. mass
        self._randomizeMass(0.9, 1.1)

        # 2. joint damping
        self._randomizeDamping(0.9, 1.1)

        # 3. torque scale
        self._randomizeTorqueScale(0.9, 1.1)

    def getRandomizedParameters(self) -> np.ndarray:
        # 1. Mass: Skip World (idx 0) -> 7 bodies
        mass = self.model.body_mass[1:].copy()
        # 2. Damping (9 DoF: 3 Root + 6 Joints). Skip root -> 6 dims
        damp = self.model.dof_damping[self.root_dof_idx :].copy()
        # 3. Torque
        torque = np.array([self.torque_scale])

        # Total Check: 8 + 6 = 14 Parameters
        return np.concatenate([mass, damp, torque])


class HalfCheetahDynamicsWrapper(BasePhysicsWrapper):
    def randomizeDynamics(self):
        # 1. friction [0.4, 1.0]
        self._randomizeSlideFriction(0.4, 1.0)

        # 2. body mass, originally 6.0 kg
        self._randomizeMass(0.5, 2.0)

        # 3. joint damping
        self._randomizeDamping(0.5, 2.0)

        # 3. torque scale
        self._randomizeTorqueScale(0.5, 1.5)

    def randomizeDynamicsTest(self):
        # 1. friction [0.4, 1.0]
        self._randomizeSlideFriction(0.9, 1.1)

        # 2. body mass, originally 6.0 kg
        self._randomizeMass(0.9, 1.1)

        # 3. joint damping
        self._randomizeDamping(0.9, 1.1)

        # 3. torque scale
        self._randomizeTorqueScale(0.9, 1.1)

    def getRandomizedParameters(self) -> np.ndarray:
        # 1. Slide Friction (9 Geoms: Floor + 8 Links)
        fric = self.model.geom_friction[:, 0].copy()
        # 2. Body Mass (8 Bodies). Skip World -> 7 Bodies
        mass = self.model.body_mass[1:].copy()
        # 3. Joint Damping (9 DoF: 3 Root + 6 Joints) -> 6 dims
        damp = self.model.dof_damping[self.root_dof_idx :].copy()
        # 3. Torque Scale -> 1 Dim
        torque = np.array([self.torque_scale])

        # Total Check: 9 + 7 + 6 + 1 = 23 Parameters
        return np.concatenate([fric, mass, damp, torque])


class AntDynamicsWrapper(BasePhysicsWrapper):
    def randomizeDynamics(self):
        # 1. friction
        self._randomizeSlideFriction(0.2, 1.0)

        # 2. body mass, scale [0.5x, 2.0x]
        self._randomizeMass(0.5, 2.0)

        # 3. joint damping
        self._randomizeDamping(0.5, 2.0)

        # 4. torque scale
        self._randomizeTorqueScale(0.5, 1.5)

    def randomizeDynamicsTest(self):
        # 1. friction
        self._randomizeSlideFriction(0.9, 1.1)

        # 2. body mass, scale [0.5x, 2.0x]
        self._randomizeMass(0.9, 1.1)

        # 3. joint damping
        self._randomizeDamping(0.9, 1.1)

        # 4. torque scale
        self._randomizeTorqueScale(0.9, 1.1)

    def getRandomizedParameters(self) -> np.ndarray:
        # 1. Slide Friction (14 Geoms: Floor + 13 Links)
        fric = self.model.geom_friction[:, 0].copy()
        # 2. Body Mass (14 Bodies). Skip World -> 13 Bodies
        mass = self.model.body_mass[1:].copy()
        # 3. Damping (14 DoF: 6 Root + 8 Joints) -> 8 dims
        damp = self.model.dof_damping[self.root_dof_idx :].copy()
        # 4. Torque Scale -> 1 Dim
        torque = np.array([self.torque_scale])

        # Total Check: 14 + 13 + 8 + 1 = 36 Parameters
        return np.concatenate([fric, mass, damp, torque])
