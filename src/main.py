from importlib.metadata import version

import numpy as np
import torch
from Basilisk.architecture import bskLogging

from PPOTrainer import PPOGymTrainer
from RSOInspectionEnv import RSOInspectionEnv
from bsk_rl import scene, data
from bsk_rl.sim import fsw


def main(run_name):
    rso_sat_args = dict(
        conjunction_radius=2.0,
        K=7.0 / 20,
        P=35.0 / 20,
        Ki=1e-6,
        dragCoeff=0.0,
        batteryStorageCapacity=1e9,
        storedCharge_Init=1e9,
        wheelSpeeds=[0.0, 0.0, 0.0],
        u_max=1.0,
    )

    inspector_sat_args = dict(
        imageAttErrorRequirement=1.0,
        imageRateErrorRequirement=None,
        instrumentBaudRate=1,
        dataStorageCapacity=1e6,
        batteryStorageCapacity=1e9,
        storedCharge_Init=1e9,
        conjunction_radius=2.0,
        dv_available_init=10.0,
        max_range_radius=1000,
        chief_name="RSO",
        u_max=1.0,
    )

    scenario = scene.SphericalRSO(
        n_points=100,
        radius=1.0,
        theta_max=np.radians(30),
        range_max=250,
        theta_solar_max=np.radians(60),
    )

    rewarders = (
        data.RSOInspectionReward(
            completion_bonus=1.0,
            completion_threshold=0.90,
        ),
        data.ResourceReward(
            resource_fn=lambda sat: sat.fsw.dv_available
            if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel)
            else 0.0,
            reward_weight=np.random.uniform(0.0, 0.5),
        ),
    )
    env = RSOInspectionEnv(rso_sat_args, inspector_sat_args, scenario, rewarders)

    trainer = PPOGymTrainer(
        env,
        run_name=run_name,
        log_dir="./runs",
        ray_local_mode=True,  # easiest debugging; set False for faster execution
        num_gpus=torch.cuda.device_count()
    )

    print("Running on", ("cuda with " + str(torch.cuda.device_count()) + " GPU(s)") if torch.cuda.is_available() else "CPU(s)")

    result = trainer.train(stop_iters=50, stop_reward=475.0, checkpoint_every=10)
    algo = result.algo

    print("TensorBoard logdir:", trainer.tensorboard_logdir)

    env.reset()
    for i in range(4):
        env.step(dict(RSO=0, Inspector=env.action_space("Inspector").sample()))

if __name__ == "__main__":
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
    version("ray")  # Parent package of RLlib
    main("constellation_tasking")
