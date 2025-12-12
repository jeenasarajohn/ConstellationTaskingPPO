from importlib.metadata import version

import gymnasium

from PPOTrainer import PPOGymTrainer
from bsk_rl import sats, obs, act, ConstellationTasking, scene, data
from bsk_rl.obs.relative_observations import rso_imaged_regions
from bsk_rl.utils.orbital import fibonacci_sphere
from bsk_rl.sim import dyn, fsw
import types
import numpy as np
from Basilisk.architecture import bskLogging
from functools import partial
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP


class RSOSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(dict(prop="one", fn=lambda _: 1.0)),
    ]
    action_spec = [act.Downlink(duration=1e9)]
    dyn_type = types.new_class(
        "Dyn", (dyn.ImagingDynModel, dyn.ConjunctionDynModel, dyn.RSODynModel)
    )
    fsw_type = fsw.ContinuousImagingFSWModel

def sun_hat_chief(self, other):
    r_SN_N = (
        self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = self.dynamics.r_BN_N
    r_SN_N = np.array(r_SN_N)
    r_SB_N = r_SN_N - r_BN_N
    r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)
    HN = other.dynamics.HN
    return HN @ r_SB_N_hat


class InspectorSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="dv_available", norm=10),
            dict(prop="inclination", norm=np.pi),
            dict(prop="eccentricity", norm=0.1),
            dict(prop="semi_major_axis", norm=7000),
            dict(prop="ascending_node", norm=2 * np.pi),
            dict(prop="argument_of_periapsis", norm=2 * np.pi),
            dict(prop="true_anomaly", norm=2 * np.pi),
            dict(prop="beta_angle", norm=np.pi),
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500),
            dict(prop="v_DC_Hc", norm=5),
            dict(
                prop="rso_imaged_regions",
                fn=partial(
                    rso_imaged_regions,
                    region_centers=fibonacci_sphere(15),
                    frame="chief_hill",
                ),
            ),
            dict(prop="sun_hat_Hc", fn=sun_hat_chief),
            chief_name="RSO",
        ),
        obs.Eclipse(norm=5700),
        obs.Time(),
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=1.0,
            max_drift_duration=5700.0 * 2,
            fsw_action="action_inspect_rso",
        )
    ]
    dyn_type = types.new_class(
        "Dyn",
        (
            dyn.MaxRangeDynModel,
            dyn.ConjunctionDynModel,
            dyn.RSOInspectorDynModel,
        ),
    )
    fsw_type = types.new_class(
        "FSW",
        (
            fsw.SteeringFSWModel,
            fsw.MagicOrbitalManeuverFSWModel,
            fsw.RSOInspectorFSWModel,
        ),
    )

def sat_arg_randomizer(satellites):
    # Generate the RSO orbit
    R_E = 6371.0  # km
    a = R_E + np.random.uniform(500, 1100)
    e = np.random.uniform(0.0, min(1 - (R_E + 500) / a, (R_E + 1100) / a - 1))
    chief_orbit = random_orbit(a=a, e=e)

    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]

    # Generate the inspector initial states.
    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO",
            chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate(
                    (
                        random_unit_vector() * np.random.uniform(250, 750),
                        random_unit_vector() * np.random.uniform(0, 1.0),
                    )
                ),
            },
        )
        args.update(relative_randomizer([rso, inspector]))

    # Align RSO Hill frame for initial nadir pointing
    mu = rso.sat_args_generator["mu"]
    r_N, v_N = elem2rv(mu, args[rso]["oe"])

    r_hat = r_N / np.linalg.norm(r_N)
    v_hat = v_N / np.linalg.norm(v_N)
    x = r_hat
    z = np.cross(r_hat, v_hat)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    HN = np.array([x, y, z])
    BH = np.eye(3)

    a = chief_orbit.a
    T = np.sqrt(a**3 / mu) * 2 * np.pi
    omega_BN_N = z * 2 * np.pi / T

    args[rso]["sigma_init"] = C2MRP(BH @ HN)
    args[rso]["omega_init"] = BH @ HN @ omega_BN_N

    return args


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

    env = ConstellationTasking(
        satellites=[
            RSOSat("RSO", sat_args=rso_sat_args),
            InspectorSat("Inspector", sat_args=inspector_sat_args, obs_type=dict),
        ],
        sat_arg_randomizer=sat_arg_randomizer,
        scenario=scenario,
        rewarder=rewarders,
        time_limit=60000,
        sim_rate=5.0,
        log_level="INFO",
    )

    print(isinstance(env, gymnasium.Env))

    trainer = PPOGymTrainer(
        env,
        run_name=run_name,
        log_dir="./runs",
        ray_local_mode=True,  # easiest debugging; set False for faster execution
    )

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