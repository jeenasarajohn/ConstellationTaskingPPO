import types
from functools import partial
from typing import Any

import gymnasium
import gymnasium as gym
import numpy as np
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from Basilisk.utilities.orbitalMotion import elem2rv
from numpy import dtype, ndarray

from bsk_rl import sats, obs, act, ConstellationTasking
from bsk_rl.obs.relative_observations import rso_imaged_regions
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import fibonacci_sphere
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief


class RSOSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(dict(prop="one", fn=lambda _: 1.0)),
    ]
    action_spec = [act.Downlink(duration=1e9)]
    dyn_type = types.new_class(
        "Dyn", (dyn.ImagingDynModel, dyn.ConjunctionDynModel, dyn.RSODynModel)
    )
    fsw_type = fsw.ContinuousImagingFSWModel


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def get_val_out_of_dict(val) -> ndarray:
    out = []
    if isinstance(val, dict):
        for v in val.values():
            if isinstance(v, dict):
                [out.append(i) for i in get_val_out_of_dict(v)]
            elif is_iterable(v):
                [out.append(i) for i in v]
            else:
                out.append(v)
    return np.array(out)



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


class RSOInspectionEnv(gymnasium.Env):
    def __init__(self, rso_sat_args, inspector_sat_args, scenario, rewarders):
        super().__init__()
        self.base_env = ConstellationTasking(
            satellites=[
                RSOSat("RSO", sat_args=rso_sat_args),
                InspectorSat("Inspector", sat_args=inspector_sat_args, obs_type=dict),
            ],
            sat_arg_randomizer=sat_arg_randomizer,
            scenario=scenario,
            rewarder=rewarders,
            time_limit=60000,
            sim_rate=5.0,
            log_level="ERROR",
        )

        self.base_env.reset()

        obs_sample = self.base_env.observation_spaces["Inspector"].sample()
        obs_shape = len(get_val_out_of_dict(obs_sample))


        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float64,
        )

        self.action_space = self.base_env.action_space("Inspector")

    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        return get_val_out_of_dict(obs["Inspector"]), info.get("Inspector", {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step({"Inspector": action})
        terminated = terminated["Inspector"]
        truncated = truncated["Inspector"]
        obs = get_val_out_of_dict(obs["Inspector"])
        reward = reward["Inspector"]
        info = info["Inspector"]
        return obs, reward, terminated, truncated, info
