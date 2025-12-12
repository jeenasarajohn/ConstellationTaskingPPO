from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    raise ImportError("Install gymnasium: pip install gymnasium")

from torch.utils.tensorboard import SummaryWriter


_ENV_SINGLETON = None



def make_singleton_env(_cfg):
    # IMPORTANT: must be top-level so Ray can pickle the function reference
    # without trying to pickle the environment instance.
    global _ENV_SINGLETON
    if _ENV_SINGLETON is None:
        raise RuntimeError("ENV singleton not set. Set _ENV_SINGLETON before building the algo.")
    return _ENV_SINGLETON


@dataclass(frozen=True)
class TrainResult:
    algo: Any  # RLlib Algorithm (PPO)
    last_metrics: Dict[str, Any]
    checkpoint_path: Optional[str]


class PPOGymTrainer:
    """
    Explicit PPO training loop using RLlib (no Ray Tune).
    Accepts a *fully initialized* Gymnasium env instance.

    Constraints:
      - When passing an env instance, we enforce num_workers=0 (single process),
        because multiple RLlib workers require multiple env instances.

    TensorBoard:
      - Uses SummaryWriter and logs key metrics each iteration to `tensorboard_logdir`.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        run_name: str = "rllib_ppo",
        log_dir: str = "./runs",
        seed: int = 0,
        num_gpus: int = 0,
        # PPO hyperparams (minimal sensible defaults)
        train_batch_size: int = 4000,
        sgd_minibatch_size: int = 256,
        num_sgd_iter: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        # Ray init options (RLlib needs Ray; we keep it minimal)
        ray_local_mode: bool = True,
    ):
        if not isinstance(env, gym.Env):
            raise TypeError("env must be an initialized Gymnasium Env instance.")

        self._env = env
        self._seed = seed
        self._num_gpus = num_gpus
        self._train_batch_size = train_batch_size
        self._sgd_minibatch_size = sgd_minibatch_size
        self._num_sgd_iter = num_sgd_iter
        self._lr = lr
        self._gamma = gamma
        self._ray_local_mode = ray_local_mode

        ts = time.strftime("%Y%m%d-%H%M%S")
        self._exp_dir = os.path.abspath(os.path.join(log_dir, f"{run_name}_{ts}"))
        os.makedirs(self._exp_dir, exist_ok=True)

        self._env_name = f"env_{run_name}_{ts}"
        self._writer = SummaryWriter(log_dir=self._exp_dir)

        self._algo = None

    @property
    def tensorboard_logdir(self) -> str:
        return self._exp_dir

    def _register_env(self) -> None:
        global _ENV_SINGLETON
        _ENV_SINGLETON = self._env  # store the *already-initialized* env object
        register_env(self._env_name, make_singleton_env)

    def _build_algo(self):
        config = (
            PPOConfig()
            .framework("torch")
            .environment(env=self._env_name, env_config={})
            .env_runners(num_env_runners=0)
            .resources(num_gpus=self._num_gpus)
            .training(
                lr=self._lr,
                gamma=self._gamma,
                train_batch_size=self._train_batch_size,
                minibatch_size=self._sgd_minibatch_size,
                num_epochs=self._num_sgd_iter,
            )
            .debugging(seed=self._seed)
        )
        return config.build_algo()

    def _log_to_tensorboard(self, metrics: Dict[str, Any], step: int) -> None:
        # Log a few common RLlib fields (existence varies by setup).
        def add_scalar(tag: str, key: str):
            v = metrics.get(key, None)
            if isinstance(v, (int, float)):
                self._writer.add_scalar(tag, v, step)

        add_scalar("charts/episode_reward_mean", "episode_reward_mean")
        add_scalar("charts/episode_len_mean", "episode_len_mean")
        add_scalar("charts/timesteps_total", "timesteps_total")
        add_scalar("charts/time_this_iter_s", "time_this_iter_s")

        # Learner stats often live under "info" -> "learner" depending on RLlib version/config.
        info = metrics.get("info", {})
        if isinstance(info, dict):
            learner = info.get("learner", {})
            if isinstance(learner, dict):
                # Sometimes it's keyed by policy id, e.g. {"default_policy": {...}}
                for _, pol in learner.items():
                    if isinstance(pol, dict):
                        ls = pol.get("learner_stats", {})
                        if isinstance(ls, dict):
                            for k in ("total_loss", "policy_loss", "vf_loss", "entropy", "kl"):
                                v = ls.get(k, None)
                                if isinstance(v, (int, float)):
                                    self._writer.add_scalar(f"losses/{k}", v, step)
                        break  # log first policy only

        self._writer.flush()

    def train(
        self,
        *,
        stop_iters: int = 100,
        stop_reward: Optional[float] = None,
        checkpoint_every: int = 0,
    ) -> TrainResult:
        """
        Explicit loop:
          for i in range(stop_iters):
              metrics = algo.train()
              log(metrics)
              (optional) checkpoint
        """
        # RLlib requires Ray runtime, but we don't use Tune.
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=self._ray_local_mode,
            log_to_driver=False,
        )

        checkpoint_path: Optional[str] = None
        last: Dict[str, Any] = {}

        try:
            self._register_env()
            self._algo = self._build_algo()

            for i in range(1, stop_iters + 1):
                last = self._algo.train()

                r = last.get("episode_reward_mean", float("nan"))
                t = last.get("timesteps_total", 0)
                print(f"[iter {i:04d}] reward_mean={r:.2f} timesteps_total={t}")

                self._log_to_tensorboard(last, step=i)

                if checkpoint_every and (i % checkpoint_every == 0):
                    checkpoint_path = self._algo.save(self._exp_dir)
                    print(f"Checkpoint: {checkpoint_path}")

                if stop_reward is not None:
                    rr = last.get("episode_reward_mean", None)
                    if isinstance(rr, (int, float)) and rr >= stop_reward:
                        break

            # final checkpoint (optional but useful)
            checkpoint_path = self._algo.save(self._exp_dir)

            return TrainResult(algo=self._algo, last_metrics=last, checkpoint_path=checkpoint_path)

        finally:
            try:
                self._writer.close()
            finally:
                ray.shutdown()


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1")  # finished initialized object

    trainer = PPOGymTrainer(
        env,
        run_name="cartpole_explicit_loop",
        log_dir="./runs",
        ray_local_mode=True,  # easiest debugging; set False for faster execution
    )

    result = trainer.train(stop_iters=50, stop_reward=475.0, checkpoint_every=10)
    algo = result.algo

    print("TensorBoard logdir:", trainer.tensorboard_logdir)
    # Run: tensorboard --logdir ./runs
