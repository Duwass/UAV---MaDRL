from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from envs.uav_backscatter_env import UAVBackscatterEnv, load_config
from marl.qmix.qmix_agent import QMIXAgent
from marl.qmix.replay_buffer import EpisodeReplayBuffer


ROOT = Path(__file__).resolve().parents[2]


class QMIXTrainer:
    def __init__(
        self,
        config_path: str | Path,
        episodes_override: int | None = None,
        seed_override: int | None = None,
    ):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.training_cfg = dict(self.config.get("training", {}))
        self.interface_cfg = dict(self.config.get("training_interface", {}))
        self.qmix_cfg = dict(self.config.get("qmix", {}))
        self.logging_cfg = dict(self.config.get("logging", {}))
        if episodes_override is not None:
            self.training_cfg["total_episodes"] = int(episodes_override)
        if seed_override is not None:
            self.training_cfg["seed"] = int(seed_override)

        if str(self.interface_cfg.get("type", "hierarchical")).lower() != "hierarchical":
            raise ValueError("QMIX prototype currently requires training_interface.type=hierarchical.")

        self.seed = int(self.training_cfg.get("seed", 42))
        self.env_config_path = self._resolve_path(self.config["env_config"])
        self.env_config = load_config(self.env_config_path)
        executor_config = dict(self.config.get("executor", self.config.get("executor_config", {})) or {})
        hierarchical_actions_config = dict(self.config.get("hierarchical_actions", {}) or {})
        self.env = HierarchicalUAVBackscatterEnv(
            UAVBackscatterEnv(self.env_config),
            executor_config,
            hierarchical_actions_config,
        )
        self.output_prefix = str(self.logging_cfg.get("output_prefix", self.config_path.stem))
        self.csv_dir = ROOT / "results" / "csv"
        self.checkpoint_dir = ROOT / "results" / "checkpoints"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.env.reset(seed=self.seed)
        self.n_agents = int(self.env.num_uav)
        self.action_dim = int(self.env.action_size)
        self.obs_dim = int(np.asarray(self.env.get_local_observation(0)).shape[0])
        self.state_dim = int(np.asarray(self.env.get_global_state()).shape[0])
        self.agent = QMIXAgent(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_agent_id=bool(self.qmix_cfg.get("use_agent_id", True)),
            agent_hidden_sizes=list(self.qmix_cfg.get("agent_hidden_sizes", [128, 128])),
            mixing_embed_dim=int(self.qmix_cfg.get("mixing_embed_dim", 32)),
            hypernet_hidden_dim=int(self.qmix_cfg.get("hypernet_hidden_dim", 64)),
            learning_rate=float(self.qmix_cfg.get("learning_rate", 5.0e-4)),
            gamma=float(self.qmix_cfg.get("gamma", 0.99)),
            target_update_steps=int(self.qmix_cfg.get("target_update_steps", 200)),
            gradient_clip_norm=float(self.qmix_cfg.get("gradient_clip_norm", 10.0)),
            epsilon_start=float(self.qmix_cfg.get("epsilon_start", 1.0)),
            epsilon_end=float(self.qmix_cfg.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(self.qmix_cfg.get("epsilon_decay_steps", 25000)),
            double_q=bool(self.qmix_cfg.get("double_q", True)),
            device=str(self.qmix_cfg.get("device", "auto")),
            seed=self.seed,
        )
        self.replay = EpisodeReplayBuffer(int(self.qmix_cfg.get("replay_capacity", 5000)), seed=self.seed)
        self.best_eval_metric = -np.inf
        self.train_rows: list[dict[str, Any]] = []
        self.eval_rows: list[dict[str, Any]] = []
        print(
            f"[QMIX] Training interface: hierarchical n_agents={self.n_agents} "
            f"action_dim={self.action_dim} obs_dim={self.obs_dim} state_dim={self.state_dim}"
        )

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    def _current_observations(self) -> np.ndarray:
        return np.stack([self.env.get_local_observation(agent_id) for agent_id in range(self.n_agents)]).astype(np.float32)

    def _current_masks(self) -> np.ndarray:
        return np.stack([self.env.get_action_mask(agent_id) for agent_id in range(self.n_agents)]).astype(np.float32)

    def collect_episode(self, episode_seed: int, deterministic: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
        self.env.reset(seed=episode_seed)
        max_steps = int(self.training_cfg.get("max_steps_per_episode", self.env.max_steps))
        episode: dict[str, list[Any]] = {
            "observations": [],
            "global_states": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "next_global_states": [],
            "dones": [],
            "action_masks": [],
            "next_action_masks": [],
        }
        terminated = False
        truncated = False
        step_count = 0
        info: dict[str, Any] = {}

        while not (terminated or truncated) and step_count < max_steps:
            observations = self._current_observations()
            global_state = self.env.get_global_state().astype(np.float32)
            action_masks = self._current_masks()
            actions, fallback_count = self.agent.select_actions(observations, action_masks, deterministic=deterministic)
            _, reward, terminated, truncated, info = self.env.step(actions)
            done = bool(terminated or truncated or step_count + 1 >= max_steps)
            next_observations = self._current_observations()
            next_global_state = self.env.get_global_state().astype(np.float32)
            next_action_masks = self._current_masks() if not done else np.ones_like(action_masks, dtype=np.float32)

            episode["observations"].append(observations)
            episode["global_states"].append(global_state)
            episode["actions"].append(np.asarray(actions, dtype=np.int64))
            episode["rewards"].append(float(reward))
            episode["next_observations"].append(next_observations)
            episode["next_global_states"].append(next_global_state)
            episode["dones"].append(float(done))
            episode["action_masks"].append(action_masks)
            episode["next_action_masks"].append(next_action_masks)
            step_count += 1

        episode_arrays = {key: np.asarray(value) for key, value in episode.items()}
        return episode_arrays, dict(info.get("episode_metrics", {})), step_count

    def train(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        total_episodes = int(self.training_cfg.get("total_episodes", 500))
        eval_interval = int(self.training_cfg.get("eval_interval_episodes", 10))
        eval_episodes = int(self.training_cfg.get("eval_episodes", 30))
        save_interval = int(self.training_cfg.get("save_interval_episodes", 50))
        min_replay = int(self.qmix_cfg.get("min_replay_size", 10))
        batch_size = int(self.qmix_cfg.get("batch_size", 16))
        updates_per_episode = int(self.qmix_cfg.get("updates_per_episode", 1))
        start_time = time.time()

        for episode in range(1, total_episodes + 1):
            episode_data, episode_metrics, step_count = self.collect_episode(self.seed + episode, deterministic=False)
            self.replay.push_episode(episode_data)
            losses: list[float] = []
            q_totals: list[float] = []
            if len(self.replay) >= min_replay:
                for _ in range(max(updates_per_episode, 1)):
                    metrics = self.agent.train_step(self.replay.sample(batch_size))
                    losses.append(metrics["loss"])
                    q_totals.append(metrics["avg_q_tot"])

            self.train_rows.append(self._build_train_row(episode, step_count, episode_metrics, losses, q_totals))
            self._save_train_logs()

            should_eval = episode == 1 or episode % eval_interval == 0 or episode == total_episodes
            if should_eval:
                eval_metrics = self.evaluate(eval_episodes, seed_offset=100000 + episode)
                eval_metrics["episode"] = episode
                self.eval_rows.append(eval_metrics)
                self._save_eval_logs()
                score = float(eval_metrics.get("eval_avg_throughput_per_frame", 0.0))
                if score > self.best_eval_metric:
                    self.best_eval_metric = score
                    self._save_checkpoint("best", episode, eval_metrics)

            if bool(self.logging_cfg.get("save_checkpoints", True)) and (episode % save_interval == 0 or episode == total_episodes):
                self._save_checkpoint("latest", episode, {"elapsed_seconds": time.time() - start_time})

            print(
                f"episode={episode}/{total_episodes} reward={episode_metrics.get('total_reward', 0.0):.2f} "
                f"throughput={episode_metrics.get('avg_throughput_per_frame', 0.0):.3f} "
                f"eps={self.agent.epsilon:.3f} replay={len(self.replay)} "
                f"loss={(np.mean(losses) if losses else 0.0):.4f}"
            )

        self._save_checkpoint("latest", total_episodes, {"elapsed_seconds": time.time() - start_time})
        return pd.DataFrame(self.train_rows), pd.DataFrame(self.eval_rows)

    def evaluate(self, episodes: int, checkpoint: str | Path | None = None, seed_offset: int = 200000) -> dict[str, float]:
        rows = self.evaluate_episodes(episodes, checkpoint=checkpoint, seed_offset=seed_offset)
        return self._summarize_eval_rows(rows)

    def evaluate_episodes(
        self,
        episodes: int,
        checkpoint: str | Path | None = None,
        seed_offset: int = 300000,
    ) -> pd.DataFrame:
        if checkpoint is not None:
            self.agent.load_checkpoint(checkpoint)
        rows: list[dict[str, Any]] = []
        for ep in range(int(episodes)):
            _, metrics, _ = self.collect_episode(self.seed + seed_offset + ep, deterministic=True)
            row = dict(metrics)
            row["episode_id"] = ep
            row["policy_name"] = "qmix"
            row["scenario_name"] = Path(self.config["env_config"]).stem
            rows.append(row)
        return pd.DataFrame(rows)

    def _summarize_eval_rows(self, df: pd.DataFrame) -> dict[str, float]:
        return {
            "eval_avg_reward": self._mean(df, "total_reward"),
            "eval_avg_throughput": self._mean(df, "total_throughput"),
            "eval_avg_throughput_per_frame": self._mean(df, "avg_throughput_per_frame"),
            "eval_packet_drop_rate": self._mean(df, "packet_drop_rate"),
            "eval_jamming_failure_rate": self._mean(df, "jamming_failure_rate"),
            "eval_fairness": self._mean(df, "fairness_index"),
            "eval_energy_efficiency": self._mean(df, "energy_efficiency"),
            "eval_backscatter_success_rate": self._mean(df, "backscatter_success_rate"),
            "eval_active_success_rate": self._mean(df, "active_success_rate"),
            "eval_mode_usage_harvest": self._mean(df, "mode_usage_harvest"),
            "eval_mode_usage_backscatter": self._mean(df, "mode_usage_backscatter"),
            "eval_mode_usage_active": self._mean(df, "mode_usage_active"),
            "eval_fallback_rate": self._mean(df, "fallback_rate"),
        }

    def _build_train_row(
        self,
        episode: int,
        step_count: int,
        metrics: dict[str, Any],
        losses: list[float],
        q_totals: list[float],
    ) -> dict[str, Any]:
        row = {
            "episode": episode,
            "step_count": step_count,
            "train_reward": float(metrics.get("total_reward", 0.0)),
            "train_throughput": float(metrics.get("total_throughput", 0.0)),
            "train_drop_rate": float(metrics.get("packet_drop_rate", 0.0)),
            "train_jamming_failure_rate": float(metrics.get("jamming_failure_rate", 0.0)),
            "train_fairness": float(metrics.get("fairness_index", 0.0)),
            "epsilon": float(self.agent.epsilon),
            "replay_size": len(self.replay),
            "updates_per_episode": int(self.qmix_cfg.get("updates_per_episode", 1)),
            "train_steps": int(self.agent.train_steps),
            "target_update_count": int(self.agent.target_update_count),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_q_tot": float(np.mean(q_totals)) if q_totals else 0.0,
            "mode_usage_harvest": float(metrics.get("mode_usage_harvest", 0.0)),
            "mode_usage_backscatter": float(metrics.get("mode_usage_backscatter", 0.0)),
            "mode_usage_active": float(metrics.get("mode_usage_active", 0.0)),
            "backscatter_success_rate": float(metrics.get("backscatter_success_rate", 0.0)),
            "active_success_rate": float(metrics.get("active_success_rate", 0.0)),
            "fallback_rate": float(metrics.get("fallback_rate", 0.0)),
            "device": str(self.agent.device),
        }
        for idx in range(self.action_dim):
            key = f"high_level_action_{idx}_count"
            row[key] = float(metrics.get(key, 0.0))
        return row

    def _mean(self, df: pd.DataFrame, key: str) -> float:
        return float(df[key].mean()) if key in df.columns and not df.empty else 0.0

    def _save_train_logs(self) -> Path:
        path = self.csv_dir / f"{self.output_prefix}_train_log.csv"
        pd.DataFrame(self.train_rows).to_csv(path, index=False)
        return path

    def _save_eval_logs(self) -> Path:
        path = self.csv_dir / f"{self.output_prefix}_eval_log.csv"
        pd.DataFrame(self.eval_rows).to_csv(path, index=False)
        return path

    def _save_checkpoint(self, suffix: str, episode: int, extra: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / f"{self.output_prefix}_{suffix}.pt"
        payload = {
            "episode": episode,
            "config_path": str(self.config_path),
            "env_config": self.config["env_config"],
            "output_prefix": self.output_prefix,
            **extra,
        }
        self.agent.save_checkpoint(path, payload)
        return path
