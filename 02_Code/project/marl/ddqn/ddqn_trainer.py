from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from envs.uav_backscatter_env import UAVBackscatterEnv, load_config
from envs.hierarchical_env import HierarchicalUAVBackscatterEnv
from marl.ddqn.ddqn_agent import CentralizedFactorizedDDQNAgent
from marl.ddqn.replay_buffer import ReplayBuffer
from marl.ddqn.reward_processing import RewardProcessor


ROOT = Path(__file__).resolve().parents[2]


class DDQNTrainer:
    def __init__(
        self,
        config_path: str | Path,
        episodes_override: int | None = None,
        seed_override: int | None = None,
        resume_checkpoint: str | Path | None = None,
    ):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.training_cfg = dict(self.config.get("training", {}))
        self.interface_cfg = dict(self.config.get("training_interface", {}))
        self.ddqn_cfg = dict(self.config.get("ddqn", {}))
        self.reward_processing_cfg = dict(self.config.get("reward_processing", {}))
        self.logging_cfg = dict(self.config.get("logging", {}))
        if episodes_override is not None:
            self.training_cfg["total_episodes"] = int(episodes_override)
        if seed_override is not None:
            self.training_cfg["seed"] = int(seed_override)

        env_config_path = self._resolve_path(self.config["env_config"])
        self.env_config = load_config(env_config_path)
        base_env = UAVBackscatterEnv(self.env_config)
        if str(self.interface_cfg.get("type", "flat")).lower() == "hierarchical":
            executor_config = dict(self.config.get("executor", self.config.get("executor_config", {})) or {})
            hierarchical_actions_config = dict(self.config.get("hierarchical_actions", {}) or {})
            self.env = HierarchicalUAVBackscatterEnv(base_env, executor_config, hierarchical_actions_config)
        else:
            self.env = base_env
        self.seed = int(self.training_cfg.get("seed", 42))
        self.output_prefix = str(self.logging_cfg.get("output_prefix", self.config_path.stem))
        self.csv_dir = ROOT / "results" / "csv"
        self.checkpoint_dir = ROOT / "results" / "checkpoints"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = self.env.reset(seed=self.seed)
        self.state_type = str(self.ddqn_cfg.get("state_type", "concat_global_local"))
        sample_state = self.build_state(0)
        self.state_dim = int(sample_state.shape[0])
        self.action_dim = int(self.env.action_size)
        self.num_uav = int(self.env.num_uav)
        print(f"[DDQN] Training interface: {str(self.interface_cfg.get('type', 'flat')).lower()} action_dim={self.action_dim}")

        self.agent = CentralizedFactorizedDDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_uav=self.num_uav,
            hidden_sizes=list(self.ddqn_cfg.get("hidden_sizes", [256, 256])),
            layer_norm=bool(self.ddqn_cfg.get("layer_norm", False)),
            learning_rate=float(self.ddqn_cfg.get("learning_rate", 1.0e-3)),
            gamma=float(self.ddqn_cfg.get("gamma", 0.99)),
            batch_size=int(self.ddqn_cfg.get("batch_size", 128)),
            target_update_steps=int(self.ddqn_cfg.get("target_update_steps", 1000)),
            gradient_clip_norm=float(self.ddqn_cfg.get("gradient_clip_norm", 10.0)),
            epsilon_start=float(self.ddqn_cfg.get("epsilon_start", 1.0)),
            epsilon_end=float(self.ddqn_cfg.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(self.ddqn_cfg.get("epsilon_decay_steps", 50000)),
            network_type=str(self.ddqn_cfg.get("network_type", "standard")),
            device=str(self.ddqn_cfg.get("device", "auto")),
            seed=self.seed,
        )
        self.reward_processor = RewardProcessor.from_config(self.reward_processing_cfg)
        self.replay = ReplayBuffer(int(self.ddqn_cfg.get("replay_capacity", 100000)), seed=self.seed)
        self.best_eval_metric = -np.inf
        self.train_rows: list[dict[str, Any]] = []
        self.eval_rows: list[dict[str, Any]] = []
        self.start_episode = 0
        if resume_checkpoint is not None:
            self._load_resume_state(resume_checkpoint)

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    def build_state(self, uav_id: int) -> np.ndarray:
        if self.state_type == "global_state":
            return self.env.get_global_state().astype(np.float32)
        if self.state_type == "concat_global_local":
            return np.concatenate([self.env.get_global_state(), self.env.get_local_observation(uav_id)]).astype(np.float32)
        raise ValueError(f"Unknown ddqn.state_type={self.state_type}")

    def train(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        total_episodes = int(self.training_cfg.get("total_episodes", 1000))
        max_steps = int(self.training_cfg.get("max_steps_per_episode", self.env.max_steps))
        eval_interval = int(self.training_cfg.get("eval_interval_episodes", 50))
        eval_episodes = int(self.training_cfg.get("eval_episodes", 20))
        save_interval = int(self.training_cfg.get("save_interval_episodes", 100))
        min_replay = int(self.ddqn_cfg.get("min_replay_size", 1000))
        batch_size = int(self.ddqn_cfg.get("batch_size", 128))
        start_time = time.time()

        for episode in range(self.start_episode + 1, total_episodes + 1):
            obs, info = self.env.reset(seed=self.seed + episode)
            done = False
            step_count = 0
            losses: list[float] = []
            q_values: list[float] = []
            while not done and step_count < max_steps:
                states = [self.build_state(uav_id) for uav_id in range(self.num_uav)]
                masks = [self.env.get_action_mask(uav_id) for uav_id in range(self.num_uav)]
                actions = [
                    self.agent.select_action(states[uav_id], uav_id, masks[uav_id], deterministic=False)
                    for uav_id in range(self.num_uav)
                ]
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                processed_reward = self.reward_processor.process(float(reward))
                done = bool(terminated or truncated or step_count + 1 >= max_steps)
                next_states = [self.build_state(uav_id) for uav_id in range(self.num_uav)]
                next_masks = [self.env.get_action_mask(uav_id) if not done else np.ones(self.action_dim, dtype=np.int8) for uav_id in range(self.num_uav)]

                for uav_id in range(self.num_uav):
                    self.replay.push(
                        states[uav_id],
                        uav_id,
                        actions[uav_id],
                        processed_reward,
                        next_states[uav_id],
                        done,
                        masks[uav_id],
                        next_masks[uav_id],
                    )
                if len(self.replay) >= min_replay:
                    metrics = self.agent.train_step(self.replay.sample(batch_size))
                    losses.append(metrics["loss"])
                    q_values.append(metrics["mean_q"])
                step_count += 1

            episode_metrics = dict(info.get("episode_metrics", {}))
            self.train_rows.append(self._build_train_row(episode, step_count, episode_metrics, losses, q_values))
            self._save_train_logs()

            should_eval = episode == 1 or episode % eval_interval == 0 or episode == total_episodes
            if should_eval:
                eval_metrics = self.evaluate(eval_episodes, seed_offset=100000 + episode)
                eval_metrics["episode"] = episode
                self.eval_rows.append(eval_metrics)
                self._save_eval_logs()
                eval_score = float(eval_metrics.get("eval_avg_throughput_per_frame", eval_metrics.get("eval_avg_reward", 0.0)))
                if eval_score > self.best_eval_metric:
                    self.best_eval_metric = eval_score
                    self._save_checkpoint("best", episode, eval_metrics)

            if bool(self.logging_cfg.get("save_checkpoints", True)) and (episode % save_interval == 0 or episode == total_episodes):
                self._save_checkpoint("latest", episode, {"elapsed_seconds": time.time() - start_time})

            print(
                f"episode={episode}/{total_episodes} reward={episode_metrics.get('total_reward', 0.0):.2f} "
                f"throughput={episode_metrics.get('avg_throughput_per_frame', 0.0):.3f} "
                f"eps={self.agent.epsilon:.3f} replay={len(self.replay)}"
            )

        self._save_checkpoint("latest", total_episodes, {"elapsed_seconds": time.time() - start_time})
        return pd.DataFrame(self.train_rows), pd.DataFrame(self.eval_rows)

    def _load_resume_state(self, checkpoint: str | Path) -> None:
        extra = self.agent.load_checkpoint(checkpoint)
        self.start_episode = int(extra.get("episode", 0))
        train_path = self.csv_dir / f"{self.output_prefix}_train_log.csv"
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            if "episode" in train_df.columns and self.start_episode > 0:
                train_df = train_df[train_df["episode"] <= self.start_episode]
            self.train_rows = train_df.to_dict("records")
        eval_path = self.csv_dir / f"{self.output_prefix}_eval_log.csv"
        if eval_path.exists():
            eval_df = pd.read_csv(eval_path)
            if "episode" in eval_df.columns and self.start_episode > 0:
                eval_df = eval_df[eval_df["episode"] <= self.start_episode]
            self.eval_rows = eval_df.to_dict("records")
            if "eval_avg_throughput_per_frame" in eval_df.columns and not eval_df.empty:
                self.best_eval_metric = float(eval_df["eval_avg_throughput_per_frame"].max())
        print(
            f"[DDQN] Resumed from {checkpoint} at episode={self.start_episode} "
            f"train_steps={self.agent.train_steps} replay_size={len(self.replay)}"
        )

    def evaluate(self, episodes: int, checkpoint: str | Path | None = None, seed_offset: int = 200000) -> dict[str, float]:
        if checkpoint is not None:
            self.agent.load_checkpoint(checkpoint)
        rows: list[dict[str, float]] = []
        for ep in range(int(episodes)):
            obs, info = self.env.reset(seed=self.seed + seed_offset + ep)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                states = [self.build_state(uav_id) for uav_id in range(self.num_uav)]
                masks = [self.env.get_action_mask(uav_id) for uav_id in range(self.num_uav)]
                actions = [
                    self.agent.select_action(states[uav_id], uav_id, masks[uav_id], deterministic=True)
                    for uav_id in range(self.num_uav)
                ]
                obs, reward, terminated, truncated, info = self.env.step(actions)
            rows.append(dict(info.get("episode_metrics", {})))
        df = pd.DataFrame(rows)
        return {
            "eval_avg_reward": float(df["total_reward"].mean()),
            "eval_avg_throughput": float(df["total_throughput"].mean()),
            "eval_avg_throughput_per_frame": float(df["avg_throughput_per_frame"].mean()),
            "eval_packet_drop_rate": float(df["packet_drop_rate"].mean()),
            "eval_jamming_failure_rate": float(df["jamming_failure_rate"].mean()),
            "eval_fairness": float(df["fairness_index"].mean()),
            "eval_energy_efficiency": float(df["energy_efficiency"].mean()),
            "eval_backscatter_success_rate": float(df.get("backscatter_success_rate", pd.Series([0.0])).mean()),
            "eval_active_success_rate": float(df.get("active_success_rate", pd.Series([0.0])).mean()),
            "eval_mode_usage_harvest": float(df.get("mode_usage_harvest", pd.Series([0.0])).mean()),
            "eval_mode_usage_backscatter": float(df.get("mode_usage_backscatter", pd.Series([0.0])).mean()),
            "eval_mode_usage_active": float(df.get("mode_usage_active", pd.Series([0.0])).mean()),
        }

    def evaluate_episodes(self, episodes: int, checkpoint: str | Path | None = None) -> pd.DataFrame:
        if checkpoint is not None:
            self.agent.load_checkpoint(checkpoint)
        rows: list[dict[str, Any]] = []
        for ep in range(int(episodes)):
            self.env.reset(seed=self.seed + 300000 + ep)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                states = [self.build_state(uav_id) for uav_id in range(self.num_uav)]
                masks = [self.env.get_action_mask(uav_id) for uav_id in range(self.num_uav)]
                actions = [
                    self.agent.select_action(states[uav_id], uav_id, masks[uav_id], deterministic=True)
                    for uav_id in range(self.num_uav)
                ]
                _, _, terminated, truncated, info = self.env.step(actions)
            row = dict(info.get("episode_metrics", {}))
            row["episode_id"] = ep
            row["policy_name"] = "ddqn"
            row["scenario_name"] = Path(self.config["env_config"]).stem
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_train_row(
        self,
        episode: int,
        step_count: int,
        metrics: dict[str, Any],
        losses: list[float],
        q_values: list[float],
    ) -> dict[str, Any]:
        return {
            "episode": episode,
            "step_count": step_count,
            "train_reward": float(metrics.get("total_reward", 0.0)),
            "train_throughput": float(metrics.get("total_throughput", 0.0)),
            "train_drop_rate": float(metrics.get("packet_drop_rate", 0.0)),
            "train_jamming_failure_rate": float(metrics.get("jamming_failure_rate", 0.0)),
            "train_fairness": float(metrics.get("fairness_index", 0.0)),
            "epsilon": float(self.agent.epsilon),
            "device": str(self.agent.device),
            "network_type": self.agent.network_type,
            "reward_processing_mode": self.reward_processor.mode if self.reward_processor.enabled else "none",
            "reward_processing_scale": self.reward_processor.scale,
            "replay_size": len(self.replay),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_q": float(np.mean(q_values)) if q_values else 0.0,
            "mode_usage_harvest": float(metrics.get("mode_usage_harvest", 0.0)),
            "mode_usage_backscatter": float(metrics.get("mode_usage_backscatter", 0.0)),
            "mode_usage_active": float(metrics.get("mode_usage_active", 0.0)),
            "invalid_action_rate": float(metrics.get("invalid_action_rate", 0.0)),
            "backscatter_success_rate": float(metrics.get("backscatter_success_rate", 0.0)),
            "active_success_rate": float(metrics.get("active_success_rate", 0.0)),
            **self._hierarchical_train_metrics(metrics),
        }

    def _hierarchical_train_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        keys = [
            "fallback_count",
            "fallback_rate",
            "executor_idle_count",
            "executor_harvest_count",
            "executor_backscatter_count",
            "executor_active_count",
            "executor_avoid_jammer_count",
        ]
        out = {key: float(metrics.get(key, 0.0)) for key in keys}
        for idx in range(int(getattr(self.env, "action_size", 0))):
            key = f"high_level_action_{idx}_count"
            if key in metrics:
                out[key] = float(metrics.get(key, 0.0))
        return out

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
            "reward_processing": self.reward_processor.to_dict(),
            **extra,
        }
        self.agent.save_checkpoint(path, payload)
        return path
