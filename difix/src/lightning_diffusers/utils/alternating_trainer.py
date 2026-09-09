from typing import Dict, Any, Optional, List
from loguru import logger


class RatioAlternatingTrainer:


    def __init__(
        self,
        normal_steps: int,
        identity_steps: int,
        start_step: int = 0
    ):

        if normal_steps <= 0 or identity_steps <= 0:
            raise ValueError("Both normal_steps and identity_steps must be positive")

        self.normal_steps = normal_steps
        self.identity_steps = identity_steps
        self.cycle_length = normal_steps + identity_steps
        self.step_counter = start_step

        logger.info(f"RatioAlternatingTrainer initialized:")
        logger.info(f"  Pattern: {normal_steps} normal : {identity_steps} identity")
        logger.info(f"  Cycle length: {self.cycle_length}")
        logger.info(f"  Starting step: {start_step}")

    def get_current_step_type(self) -> str:

        cycle_position = self.step_counter % self.cycle_length

        if cycle_position < self.normal_steps:
            return "normal"
        else:
            return "identity"

    def advance_step(self) -> str:

        self.step_counter += 1
        return self.get_current_step_type()

    def get_cycle_progress(self) -> Dict[str, Any]:

        cycle_position = self.step_counter % self.cycle_length
        current_cycle = self.step_counter // self.cycle_length


        if cycle_position < self.normal_steps:
            current_phase = "normal"
            phase_step = cycle_position
            phase_total = self.normal_steps
        else:
            current_phase = "identity"
            phase_step = cycle_position - self.normal_steps
            phase_total = self.identity_steps

        return {
            "step_counter": self.step_counter,
            "current_cycle": current_cycle,
            "cycle_position": cycle_position,
            "cycle_length": self.cycle_length,
            "current_phase": current_phase,
            "phase_step": phase_step + 1,
            "phase_total": phase_total,
            "normal_steps": self.normal_steps,
            "identity_steps": self.identity_steps
        }

    def get_step_type_counts(self) -> Dict[str, int]:

        completed_cycles = self.step_counter // self.cycle_length
        remaining_steps = self.step_counter % self.cycle_length


        normal_count = completed_cycles * self.normal_steps
        identity_count = completed_cycles * self.identity_steps


        if remaining_steps <= self.normal_steps:
            normal_count += remaining_steps
        else:
            normal_count += self.normal_steps
            identity_count += (remaining_steps - self.normal_steps)

        return {
            "normal_steps": normal_count,
            "identity_steps": identity_count,
            "total_steps": self.step_counter
        }

    def is_cycle_complete(self) -> bool:

        return (self.step_counter % self.cycle_length) == 0 and self.step_counter > 0

    def reset_to_step(self, step: int) -> None:

        if step < 0:
            raise ValueError("Step must be non-negative")

        old_step = self.step_counter
        self.step_counter = step

        logger.info(f"Reset alternating trainer from step {old_step} to step {step}")

    def get_pattern_string(self) -> str:

        pattern_parts = ["N"] * self.normal_steps + ["I"] * self.identity_steps
        return "-".join(pattern_parts)

    def predict_next_steps(self, num_steps: int = 10) -> List[str]:

        future_steps = []
        temp_counter = self.step_counter

        for _ in range(num_steps):
            cycle_position = temp_counter % self.cycle_length

            if cycle_position < self.normal_steps:
                future_steps.append("normal")
            else:
                future_steps.append("identity")

            temp_counter += 1

        return future_steps

    def get_training_statistics(self) -> Dict[str, Any]:

        progress = self.get_cycle_progress()
        counts = self.get_step_type_counts()


        total_steps = counts["total_steps"]
        normal_ratio = counts["normal_steps"] / total_steps if total_steps > 0 else 0
        identity_ratio = counts["identity_steps"] / total_steps if total_steps > 0 else 0

        return {

            "current_step": self.step_counter,
            "current_step_type": self.get_current_step_type(),
            "current_cycle": progress["current_cycle"],
            "cycle_progress": f"{progress['phase_step']}/{progress['phase_total']} {progress['current_phase']}",


            "pattern": self.get_pattern_string(),
            "normal_steps_per_cycle": self.normal_steps,
            "identity_steps_per_cycle": self.identity_steps,
            "cycle_length": self.cycle_length,


            "total_normal_steps": counts["normal_steps"],
            "total_identity_steps": counts["identity_steps"],
            "total_steps": total_steps,


            "normal_step_ratio": normal_ratio,
            "identity_step_ratio": identity_ratio,
            "normal_percentage": normal_ratio * 100,
            "identity_percentage": identity_ratio * 100,


            "next_5_steps": self.predict_next_steps(5)
        }

    def log_current_status(self, level: str = "info") -> None:

        stats = self.get_training_statistics()

        log_func = getattr(logger, level, logger.info)

        log_func(f"Alternating Training Status:")
        log_func(f"  Step {stats['current_step']}: {stats['current_step_type']} "
                f"({stats['cycle_progress']})")
        log_func(f"  Pattern: {stats['pattern']} | "
                f"Normal: {stats['total_normal_steps']} ({stats['normal_percentage']:.1f}%) | "
                f"Identity: {stats['total_identity_steps']} ({stats['identity_percentage']:.1f}%)")
        log_func(f"  Next steps: {'-'.join(stats['next_5_steps'][:5])}")


class AlternatingTrainingConfig:


    def __init__(
        self,
        normal_steps: int = 2,
        identity_steps: int = 1,
        enabled: bool = True,
    ):

        self.normal_steps = normal_steps
        self.identity_steps = identity_steps
        self.enabled = enabled

        self.validate()

    def validate(self) -> None:

        if self.normal_steps <= 0:
            raise ValueError(f"normal_steps must be positive, got {self.normal_steps}")

        if self.identity_steps <= 0:
            raise ValueError(f"identity_steps must be positive, got {self.identity_steps}")

    def create_trainer(self, start_step: int = 0) -> Optional[RatioAlternatingTrainer]:

        if not self.enabled:
            logger.info("Alternating training disabled")
            return None

        return RatioAlternatingTrainer(
            normal_steps=self.normal_steps,
            identity_steps=self.identity_steps,
            start_step=start_step
        )

    def get_ratio_string(self) -> str:

        return f"{self.normal_steps}:{self.identity_steps}"

    def to_dict(self) -> Dict[str, Any]:

        return {
            "normal_steps": self.normal_steps,
            "identity_steps": self.identity_steps,
            "enabled": self.enabled,
            "ratio": self.get_ratio_string()
        }

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "AlternatingTrainingConfig":

        return cls(
            normal_steps=config.get("normal_steps", 2),
            identity_steps=config.get("identity_steps", 1),
            enabled=config.get("enabled", True)
        )
