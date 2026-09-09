import time
from typing import Dict, Optional

import psutil
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from lightning_diffusers.utils import pylogger

log = pylogger.get_pylogger(__name__)


class PerformanceMonitor(Callback):


    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_memory: bool = True,
        log_speed: bool = True,
        log_system: bool = False,
    ):

        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_memory = log_memory
        self.log_speed = log_speed
        self.log_system = log_system


        self._step_start_time: Optional[float] = None
        self._batch_sizes: list = []
        self._step_times: list = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        if self.log_speed:
            self._step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not self._should_log(trainer):
            return

        metrics = {}


        if self.log_speed and self._step_start_time is not None:
            step_time = time.time() - self._step_start_time
            self._step_times.append(step_time)


            if len(self._step_times) > 100:
                self._step_times = self._step_times[-100:]

            avg_step_time = sum(self._step_times) / len(self._step_times)

            metrics.update({
                "perf/step_time_sec": step_time,
                "perf/avg_step_time_sec": avg_step_time,
                "perf/steps_per_sec": 1.0 / avg_step_time,
            })


            try:
                if hasattr(batch, "__len__"):
                    batch_size = len(batch)
                elif isinstance(batch, dict) and len(batch) > 0:

                    first_key = next(iter(batch.keys()))
                    batch_size = batch[first_key].shape[0] if hasattr(batch[first_key], 'shape') else 1
                else:
                    batch_size = 1

                self._batch_sizes.append(batch_size)
                if len(self._batch_sizes) > 100:
                    self._batch_sizes = self._batch_sizes[-100:]

                avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
                metrics.update({
                    "perf/batch_size": batch_size,
                    "perf/avg_batch_size": avg_batch_size,
                    "perf/samples_per_sec": avg_batch_size / avg_step_time,
                })

            except Exception as e:
                log.debug(f"Could not compute batch size metrics: {e}")


        if self.log_memory and torch.cuda.is_available():
            try:
                device = pl_module.device
                if device.type == "cuda":
                    gpu_id = device.index or 0


                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

                    memory_utilization = (memory_allocated / memory_total) * 100

                    metrics.update({
                        "perf/gpu_memory_allocated_gb": memory_allocated,
                        "perf/gpu_memory_reserved_gb": memory_reserved,
                        "perf/gpu_memory_total_gb": memory_total,
                        "perf/gpu_memory_utilization_pct": memory_utilization,
                    })

            except Exception as e:
                log.debug(f"Could not get GPU memory stats: {e}")


        if self.log_system:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                metrics.update({
                    "perf/cpu_utilization_pct": cpu_percent,
                    "perf/ram_utilization_pct": memory.percent,
                    "perf/ram_available_gb": memory.available / 1e9,
                })

            except Exception as e:
                log.debug(f"Could not get system stats: {e}")


        if metrics:
            self._log_metrics(trainer, metrics)

    @rank_zero_only
    def _log_metrics(self, trainer, metrics: Dict[str, float]):


        if trainer.loggers:
            for logger in trainer.loggers:
                try:
                    logger.log_metrics(metrics, step=trainer.global_step)
                except Exception as e:
                    log.debug(f"Could not log to {type(logger).__name__}: {e}")


        if trainer.global_step % (self.log_every_n_steps * 5) == 0:
            perf_summary = []

            if "perf/steps_per_sec" in metrics:
                perf_summary.append(f"Speed: {metrics['perf/steps_per_sec']:.2f} steps/s")

            if "perf/samples_per_sec" in metrics:
                perf_summary.append(f"{metrics['perf/samples_per_sec']:.1f} samples/s")

            if "perf/gpu_memory_utilization_pct" in metrics:
                perf_summary.append(f"GPU: {metrics['perf/gpu_memory_utilization_pct']:.1f}%")

            if perf_summary:
                log.info(f"Performance: {' | '.join(perf_summary)}")

    def _should_log(self, trainer) -> bool:

        return (trainer.global_step % self.log_every_n_steps) == 0


def log_model_summary(model: torch.nn.Module, logger_name: str = "model_summary") -> Dict[str, int]:

    log = pylogger.get_pylogger(logger_name)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params


    model_size_mb = (total_params * 4) / (1024 * 1024)


    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1

    stats = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "model_size_mb": model_size_mb,
    }


    log.info(f"Model Summary:")
    log.info(f"  Total parameters: {total_params:,}")
    log.info(f"  Trainable parameters: {trainable_params:,}")
    log.info(f"  Frozen parameters: {frozen_params:,}")
    log.info(f"  Estimated size: {model_size_mb:.1f} MB")


    if layer_counts:
        sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
        log.info(f"  Top layer types:")
        for layer_type, count in sorted_layers[:5]:
            log.info(f"    {layer_type}: {count}")

    return stats


def estimate_memory_requirements(
    batch_size: int,
    sequence_length: int = 512,
    model_params: int = 1000000,
    precision: str = "float32"
) -> Dict[str, float]:


    bytes_per_element = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1
    }.get(precision, 4)


    model_memory = (model_params * bytes_per_element) / 1e9


    optimizer_memory = model_memory * 2


    gradient_memory = model_memory


    activation_memory = (batch_size * sequence_length * 512 * bytes_per_element) / 1e9


    cuda_overhead = 1.0

    estimates = {
        "model_weights_gb": model_memory,
        "optimizer_states_gb": optimizer_memory,
        "gradients_gb": gradient_memory,
        "activations_gb": activation_memory,
        "cuda_overhead_gb": cuda_overhead,
        "total_estimated_gb": model_memory + optimizer_memory + gradient_memory + activation_memory + cuda_overhead
    }

    return estimates


def get_gpu_memory_summary() -> Optional[Dict[str, float]]:

    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - reserved,
            "total_gb": total,
            "utilization_pct": (allocated / total) * 100
        }

    except Exception as e:
        log.debug(f"Could not get GPU memory summary: {e}")
        return None
