import os
import torch
from utils import is_main_process


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


class ModelLogger:
    def __init__(self, enable: bool, module):
        self.enable = enable
        if not enable:
            return

        self.module = module
        self.id_to_name = {
            id(module): str(name) for name, module in module.named_modules()
        }
        self.activations = {
            id(module): None for module in module.modules()
        }
        self.hooks = []

    def __enter__(self, *args, **kwargs):
        if not self.enable:
            return self

        def log_activations(m, m_in, m_out):
            if isinstance(m_out, (tuple, list)):
                m_out = m_out[0]
            self.activations[id(m)] = m_out.detach().cpu()

        for m in self.module.modules():
            self.hooks.append(m.register_forward_hook(log_activations))

        return self

    @torch.no_grad()
    def _log_activations(self):
        wandb.log(
            {
                f"activations_mean/{self.id_to_name[m_id]}": a.mean().item()
                for m_id, a in self.activations.items()
                if a is not None
            },
            commit=False
        )
        wandb.log(
            {
                f"activations_std/{self.id_to_name[m_id]}": a.std().item()
                for m_id, a in self.activations.items()
                if a is not None
            },
            commit=False
        )

    @torch.no_grad()
    def _log_parameter_histograms(self):
        for name, param in self.module.named_parameters():
            wandb.log(
                {
                    f"parameters_mean/{name}": param.data.mean().cpu().item(),
                    f"parameters_norm/{name}": torch.linalg.norm(param.data).cpu().item(),
                    f"parameters_std/{name}": param.data.std().cpu().item(),
                },
                commit=False
            )

    @torch.no_grad()
    def _log_gradients_histograms(self):
        for name, param in self.module.named_parameters():
            if param.grad is not None:
                wandb.log(
                    {
                        f"gradients_mean/{name}": param.grad.mean().cpu().item(),
                        f"gradients_norm/{name}": torch.linalg.norm(param.grad).cpu().item(),
                        f"gradients_std/{name}": param.grad.std().cpu().item(),
                    },
                    commit=False
                )

    def __exit__(self, *args, **kwargs):
        if not self.enable:
            return

        if is_main_process():
            self._log_activations()
            self._log_parameter_histograms()
            self._log_gradients_histograms()

        for hook in self.hooks:
            hook.remove()
