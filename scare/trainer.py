import torch
import torch.distributed as dist
from trl import GRPOTrainer

world = 1
rank = 0

class ScaffoldTrainer(GRPOTrainer):

    def training_step(self, model, inputs, num_items_in_batch=None):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and (not getattr(model, '_scare_fp64_hook', False)):

            def hook(state, bucket):
                flat = bucket.buffer().view(-1)
                for start in range(0, flat.numel(), 4 * 1024 * 1024):
                    chunk = flat[start:start + 4 * 1024 * 1024]
                    buffer = chunk.to(torch.float64)
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=False)
                    buffer.div_(world)
                    chunk.copy_(buffer)
                    del buffer
                future = torch.futures.Future()
                future.set_result(bucket.buffer())
                return future
            model.register_comm_hook(None, hook)
            model._scare_fp64_hook = True
        return super().training_step(model, inputs, num_items_in_batch)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        from accelerate.utils import DistributedDataParallelKwargs
        if self.accelerator.ddp_handler is None:
            self.accelerator.ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator.ddp_handler.gradient_as_bucket_view = True
        self.accelerator.ddp_handler.static_graph = False
        self.accelerator.ddp_handler.bucket_cap_mb = 25

    def create_optimizer(self):
        result = super().create_optimizer()
        if not getattr(self, 'guarded', False):

            def guard(opt, args, kwargs):
                finite = torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]).all()
                if not finite.item():
                    raise FloatingPointError(f'Nonfinite gradient rank{rank} step{self.state.global_step + 1}')
            self.optimizer.register_step_pre_hook(guard)
            self.guarded = True
        return result

    def _compute_loss(self, model, inputs):
        mask = inputs['completion_mask']
        old_length = mask.shape[1]
        occupied = mask.bool().any(dim=0).nonzero()
        length = int(occupied[-1].item()) + 1 if occupied.numel() else 1
        inputs = {k: v[:, :length] if isinstance(v, torch.Tensor) and v.ndim == 2 and (v.shape[1] == old_length) and (k not in ['prompt_ids', 'prompt_mask', 'advantages']) else v for k, v in inputs.items()}
        loss = super()._compute_loss(model, inputs)
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError('nonfinite loss')
        return loss
