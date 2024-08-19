import torch
from torch.nn.parallel import DataParallel

class ZDataParallel(DataParallel):
    def __init__(self, *args, dim=0, **kwargs):
        super(ZDataParallel, self).__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def forward(self,*inputs, **kwargs):
        return super().forward(*inputs, **kwargs)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.train_step(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.train_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.val_step(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.val_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
