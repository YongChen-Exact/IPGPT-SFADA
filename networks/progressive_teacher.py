import logging
import torch
import torch.nn as nn

class ProT(nn.Module):

    def __init__(self, stu_model, tch_model, m=0.98, checkpoint_path=None):
        super(ProT, self).__init__()
        self.m = m
        self.queue_ptr = 0
        self.stu_model = stu_model
        self.tch_model = tch_model
        self.tch_model.requires_grad_(False)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            name = name[len("module.") :] if name.startswith("module.") else name
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}")

    @torch.no_grad()
    def _ema_model_update(self):
        for param_s, param_t in zip(self.stu_model.parameters(), self.tch_model.parameters()):
            param_t.data = param_t.data * self.m + param_s.data * (1.0 - self.m)

    def forward(self, x):
        logits = self.stu_model(x)
        with torch.no_grad():                    
            self._ema_model_update()
        return logits