import torch
import torch.nn as nn

#from models.baseline_model import BaselineResNet
#from models.se_resnet34 import se_resnet34_model


MODEL_REGISTRY = {
    "baseline_resnet34": lambda: BaselineResNet(num_classes=50),
    "se_resnet34":      lambda: se_resnet34_model(num_classes=50),
}


class AudioClassificationModel:
    def __init__(self, model_name: str, checkpoint_path: str, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, checkpoint_path)
        self.model.eval()

    def _load_model(self, model_name: str, checkpoint_path: str):
        # model selection
        if model_name not in MODEL_REGISTRY:
            raise ValueError("Invalid model name.")

        # instantiate architecture EXACTLY matching training
        model = MODEL_REGISTRY[model_name]()
        model = model.to(self.device)

        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # extract state_dict
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # strict load → forces architecture match
        model.load_state_dict(state_dict, strict=True)

        return model

    @torch.no_grad()
    def predict(self, tensor):
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        return probs
