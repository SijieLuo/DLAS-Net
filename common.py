import copy
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # [N, ...] -> [N, 1, L] -> GAP1D -> [N, D]
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=288):
        self.device = device
        self.target_size = target_size
        self.smoothing = 1

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """
        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone

        # clean old hooks if any
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.backbone.hook_handles = []

        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            self.register_hook(extract_layer)

        self.to(self.device)

    def forward(self, images, eval=True):
        """
        当 images.requires_grad=True（例如做 Grad-CAM）时，强制开启梯度；
        否则使用 no_grad() 以提升推理速度。
        若 train_backbone=True 且 eval=False，则按常规训练路径执行（默认有梯度）。
        """
        self.outputs.clear()

        if self.train_backbone and not eval:
            # 训练主干：保留梯度
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        else:
            # 推理路径：根据输入是否需要梯度自动选择上下文
            ctx = torch.enable_grad() if images.requires_grad else torch.no_grad()
            with ctx:
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass

        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        # 这里不需要梯度
        _input.requires_grad_(False)
        _output = self(_input, eval=True)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name):
        module = self.find_module(self.backbone, layer_name)
        if module is not None:
            forward_hook = ForwardHook(self.outputs, layer_name, self.layers_to_extract_from[-1])
            if isinstance(module, torch.nn.Sequential):
                hook = module[-1].register_forward_hook(forward_hook)
            else:
                hook = module.register_forward_hook(forward_hook)
            self.backbone.hook_handles.append(hook)
        else:
            raise ValueError(f"Module {layer_name} not found in the model")

    def find_module(self, model, module_name):
        # 支持嵌套命名如 "layer3.2.conv3" 或 "layer3[2].conv3"
        for name, module in model.named_modules():
            if name == module_name:
                return module
            elif "." in module_name:
                father, child = module_name.split(".", 1)
                # 兼容 "layer3[2]" 这种写法
                if "[" in father and father.endswith("]"):
                    base, idx = father.split("[")
                    idx = int(idx[:-1])
                    if name == base:
                        seq = getattr(model, base, None)
                        if seq is not None and hasattr(seq, "__getitem__") and idx < len(seq):
                            return self.find_module(seq[idx], child)
                else:
                    if name == father:
                        return self.find_module(module, child)
        return None


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # 保留以兼容旧代码；当前不使用主动抛异常的短路逻辑
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None


class LastLayerToExtractReachedException(Exception):
    pass
