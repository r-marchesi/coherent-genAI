from lib.diffusion_models import (
    MultiConditioningDiffusionModelUnet,
    MultiConditioningDiffusionModelUnetV2,
    MultiConditioningDiffusionModelMLP,
    MultiConditioningDiffusionModelNormMLP,
    MultiConditioningDiffusionModelNormMLP_v2,
)

_MODEL_REGISTRY = {
    "unet": MultiConditioningDiffusionModelUnet,
    "unet_v2": MultiConditioningDiffusionModelUnetV2,
    "mlp": MultiConditioningDiffusionModelMLP,
    "norm_mlp": MultiConditioningDiffusionModelNormMLP,
    "norm_mlp_v2": MultiConditioningDiffusionModelNormMLP_v2,
}

def get_diffusion_model(
    model_name: str,
    diffusion,
    cfg,
    x_dim: int,
    cond_dims,
    fusion_method: str = "concat"
):
    """
    Build both the GaussianDiffusion process *and* the model itself.
    
    Args:
      model_name: one of _MODEL_REGISTRY keys
      cfg:      SimpleNamespace with your hyperparams (initial_size, bottleneck_size, …)
      x_dim:    number of features in x
      cond_dims: int or list[int] of conditioning dims
      fusion_method: only used for multi-conditioning
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{model_name}'. "
            f"Available options: {list(_MODEL_REGISTRY.keys())}"
        )

    ModelClass = _MODEL_REGISTRY[model_name]

    # UNet variants
    if model_name == "unet":
        return ModelClass(
            diffusion,
            num_features = x_dim,
            cond_dims = cond_dims,
            cond_embedding_dims = cfg.cond_embedding_dim,
            fusion_method = fusion_method,
            initial_size = cfg.initial_size,
            bottleneck_size = cfg.bottleneck_size,
            n_layers = cfg.n_layers,
            time_embedding_dim = cfg.time_embedding_dimension,
        )

    elif model_name == "unet_v2":
        return ModelClass(
            diffusion,
            num_features = x_dim,
            cond_dims = cond_dims,
            cond_embedding_dims = cfg.cond_embedding_dim,
            fusion_method = fusion_method,
            initial_size = cfg.initial_size,
            bottleneck_size = cfg.bottleneck_size,
            n_layers = cfg.n_layers,
            time_embedding_dim = cfg.time_embedding_dimension,
        )

    # MLP variants
    elif model_name == "mlp":
        return ModelClass(
            diffusion,
            num_features = x_dim,
            cond_dims = cond_dims,
            cond_embedding_dims = cfg.cond_embedding_dim,
            fusion_method = fusion_method,
            hidden_size = cfg.initial_size,
            n_layers = cfg.n_layers,
            time_embedding_dim = cfg.time_embedding_dimension,
        )

    elif model_name == "norm_mlp":
        return ModelClass(
            diffusion,
            num_features = x_dim,
            cond_dims = cond_dims,
            cond_embedding_dims = cfg.cond_embedding_dim,
            fusion_method = fusion_method,
            hidden_size = cfg.initial_size,
            n_layers = cfg.n_layers,
            time_embedding_dim = cfg.time_embedding_dimension,
        )    

    
    elif model_name == "norm_mlp_v2":
        return ModelClass(
            diffusion,
            num_features = x_dim,
            cond_dims = cond_dims,
            # Note: cond_embedding_dims and fusion_method are not needed
            hidden_size = cfg.initial_size,
            n_layers = cfg.n_layers,
            time_embedding_dim = cfg.time_embedding_dimension,
        )

    