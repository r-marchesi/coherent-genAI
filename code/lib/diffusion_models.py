# torch libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoidal_embedding(t, embedding_dim):
    """
    t: Tensor of shape (batch_size,) containing time step indices.
    embedding_dim: Dimension of the output embedding.
    Returns: Tensor of shape (batch_size, embedding_dim)
    """
    device = t.device
    # Make sure t is float for multiplication
    t = t.float().unsqueeze(1)  # (batch_size, 1)
    half_dim = embedding_dim // 2
    # Compute the frequencies
    # We use log(10000) divided by (half_dim - 1)
    emb_scale = math.log(10000) / (half_dim - 1) if half_dim > 1 else 1.0
    # Create a tensor [0, 1, ..., half_dim-1] and scale it.
    freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)  # (half_dim,)
    # (batch_size, half_dim)
    emb = t * freq.unsqueeze(0)
    # Concatenate sin and cos. If embedding_dim is odd, you can pad one extra dimension.
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (batch_size, embedding_dim)



class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
       
        # Create noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
       
        # Store as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
 
    def get_beta(self, t):
        """Retrieve beta_t given a timestep t"""
        return self.betas[t] if isinstance(t, int) else self.betas[t].view(-1, 1)
 
    def get_alpha_bar(self, t):
        """Retrieve alpha_bar_t given a timestep t"""
        return self.alphas_cumprod[t] if isinstance(t, int) else self.alphas_cumprod[t].view(-1, 1)
 
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
       
        sqrt_alpha_bar_t = torch.sqrt(self.get_alpha_bar(t))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.get_alpha_bar(t))
       
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise



class MultiConditioningDiffusionModelUnet(nn.Module):
    def __init__(
        self,
        diffusion,
        num_features,
        cond_dims,            # int or list[int]: dimensions of conditioning vectors.
        cond_embedding_dims,  # int or list[int]: corresponding embedding dimensions.
        fusion_method='concat',  # 'concat' or 'sum'
        initial_size=1024,
        bottleneck_size=64,
        n_layers=4,
        time_embedding_dim=128
    ):
        """
        A diffusion model that supports both multi-conditioning (with masking) and the 
        simple case of a single conditioning vector.
        
        Args:
            diffusion: Diffusion process object.
            num_features (int): Dimensionality of the input data.
            cond_dims (int or list[int]): Conditioning dimensions.
            cond_embedding_dims (int or list[int]): Desired embedding dimensions.
            fusion_method (str): Method to fuse conditioning embeddings ('concat' or 'sum').
            initial_size, bottleneck_size, n_layers, time_embedding_dim: Network hyperparameters.
        """
        super().__init__()
        self.diffusion = diffusion
        self.num_features = num_features
        self.time_embedding_dim = time_embedding_dim
        self.n_layers = n_layers
        self.fusion_method = fusion_method

        # Ensure conditioning parameters are lists.
        if not isinstance(cond_dims, list):
            cond_dims = [cond_dims]
        if not isinstance(cond_embedding_dims, list):
            cond_embedding_dims = [cond_embedding_dims]*len(cond_dims)
        self.cond_dims = cond_dims
        self.cond_embedding_dims = cond_embedding_dims
        
        # Determine total conditioning embedding dimension.
        if fusion_method == 'concat':
            self.total_cond_emb_dim = sum(cond_embedding_dims)
        elif fusion_method == 'sum':
            if len(set(cond_embedding_dims)) != 1:
                raise ValueError("For sum fusion, all cond_embedding_dims must be equal.")
            self.total_cond_emb_dim = cond_embedding_dims[0]
        else:
            raise ValueError("Unsupported fusion_method. Use 'concat' or 'sum'.")

        # Create a set of embedding layers (one per conditioning vector).
        self.cond_embeddings = nn.ModuleList([
            nn.Linear(dim, emb_dim) for dim, emb_dim in zip(self.cond_dims, self.cond_embedding_dims)
        ])

        self.time_embedding_dense = nn.Linear(time_embedding_dim, time_embedding_dim)
        
        # === ENCODER ===
        # The first layer input dimension: [x, t_emb, cond_emb]
        encoder_input_dim = num_features + time_embedding_dim + self.total_cond_emb_dim
        self.encoder_sizes = [
            int(initial_size * (bottleneck_size / initial_size) ** (i / float(n_layers)))
            for i in range(n_layers)
        ]
        self.encoder_layers = nn.ModuleList([nn.Linear(encoder_input_dim, self.encoder_sizes[0])])
        for i in range(1, n_layers):
            self.encoder_layers.append(nn.Linear(self.encoder_sizes[i-1], self.encoder_sizes[i]))
        
        self.encoder_time_projs = nn.ModuleList([
            nn.Linear(time_embedding_dim, self.encoder_sizes[i]) for i in range(1, n_layers)
        ])
        self.encoder_cond_projs = nn.ModuleList([
            nn.Linear(self.total_cond_emb_dim, self.encoder_sizes[i]) for i in range(1, n_layers)
        ])

        # === BOTTLENECK ===
        self.bottleneck_time_proj = nn.Linear(time_embedding_dim, self.encoder_sizes[-1])
        self.bottleneck_cond_proj = nn.Linear(self.total_cond_emb_dim, self.encoder_sizes[-1])
        self.bottleneck = nn.Linear(self.encoder_sizes[-1], bottleneck_size)

        # === DECODER ===
        rev_sizes = list(reversed(self.encoder_sizes))
        self.decoder_layers = nn.ModuleList()
        self.decoder_time_projs = nn.ModuleList()
        self.decoder_cond_projs = nn.ModuleList()
        d_in = bottleneck_size
        self.decoder_sizes = []
        for i in range(n_layers):
            skip_dim = rev_sizes[i]
            dec_in_dim = d_in + skip_dim
            self.decoder_time_projs.append(nn.Linear(time_embedding_dim, dec_in_dim))
            self.decoder_cond_projs.append(nn.Linear(self.total_cond_emb_dim, dec_in_dim))
            self.decoder_layers.append(nn.Linear(dec_in_dim, skip_dim))
            d_in = skip_dim
            self.decoder_sizes.append(skip_dim)
        
        # === FINAL OUTPUT LAYER ===
        final_in_dim = self.decoder_sizes[-1] + num_features
        self.final_time_proj = nn.Linear(time_embedding_dim, final_in_dim)
        self.final_cond_proj = nn.Linear(self.total_cond_emb_dim, final_in_dim)
        self.output_layer = nn.Linear(final_in_dim, num_features)

    def embed_conditions(self, conds, cond_masks=None):
        """
        Embeds the conditioning inputs.
        
        Args:
            conds: Either a single tensor of shape (batch_size, cond_dim) or a list of tensors.
            cond_masks: (optional) Either a tensor mask for the single conditioning or a list of masks.
                        Each mask should be of shape (batch_size,) or (batch_size, 1).
                        
        Returns:
            Tensor: Fused conditioning embedding.
        """
        # If a single tensor is provided, wrap it into a list.
        if not isinstance(conds, list):
            conds = [conds]
        
 
        embeddings = []
        # Process each condition individually.
        for i, (condition, emb_layer) in enumerate(zip(conds, self.cond_embeddings)):
        
      
            emb = F.relu(emb_layer(condition))
            # Apply mask if provided.
            if cond_masks is not None:
                mask = cond_masks[i]
                if mask.dim() == 1: # transform into a column vector 
                    mask = mask.unsqueeze(-1)
                emb = emb * mask
            
            embeddings.append(emb)
        
        # Fuse embeddings.
        if self.fusion_method == 'concat':
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'sum':
            fused = embeddings[0]
            for e in embeddings[1:]:
                fused = fused + e
        else:
            raise ValueError("Unsupported fusion_method.")
        return fused

    def forward(self, x, t, conds, cond_masks=None):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, num_features).
            t: Tensor of shape (batch_size,) representing time steps.
            conds: Conditioning input. It can be either a single tensor or a list of tensors.
            cond_masks: (Optional) Mask(s) for conditioning; either a tensor or list of tensors.
        """
        # Embed time.
        t_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)
        t_emb = F.relu(self.time_embedding_dense(t_emb))
        
        # Embed condition(s).
        cond_emb = self.embed_conditions(conds, cond_masks)
        

        
        # --- Encoder ---
        encoder_skips = []
        h = torch.cat([x, t_emb, cond_emb], dim=-1)
        h = F.relu(self.encoder_layers[0](h))
        encoder_skips.append(h)
        
        for i in range(1, self.n_layers):
            t_emb_scaled = self.encoder_time_projs[i-1](t_emb)
            cond_emb_scaled = self.encoder_cond_projs[i-1](cond_emb)
            h = F.relu(self.encoder_layers[i](h) + t_emb_scaled + cond_emb_scaled)
            encoder_skips.append(h)
        
        # --- Bottleneck ---
        h = h + self.bottleneck_time_proj(t_emb) + self.bottleneck_cond_proj(cond_emb)
        h = F.relu(self.bottleneck(h))
        
        # --- Decoder ---
        for i, dec_layer in enumerate(self.decoder_layers):
            skip = encoder_skips[-(i + 1)]
            dec_in = torch.cat([h, skip], dim=-1)
            dec_in = dec_in + self.decoder_time_projs[i](t_emb) + self.decoder_cond_projs[i](cond_emb)
            h = F.relu(dec_layer(dec_in))
        
        # --- Final Output ---
        final_in = torch.cat([h, x], dim=-1)
        final_in = final_in + self.final_time_proj(t_emb) + self.final_cond_proj(cond_emb)
        out = self.output_layer(final_in)
        return out


class MultiConditioningDiffusionModelUnetV2(nn.Module): # conditional embeddings only at the beginning and in the bottleneck
    def __init__(
        self,
        diffusion,
        num_features,
        cond_dims,            # int or list[int]: dimensions of conditioning vectors.
        cond_embedding_dims,  # int or list[int]: corresponding embedding dimensions.
        fusion_method='concat',  # 'concat' or 'sum'
        initial_size=1024,
        bottleneck_size=64,
        n_layers=4,
        time_embedding_dim=128
    ):
        super().__init__()
        self.diffusion = diffusion
        self.num_features = num_features
        self.time_embedding_dim = time_embedding_dim
        self.n_layers = n_layers
        self.fusion_method = fusion_method

        # Ensure conditioning parameters are lists.
        if not isinstance(cond_dims, list):
            cond_dims = [cond_dims]
        if not isinstance(cond_embedding_dims, list):
            cond_embedding_dims = [cond_embedding_dims] * len(cond_dims)
        self.cond_dims = cond_dims
        self.cond_embedding_dims = cond_embedding_dims

        # Determine total conditioning embedding dimension.
        if fusion_method == 'concat':
            self.total_cond_emb_dim = sum(cond_embedding_dims)
        elif fusion_method == 'sum':
            if len(set(cond_embedding_dims)) != 1:
                raise ValueError("For sum fusion, all cond_embedding_dims must be equal.")
            self.total_cond_emb_dim = cond_embedding_dims[0]
        else:
            raise ValueError("Unsupported fusion_method. Use 'concat' or 'sum'.")

        # Embedding layers for each conditioning vector
        self.cond_embeddings = nn.ModuleList([
            nn.Linear(dim, emb_dim) for dim, emb_dim in zip(self.cond_dims, self.cond_embedding_dims)
        ])

        # Time embedding
        self.time_embedding_dense = nn.Linear(time_embedding_dim, time_embedding_dim)

        # Compute encoder sizes
        self.encoder_sizes = [
            int(initial_size * (bottleneck_size / initial_size) ** (i / float(n_layers)))
            for i in range(n_layers)
        ]

        # --- ENCODER ---
        # First layer sees x + t_emb + cond_emb
        encoder_input_dim = num_features + time_embedding_dim + self.total_cond_emb_dim
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(encoder_input_dim, self.encoder_sizes[0]))
        for i in range(1, n_layers):
            self.encoder_layers.append(nn.Linear(self.encoder_sizes[i-1], self.encoder_sizes[i]))

        # Time projections for encoder layers 1..n_layers-1
        self.encoder_time_projs = nn.ModuleList([
            nn.Linear(time_embedding_dim, self.encoder_sizes[i]) for i in range(1, n_layers)
        ])

        # --- BOTTLENECK ---
        self.bottleneck_time_proj = nn.Linear(time_embedding_dim, self.encoder_sizes[-1])
        self.bottleneck_cond_proj = nn.Linear(self.total_cond_emb_dim, self.encoder_sizes[-1])
        self.bottleneck = nn.Linear(self.encoder_sizes[-1], bottleneck_size)

        # --- DECODER ---
        rev_sizes = list(reversed(self.encoder_sizes))
        self.decoder_layers = nn.ModuleList()
        self.decoder_time_projs = nn.ModuleList()
        d_in = bottleneck_size
        for i in range(n_layers):
            skip_dim = rev_sizes[i]
            dec_in_dim = d_in + skip_dim
            self.decoder_time_projs.append(nn.Linear(time_embedding_dim, dec_in_dim))
            self.decoder_layers.append(nn.Linear(dec_in_dim, skip_dim))
            d_in = skip_dim

        # --- FINAL OUTPUT ---
        final_in_dim = self.decoder_layers[-1].out_features + num_features
        self.final_time_proj = nn.Linear(time_embedding_dim, final_in_dim)
        self.output_layer = nn.Linear(final_in_dim, num_features)

    def embed_conditions(self, conds, cond_masks=None):
        """
        Embeds the conditioning inputs once at the start.
        """
        if not isinstance(conds, list):
            conds = [conds]
        embeddings = []
        for i, (condition, emb_layer) in enumerate(zip(conds, self.cond_embeddings)):
            emb = F.relu(emb_layer(condition))
            if cond_masks is not None:
                mask = cond_masks[i]
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                emb = emb * mask
            embeddings.append(emb)

        if self.fusion_method == 'concat':
            return torch.cat(embeddings, dim=-1)
        else:
            fused = embeddings[0]
            for e in embeddings[1:]:
                fused = fused + e
            return fused

    def forward(self, x, t, conds, cond_masks=None):
        # Time embedding
        t_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)
        t_emb = F.relu(self.time_embedding_dense(t_emb))

        # Conditioning embedding (only used in first layer & bottleneck)
        cond_emb = self.embed_conditions(conds, cond_masks)

        # --- ENCODER ---
        skips = []
        h = torch.cat([x, t_emb, cond_emb], dim=-1)
        h = F.relu(self.encoder_layers[0](h))
        skips.append(h)

        for i in range(1, self.n_layers):
            h = self.encoder_layers[i](h) + self.encoder_time_projs[i-1](t_emb)
            h = F.relu(h)
            skips.append(h)

        # --- BOTTLENECK ---
        h = h + self.bottleneck_time_proj(t_emb) + self.bottleneck_cond_proj(cond_emb)
        h = F.relu(self.bottleneck(h))

        # --- DECODER ---
        for i, dec_layer in enumerate(self.decoder_layers):
            skip = skips[-(i + 1)]
            dec_in = torch.cat([h, skip], dim=-1)
            dec_in = dec_in + self.decoder_time_projs[i](t_emb)
            h = F.relu(dec_layer(dec_in))

        # --- FINAL OUTPUT ---
        final_in = torch.cat([h, x], dim=-1)
        final_in = final_in + self.final_time_proj(t_emb)
        return self.output_layer(final_in)



class MultiConditioningDiffusionModelMLP(nn.Module):
    def __init__(
        self,
        diffusion,
        num_features,
        cond_dims,            # int or list[int]: dimensions of conditioning vectors.
        cond_embedding_dims,  # int or list[int]: corresponding embedding dimensions.
        fusion_method='concat',  # 'concat' or 'sum'
        hidden_size=512,      # Hidden layer size for the MLP.
        n_layers=4,           # Number of layers in the MLP.
        time_embedding_dim=128
    ):
        """
        A diffusion model that supports both multi-conditioning (with masking) and the 
        simple case of a single conditioning vector, implemented as an MLP.
        
        Args:
            diffusion: Diffusion process object.
            num_features (int): Dimensionality of the input data.
            cond_dims (int or list[int]): Conditioning dimensions.
            cond_embedding_dims (int or list[int]): Desired embedding dimensions.
            fusion_method (str): Method to fuse conditioning embeddings ('concat' or 'sum').
            hidden_size (int): Size of hidden layers in the MLP.
            n_layers (int): Number of layers in the MLP.
            time_embedding_dim (int): Dimensionality of the time embedding.
        """
        super().__init__()
        self.diffusion = diffusion
        self.num_features = num_features
        self.time_embedding_dim = time_embedding_dim
        self.n_layers = n_layers
        self.fusion_method = fusion_method

        # Ensure conditioning parameters are lists.
        if not isinstance(cond_dims, list):
            cond_dims = [cond_dims]
        if not isinstance(cond_embedding_dims, list):
            cond_embedding_dims = [cond_embedding_dims] * len(cond_dims)
        self.cond_dims = cond_dims
        self.cond_embedding_dims = cond_embedding_dims
        
        # Determine total conditioning embedding dimension.
        if fusion_method == 'concat':
            self.total_cond_emb_dim = sum(cond_embedding_dims)
        elif fusion_method == 'sum':
            if len(set(cond_embedding_dims)) != 1:
                raise ValueError("For sum fusion, all cond_embedding_dims must be equal.")
            self.total_cond_emb_dim = cond_embedding_dims[0]
        else:
            raise ValueError("Unsupported fusion_method. Use 'concat' or 'sum'.")

        # Create a set of embedding layers (one per conditioning vector).
        self.cond_embeddings = nn.ModuleList([
            nn.Linear(dim, emb_dim) for dim, emb_dim in zip(self.cond_dims, self.cond_embedding_dims)
        ])

        self.time_embedding_dense = nn.Linear(time_embedding_dim, time_embedding_dim)
        
        # === MLP Layers ===
        # Input dimension: [x, t_emb, cond_emb]
        input_dim = num_features + time_embedding_dim + self.total_cond_emb_dim
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            out_dim = hidden_size if i < n_layers - 1 else num_features
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))

    def embed_conditions(self, conds, cond_masks=None):
        """
        Embeds the conditioning inputs.
        
        Args:
            conds: Either a single tensor of shape (batch_size, cond_dim) or a list of tensors.
            cond_masks: (optional) Either a tensor mask for the single conditioning or a list of masks.
                        Each mask should be of shape (batch_size,) or (batch_size, 1).
                        
        Returns:
            Tensor: Fused conditioning embedding.
        """
        # If a single tensor is provided, wrap it into a list.
        if not isinstance(conds, list):
            conds = [conds]
        
        embeddings = []
        # Process each condition individually.
        for i, (condition, emb_layer) in enumerate(zip(conds, self.cond_embeddings)):
            emb = F.relu(emb_layer(condition))
            # Apply mask if provided.
            if cond_masks is not None:
                mask = cond_masks[i]
                if mask.dim() == 1:  # Transform into a column vector.
                    mask = mask.unsqueeze(-1)
                emb = emb * mask
            embeddings.append(emb)
        
        # Fuse embeddings.
        if self.fusion_method == 'concat':
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'sum':
            fused = embeddings[0]
            for e in embeddings[1:]:
                fused = fused + e
        else:
            raise ValueError("Unsupported fusion_method.")
        return fused

    def forward(self, x, t, conds, cond_masks=None):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, num_features).
            t: Tensor of shape (batch_size,) representing time steps.
            conds: Conditioning input. It can be either a single tensor or a list of tensors.
            cond_masks: (Optional) Mask(s) for conditioning; either a tensor or list of tensors.
        """
        # Embed time.
        t_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)
        t_emb = F.relu(self.time_embedding_dense(t_emb))
        
        # Embed condition(s).
        cond_emb = self.embed_conditions(conds, cond_masks)
        
        # Concatenate inputs.
        h = torch.cat([x, t_emb, cond_emb], dim=-1)
        
        # Pass through MLP layers.
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < self.n_layers - 1:  
                h = F.relu(h)
        
        return h



class MultiConditioningDiffusionModelNormMLP(nn.Module):
    def __init__(
        self,
        diffusion,
        num_features,
        cond_dims,            # int or list[int]: dimensions of conditioning vectors.
        cond_embedding_dims,  # int or list[int]: corresponding embedding dimensions.
        fusion_method='concat',  # 'concat' or 'sum'
        hidden_size=512,      # Hidden layer size for the MLP.
        n_layers=4,           # Number of layers in the MLP.
        time_embedding_dim=128
    ):
        """
        A diffusion model that supports both multi-conditioning (with masking) and the 
        simple case of a single conditioning vector, implemented as an MLP.
        
        Args:
            diffusion: Diffusion process object.
            num_features (int): Dimensionality of the input data.
            cond_dims (int or list[int]): Conditioning dimensions.
            cond_embedding_dims (int or list[int]): Desired embedding dimensions.
            fusion_method (str): Method to fuse conditioning embeddings ('concat' or 'sum').
            hidden_size (int): Size of hidden layers in the MLP.
            n_layers (int): Number of layers in the MLP.
            time_embedding_dim (int): Dimensionality of the time embedding.
        """
        super().__init__()
        self.diffusion = diffusion
        self.num_features = num_features
        self.time_embedding_dim = time_embedding_dim
        self.n_layers = n_layers
        self.fusion_method = fusion_method

        # Ensure conditioning parameters are lists.
        if not isinstance(cond_dims, list):
            cond_dims = [cond_dims]
        if not isinstance(cond_embedding_dims, list):
            cond_embedding_dims = [cond_embedding_dims] * len(cond_dims)
        self.cond_dims = cond_dims
        self.cond_embedding_dims = cond_embedding_dims
        
        # Determine total conditioning embedding dimension.
        if fusion_method == 'concat':
            self.total_cond_emb_dim = sum(cond_embedding_dims)
        elif fusion_method == 'sum':
            if len(set(cond_embedding_dims)) != 1:
                raise ValueError("For sum fusion, all cond_embedding_dims must be equal.")
            self.total_cond_emb_dim = cond_embedding_dims[0]
        else:
            raise ValueError("Unsupported fusion_method. Use 'concat' or 'sum'.")

        # Create a set of embedding layers (one per conditioning vector).
        self.cond_embeddings = nn.ModuleList([
            nn.Linear(dim, emb_dim) for dim, emb_dim in zip(self.cond_dims, self.cond_embedding_dims)
        ])

        self.time_embedding_dense = nn.Linear(time_embedding_dim, time_embedding_dim)
        
        # === MLP Layers ===
        # Input dimension: [x, t_emb, cond_emb]
        input_dim = num_features + time_embedding_dim + self.total_cond_emb_dim
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size + time_embedding_dim
            out_dim = hidden_size if i < n_layers - 1 else num_features
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))

        # === BatchNorm after each hidden layer ===
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size)
            for _ in range(n_layers - 1)
        ])


    def embed_conditions(self, conds, cond_masks=None):
        """
        Embeds the conditioning inputs.
        
        Args:
            conds: Either a single tensor of shape (batch_size, cond_dim) or a list of tensors.
            cond_masks: (optional) Either a tensor mask for the single conditioning or a list of masks.
                        Each mask should be of shape (batch_size,) or (batch_size, 1).
                        
        Returns:
            Tensor: Fused conditioning embedding.
        """
        # If a single tensor is provided, wrap it into a list.
        if not isinstance(conds, list):
            conds = [conds]
        
        embeddings = []
        # Process each condition individually.
        for i, (condition, emb_layer) in enumerate(zip(conds, self.cond_embeddings)):
            emb = F.relu(emb_layer(condition))
            # Apply mask if provided.
            if cond_masks is not None:
                mask = cond_masks[i]
                if mask.dim() == 1:  # Transform into a column vector.
                    mask = mask.unsqueeze(-1)
                emb = emb * mask
            embeddings.append(emb)
        
        # Fuse embeddings.
        if self.fusion_method == 'concat':
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'sum':
            fused = embeddings[0]
            for e in embeddings[1:]:
                fused = fused + e
        else:
            raise ValueError("Unsupported fusion_method.")
        return fused

    def forward(self, x, t, conds, cond_masks=None):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, num_features).
            t: Tensor of shape (batch_size,) representing time steps.
            conds: Conditioning input. It can be either a single tensor or a list of tensors.
            cond_masks: (Optional) Mask(s) for conditioning; either a tensor or list of tensors.
        """
        # Embed time.
        t_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)
        t_emb = F.relu(self.time_embedding_dense(t_emb))
        
        # Embed condition(s).
        cond_emb = self.embed_conditions(conds, cond_masks)
        
        # Concatenate inputs.
        h = torch.cat([x, t_emb, cond_emb], dim=-1)
        
        # Pass through MLP layers.
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            # apply BN + ReLU on all but last layer
            if i < self.n_layers - 1:
                h = self.bn_layers[i](h)
                h = F.relu(h)
                h = torch.cat([h, t_emb], dim=-1)
        
        return h



class MultiConditioningDiffusionModelNormMLP_v2(nn.Module):
    def __init__(
        self,
        diffusion,
        num_features,
        cond_dims,          # int or list[int]: dimensions of conditioning vectors.
        hidden_size=512,     # Hidden layer size for the MLP.
        n_layers=4,          # Number of layers in the MLP.
        time_embedding_dim=128
    ):
        """
        A diffusion model that supports multi-conditioning by concatenating raw
        conditioning data and masks directly to the input.

        Args:
            diffusion: An instance of the GaussianDiffusion process.
            num_features (int): Dimensionality of the input data (x).
            cond_dims (int or list[int]): Dimensions of the conditioning vectors.
            hidden_size (int): Size of hidden layers in the MLP.
            n_layers (int): Number of layers in the MLP.
            time_embedding_dim (int): Dimensionality of the time embedding.
        """
        super().__init__()
        self.diffusion = diffusion
        self.num_features = num_features
        self.time_embedding_dim = time_embedding_dim
        self.n_layers = n_layers

        # Ensure cond_dims is a list for consistent processing.
        if not isinstance(cond_dims, list):
            cond_dims = [cond_dims]
        self.cond_dims = cond_dims

        self.time_embedding_dense = nn.Linear(time_embedding_dim, time_embedding_dim)
        
        # === MLP Layers ===
        # Calculate the total input dimension for the first layer of the MLP.
        # Input: [x, t_emb, all_conds, all_masks]
        total_cond_dim = sum(self.cond_dims)
        num_masks = len(self.cond_dims) # One mask for each conditioning input.
        input_dim = num_features + time_embedding_dim + total_cond_dim + num_masks
        
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            # Input dimension changes for the first layer.
            # Subsequent layers have a skip connection with the time embedding.
            in_dim = input_dim if i == 0 else hidden_size + time_embedding_dim
            out_dim = hidden_size if i < n_layers - 1 else num_features
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))

        # === BatchNorm Layers ===
        # We need a batch normalization layer for each hidden layer.
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size)
            for _ in range(n_layers - 1)
        ])

    def forward(self, x, t, conds, cond_masks=None):
        """
        Performs the forward pass of the model.

        Args:
            x (Tensor): The input data, shape (batch_size, num_features).
            t (Tensor): The current timestep, shape (batch_size,).
            conds (Tensor or list[Tensor]): Conditioning inputs.
            cond_masks (Tensor or list[Tensor], optional): Masks for the conditions.
                                                        If None, conditions are assumed to be present.

        Returns:
            Tensor: The predicted noise.
        """
        # 1. Embed the time step.
        t_emb = get_sinusoidal_embedding(t, self.time_embedding_dim)
        t_emb = F.relu(self.time_embedding_dense(t_emb))
        
        # 2. Prepare conditioning vectors and masks for concatenation.
        # Ensure conds is a list.
        if not isinstance(conds, list):
            conds = [conds]

        # Process masks. If not provided, create masks of ones (indicating presence).
        processed_masks = []
        if cond_masks is None:
            # If no masks are given, all conditions are considered present (mask = 1).
            for c in conds:
                batch_size = c.shape[0]
                mask = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
                processed_masks.append(mask)
        else:
            # If masks are provided, ensure they are in list format and have the correct shape.
            if not isinstance(cond_masks, list):
                cond_masks = [cond_masks]
            for mask in cond_masks:
                # Reshape from (batch_size,) to (batch_size, 1) if necessary.
                processed_masks.append(mask.unsqueeze(-1) if mask.dim() == 1 else mask)
        
        # 3. Concatenate all inputs: [x, t_emb, cond_vectors..., mask_vectors...]
        all_conds = torch.cat(conds, dim=-1)
        all_masks = torch.cat(processed_masks, dim=-1)
        h = torch.cat([x, t_emb, all_conds, all_masks], dim=-1)
        
        # 4. Pass the combined tensor through the MLP.
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            # Apply BatchNorm, ReLU, and skip connection for all but the last layer.
            if i < self.n_layers - 1:
                h = self.bn_layers[i](h)
                h = F.relu(h)
                # Add skip connection with the time embedding.
                h = torch.cat([h, t_emb], dim=-1)
        
        return h