import torch
import torch.nn as nn

class ZINBAE(nn.Module):
    def __init__(
        self,
        seq_length=2000,
        feature_dim=8,
        layers=(512, 256, 128),
        use_conv=False,
        conv_channels=64,
        pool_size=2,
        kernel_size=3,
        padding='same',
        stride=1,
        dropout=0.0,
        mu_offset=0,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'ZINBAE'
        self.use_conv = use_conv
        self.dropout = dropout
        self.mu_offset = mu_offset
        
        # ----- Optional Conv1D Layer -----
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            
            # Calculate correct sequence length after conv and pooling
            if padding == 'same':
                # With padding='same', length is preserved (only works with stride=1)
                conv_seq_length = seq_length
            else:
                # With valid padding (padding=0 or numeric), length reduces
                if isinstance(padding, str):
                    padding = 0
                conv_seq_length = (seq_length + 2 * padding - kernel_size) // stride + 1
            
            pooled_seq_length = conv_seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels
        else:
            input_dim = seq_length * feature_dim
        
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ----- Latent bottleneck layer -----
        # Compress to half the size of last encoder layer
        self.latent_dim = layers[-1] // 2
        self.latent_layer = nn.Linear(layers[-1], self.latent_dim)
        self.latent_relu = nn.ReLU()
        
        # ----- Decoder "body" (shared) -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # First expand from latent back to last encoder layer size
        decoder_layers.append(nn.Linear(prev_dim, layers[-1]))
        decoder_layers.append(nn.ReLU())
        if dropout > 0:
            decoder_layers.append(nn.Dropout(p=dropout))
        prev_dim = layers[-1]
        
        # Mirror all encoder layers in reverse (except last which we just handled)
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        
        # Count only Linear layers
        num_encoder_linear = sum(1 for layer in encoder_layers if isinstance(layer, nn.Linear))
        num_decoder_linear = sum(1 for layer in decoder_layers if isinstance(layer, nn.Linear))
        print(f"Number of encoder layers: {num_encoder_linear}")
        print(f"Number of decoder layers (excluding heads): {num_decoder_linear}")
        print(f"Total number of layers including latent layer: {num_encoder_linear + 1 + num_decoder_linear}")
        
        # this shared decoder output D will feed μ, θ, π heads
        self.decoder_shared = nn.Sequential(*decoder_layers)
        decoder_out_dim = prev_dim  # last h in loop above
        
        # ----- ZINB heads -----
        # Each outputs seq_length parameters (one per position)
        self.mu_layer    = nn.Linear(decoder_out_dim, seq_length)
        self.theta_layer = nn.Linear(decoder_out_dim, seq_length)
        self.pi_layer    = nn.Linear(decoder_out_dim, seq_length)
        
        # Initialize ZINB output layers with smaller weights to prevent initial explosion
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.theta_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.pi_layer.weight, gain=0.1)
        # Initialize biases to reasonable starting values
        nn.init.constant_(self.mu_layer.bias, 0.0)  # Will result in mu_hat ~= 1 after exp
        nn.init.constant_(self.theta_layer.bias, 0.0)  # Will result in theta ~= 1 after exp
        nn.init.constant_(self.pi_layer.bias, -2.0)  # Will result in pi ~= 0.12 after sigmoid
    
    def forward(self, x_in, size_factors):
        """
        x_in: preprocessed input (e.g. log1p(CPM), maybe scaled)
              shape (batch, seq_length, feature_dim)
        size_factors: library-size factors for each sample
              shape (batch,) or (batch, 1)
        """
        batch_size = x_in.size(0)
        
        if self.use_conv:
            # (batch, seq_length, feature_dim) -> (batch, feature_dim, seq_length)
            x = x_in.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.conv_relu(x)
            x = self.pool(x)  # (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x_in
        
        x = x.view(batch_size, -1)  # flatten
        
        # Encode
        h = self.encoder(x)
        
        # Latent bottleneck
        z = self.latent_layer(h)
        z = self.latent_relu(z)
        
        # Decode shared representation
        D = self.decoder_shared(z)  # shape (batch, decoder_out_dim)
        
        # ZINB parameters with clamping to prevent overflow
        # Use softplus or clamped exp to prevent exploding values
        mu_hat_logits = self.mu_layer(D)                 # log-mean (unscaled)
        # mu_hat_logits = torch.clamp(mu_hat_logits, -20, 20)

        if size_factors.dim() == 1:
            size_factors = size_factors.unsqueeze(1)
        log_sf = torch.log(size_factors.clamp_min(1e-8))

        log_mu = mu_hat_logits + log_sf
        log_mu = torch.clamp(log_mu, min=-20, max=20)
        # mu = torch.exp(log_mu) 
        mu=torch.exp(log_mu) + self.mu_offset
        # mu = torch.nn.functional.softplus(log_mu) + 1e-4  # ensure positivity 
        
        theta_logits = self.theta_layer(D)
        theta = torch.clamp(theta_logits, min=-20, max=10)
        # theta = torch.exp(theta)    # (batch, seq_length), positive
        theta = torch.exp(theta).clamp(min=1)
        # theta = torch.nn.functional.softplus(theta) + 1e-4
        
        pi = torch.sigmoid(self.pi_layer(D))
        pi = pi.clamp(1e-5, 1 - 1e-5)

        return mu, theta, pi, z
    
class ZINBVAE(nn.Module):
    def __init__(
        self,
        seq_length=2000,
        feature_dim=8,
        layers=(512, 256, 128),
        use_conv=False,
        conv_channels=64,
        pool_size=2,
        kernel_size=3,
        padding=1,
        stride=1,
        dropout=0.0,
        mu_offset=0,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'ZINBVAE'
        self.use_conv = use_conv
        self.dropout = dropout
        self.mu_offset = mu_offset
        
        # ----- Optional Conv1D Layer -----
        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            pooled_seq_length = seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels
        else:
            input_dim = seq_length * feature_dim
            
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ----- Latent bottleneck layer -----
        # Compress to half the size of last encoder layer
        encoder_out_dim = layers[-1]
        self.latent_dim = encoder_out_dim // 2
        self.latent_layer = nn.Linear(encoder_out_dim, self.latent_dim)
        self.latent_relu = nn.ReLU()
        
        # Latent space layers (for VAE reparameterization)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        
        # ----- Decoder -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # First expand from latent back to last encoder layer size
        decoder_layers.append(nn.Linear(prev_dim, encoder_out_dim))
        decoder_layers.append(nn.ReLU())
        if dropout > 0:
            decoder_layers.append(nn.Dropout(p=dropout))
        prev_dim = encoder_out_dim
        
        # Mirror all encoder layers in reverse (except last which we just handled)
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h
            
        # this shared decoder output D will feed μ, θ, π heads
        self.decoder_shared = nn.Sequential(*decoder_layers)
        decoder_out_dim = prev_dim  # last h in loop above
        
        # ----- ZINB heads -----
        # Each outputs seq_length parameters (one per position)
        self.mu_layer    = nn.Linear(decoder_out_dim, seq_length)
        self.theta_layer = nn.Linear(decoder_out_dim, seq_length)
        self.pi_layer    = nn.Linear(decoder_out_dim, seq_length)
        
        # Initialize ZINB output layers with smaller weights to prevent initial explosion
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.theta_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.pi_layer.weight, gain=0.1)
        # Initialize biases to reasonable starting values
        nn.init.constant_(self.mu_layer.bias, 0.0)  # Will result in mu_hat ~= 1 after exp
        nn.init.constant_(self.theta_layer.bias, 0.0)  # Will result in theta ~= 1 after exp
        nn.init.constant_(self.pi_layer.bias, -2.0)  # Will result in pi ~= 0.12 after sigmoid
    
    def encode(self, x):
        h = self.encoder(x)
        # Latent bottleneck
        z_pre = self.latent_layer(h)
        z_pre = self.latent_relu(z_pre)
        # VAE sampling parameters
        mu = self.fc_mu(z_pre)
        logvar = self.fc_logvar(z_pre)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x_in, size_factors):
        batch_size = x_in.size(0)
        
        if self.use_conv:
            # (batch, seq_length, feature_dim) -> (batch, feature_dim, seq_length)
            x = x_in.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.conv_relu(x)
            x = self.pool(x)  # (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x_in
            
        x = x.view(batch_size, -1)  # flatten
            
        # Encode
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        
        D = self.decoder_shared(z)  # shape (batch, decoder_out_dim)
        # ZINB parameters with clamping to prevent overflow
        # Compute mu similar to ZINBAE: mu_hat logits + log(size_factors) then exp
        mu_hat_logits = self.mu_layer(D)
        # mu_hat_logits = torch.clamp(mu_hat_logits, -20, 20)

        if size_factors.dim() == 1:
            size_factors = size_factors.unsqueeze(1)
        log_sf = torch.log(size_factors.clamp_min(1e-8))

        log_mu = mu_hat_logits + log_sf
        # mu = torch.nn.functional.softplus(log_mu) + 1e-4  # ensure positivity
        mu = torch.exp(log_mu) + self.mu_offset

        # theta via softplus (positive, stable)
        theta_logits = self.theta_layer(D)
        theta = torch.clamp(theta_logits, min=-20, max=10)
        theta = torch.exp(theta)
        # theta = torch.nn.functional.softplus(theta) + 1e-4

        # dropout / zero-inflation probability
        pi = torch.sigmoid(self.pi_layer(D))
        pi = pi.clamp(1e-5, 1 - 1e-5)

        return mu, theta, pi, z, mu_z, logvar_z
        