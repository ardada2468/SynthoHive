import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, List, Any, Tuple
from .base import ConditionalGenerativeModel
from .layers import ResidualLayer, Discriminator, EntityEmbeddingLayer
from syntho_hive.core.data.transformer import DataTransformer

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculate the WGAN-GP gradient penalty term.

    Args:
        discriminator: Discriminator network used to score samples.
        real_samples: Tensor of real samples after preprocessing.
        fake_samples: Tensor of generated samples.
        device: Torch device for computation.

    Returns:
        Scalar gradient penalty encouraging Lipschitz continuity.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones((real_samples.size(0), 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class CTGAN(ConditionalGenerativeModel):
    """Conditional Tabular GAN with entity embeddings and parent context."""
    def __init__(
        self,
        metadata: Any,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, int] = (256, 256),
        discriminator_dim: Tuple[int, int] = (256, 256),
        batch_size: int = 500,
        epochs: int = 300,
        device: str = "cpu",
        embedding_threshold: int = 50,
        discriminator_steps: int = 5
    ):
        """Create a CTGAN instance configured for tabular synthesis.

        Args:
            metadata: Table metadata describing columns and constraints.
            embedding_dim: Dimension of input noise vector.
            generator_dim: Hidden layer widths for the generator.
            discriminator_dim: Hidden layer widths for the discriminator.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            embedding_threshold: Cardinality threshold for switching to embeddings.
            discriminator_steps: Number of discriminator steps per generator step.
        """
        self.metadata = metadata
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs = epochs
        self.device = torch.device(device)
        self.discriminator_steps = discriminator_steps
        # Prioritize init arg, fallback to metadata if available, else default (already 50)
        self.embedding_threshold = embedding_threshold
        
        self.generator = None
        self.discriminator = None
        self.transformer = DataTransformer(metadata, embedding_threshold=self.embedding_threshold)
        self.context_transformer = DataTransformer(metadata, embedding_threshold=self.embedding_threshold)
        
        # Embedding Layers
        self.embedding_layers = nn.ModuleDict()
        self.data_column_info = [] # List of tuples: (dim, type, related_info)


    def _compile_layout(self, transformer):
        """Analyze transformer output to map column indices and types.

        Args:
            transformer: Fitted ``DataTransformer`` for the child table.
        """
        self.data_column_info = []
        self.embedding_layers = nn.ModuleDict()
        
        current_idx = 0
        for col, info in transformer._column_info.items():
            if info['type'] == 'categorical_embedding':
                # Create Embedding Layer
                num_categories = info['num_categories']
                # Heuristic for embedding dimension: min(50, num_categories/2)
                emb_dim = min(50, (num_categories + 1) // 2)
                
                self.embedding_layers[col] = EntityEmbeddingLayer(num_categories, emb_dim).to(self.device)
                
                self.data_column_info.append({
                    'name': col,
                    'type': 'embedding', 
                    'input_idx': current_idx, 
                    'input_dim': 1,
                    'output_dim': emb_dim,
                    'num_categories': num_categories 
                })
                current_idx += 1
            else:
                self.data_column_info.append({
                    'name': col,
                    'type': 'normal',
                    'input_idx': current_idx,
                    'input_dim': info['dim'],
                    'output_dim': info['dim']
                })
                current_idx += info['dim']
                
    def _apply_embeddings(self, data, is_fake=False):
        """Convert a mixed categorical/continuous tensor into embedding space.

        Args:
            data: Input tensor with mixed column representations.
            is_fake: Whether the tensor came from the generator (logits) or real data (indices).

        Returns:
            Tensor with embeddings applied to categorical columns.
        """
        parts = []
        for info in self.data_column_info:
            idx = info['input_idx']
            dim = info['input_dim']
            col_data = data[:, idx:idx+dim]
            
            if info['type'] == 'embedding':
                layer = self.embedding_layers[info['name']]
                if is_fake:
                    # col_data contains Softmax logits from Generator
                    # Needs hard Gumbel-Softmax or Softmax? Generator outputs unnormalized logits usually.
                    # Ideally Generator outputs (N, num_cats). 
                    # Wait, 'data' passed here is strictly what Generator produced.
                    # Discriminator expects (N, EmbDim).
                    
                    # Logic: Generator outputs Logits. We apply Softmax -> Dense.
                    # But wait, logic above says Generator outputs: 
                    # Embedding: Logits (dim=num_cats)
                    # Normal: Values (dim=original_dim)
                    
                    # So 'dim' in loop here must match GENERATOR output structure, not Transformer output.
                    # Compile Layout logic is slightly tricky because Generator output shape != Transformer output shape for Embeddings.
                    
                    # RE-THINK:
                    # Transformer Output (Real): [Index] (1 dim)
                    # Generator Output (Fake): [Logits] (num_cats dim)
                    
                    # This function strictly transforms Real Data (Index) -> Embedding.
                    # Or Fake Data (Logits) -> Soft Embedding.
                    
                    # Problem: input 'data' has different shapes for Real vs Fake.
                    # We need to handle them separately or have this function assume inputs are already sliced?
                    # Let's pass sliced inputs or rely on info having both dims.
                    pass
                else:
                    # Real Data: Indices -> Embedding
                    # input is (N, 1) indices
                    embeddings = layer(col_data.long().squeeze(1))
                    parts.append(embeddings)
            else:
                parts.append(col_data)
                
        # Re-implementing clearer separated logic in Build Model / Forward
        return torch.cat(parts, dim=1)

    def _build_model(self, transformer_output_dim: int, context_dim: int = 0):
        """Instantiate generator and discriminator modules.

        Args:
            transformer_output_dim: Flattened dimension of transformed child data.
            context_dim: Flattened dimension of transformed context (if any).
        """
        # 1. Compile Layout first
        self._compile_layout(self.transformer)
        
        # 2. Calculate Generator Output Dim & Discriminator Input Dim
        gen_output_dim = 0
        disc_input_dim_base = 0
        
        for info in self.data_column_info:
            if info['type'] == 'embedding':
                gen_output_dim += info['num_categories'] # Generator outputs logits
                disc_input_dim_base += info['output_dim'] # D sees embeddings
            else:
                gen_output_dim += info['output_dim']
                disc_input_dim_base += info['output_dim']

        # Generator: Noise + Context -> Data (Logits/Values)
        gen_input_dim = self.embedding_dim + context_dim
        
        self.generator = nn.Sequential(
            ResidualLayer(gen_input_dim, self.generator_dim[0]),
            ResidualLayer(self.generator_dim[0], self.generator_dim[1]),
            nn.Linear(self.generator_dim[1], gen_output_dim)
        ).to(self.device)
        
        # Discriminator: Data(Embeddings) + Context -> Score
        disc_input_dim = disc_input_dim_base + context_dim
        
        self.discriminator = Discriminator(disc_input_dim, self.discriminator_dim[0]).to(self.device)
        
    def fit(self, data: pd.DataFrame, context: Optional[pd.DataFrame] = None, table_name: Optional[str] = None, **kwargs: Any) -> None:
        """Train the CTGAN model on tabular data.

        Args:
            data: Child table data (target) to model.
            context: Parent attributes to condition on (aligned row-wise).
            table_name: Table name for metadata lookup and constraint handling.
            **kwargs: Extra training options (unused placeholder for compatibility).
        """
        # 1. Fit and Transform Data
        self.transformer.fit(data, table_name=table_name)
        train_data = self.transformer.transform(data)
        train_data = torch.from_numpy(train_data).float().to(self.device)
        
        # 2. Handle Context
        if context is not None:
            assert len(data) == len(context), "Data and context must have same number of rows"
            
            # Use dedicated transformer for context
            # NOTE: We abuse metdata here slightly. Ideally context comes from a known table (Parent).
            # But context might be a mix of parent columns. 
            # For fit, we pass table_name=None to fit on just the columns present in context df.
            self.context_transformer.fit(context)
            context_transformed = self.context_transformer.transform(context)
            context_data = torch.from_numpy(context_transformed).float().to(self.device)
            context_dim = context_data.shape[1]
        else:
            context_data = None
            context_dim = 0
            
        data_dim = train_data.shape[1]
        
        # 3. Build Model
        if self.generator is None:
            self._build_model(data_dim, context_dim)
            
        optimizer_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        
        # 4. Training Loop (WGAN-GP)
        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        
        for epoch in range(self.epochs):
            for i in range(steps_per_epoch):
                # --- Train Discriminator ---
                for _ in range(self.discriminator_steps):
                    # Sample real data
                    idx = np.random.randint(0, len(train_data), self.batch_size)
                    real_data_batch = train_data[idx]
                    if context_data is not None:
                        real_context_batch = context_data[idx]
                        real_input = torch.cat([real_data_batch, real_context_batch], dim=1)
                    else:
                        real_context_batch = None
                        real_input = real_data_batch

                    # Generate fake data
                    noise = torch.randn(self.batch_size, self.embedding_dim, device=self.device)
                    if real_context_batch is not None:
                        gen_input = torch.cat([noise, real_context_batch], dim=1)
                    else:
                        gen_input = noise
                        
                    fake_raw = self.generator(gen_input)
                    
                    # Apply Embeddings / Softmax to Fake Data
                    fake_parts = []
                    fake_ptr = 0
                    for info in self.data_column_info:
                        if info['type'] == 'embedding':
                            dim = info['num_categories']
                            logits = fake_raw[:, fake_ptr:fake_ptr+dim]
                            fake_ptr += dim
                            
                            # Gumbel Softmax or Softmax? WGAN prefers generic softmax for differentiability
                            # Note: Gumbel Softmax allows hard sampling with gradients.
                            probs = F.softmax(logits, dim=1)
                            emb_vect = self.embedding_layers[info['name']].forward_soft(probs)
                            fake_parts.append(emb_vect)
                        else:
                            dim = info['output_dim']
                            val = fake_raw[:, fake_ptr:fake_ptr+dim]
                            fake_ptr += dim
                            fake_parts.append(val)
                    
                    fake_data_batch = torch.cat(fake_parts, dim=1)
                    
                    # Apply Embeddings to Real Data
                    real_parts = []
                    real_ptr = 0
                    # Need to iterate column info again to slice real data correctly
                    # Real data from transformer is concatenated (Indices, Values...)
                    for info in self.data_column_info:
                        dim = info['input_dim'] # 1 for embedding (index)
                        col_data = real_data_batch[:, real_ptr:real_ptr+dim]
                        real_ptr += dim
                        
                        if info['type'] == 'embedding':
                            emb_vect = self.embedding_layers[info['name']](col_data.long().squeeze(1))
                            real_parts.append(emb_vect)
                        else:
                            real_parts.append(col_data)
                    
                    real_data_processed = torch.cat(real_parts, dim=1)

                    if real_context_batch is not None:
                        fake_input = torch.cat([fake_data_batch, real_context_batch], dim=1)
                        real_input_processed = torch.cat([real_data_processed, real_context_batch], dim=1)
                    else:
                        fake_input = fake_data_batch
                        real_input_processed = real_data_processed

                    # Compute WGAN loss
                    d_real = self.discriminator(real_input_processed)
                    d_fake = self.discriminator(fake_input)
                    
                    # Gradient Penalty
                    gp = compute_gradient_penalty(self.discriminator, real_input_processed, fake_input, self.device)
                    
                    loss_D = -torch.mean(d_real) + torch.mean(d_fake) + 10.0 * gp
                    
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()

                # --- Train Generator ---
                # Train generator once after n_critic discriminator steps
                noise = torch.randn(self.batch_size, self.embedding_dim, device=self.device)
                if real_context_batch is not None:
                    # Re-sample context for generator training?? 
                    # Ideally yes, but reusing batch is fine for conditional stability
                    # We'll stick to reusing the last seen batch for simplicity/stability
                    gen_input = torch.cat([noise, real_context_batch], dim=1)
                else:
                    gen_input = noise
                    
                fake_raw = self.generator(gen_input)
                
                # Apply Embeddings / Softmax (Same logic as above)
                fake_parts = []
                fake_ptr = 0
                for info in self.data_column_info:
                    if info['type'] == 'embedding':
                        dim = info['num_categories']
                        logits = fake_raw[:, fake_ptr:fake_ptr+dim]
                        fake_ptr += dim
                        probs = F.softmax(logits, dim=1)
                        emb_vect = self.embedding_layers[info['name']].forward_soft(probs)
                        fake_parts.append(emb_vect)
                    else:
                        dim = info['output_dim']
                        val = fake_raw[:, fake_ptr:fake_ptr+dim]
                        fake_ptr += dim
                        fake_parts.append(val)
                
                fake_data_batch = torch.cat(fake_parts, dim=1)
                
                if real_context_batch is not None:
                    fake_input = torch.cat([fake_data_batch, real_context_batch], dim=1)
                else:
                    fake_input = fake_data_batch
                    
                d_fake = self.discriminator(fake_input)
                loss_G = -torch.mean(d_fake)
                
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss D={loss_D.item():.4f}, Loss G={loss_G.item():.4f}")

    def sample(self, num_rows: int, context: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.DataFrame:
        """Generate synthetic samples, optionally conditioned on parent context.

        Args:
            num_rows: Number of rows to generate.
            context: Optional parent attributes aligned to the requested rows.
            **kwargs: Additional sampling controls (unused placeholder).

        Returns:
            DataFrame of synthetic rows mapped back to original schema.
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_rows, self.embedding_dim, device=self.device)
            
            if context is not None:
                # Assuming context is provided for exactly num_rows
                assert len(context) == num_rows
                
                # Transform context using the fitted context transformer
                context_transformed = self.context_transformer.transform(context)
                context_data = torch.from_numpy(context_transformed).float().to(self.device)
                
                gen_input = torch.cat([noise, context_data], dim=1)
            else:
                gen_input = noise
                
            fake_raw = self.generator(gen_input)
            
            # Post-process logits to indices for output
            output_parts = []
            fake_ptr = 0
            for info in self.data_column_info:
                if info['type'] == 'embedding':
                    dim = info['num_categories']
                    logits = fake_raw[:, fake_ptr:fake_ptr+dim]
                    fake_ptr += dim
                    
                    # Argmax to get index
                    indices = torch.argmax(logits, dim=1, keepdim=True)
                    output_parts.append(indices.cpu().numpy())
                else:
                    dim = info['output_dim']
                    val = fake_raw[:, fake_ptr:fake_ptr+dim]
                    fake_ptr += dim
                    output_parts.append(val.cpu().numpy())
            
            fake_data_np = np.concatenate(output_parts, axis=1)
            
        return self.transformer.inverse_transform(fake_data_np)
        
    def save(self, path: str) -> None:
        """Persist generator and discriminator state dicts to disk.

        Args:
            path: Filesystem path to write the checkpoint to.
        """
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            # Ideally we pickle the transformer, but for now we assume it's reconstructible or part of the object state
            # A distinct save mechanism for the full object is better (e.g. pickle or joblib)
        }, path)

    def load(self, path: str) -> None:
        """Load generator and discriminator weights from disk.

        Args:
            path: Filesystem path containing a saved checkpoint.
        """
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
