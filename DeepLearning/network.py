import sys
import torch
import pytorch_lightning as pl
from torchir.metrics import NMI
from torchir.networks import RigidIRNet
from torchir.transformers import AffineTransformer

class LitAIRNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.airnet = RigidIRNet(ndim=3)  # Your registration network
        self.global_transformer = AffineTransformer(ndim=3)  # Affine transformer
        self.metric = NMI()  # Use NMI as the metric for registration
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = torch.optim.Adam(self.airnet.parameters(), lr=lr, amsgrad=True)
        return optimizer

    def forward(self, fixed, moving):
        # Get rotation and translation from the model
        rotation, translation = self.airnet(fixed, moving)

        print(f"rotation shape in forward: {rotation.shape}")

        # Create identity scale and shear
        batch_size = rotation.shape[0]
        scale = torch.ones((batch_size, 3), device=rotation.device)  # Identity scale
        shear = torch.zeros((batch_size, 3), device=rotation.device)  # Identity shear

        # Pass all four parameters (rotation, translation, scale, shear) to the transformer
        parameters = (translation, rotation, scale, shear)
        warped = self.global_transformer(parameters, fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        # Get the fixed and moving images from the batch
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)  # Compute NMI loss
        self.log('NMI/training', loss)  # Log the training loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Apply the forward pass for validation
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)  # Compute NMI loss for validation
        self.log('NMI/validation', loss)  # Log the validation loss
        return loss
