import torch
import torch.nn as nn
import pytorch_lightning as pl

class LightningWrapper(pl.LightningModule):
    def __init__(self, window_size=4, num_features=5, model=None, player_feature_names=["total_points", "ict_index", "clean_sheets", "saves", "assists"], lr=1e-3, weight_decay=0):
        super().__init__()
        self.window_size = window_size
        self.dim = window_size * num_features
        self.feature_string = ','.join(player_feature_names)
        self.model = model
        self.lr = lr 
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        inputs, outputs = batch
        predictions = self.model.forward(inputs)
        loss = nn.MSELoss()(predictions, outputs)
        self.log("train_loss", loss)
        #self.log(f"features : {self.feature_string} model : {self.model_type} train_loss", loss)
        #self.logger.experiment.add_scalars('1',{f'{self.feature_string} train':loss})
        return loss 

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        #self.log(f"features : {self.feature_string} model : {self.model_type} val_loss", loss)
        #self.logger.experiment.add_scalars('1',{f'{self.feature_string} val':loss})

    def configure_optimizers(self):
        if len(list(self.parameters())) > 0:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            return optimizer
        return None