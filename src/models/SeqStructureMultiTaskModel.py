from typing import Any, List, Optional, Union
from pytorch_lightning import LightningModule
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from src.models.components.bert import ProteinBERT
from src.models.components.mpnn import graph_MPNN
from src.models.components.interaction import DilatedResNet2D
from src.models.components.decoder import BertGNNDecoder, BertDecoder
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SeqStructureMultiTaskModel(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        bert_cfg: DictConfig,
        graph_cfg: DictConfig,
        inter_cfg: DictConfig,
        task_cfg: DictConfig,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        ## bert module ##
        self.bert = ProteinBERT(
            num_tokens=self.hparams.bert_cfg.num_tokens,
            dim=self.hparams.bert_cfg.dim,
            depth=self.hparams.bert_cfg.depth,
            narrow_conv_kernel=self.hparams.bert_cfg.narrow_conv_kernel,
            wide_conv_kernel=self.hparams.bert_cfg.wide_conv_kernel,
            wide_conv_dilation=self.hparams.bert_cfg.wide_conv_dilation,
            attn_heads=self.hparams.bert_cfg.attn_heads,
            attn_dim_head=self.hparams.bert_cfg.attn_dim_head,
            local_self_attn=self.hparams.bert_cfg.local_self_attn,
            output_attentions=self.hparams.bert_cfg.output_attentions,
            output_hidden_states=self.hparams.bert_cfg.output_hidden_states
        )

        ## graph module ##
        self.graph_module = graph_MPNN(
            input_node_size=self.hparams.bert_cfg.dim, 
            input_edge_size=self.hparams.bert_cfg.depth * self.hparams.bert_cfg.attn_heads, 
            lm_hidden_size=self.hparams.graph_cfg.lm_hidden_size, 
            node_feat_channels=self.hparams.graph_cfg.node_feat_channels, 
            edge_feat_channels=self.hparams.graph_cfg.edge_feat_channels,
            atten_heads=self.hparams.graph_cfg.atten_heads, 
            dropout_rate=self.hparams.graph_cfg.dropout_rate,
            num_layers=self.hparams.graph_cfg.num_layers
        )

        ## interaction module ##
        self.interact_reduct = nn.Conv2d(self.hparams.graph_cfg.lm_hidden_size*2, 
                                         self.hparams.inter_cfg.init_channels, 
                                         kernel_size=(1, 1), 
                                         padding=(0, 0))
        
        self.dilated_resnet_2d = DilatedResNet2D(self.hparams.inter_cfg.init_channels, 
                                                 self.hparams.inter_cfg.num_bins,
                                                 self.hparams.inter_cfg.dilation_rates, 
                                                 self.hparams.inter_cfg.num_residual_blocks)

        ## prediction head ##
        self.bertGNN_decoder = BertGNNDecoder(self.hparams.graph_cfg.lm_hidden_size, 
                                              self.hparams.task_cfg.num_classes_aa, 
                                              self.hparams.task_cfg.num_classes_statics, 
                                              self.hparams.task_cfg.num_classes_effect,
                                              self.hparams.task_cfg.num_bins)
        self.bert_decoder = BertDecoder(self.hparams.bert_cfg.dim, 
                                        self.hparams.task_cfg.num_classes_aa, 
                                        self.hparams.task_cfg.num_classes_statics, 
                                        self.hparams.task_cfg.num_classes_effect)

        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # metrics
        loss_types = ["loss", "aa_loss", "statics_loss", "effect_loss", "dist_loss"]

        self.train_metrics = {f"train_{loss_type}": torchmetrics.MeanMetric().to(self.device) for loss_type in loss_types}
        self.val_metrics = {f"val_{loss_type}": torchmetrics.MeanMetric().to(self.device) for loss_type in loss_types}
        self.test_metrics = {f"test_{loss_type}": torchmetrics.MeanMetric().to(self.device) for loss_type in loss_types}


    def construct_interaction_tensor(self, node_feat: torch.Tensor):
        new_node_feat = node_feat.permute(0,2,1)
        seq_pad_len = node_feat.shape[1]
        # interact_tensor size [batch_size,2*hidden_dim,seq_length,seq_length]
        interact_tensor = torch.cat((torch.repeat_interleave(new_node_feat.unsqueeze(3), repeats=seq_pad_len, dim=3),
                                    torch.repeat_interleave(new_node_feat.unsqueeze(2), repeats=seq_pad_len, dim=2)), dim=1)
        interact_tensor_t = interact_tensor.transpose(2,3) # sharing its underlying storage with the input tensor (since input is strided tensor)
        triu_idx_i,triu_idx_j =  torch.triu_indices(seq_pad_len, seq_pad_len, 1)
        interact_tensor_t[:,:,triu_idx_i,triu_idx_j] = interact_tensor[:,:,triu_idx_i,triu_idx_j] # interact_tensor is also changed
        interact_tensor.to(node_feat.device)
        #assert (interact_tensor.transpose(2,3) == interact_tensor).all() == True
        return interact_tensor

        
    def forward(self, batch: Any):
        bert_outputs = self.bert(seq=batch.seq, mask=batch.mask)
        bert_mut_outputs = self.bert(seq=batch.mut_seq)
        residue_embeddings, bert_attentions = bert_outputs[0], bert_outputs[-1]
        residue_mut_embeddings, bert_mut_attentions = bert_mut_outputs[0], bert_mut_outputs[-1]

        loss = 0.

        if hasattr(batch, 'edge_index'):
            batch_node_feats = []
            batch_mut_node_feats = []
            
            for i in range(batch.seq.shape[0]):
                node_feats = self.graph_module(residue_embeddings[i],  
                                               batch.edge_index[:, batch.batch_edge_labels == i], 
                                               bert_attentions[:, i, ...],
                                               batch.align_mask[i])
                mut_node_feats = self.graph_module(residue_mut_embeddings[i],  
                                                   batch.edge_index[:, batch.batch_edge_labels == i], 
                                                   bert_mut_attentions[:, i, ...],
                                                   batch.align_mask[i])
                
                batch_node_feats.append(node_feats)
                batch_mut_node_feats.append(mut_node_feats)
            
            batch_node_feats = torch.stack(batch_node_feats, dim=0)
            batch_mut_node_feats = torch.stack(batch_mut_node_feats, dim=0)

            interact_tensor = self.construct_interaction_tensor(batch_node_feats)
            interact_tensor_reduct = self.interact_reduct(interact_tensor)
            dilat_res2d_output = self.dilated_resnet_2d(interact_tensor_reduct)
            
            aa_out, statics_out, effect_out, dist_out, aa_loss, statics_loss, effect_loss, dist_loss = \
            self.bertGNN_decoder(batch_node_feats, batch_mut_node_feats, dilat_res2d_output, batch)

        else:
            aa_out, statics_out, effect_out, aa_loss, statics_loss, effect_loss = \
            self.bert_decoder(residue_embeddings, residue_mut_embeddings, batch)

        losses = {
        "aa_loss": aa_loss,
        "statics_loss": statics_loss,
        "effect_loss": effect_loss,
        }

        outputs = {
        "aa_out": aa_out,
        "statics_out": statics_out,
        "effect_out": effect_out,
        }
    
        loss += self.hparams.task_cfg.aa_loss_weight * aa_loss + \
                      self.hparams.task_cfg.statics_loss_weight * statics_loss + \
                      self.hparams.task_cfg.effect_loss_weight * effect_loss
    
        if hasattr(batch, 'edge_index'):
            losses["dist_loss"] = dist_loss
            outputs["dist_out"] = dist_out
            loss += self.hparams.task_cfg.dist_loss_weight * dist_loss
    
        return loss, losses, outputs


    def step(self, batch: Any):
        loss, losses, _ = self.forward(batch)
        return loss, losses

    def on_train_start(self):
        self._reset_metrics(self.train_metrics)

    def _reset_metrics(self, metrics_dict):
        for metric in metrics_dict.values():
            metric.reset()
    
    def _update_metrics(self, metrics_dict, losses, phase):
        for loss_name, loss_value in losses.items():
            metric_name = f"{phase}_{loss_name}"
            if metric_name in metrics_dict:
                metrics_dict[metric_name] = metrics_dict[metric_name].to(self.device)
                metrics_dict[metric_name].update(loss_value.detach())
    
    def _log_metrics(self, metrics_dict, phase):
        for metric_name, metric in metrics_dict.items():
            if metric_name.startswith(phase):
                if metric._update_count == 0:  
                    continue
                self.log(metric_name, metric.compute(), sync_dist=True)

    def training_step(self, batch: Any, batch_idx: int):
        try:
            loss, losses = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                log.info(f"Skipping training batch with index {batch_idx} due to OOM error...")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
            else:
                raise e
    
        if loss.isnan().any() or loss.isinf().any():
            log.info(f"Loss for batch with index {batch_idx} is invalid. Skipping...")
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
    
        self._update_metrics(self.train_metrics, losses, phase="train")
        self.train_metrics["train_loss"].to(self.device).update(loss.detach())
    
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        # log metric(s)
        self._log_metrics(self.train_metrics, phase="train")
    
    def validation_step(self, batch: Any, batch_idx: int):
        try:
            loss, losses = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=False)}
            else:
                raise e
    
        self._update_metrics(self.val_metrics, losses, phase="val")
        self.val_metrics["val_loss"].to(self.device).update(loss.detach())
    
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        # log metric(s)
        self._log_metrics(self.val_metrics, phase="val")
    
    def test_step(self, batch: Any, batch_idx: int):
        try:
            loss, losses = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                log.info(f"Skipping test batch with index {batch_idx} due to OOM error...")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=False)}
            else:
                raise e
    
        # update metric(s)
        self._update_metrics(self.test_metrics, losses, phase="test")
        self.test_metrics["test_loss"].update(loss.detach())
    
        return {"loss": loss}


    def on_test_epoch_end(self):
        # log metric(s)
        self._log_metrics(self.test_metrics, phase="test")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
