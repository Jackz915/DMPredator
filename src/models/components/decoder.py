import torch
import torch.nn as nn
import torch.nn.functional as F

class BertGNNDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes_aa, num_classes_statics, num_classes_effect, num_bins):
        super(BertGNNDecoder, self).__init__()
        
        self.aa_decoder = nn.Linear(hidden_size, num_classes_aa)  
        self.statics_decoder = nn.Linear(hidden_size, num_classes_statics)  
        self.effect_decoder = nn.Linear(hidden_size, num_classes_effect)  
        self.num_bins = num_bins
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch_node_feats, batch_mut_node_feats, dilat_res2d_output, batch):
        aa_out = self.aa_decoder(batch_node_feats) 
        statics_out = self.statics_decoder(batch_node_feats) 
        effect_out = self.effect_decoder((batch_mut_node_feats - batch_node_feats).max(dim=1)[0])  
        
        target_seq = torch.argmax(batch.seq, dim=-1)  
        aa_loss = self.cross_entropy_loss(aa_out.view(-1, aa_out.size(-1)), target_seq.view(-1)) 
        statics_loss = self.mse_loss(statics_out, batch.target_static)  
        effect_loss = self.cross_entropy_loss(effect_out, batch.target_label.long())  

        edge_index = batch.edge_index  
        edge_attr = batch.edge_attr
        batch_edge_labels = batch.batch_edge_labels  

        if edge_attr.numel() == 0:  
            dist_loss = torch.zeros(1, device=batch_node_feats.device, requires_grad=True)
            return aa_out, statics_out, effect_out, dilat_res2d_output, aa_loss, statics_loss, effect_loss, dist_loss
            
        else:
            predicted_distances = []
        
            for i in range(dilat_res2d_output.size(0)):  
                current_batch_edge_index = edge_index[:, batch_edge_labels == i]  
                pred_dist = dilat_res2d_output[i, :, current_batch_edge_index[0], current_batch_edge_index[1]]  
                predicted_distances.append(pred_dist)
            
            predicted_distances = torch.cat(predicted_distances, dim=1).view(-1, self.num_bins)
            
            dist_loss = self.cross_entropy_loss(
                predicted_distances,  
                edge_attr.squeeze().long()  
            )

        return aa_out, statics_out, effect_out, dilat_res2d_output, aa_loss, statics_loss, effect_loss, dist_loss

        
        
class BertDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes_aa, num_classes_statics, num_classes_effect):
        super(BertDecoder, self).__init__()
        
        self.aa_decoder = nn.Linear(hidden_size, num_classes_aa) 
        self.statics_decoder = nn.Linear(hidden_size, num_classes_statics) 
        self.effect_decoder = nn.Linear(hidden_size, num_classes_effect) 

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, residue_embeddings, residue_mut_embeddings, batch):
        aa_out = self.aa_decoder(residue_embeddings)
        statics_out = self.statics_decoder(residue_embeddings)
        effect_out = self.effect_decoder((residue_mut_embeddings - residue_embeddings).max(dim=1)[0]) # mean(dim=1) max(dim=1)[0]

        target_seq = torch.argmax(batch.seq, dim=-1)
        aa_loss = self.cross_entropy_loss(aa_out.view(-1, aa_out.size(-1)), target_seq.view(-1))
        statics_loss = self.mse_loss(statics_out, batch.target_static)
        effect_loss = self.cross_entropy_loss(effect_out, batch.target_label)
        
        return aa_out, statics_out, effect_out, aa_loss, statics_loss, effect_loss
