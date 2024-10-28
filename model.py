import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class AttentionDTI(nn.Module):
    def __init__(self, hp, protein_MAX_LENGH=1000, drug_MAX_LENGH=100):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        
        # Load ProtBERT for protein embeddings
        self.protein_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        self.protein_model = AutoModel.from_pretrained("Rostlab/prot_bert")
        
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        
        # Calculate output sizes for CNNs
        drug_conv_out = drug_MAX_LENGH - sum(k - 1 for k in self.drug_kernel)
        protein_conv_out = protein_MAX_LENGH - sum(k - 1 for k in self.protein_kernel)
        
        # Define CNN layers
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(drug_conv_out)
        
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_model.config.hidden_size, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(protein_conv_out)
        
        # Calculate the total size of flattened features after pooling
        total_conv_out = (self.conv * 4) * 2  # Multiply by 2 because we concatenate drug and protein features
        
        # Update fully connected layers with correct input size
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(total_conv_out, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        device = next(self.parameters()).device
        
        # Process drug embeddings
        drugembed = self.drug_embed(drug.to(device))
        drugembed = drugembed.permute(0, 2, 1)  # [batch_size, dim, drug_length]
        
        # Process protein embeddings batch-wise
        batch_size = len(protein)
        protein_hidden_size = self.protein_model.config.hidden_size
        combined_embeddings = torch.zeros((batch_size, protein_hidden_size, self.protein_MAX_LENGH)).to(device)
        
        for i, seq in enumerate(protein):
            seq = str(seq)
            first_half_len = self.protein_MAX_LENGH // 2
            
            tokens_part1 = self.protein_tokenizer(seq[:first_half_len], 
                                                  return_tensors="pt", 
                                                  padding='max_length',
                                                  max_length=first_half_len,
                                                  truncation=True).to(device)
            
            tokens_part2 = self.protein_tokenizer(seq[first_half_len:self.protein_MAX_LENGH], 
                                                  return_tensors="pt", 
                                                  padding='max_length',
                                                  max_length=first_half_len,
                                                  truncation=True).to(device)
            
            with torch.no_grad():
                part1_embedding = self.protein_model(**tokens_part1).last_hidden_state
                part2_embedding = self.protein_model(**tokens_part2).last_hidden_state
                
                part1_embedding = part1_embedding.squeeze(0).transpose(0, 1)
                part2_embedding = part2_embedding.squeeze(0).transpose(0, 1)
                
                combined_embedding = torch.cat([part1_embedding, part2_embedding], dim=1)
                combined_embeddings[i] = combined_embedding
        
        proteinembed = combined_embeddings
        
        # Print shapes for debugging
        # print(f"Drug embed shape: {drugembed.shape}")
        # print(f"Protein embed shape: {proteinembed.shape}")
        
        # Apply CNNs
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        
        # print(f"Drug conv shape before pool: {drugConv.shape}")
        # print(f"Protein conv shape before pool: {proteinConv.shape}")
        
        # Apply max pooling
        drugConv = self.Drug_max_pool(drugConv)
        proteinConv = self.Protein_max_pool(proteinConv)
        
        # print(f"Drug conv shape after pool: {drugConv.shape}")
        # print(f"Protein conv shape after pool: {proteinConv.shape}")
        
        # Flatten both tensors to 2D consistently
        drugConv = drugConv.view(batch_size, -1)  # [batch_size, conv*4]
        proteinConv = proteinConv.view(batch_size, -1)  # [batch_size, conv*4]
        
        # print(f"Drug conv shape after flatten: {drugConv.shape}")
        # print(f"Protein conv shape after flatten: {proteinConv.shape}")
        
        # Concatenate the flattened tensors
        pair = torch.cat([drugConv, proteinConv], dim=1)
        
        # print(f"Pair shape after concat: {pair.shape}")
        
        # Process through fully connected layers
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        
        return predict
