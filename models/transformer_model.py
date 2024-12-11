# transformer_model.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Transformer Encoder层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出特征维度，可以根据需要调整
        self.feature_dim = hidden_dim
        
    def forward(self, x):
        """
        x: 输入数据，形状为 (batch_num, node_num, feature_dim)
        """
        # 加入位置编码（可选）
        # 如果需要，可以在这里添加位置编码
        
        # Transformer需要的输入格式是 (seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)  # (batch_num, node_num, feature_dim) -> (node_num, batch_num, feature_dim)
        
        # 通过Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (node_num, batch_num, feature_dim)
        
        # 选择最后一个时间步的输出，或使用池化
        # 这里使用池化（例如平均池化）
        transformer_features = transformer_out.mean(dim=0)  # (batch_num, feature_dim)
        
        return transformer_features  # 返回 (batch_num, feature_dim)
