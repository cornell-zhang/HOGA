import torch
import torch.nn as nn

class MultiheadAtt(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAtt, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        print("use our att model!!!!")

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape the projected queries, keys, and values
        query = query.view(batch_size * self.num_heads, -1, self.head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.head_dim)

        # Compute the scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply the mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the output of the attention heads
        attention_output = torch.bmm(attention_probs, value)

        # Reshape and project the output of the attention heads
        attention_output = attention_output.view(batch_size, -1, self.input_dim)
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs
