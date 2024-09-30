import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def forward_linear_with_scale_and_bias(x, module, scale=None, bias=None):
    if scale is not None:
        x = x * scale
    x = module(x)
    if bias is not None:
        x = x + bias
    return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(torch.float32))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    

class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs)
        output = torch.max(trajs, dim=-2).values

        return output


class GMMPredictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMPredictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return traj, score


class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.llm_adapt_attention = AdaptiveBlock()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None, llm_feature=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.llm_adapt_attention(inputs, llm_feature, attention_output)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output
    
    
class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.llm_adapt_attention = AdaptiveBlock()
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None, llm_feature=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        # W_q = self.cross_attention.in_proj_weight.chunk(3)
        # b_q, _, _ = self.cross_attention.in_proj_bias.chunk(3)
        # projected_q = torch.matmul(query, W_q) + b_q
        if llm_feature is None:
            import pdb; pdb.set_trace()
            return attention_output
        if (attention_output != self.llm_adapt_attention(query, llm_feature, attention_output)).any():
            print('!!!!!!!!!!!!!!!!adapter has been updated!!!!!!!!!!!!!!!!')
        attention_output = self.llm_adapt_attention(query, llm_feature, attention_output)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output
    
class AdaptiveBlock(nn.Module):
    def __init__(self, heads=8, dim=256):
        super().__init__()
        self.head_dim = dim // heads
        self.gate = torch.nn.Parameter(torch.zeros(1, heads, 1, 1, device='cuda'))
        self.n_local_heads = heads
        # self.n_local_heads = heads // int(os.environ["WORLD_SIZE"])
        # self.head_start = self.n_local_heads * int(os.environ["LOCAL_RANK"])
        # self.head_end = self.n_local_heads * (int(os.environ["LOCAL_RANK"]) + 1)
        
        self.wq = nn.Linear(dim, heads * self.head_dim)
        self.wk = nn.Linear(dim, heads * self.head_dim)
        self.wv = nn.Linear(dim, heads * self.head_dim)
        self.wo = nn.Linear(heads * self.head_dim, dim)
        
    def forward(self, query, llm_feature, output):
        bsz, adapter_len = llm_feature.shape[0], llm_feature.shape[1]
        seqlen = output.shape[1]
        
        projected_query = self.wq(query)
        projected_query = projected_query.view(bsz, -1, self.n_local_heads, self.head_dim)
        adapter_k = self.wk(llm_feature)
        adapter_k = adapter_k.view(bsz, adapter_len, self.n_local_heads, self.head_dim)
        adapter_v = self.wv(llm_feature)
        adapter_v = adapter_v.view(bsz, adapter_len, self.n_local_heads, self.head_dim)
        projected_query = projected_query.transpose(1, 2)
        adapter_k = adapter_k.transpose(1, 2)
        adapter_v = adapter_v.transpose(1, 2)
        
        # adaptive_attention = self.gate[
        #         :, self.head_start : self.head_end
        #     ].tanh().half() * self._forward_scaled_dot_product_attention(projected_query, adapter_k, adapter_v)
        adaptive_attention = self.gate.tanh().half() * self._forward_scaled_dot_product_attention(projected_query, adapter_k, adapter_v)
        adaptive_attention = adaptive_attention.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # print((output == output+adaptive_attention).all())
        output += adaptive_attention
        
        # return self.wo(output)
        return output
    
    def _forward_scaled_dot_product_attention(self, q, k, v, mask=None):
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, mask >= 0 if mask is not None else None)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, v)
            return output


class InitialPredictionDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256):
        super(InitialPredictionDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())

    def forward(self, current_states, encoding, mask, llm_feature):
        N = self._agents
        multi_modal_query = self.multi_modal_query_embedding(self.modal) #[6, 256]
        query = encoding[:, :N, None, :] + multi_modal_query[None, None, :, :] #[16, 11, 6, 256] [B,num_agent,modalities,ch]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask, llm_feature) for i in range(N)], dim=1)
        predictions, scores = self.predictor(query_content) #[16, 11, 6, 80, 4], ([16, 11, 6]
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, modalities, future_encoder):
        super(InteractionDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

    def forward(self, current_states, actors, scores, last_content, encoding, mask, llm_feature):
        # current_states, last_predictions, last_scores, last_query_content, encoding, mask
        N = actors.shape[1]
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N]) #[16, 11, 6, 256]
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2) #[16, 11, 256]
        interaction = self.interaction_encoder(futures, mask[:, :N], llm_feature)
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1)

        query = last_content + multi_futures #[16, 11, 6, 256]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask, llm_feature) for i in range(N)], dim=1)
        trajectories, scores = self.decoder(query_content)
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores