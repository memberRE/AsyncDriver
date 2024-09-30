import torch
try:
    from .predictor_modules_adapter import *
except:
    from predictor_modules_adapter import *


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        scene = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        encoding = self.fusion_encoder(scene, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes']
        }

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs, llm_feature=None):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1] # 1 21(ego+agents) 5(xyzvxvy)
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']
        # if llm_feature is not None:
        #     mask = torch.cat([mask, torch.zeros_like(llm_feature[...,0])], dim=1)
        #     encoding = torch.cat([encoding, llm_feature], dim=1)

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask, llm_feature)
        # query_content, predictions, scores
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask, llm_feature)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
        
        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding


class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_encoder = VectorMapEncoder(3, 50)
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))

    def forward(self, env_encoding, route_lanes, llm_feature):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        env_encoding = torch.max(env_encoding, dim=1, keepdim=True)[0]
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask, llm_feature)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        plan = self.plan_decoder(env_route_encoding.squeeze(1))
        plan = plan.reshape(plan.shape[0], self._future_len, 2)

        return plan


class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10): # 3 2 6 20
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes)

        return decoder_outputs, ego_plan

class LLMEnhancedGameFormer_Adapter(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10, share_encoder=None):
        super(LLMEnhancedGameFormer_Adapter, self).__init__()
        self.neighbors = neighbors
        self.share_encoder = share_encoder
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward_not_share_encoder(self, inputs, llm_feature):
        # TODO: add fusion module
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs, llm_feature=llm_feature)
        ego_plan = self.planner(env_encoding, route_lanes, llm_feature)

        return decoder_outputs, ego_plan
    
    def forward_share_encoder(self, encoder_outputs, llm_feature):
        # TODO: add fusion module
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs, llm_feature=llm_feature)
        ego_plan = self.planner(env_encoding, route_lanes, llm_feature)

        return decoder_outputs, ego_plan
    
    def forward(self, input):
        if self.share_encoder:
            encoder_outputs, llm_feature = input
            return self.forward_share_encoder(encoder_outputs, llm_feature)
        else:
            raw_inputs, llm_feature = input
            return self.forward_not_share_encoder(raw_inputs, llm_feature)