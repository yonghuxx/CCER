import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast  
from transformers import LlamaTokenizer
from .modeling_explainer import LlamaForCausalLM
import numpy as np
class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps=8, layers=[128, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps 
        self.noisy_gating = noise
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) 
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] 
        expert_outputs = torch.cat(expert_outputs, dim=-2) 
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)
        
class Explainer(torch.nn.Module):
    def __init__(self, token_size=4096, user_embed_size=128, item_embed_size=128):
        super(Explainer, self).__init__()
        from huggingface_hub import login
        model_name = "/root/autodl-tmp/Llama-2-7b-chat-hf"
        self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.attn_weights = nn.Parameter(torch.ones(5))
        self.T=5
        self.predictlayer = MLP_predict(token_size,1)
        self.attention = Attention_liner(token_size) 
        special_tokens_dict = {"additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token = "<pad>"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.MLPZ=MLPZ(token_size,token_size)
        for param in self.model.parameters():
            param.requires_grad = False
        self.user_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[user_embed_size, token_size], dropout=0.2,
                                                        noise=True)
        self.item_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[item_embed_size, token_size], dropout=0.2,
                                                        noise=True)
        self.user_time_attention = nn.Linear(token_size, self.T)  
        self.item_time_attention = nn.Linear(token_size, self.T)      
        nn.init.xavier_uniform_(self.user_time_attention.weight)
        nn.init.constant_(self.user_time_attention.bias, 0)
        nn.init.xavier_uniform_(self.item_time_attention.weight)
        nn.init.constant_(self.item_time_attention.bias, 0)        
    def process_text(self, text, last_embedding,device):
        result=[]
        alignment_losses=[]
        for i in range(len(text)):
            if not text[i] or text[i].strip() == "" or text[i].lower() == "nan" or text[i].lower() == "null":
                result.append(last_embedding[i])
                continue
            tokenized_inputs = self.tokenizer(text[i], padding=True, return_tensors="pt")
            embeddings = self.model.get_input_embeddings()(tokenized_inputs['input_ids'])
            pooled_embedding = torch.mean(embeddings, dim=1) 
            pooled_embedding = pooled_embedding.squeeze(0)
            alignment_loss = F.mse_loss(last_embedding[i], pooled_embedding) 
            alignment_losses.append(alignment_loss)
            combined_embedding = torch.stack([last_embedding[i], pooled_embedding], dim=0) 
            attention_scores = self.attention(combined_embedding)
            attn_weights = F.softmax(attention_scores, dim=0)
            fused_embedding = torch.sum(attn_weights * combined_embedding, dim=0)
            output=self.MLPZ(fused_embedding)
            result.append(output)
        return torch.stack(result,dim=0)
    def forward(self, user_embedding, item_embedding,user_text_T,item_text_T, input_text,device):
        zu_T,zi_T=[],[]
        zu_last = self.user_embedding_converter(user_embedding).half()
        zi_last = self.item_embedding_converter(item_embedding).half()
        user_text_T=np.array(user_text_T).T.tolist()
        item_text_T=np.array(item_text_T).T.tolist()
        for t in range(5):
            zu_t = self.process_text(user_text_T[t], zu_last,device)
            zu_T.append(zu_t)
            zu_last = zu_t 
            zi_t = self.process_text(item_text_T[t], zi_last,device)
            zi_T.append(zi_t)
            zi_last = zi_t
        zu_T = torch.stack(zu_T, dim=0)
        zi_T = torch.stack(zi_T, dim=0)  
        with autocast():
            user_time_weights = F.softmax(
                self.user_time_attention(zu_last), dim=-1
            )          
        zu_T_batch_first = zu_T.permute(1, 0, 2)
        weighted_zu = torch.sum(
            user_time_weights.unsqueeze(-1) * zu_T_batch_first, dim=1
        )
        with autocast():
            item_time_weights = F.softmax(
                self.item_time_attention(zi_last), dim=-1 
            )         
        zi_T_batch_first = zi_T.permute(1, 0, 2) 
        weighted_zi = torch.sum(
            item_time_weights.unsqueeze(-1) * zi_T_batch_first, dim=1
        )
        weighted_zu = weighted_zu.squeeze()
        weighted_zi = weighted_zi.squeeze()
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs['input_ids'])
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")
        user_embed_position = (tokenized_inputs['input_ids'] == user_embed_token_id).nonzero()[:, 1:]
        item_embed_position = (tokenized_inputs['input_ids'] == item_embed_token_id).nonzero()[:, 1:]
        explain_pos_position = (tokenized_inputs['input_ids'] == explain_pos_token_id).nonzero()[:, 1:]
        weighted_zu=weighted_zu.half()
        weighted_zi=weighted_zi.half()ize
        inputs_embeds[torch.arange(user_embed_position.shape[0]), user_embed_position[:, 0],
        :] = weighted_zu
        inputs_embeds[torch.arange(item_embed_position.shape[0]), item_embed_position[:, 0],
        :] = weighted_zi
        outputs = self.model(inputs_embeds=inputs_embeds, user_embed=weighted_zu,
                                 item_embed=weighted_zi, user_embed_pos=user_embed_position,
                                 item_embed_pos=item_embed_position)
        return tokenized_inputs['input_ids'], outputs, explain_pos_position.flatten()
    def loss(self, input_ids, outputs, explain_pos_position, device,lambda_reg=1e-5):
        interval = torch.arange(input_ids.shape[1]).to(device)
        mask = interval[None, :] < explain_pos_position[:, None]
        input_ids[mask] = -100
        logits = outputs.logits
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        main_loss = nn.CrossEntropyLoss()(shift_logits, shift_labels)
        return main_loss

    def generate(self, user_embedding, item_embedding,user_text_T,item_text_T, input_text,device):
        zu_T,zi_T=[],[]
        zu_last = self.user_embedding_converter(user_embedding).half()
        zi_last = self.item_embedding_converter(item_embedding).half()
        user_text_T=np.array(user_text_T).T.tolist()
        item_text_T=np.array(item_text_T).T.tolist()
        for t in range(5):
            zu_t = self.process_text(user_text_T[t], zu_last,device)
            zu_T.append(zu_t)
            zu_last = zu_t
            zi_t = self.process_text(item_text_T[t], zi_last,device)
            zi_T.append(zi_t)
            zi_last = zi_t 
        zu_T = torch.stack(zu_T, dim=0) 
        zi_T = torch.stack(zi_T, dim=0)
        with autocast():
            user_time_weights = F.softmax(
                self.user_time_attention(zu_last), dim=-1
            )          
        zu_T_batch_first = zu_T.permute(1, 0, 2)
        weighted_zu = torch.sum(
            user_time_weights.unsqueeze(-1) * zu_T_batch_first, dim=1
        )
        with autocast():
            item_time_weights = F.softmax(
                self.item_time_attention(zi_last), dim=-1 
            )         
        zi_T_batch_first = zi_T.permute(1, 0, 2) 
        weighted_zi = torch.sum(
            item_time_weights.unsqueeze(-1) * zi_T_batch_first, dim=1
        )
        weighted_zu = weighted_zu.squeeze()
        weighted_zi = weighted_zi.squeeze()
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs['input_ids'])
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")
        user_embed_position = (tokenized_inputs['input_ids'] == user_embed_token_id).nonzero()[:, 1:]
        item_embed_position = (tokenized_inputs['input_ids'] == item_embed_token_id).nonzero()[:, 1:]
        explain_pos_position = (tokenized_inputs['input_ids'] == explain_pos_token_id).nonzero()[:, 1:]
        weighted_zu=weighted_zu.half()
        weighted_zi=weighted_zi.half()
        inputs_embeds[torch.arange(user_embed_position.shape[0]), user_embed_position[:, 0],
        :] = weighted_zu
        inputs_embeds[torch.arange(item_embed_position.shape[0]), item_embed_position[:, 0],
        :] = weighted_zi
        outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=128,
                                      user_embed=weighted_zu, item_embed=weighted_zi,
                                      user_embed_pos=user_embed_position, item_embed_pos=item_embed_position)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

class MLPZ(nn.Module):
    def __init__(self, embedding_dim, output_dim,drop_ratio=0.2):
        super(MLPZ, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(2048, output_dim),
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0) 
    def forward(self, x):
        with autocast():
            Z = self.linear(x)
        return Z
class MLP_predict(nn.Module):
    def __init__(self, embedding_dim, output_dim,drop_ratio=0.2):
        super(MLP_predict, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1)
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0) 
    def forward(self, x):
        with autocast():
            pred = self.linear(x)
        return pred

class Attention_liner(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.2):
        super(Attention_liner, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(256, 1), 
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        with autocast():
            attention_scores = self.linear(x)
        return attention_scores
class PredictLayer(nn.Module):
    def __init__(self, dims, drop_ratio=0.2):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(dims[2], dims[3]),
        )
    def forward(self, x):
        with autocast():
            out = self.linear(x)
        return out