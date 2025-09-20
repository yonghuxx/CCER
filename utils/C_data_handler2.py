import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List, Any

class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):

        return self.input_text[idx]

class DataHandler:
    def __init__(self):
        if (args.dataset == "yelp" ):
            self.system_prompt = "You will serve as an explainer assistant. Follow these steps to analyze the user's experience with the establishment:Firstly, Identify Preferences:Infer the user's general likes and dislikes in establishments.Secondly,Evaluate establishment: Assess the establishment's key strengths and weaknesses.Thirdly, Match Attributes: Compare the user's preferences with the establishment's attributes.Finally,Summarize Experience: In under 50 words, explain the user's enjoyment and dissatisfaction based on the alignment between their preferences and the establishment's features. no special characters, concise."
            self.item = "establishments"
        else:
            self.system_prompt="Explain why the user would buy with the movie within 50 words."
            self.system_prompt="You will serve as an explainer assistant. Follow these steps to analyze the user's experience with the establishment:Firstly, Identify Preferences:Infer the user's general likes and dislikes in movies.Secondly,Evaluate movie: Assess the movie's key strengths and weaknesses.Thirdly, Match Attributes: Compare the user's preferences with the movie's attributes.Finally,Summarize Experience: In under 50 words, explain the user's enjoyment and dissatisfaction based on the alignment between their preferences and the movie's features. no special characters, concise."
            self.item = "movie"

        embed_path=f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_with_embeddings1.pth"
        checkpoint=torch.load(embed_path)
        self.user_emb=checkpoint['weighted_u']
        self.item_emb=checkpoint['weighted_i']
        self.user_text_T_dict=self.load_user_text(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/trn_u_text.csv")
        self.item_text_T_dict=self.load_item_text(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/trn_i_text.csv")

    def load_user_text(self,file_path):
        text_dict={}
        df=pd.read_csv(file_path)
        conclusion_columns=[col for col in df.columns if col.startswith('conclusion')]

        for _,row in df.iterrows():
            id=row['user_id']
            text_T=[]
            for col in conclusion_columns:
                text_T.append(row[col])
            text_dict[id]=text_T
        return text_dict

    def load_item_text(self,file_path):
        text_dict={}
        df=pd.read_csv(file_path)
        conclusion_columns=[col for col in df.columns if col.startswith('conclusion')]

        for _,row in df.iterrows():
            id=row['item_id']
            text_T=[]
            for col in conclusion_columns:
                text_T.append(row[col])
            text_dict[id]=text_T
        return text_dict

    def load_data(self):
        # load data from data_loaders in data
        with open(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/train.pkl", "rb") as file:
            trn_dict = pickle.load(file)
        with open(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/valid.pkl", "rb") as file:
            val_dict = pickle.load(file)
        with open(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/test.pkl", "rb") as file:
            tst_dict = pickle.load(file)

        trn_input = []
        val_input = []
        tst_input = []
        for i in range(len(trn_dict["uid"])):
            if trn_dict["rating"][i]>=3:
                user_message = f"The user has a positive experience with the establishment . user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS> {trn_dict['explanation'][i]}"
            else:
                user_message = f"The user has a negative experience with the establishment. user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS> {trn_dict['explanation'][i]}"
            trn_input.append(
                (
                    self.user_emb[trn_dict["uid"][i]],
                    self.item_emb[trn_dict["iid"][i]],
                    trn_dict["rating"][i],
                    self.user_text_T_dict[trn_dict["uid"][i]],
                    self.item_text_T_dict[trn_dict["iid"][i]],
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                )
            )
        for i in range(len(val_dict["uid"])):
            if val_dict["rating"][i]>=3:
                user_message = f"Rating:{tst_dict["rating"][i]} The user has a positive experience with establishment . user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS>"
            else:
                user_message = f"Rating:{tst_dict["rating"][i]} The user has a negative experience with the establishment . user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS>"        
            user_emb = self.user_emb[val_dict["uid"][i]] if val_dict["uid"][i] < len(self.user_emb) else None
            item_emb = self.item_emb[val_dict["iid"][i]] if val_dict["iid"][i] < len(self.item_emb) else None
            user_text = self.user_text_T_dict.get(val_dict["uid"][i], None)
            item_text = self.item_text_T_dict.get(val_dict["iid"][i], None)
            if user_emb is None or item_emb is None or user_text is None or item_text is None:
                continue
            val_input.append(
                (
                    user_emb,
                    item_emb,
                    val_dict['rating'][i],
                    user_text,
                    item_text,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    val_dict['explanation'][i],
                )
            )

        for i in range(len(tst_dict["uid"])):
            if tst_dict["rating"][i]>=3:
                user_message = f"Rating:{tst_dict["rating"][i]} The user has a positive experience with the establishment . user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS>"
            else:
                user_message = f"Rating:{tst_dict["rating"][i]} The user has a negative experience with the establishment . user prefrences: <USER_EMBED> {self.item} attributes: <ITEM_EMBED>  <EXPLAIN_POS>"
            user_emb = self.user_emb[tst_dict["uid"][i]] if tst_dict["uid"][i] < len(self.user_emb) else None
            item_emb = self.item_emb[tst_dict["iid"][i]] if tst_dict["iid"][i] < len(self.item_emb) else None
            user_text = self.user_text_T_dict.get(tst_dict["uid"][i], None)
            item_text = self.item_text_T_dict.get(tst_dict["iid"][i], None)
            if user_emb is None or item_emb is None or user_text is None or item_text is None:
                continue
            tst_input.append(
                (
                    user_emb,
                    item_emb,
                    tst_dict['rating'][i],
                    user_text,
                    item_text,
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    tst_dict['explanation'][i],
                )
            )
        def custom_collate_fn(batch):
            user_embs = torch.stack([item[0].clone().detach().float() for item in batch])
            item_embs = torch.stack([item[1].clone().detach().float() for item in batch])
            ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            user_texts = [item[3] for item in batch]
            item_texts = [item[4] for item in batch]
            prompts = [item[5] for item in batch]
            return user_embs, item_embs, ratings,user_texts, item_texts, prompts
        def custom_collate_fn_test(batch):
            user_embs = torch.stack([item[0].clone().detach().float() for item in batch])
            item_embs = torch.stack([item[1].clone().detach().float() for item in batch])
            ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32) 
            user_texts = [item[3] for item in batch]
            item_texts = [item[4] for item in batch]
            prompts = [item[5] for item in batch] 
            explanations = [item[6] for item in batch]
            return user_embs, item_embs, ratings,user_texts, item_texts, prompts,explanations
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=custom_collate_fn)

        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=custom_collate_fn_test)

        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=custom_collate_fn_test)

        return trn_loader, val_loader, tst_loader
