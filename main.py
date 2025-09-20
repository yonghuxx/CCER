
import os
import pickle
import json
import warnings
from datetime import datetime
import torch
import torch.nn as nn
from models.C_explainer import Explainer
from utils.C_data_handler2 import DataHandler
from utils.parse import args
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CCER:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.model = Explainer().to(device)
        self.lambda_cf=1e-1
        self.lambda_allign=1
        self.data_handler = DataHandler()
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()
        self.user_embedding_converter_path = f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/user_converter.pkl"
        self.item_embedding_converter_path = f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/item_converter.pkl"
        self.attention_path=f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/attention.pkl"
        self.MLPZ_path=f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/MLPZ.pkl"
        self.user_time_attention_path=f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/user_time_attention.pkl"
        self.item_time_attention_path=f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/item_time_attention_3.pkl"        
        self.tst_predictions_path = f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/tst_predictions.pkl"
        self.tst_references_path = f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/tst_references.pkl"
    def load_model(self,optimizer):
        if os.path.exists(self.user_embedding_converter_path):
            self.model.user_embedding_converter.load_state_dict(torch.load(self.user_embedding_converter_path))
            self.model.item_embedding_converter.load_state_dict(torch.load(self.item_embedding_converter_path))
            self.model.attention.load_state_dict(torch.load(self.attention_path))
            self.model.MLPZ.load_state_dict(torch.load(self.MLPZ_path))
            self.model.user_time_attention.load_state_dict(torch.load(self.user_time_attention_path))    
            self.model.item_time_attention.load_state_dict(torch.load(self.item_time_attention_path))            
            optimizer_state = torch.load(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/optimizer_state_dict.pth")
            optimizer.load_state_dict(optimizer_state)
            print("Model loaded successfully!")
    def save_model(self, epoch, optimizer):
        print(f"Saving model after epoch {epoch}")
        base_dir = f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param"
        user_conv_path = f"{base_dir}/user_converter.pkl"
        item_conv_path = f"{base_dir}/item_converter.pkl"
        attention_path = f"{base_dir}/attention.pkl"
        user_time_attention_path = f"{base_dir}/user_time_attention.pkl"
        item_time_attention_path = f"{base_dir}/item_time_attention.pkl"        
        mlpz_path = f"{base_dir}/MLPZ.pkl"
        optimizer_path = f"{base_dir}/optimizer_state_dict.pth"
        torch.save(self.model.user_embedding_converter.state_dict(), user_conv_path)
        torch.save(self.model.item_embedding_converter.state_dict(), item_conv_path)
        torch.save(self.model.attention.state_dict(), attention_path)
        torch.save(self.model.MLPZ.state_dict(), mlpz_path)
        torch.save(self.model.user_time_attention.state_dict(), user_time_attention_path)
        torch.save(self.model.item_time_attention.state_dict(), item_time_attention_path)   
        torch.save(optimizer.state_dict(), optimizer_path)        
    def train(self):
        print(f"Loading model from {self.user_embedding_converter_path}")
        self.model.user_embedding_converter.load_state_dict(torch.load(self.user_embedding_converter_path))
        self.model.item_embedding_converter.load_state_dict(torch.load(self.item_embedding_converter_path))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)   
        total_batches = len(self.trn_loader)
        percent_interval = total_batches // 100  
        best_loss=float(20)
        counter = 0
        for epoch in range(args.epochs):
            total_loss = 0
            total_ali_loss=0
            self.model.train()
            with tqdm(enumerate(self.trn_loader), total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
                processed_batches = 0
                for i, batch in pbar:
                    user_embed, item_embed,ratings, user_text_T,item_text_T,input_text = batch
                    user_embed = user_embed.to(device)
                    item_embed = item_embed.to(device)
                    input_ids, outputs, explain_pos_position= self.model.forward(user_embed, item_embed, user_text_T,item_text_T,input_text,device)
                    input_ids = input_ids.to(device)
                    explain_pos_position = explain_pos_position.to(device)
                    optimizer.zero_grad()
                    main_loss = self.model.loss(input_ids, outputs, explain_pos_position, device)
                    batch_loss=main_loss
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                    processed_batches += 1
                    if processed_batches >= percent_interval:
                        avg_loss = total_loss / processed_batches
                        pbar.set_postfix(loss=avg_loss)  
                        percent_interval += total_batches // 100 

            avg_loss = total_loss / total_batches
            print(f"Epoch [{epoch }/{args.epochs}], avg_Loss: {avg_loss:.4f}")
            with open(f"/root/autodl-tmp/Casual_COT_LLM-base_ER/code/explainer/data/{args.dataset}/model_param/training_loss_2stage_COT.txt", "a") as f:
                f.write(f"Epoch {epoch}/{args.epochs}, avg_Loss: {avg_loss:.4f}\n")
            if avg_loss < best_loss:
                best_loss = avg_loss
                counter = 0
                self.save_model(epoch, optimizer)
            else:
                counter += 1
                if counter >= 2:
                    break
    def evaluate(self):
        warnings.filterwarnings("ignore")
        loader = self.tst_loader
        predictions_path = self.tst_predictions_path
        references_path = self.tst_references_path
        # Load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.user_time_attention.load_state_dict(torch.load(self.user_time_attention_path))    
        self.model.item_time_attention.load_state_dict(torch.load(self.item_time_attention_path))             
        self.model.attention.load_state_dict(
            torch.load(self.attention_path)
        )
        self.model.MLPZ.load_state_dict(
            torch.load(self.MLPZ_path)
        )
        self.model.eval()
        predictions = []
        references = []
        total_batch = len(loader)
        process_interval = total_batch // 100 
        print("Start time:", datetime.now().strftime("%H:%M:%S"))
        with torch.no_grad():
            with tqdm(total=total_batch, desc="Evaluating") as pbar:
                for i, batch in enumerate(loader):
                    user_embed, item_embed, ratings, user_text_T, item_text_T, input_text,explain = batch
                    user_embed = user_embed.to(device)
                    item_embed = item_embed.to(device)
                    outputs = self.model.generate(user_embed, item_embed,user_text_T,item_text_T, input_text,device)
                    end_idx = outputs[0].find("[")
                    if end_idx != -1:
                        outputs[0] = outputs[0][:end_idx]
                    predictions.append(outputs[0])
                    references.append(explain[0])
                    print(f"output:{outputs[0]}")
                    if (i + 1) % process_interval == 0:
                        pbar.update(process_interval) 
            with open(predictions_path, "wb") as file:
                pickle.dump(predictions, file)
            with open(references_path, "wb") as file:
                pickle.dump(references, file)
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved references to {references_path}")
        print("End time:", datetime.now().strftime("%H:%M:%S"))

def main():
    sample = CCER()
    if args.mode == "finetune":
        print("Finetune model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")
        sample.evaluate()

if __name__ == "__main__":
    main()
