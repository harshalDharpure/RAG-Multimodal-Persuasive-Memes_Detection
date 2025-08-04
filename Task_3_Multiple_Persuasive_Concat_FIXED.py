"""
CORRECTED Task 3: Multiple Persuasive Detection with Concatenation
Fixed version with all identified bugs corrected
"""

import warnings
import os
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
import fireworks.client
from multilingual_clip import pt_multilingual_clip
import transformers

# Configure environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
clip_model, compose = clip.load("ViT-B/32", device=device)
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
mclip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# RAG Configuration - FIXED
def get_completion(prompt, model=None, max_tokens=200):
    """Enhanced RAG function with proper error handling"""
    try:
        fw_model_dir = "accounts/fireworks/models/"
        if model is None:
            model = fw_model_dir + "llama-v2-7b"
        else:
            model = fw_model_dir + model
        
        completion = fireworks.client.Completion.create(
            model=model,
            prompt=prompt,
            n=1,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9
        )
        return completion.choices[0].text
    except Exception as e:
        print(f"RAG API error: {e}")
        return prompt  # Return original text as fallback

def get_prompt_text(text_re):
    """Generate RAG prompt"""
    return f"Retrieve relevant information about the meme with the text transcription: {text_re} Explain the sentiment with context and any additional insights associated with this meme"

# FIXED: Enhanced data processing with error handling
def get_data(data):
    """Enhanced data processing with proper error handling"""
    text = list(data['text'])
    img_path = list(data['Name'])
    name = list(data['Name'])
    persuasive_inten = list(data['persuasive_inten'])
    label = list(data['Persuasive'])
    
    # FIXED: Correct column mapping
    t3_1 = list(data['Irony'])
    t3_2 = list(data['personification'])
    t3_3 = list(data['Alliteration'])  # FIXED: Was incorrectly mapped
    t3_4 = list(data['Analogies'])
    t3_5 = list(data['Invective'])
    t3_6 = list(data['Metaphor'])
    t3_7 = list(data['puns_and_wordplays'])
    t3_8 = list(data['Satire'])
    t3_9 = list(data['Hyperboles'])

    text_features, image_features, rag_features, Name, l, ir, per, alli, ana, inv, meta, puaps, sat, hyp, persi = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for txt, img, L, n, a, b, c, d, e, f, g, h, i, v in tqdm(zip(text, img_path, label, name, t3_1, t3_2, t3_3, t3_4, t3_5, t3_6, t3_7, t3_8, t3_9, persuasive_inten)):
        try:
            img = Image.open('/content/drive/MyDrive/Gitanjali Mam/persuasive_meme/' + img)
        except Exception as e:
            print(f"Image loading error: {e}")
            continue
            
        # FIXED: Enhanced RAG with error handling
        try:
            txt_rag = get_completion(get_prompt_text(txt) + txt, model="mistral-7b-instruct-4k")
        except:
            txt_rag = txt  # Fallback to original text
            
        txt2 = txt_rag
        img = torch.stack([compose(img).to(device)])
        
        l.append(L)
        Name.append(n)
        ir.append(a)
        persi.append(v)
        per.append(b)
        alli.append(c)
        ana.append(d)
        inv.append(e)
        meta.append(f)
        puaps.append(g)
        sat.append(h)
        hyp.append(i)
        
        with torch.no_grad():
            temp_rag = mclip_model.forward(txt2, tokenizer).detach().cpu().numpy()
            rag_features.append(temp_rag)
            temp_txt = mclip_model.forward(txt, tokenizer).detach().cpu().numpy()
            text_features.append(temp_txt)
            temp_img = clip_model.encode_image(img).detach().cpu().numpy()
            image_features.append(temp_img)
            
        # FIXED: Proper memory management
        del temp_txt, temp_img, img
        torch.cuda.empty_cache()
        
    return text_features, rag_features, image_features, l, Name, ir, per, alli, ana, inv, meta, puaps, sat, hyp, persi

# FIXED: Enhanced Dataset class
class HatefulDataset(Dataset):
    def __init__(self, data):
        self.t_f, self.r_f, self.i_f, self.label, self.name, self.t3_1, self.t3_2, self.t3_3, self.t3_4, self.t3_5, self.t3_6, self.t3_7, self.t3_8, self.t3_9, self.persuasive_inten = get_data(data)

        self.t_f = np.squeeze(np.asarray(self.t_f), axis=1)
        self.r_f = np.squeeze(np.asarray(self.r_f), axis=1)
        self.i_f = np.squeeze(np.asarray(self.i_f), axis=1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        name = self.name[idx]
        label = self.label[idx]
        persuasive_inten = self.persuasive_inten[idx]
        
        # FIXED: Correct mapping of persuasion types
        t3_1 = self.t3_1[idx]
        t3_2 = self.t3_2[idx]
        t3_3 = self.t3_3[idx]  # FIXED: Was incorrectly mapped to t3_1
        t3_4 = self.t3_4[idx]
        t3_5 = self.t3_5[idx]
        t3_6 = self.t3_6[idx]
        t3_7 = self.t3_7[idx]
        t3_8 = self.t3_8[idx]
        t3_9 = self.t3_9[idx]
        
        T = self.t_f[idx, :]
        R = self.r_f[idx, :]
        I = self.i_f[idx, :]

        sample = {
            'label': label,
            'processed_txt': T,
            'processed_rag': R,
            'processed_img': I,
            'name': name,
            'persuasive_inten': persuasive_inten,
            'irony': t3_1,
            'personification': t3_2,
            'Alliteration': t3_3,
            'Analogies': t3_4,
            'Invective': t3_5,
            'Metaphor': t3_6,  # FIXED: Was incorrectly mapped to t3_5
            'puns_and_wordplays': t3_7,
            'Satire': t3_8,
            'Hyperboles': t3_9
        }
        return sample

# FIXED: Enhanced Classifier with proper dimensions
class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # FIXED: Calculate correct input dimensions
        # text_features: 768, image_features: 512, rag_features: 768
        # Total concatenated: 768 + 512 + 768 = 2048
        input_dim = 768 + 512 + 768  # 2048
        
        self.fin_old = torch.nn.Linear(input_dim, 2)
        self.fin_inten = torch.nn.Linear(input_dim, 6)
        self.fin_e1 = torch.nn.Linear(input_dim, 2)
        self.fin_e2 = torch.nn.Linear(input_dim, 2)
        self.fin_e3 = torch.nn.Linear(input_dim, 2)
        self.fin_e4 = torch.nn.Linear(input_dim, 2)
        self.fin_e5 = torch.nn.Linear(input_dim, 2)
        self.fin_e6 = torch.nn.Linear(input_dim, 2)
        self.fin_e7 = torch.nn.Linear(input_dim, 2)
        self.fin_e8 = torch.nn.Linear(input_dim, 2)
        self.fin_e9 = torch.nn.Linear(input_dim, 2)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, y, rag):
        # FIXED: Proper concatenation
        z = torch.cat((x, y, rag), dim=1)
        
        c_inten = self.fin_inten(z)
        c_e1 = self.fin_e1(z)
        c_e2 = self.fin_e2(z)
        c_e3 = self.fin_e3(z)
        c_e4 = self.fin_e4(z)
        c_e5 = self.fin_e5(z)
        c_e6 = self.fin_e6(z)
        c_e7 = self.fin_e7(z)
        c_e8 = self.fin_e8(z)
        c_e9 = self.fin_e9(z)
        c = self.fin_old(z)

        output = torch.log_softmax(c, dim=1)
        c_inten = torch.log_softmax(c_inten, dim=1)
        c_e1 = torch.log_softmax(c_e1, dim=1)
        c_e2 = torch.log_softmax(c_e2, dim=1)
        c_e3 = torch.log_softmax(c_e3, dim=1)
        c_e4 = torch.log_softmax(c_e4, dim=1)
        c_e5 = torch.log_softmax(c_e5, dim=1)
        c_e6 = torch.log_softmax(c_e6, dim=1)
        c_e7 = torch.log_softmax(c_e7, dim=1)
        c_e8 = torch.log_softmax(c_e8, dim=1)
        c_e9 = torch.log_softmax(c_e9, dim=1)
        
        return output, c_inten, c_e1, c_e2, c_e3, c_e4, c_e5, c_e6, c_e7, c_e8, c_e9

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        lab, txt, rag, img, name, intensity, e1, e2, e3, e4, e5, e6, e7, e8, e9 = train_batch
        lab = train_batch[lab]
        txt = train_batch[txt]
        rag = train_batch[rag]
        img = train_batch[img]
        name = train_batch[name]
        intensity = train_batch[intensity]
        e1 = train_batch[e1]
        e2 = train_batch[e2]
        e3 = train_batch[e3]
        e4 = train_batch[e4]
        e5 = train_batch[e5]
        e6 = train_batch[e6]
        e7 = train_batch[e7]
        e8 = train_batch[e8]
        e9 = train_batch[e9]
        
        logit_offen, logit_inten_target, a, b, c, d, e, f, g, h, i = self.forward(txt, img, rag)
        
        loss1 = self.cross_entropy_loss(logit_offen, lab)
        loss17 = self.cross_entropy_loss(logit_inten_target, intensity)
        loss4 = self.cross_entropy_loss(a, e1)
        loss5 = self.cross_entropy_loss(b, e2)
        loss6 = self.cross_entropy_loss(c, e3)
        loss7 = self.cross_entropy_loss(d, e4)
        loss8 = self.cross_entropy_loss(e, e5)
        loss9 = self.cross_entropy_loss(f, e6)
        loss10 = self.cross_entropy_loss(g, e7)
        loss11 = self.cross_entropy_loss(h, e8)
        loss12 = self.cross_entropy_loss(i, e9)
        
        loss = loss1 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss17
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        lab, txt, rag, img, name, intensity, e1, e2, e3, e4, e5, e6, e7, e8, e9 = val_batch
        lab = val_batch[lab]
        txt = val_batch[txt]
        rag = val_batch[rag]
        img = val_batch[img]
        name = val_batch[name]
        intensity = val_batch[intensity]
        e1 = val_batch[e1]
        e2 = val_batch[e2]
        e3 = val_batch[e3]
        e4 = val_batch[e4]
        e5 = val_batch[e5]
        e6 = val_batch[e6]
        e7 = val_batch[e7]
        e8 = val_batch[e8]
        e9 = val_batch[e9]
        
        logits, inten, a, b, c, d, e, f, g, h, i = self.forward(txt, img, rag)
        logits = logits.float()
        
        # FIXED: Proper ROC-AUC calculation
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        tmp = np.argmax(logits.detach().cpu().numpy(), axis=1)
        
        loss = self.cross_entropy_loss(logits, lab)
        lab = lab.detach().cpu().numpy()
        
        self.log('val_acc', accuracy_score(lab, tmp))
        self.log('val_roc_auc', roc_auc_score(lab, probabilities[:, 1]))  # FIXED
        self.log('val_loss', loss)
        
        tqdm_dict = {'val_acc': accuracy_score(lab, tmp)}
        self.validation_step_outputs.append({
            'progress_bar': tqdm_dict,
            'val_f1 offensive': f1_score(lab, tmp, average='macro')
        })

        return {
            'progress_bar': tqdm_dict,
            'val_f1 offensive': f1_score(lab, tmp, average='macro')
        }

    def on_validation_epoch_end(self):
        outs = []
        outs14 = []
        for out in self.validation_step_outputs:
            outs.append(out['progress_bar']['val_acc'])
            outs14.append(out['val_f1 offensive'])
        
        self.log('val_acc_all_offn', sum(outs) / len(outs))
        self.log('val_f1 offensive', sum(outs14) / len(outs14))
        print(f'***val_acc_all_offn at epoch end {sum(outs) / len(outs)}****')
        print(f'***val_f1 offensive at epoch end {sum(outs14) / len(outs14)}****')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        lab, txt, rag, img, name, intensity, e1, e2, e3, e4, e5, e6, e7, e8, e9 = batch
        lab = batch[lab]
        txt = batch[txt]
        rag = batch[rag]
        img = batch[img]
        name = batch[name]
        intensity = batch[intensity]
        e1 = batch[e1]
        e2 = batch[e2]
        e3 = batch[e3]
        e4 = batch[e4]
        e5 = batch[e5]
        e6 = batch[e6]
        e7 = batch[e7]
        e8 = batch[e8]
        e9 = batch[e9]
        
        logits, inten, a, b, c, d, e, f, g, h, i = self.forward(txt, img, rag)
        logits = logits.float()
        
        # FIXED: Proper ROC-AUC calculation
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        tmp = np.argmax(logits.detach().cpu().numpy(force=True), axis=-1)
        
        loss = self.cross_entropy_loss(logits, lab)
        lab = lab.detach().cpu().numpy()
        
        self.log('test_acc', accuracy_score(lab, tmp))
        self.log('test_roc_auc', roc_auc_score(lab, probabilities[:, 1]))  # FIXED
        self.log('test_loss', loss)
        
        tqdm_dict = {'test_acc': accuracy_score(lab, tmp)}
        self.test_step_outputs.append({
            'progress_bar': tqdm_dict,
            'test_acc': accuracy_score(lab, tmp),
            'test_f1_score': f1_score(lab, tmp, average='macro')
        })
        
        return {
            'progress_bar': tqdm_dict,
            'test_acc': accuracy_score(lab, tmp),
            'test_f1_score': f1_score(lab, tmp, average='macro')
        }

    def on_test_epoch_end(self):
        outs = []
        outs2 = []
        for out in self.test_step_outputs:
            outs.append(out['test_acc'])
            outs2.append(out['test_f1_score'])
        
        self.log('test_acc', sum(outs) / len(outs))
        self.log('test_f1_score', sum(outs2) / len(outs2))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer

# FIXED: Enhanced Data Module
class HmDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # Load your data here
        data = pd.read_csv('/content/personification_train.csv')
        sample_dataset = HatefulDataset(data)
        
        # FIXED: Proper train/val/test split
        torch.manual_seed(42)
        total_size = len(sample_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        self.hm_train, self.hm_val, self.hm_test = random_split(
            sample_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.hm_train, batch_size=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.hm_val, batch_size=8, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.hm_test, batch_size=8, drop_last=True)

# FIXED: Enhanced training setup
def main():
    data_module = HmDataModule()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_all_offn',
        dirpath='checkpoints/',
        filename='epoch{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="max",
    )
    
    all_callbacks = [checkpoint_callback]
    
    # FIXED: Proper seeding
    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)
    
    hm_model = Classifier()
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=20,
        precision=16,
        callbacks=all_callbacks
    )
    
    trainer.fit(hm_model, data_module)

if __name__ == "__main__":
    main() 