#mdified Code for the persuasive Meme Detectio

def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['text'])
  img_path = list(data['Name'])
  name = list(data['Name'])
  persuasive_inten = list(data['persuasive_inten'])
  # persuasive_inten = list(map(lambda x: x - 1 , persuasive_inten))
  label = list(data['Persuasive'])
  t3_1 =list(data['None'])
  t3_2 = list(data['Negatively persuasive'])
  t3_3 = list(data['Slightly Negatively persuasive'])
  t3_4 = list(data['Neutral'])
  t3_5 = list(data['Positively persuasive'])
  t3_6 = list(data['Slightly Positively persuasive'])






  text_features,image_features,rag_features,Name,l,t1,t2,t3,t4,t5,t6,persi = [],[],[],[],[],[],[],[],[],[],[],[]

  for txt,img,L,n,a,b,c,d,e,f,v in tqdm(zip(text,img_path,label,name,t3_1,t3_2,t3_3,t3_4,t3_5,t3_6,persuasive_inten)):
    try:
      img = Image.open('/content/drive/MyDrive/Gitanjali Mam/persuasive_meme/'+img)
    except Exception as e:
      print(e)
      continue
    txt_rag = get_completion(get_prompt_text(txt)+txt, model=mistral_llm)
    txt2 = txt_rag
    img = torch.stack([compose(img).to(device)])
    l.append(L)
    Name.append(n)
    t1.append(a)
    t2.append(b)
    t3.append(c)
    t4.append(d)
    t5.append(e)
    t6.append(f)
    persi.append(v)


    with torch.no_grad():
      temp_rag=model.forward(txt2, tokenizer).detach().cpu().numpy()
      rag_features.append(temp_rag)
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      text_features.append(temp_txt)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)
      del temp_txt
      del temp_img
      torch.cuda.empty_cache()
    del img
    torch.cuda.empty_cache()
  return text_features,rag_features,image_features,l,Name,t1,t2,t3,t4,t5,t6,persi


#Pre-Processing:
#Converts the opened image (img) to a PyTorch tensor and stacks it into a batch
#Uses CLIP to encode text into text_features & image to image_features
#CLIP Uses Zero-Shot Learning_


class HatefulDataset(Dataset):

  def __init__(self,data):
    self.t_f,self.r_f,self.i_f,self.label,self.name, self.t3_1, self.t3_2, self.t3_3, self.t3_4, self.t3_5, self.t3_6,self.persuasive_inten = get_data(data)

    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.r_f = np.squeeze(np.asarray(self.r_f),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)

  def __len__(self):
    return len(self.label)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    name=self.name[idx]
    label = self.label[idx]
    persuasive_inten = self.persuasive_inten[idx]
    t3_1 = self.t3_1[idx]
    t3_2 = self.t3_2[idx]
    t3_3 = self.t3_1[idx]
    t3_4 = self.t3_4[idx]
    t3_5 = self.t3_5[idx]
    t3_6 = self.t3_6[idx]



    T = self.t_f[idx,:]
    R = self.r_f[idx,:]
    I = self.i_f[idx,:]

    sample = {'label':label,'processed_txt':T,'processed_rag':R,'processed_img':I,'name':name,'persuasive_inten':persuasive_inten,'None':t3_1,
              'Negatively persuasive':t3_2,'Slightly Negatively persuasive': t3_3,'Neutral':t3_4,
              'Positively persuasive':t3_5,'Slightly Positively persuasive':t3_6}
    return sample

#Dataset Class for easily maintaing pipeline

#Init -> Pre-Process from get_data and convert to numpy array
#Len -> Total Number of Sample
#Getitem -> Sample at given idx in a List

#Text & Image features -> RAG  Output goes through CLIP
# We get otuput from multimodal MFB
# apply attention mechanism  -> concatenate & apply softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        #self.__C = __C
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)

        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat             # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z


#MFB -> Multimodal Factorized Bilinear Pooling
#used to model complex interactions between features like image and text
#MFB_K -> Number Of factors, MFB_O -> Output size,
#Init initializes linear projection layers for image and question features , dropout layer and average pooling layer

#Forward:

#exp_in = input expansion factor (default - 1)
#Linear projection of image and question features to factorized bilinear form
#Element-wise multiplication of image and question features
#APply Dropout
#Average pooling along the factorized dimension (MFB_K) to reduce the size of the output tensor
#Element-wise operations to compute the final output (z) using square root and normalization using Relu.
#The final output represents the fused representation of image and question features.

class Classifier(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.MFB = MFB(512,768,True,256,64,0.1)
    self.fin_y_shape = torch.nn.Linear(768,512)
    self.fin_old = torch.nn.Linear(128,2)
    self.fin = torch.nn.Linear(16 * 768, 64)
    self.fin_inten = torch.nn.Linear(128,6)
    self.fin_e1 = torch.nn.Linear(128,2)
    self.fin_e2 = torch.nn.Linear(128,2)
    self.fin_e3 = torch.nn.Linear(128,2)
    self.fin_e4 = torch.nn.Linear(128,2)
    self.fin_e5 = torch.nn.Linear(128,2)
    self.fin_e6 = torch.nn.Linear(128,2)
    
    self.validation_step_outputs = []
    self.test_step_outputs = []

  def forward(self, x,y,rag):
      x_,y_,rag_ = x,y,rag
      print("x.shape", x.shape)
      z = self.MFB(torch.unsqueeze(y, axis=1), torch.unsqueeze(x, axis=1))
      z_rag = self.MFB(torch.unsqueeze(y, axis=1), torch.unsqueeze(rag, axis=1))
      z_newe = torch.cat((z, z_rag), dim=2)
      z_new = torch.squeeze(z_newe, dim=1)
      c_inten = self.fin_inten(z_new)
      c_e1 = self.fin_e1(z_new)
      c_e2 = self.fin_e2(z_new)
      c_e3 = self.fin_e3(z_new)
      c_e4 = self.fin_e4(z_new)
      c_e5 = self.fin_e5(z_new)
      c_e6 = self.fin_e6(z_new)
      
      c = self.fin_old(z_new)
    
      output = torch.log_softmax(c, dim=1)
      c_inten = torch.log_softmax(c_inten, dim=1)
      c_e1 = torch.log_softmax(c_e1, dim=1)
      c_e2 = torch.log_softmax(c_e2, dim=1)
      c_e3 = torch.log_softmax(c_e3, dim=1)
      c_e4 = torch.log_softmax(c_e4, dim=1)
      c_e5 = torch.log_softmax(c_e5, dim=1)
      c_e6 = torch.log_softmax(c_e6, dim=1)
     
      return output,c_inten,c_e1,c_e2,c_e3,c_e4,c_e5,c_e6



  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6 = train_batch
      lab = train_batch[lab]
      #print(lab)
      txt = train_batch[txt]
      rag = train_batch[rag]
      img = train_batch[img]
      name= train_batch[name]
      intensity = train_batch[intensity]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      e6 = train_batch[e6]
  

      logit_offen,logit_inten_target,a,b,c,d,e,f= self.forward(txt,img,rag)
    
      loss1 = self.cross_entropy_loss(logit_offen, lab)
      loss4 = self.cross_entropy_loss(a, e1)
      loss5 = self.cross_entropy_loss(b, e2)
      loss6 = self.cross_entropy_loss(c, e3)
      loss7 = self.cross_entropy_loss(d, e4)
      loss8 = self.cross_entropy_loss(e, e5)
      loss9 = self.cross_entropy_loss(f, e6)
      
      loss17 = self.cross_entropy_loss(logit_inten_target, intensity)
      loss = loss1 + loss4 + loss5 + loss6 + loss7 + loss8 +loss9 + loss17
      self.log('train_loss', loss)
      return loss

  def validation_step(self, val_batch, batch_idx):
      lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6= val_batch
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
      
      logits,inten,a,b,c,d,e,f = self.forward(txt,img,rag)
      logits=logits.float()
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      self.validation_step_outputs.append({'progress_bar': tqdm_dict,'val_f1 offensive': f1_score(lab,tmp,average='macro')})

      return {
                'progress_bar': tqdm_dict,
      'val_f1 offensive': f1_score(lab,tmp,average='macro')
      }

  def on_validation_epoch_end(self):
    outs = []
    outs14=[]
    for out in self.validation_step_outputs:
       outs.append(out['progress_bar']['val_acc'])
       outs14.append(out['val_f1 offensive'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_f1 offensive', sum(outs14)/len(outs14))
    print(f'***val_acc_all_offn at epoch end {sum(outs)/len(outs)}****')
    print(f'***val_f1 offensive at epoch end {sum(outs14)/len(outs14)}****')
    self.validation_step_outputs.clear()

  def test_step(self, batch, batch_idx):
      lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6 = batch
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
     
      logits,inten,a,b,c,d,e,f= self.forward(txt,img,rag)
      logits = logits.float()
      tmp = np.argmax(logits.detach().cpu().numpy(force=True),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test_roc_auc',roc_auc_score(lab,tmp))
      self.log('test_loss', loss)
      tqdm_dict = {'test_acc': accuracy_score(lab,tmp)}
      self.test_step_outputs.append({'progress_bar': tqdm_dict,'test_acc': accuracy_score(lab,tmp), 'test_f1_score': f1_score(lab,tmp,average='macro')})
      return {
                'progress_bar': tqdm_dict,
                'test_acc': accuracy_score(lab,tmp),
                'test_f1_score': f1_score(lab,tmp,average='macro')
      }
  def on_test_epoch_end(self):
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12= \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[]
      for out in self.test_step_outputs:
        outs.append(out['test_acc'])
        outs2.append(out['test_f1_score'])
      self.log('test_acc', sum(outs)/len(outs))
      self.log('test_f1_score', sum(outs2)/len(outs2))
      self.test_step_outputs.clear()
    
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer


"""
Main Model:
Initialize
Forward Pass
Training Step
Validation Step
Testing Step

Pp
"""
class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):
    self.hm_train = t_p
    self.hm_val = v_p
    # self.hm_test = test
    self.hm_test = te_p

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64, drop_last=True)

  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128, drop_last=True)

data_module = HmDataModule()
checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offnn',
     dirpath='mrinal/',
     filename='epoch{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
# train
from pytorch_lightning import seed_everything
seed_everything(123, workers=True)
hm_model = Classifier()
gpus=1
#if torch.cuda.is_available():gpus=0
trainer = pl.Trainer(deterministic=True,max_epochs=60,precision=16,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)


