  class Classifier(pl.LightningModule):

    def __init__(self):
      super().__init__()
      self.MFB = MFB(512,768,True,256,64,0.1)
      self.fin_y_shape = torch.nn.Linear(768,512)
      self.fin_old = torch.nn.Linear(2048,2)
      self.fin = torch.nn.Linear(16 * 768, 64)
      self.fin_inten = torch.nn.Linear(2048,6)
      self.fin_e1 = torch.nn.Linear(2048,2)
      self.fin_e2 = torch.nn.Linear(2048,2)
      self.fin_e3 = torch.nn.Linear(2048,2)
      self.fin_e4 = torch.nn.Linear(2048,2)
      self.fin_e5 = torch.nn.Linear(2048,2)
      self.fin_e6 = torch.nn.Linear(2048,2)
      self.fin_e7 = torch.nn.Linear(2048,2)
      self.fin_e8 = torch.nn.Linear(2048,2)
      self.fin_e9 = torch.nn.Linear(2048,2)
      # self.reduce_x = torch.nn.Linear(768, 512)
      # self.reduce_rag = torch.nn.Linear(768, 512)



      self.validation_step_outputs = []
      self.test_step_outputs = []


    def forward(self, x,y,rag):
        x_,y_,rag_ = x,y,rag
        print("x.shape", x.shape)
        print("y.shape",y.shape)
        # print("rag.shape",rag.shape)

        # x = self.reduce_x(x)
        # rag = self.reduce_rag(rag)

        # print("x.shape", x.shape)
        # print("y.shape",y.shape)
        # print("rag.shape",rag.shape)
        # z = self.MFB(torch.unsqueeze(y, axis=1), torch.unsqueeze(rag, axis=1))
        # z_rag = self.MFB(torch.unsqueeze(y, axis=1),torch.unsqueeze(rag, axis=1))
        # z_con = torch.cat((z, z_rag), dim=1)


        # Concatenate x with y and then with rag


        # z= torch.cat((torch.cat((x, y), dim=1), rag), dim=1)


        # # Pass concatenated x with y and x with rag through your network
        # z_new = torch.squeeze(z,dim=1)
        # print("z_new shape",z_new)

        z = torch.cat((x, y, rag), dim=1)
        z_new = torch.squeeze(z, dim=1)



        c_inten = self.fin_inten(z_new)
        c_e1 = self.fin_e1(z_new)
        c_e2 = self.fin_e2(z_new)
        c_e3 = self.fin_e3(z_new)
        c_e4 = self.fin_e4(z_new)
        c_e5 = self.fin_e5(z_new)
        c_e6 = self.fin_e6(z_new)
        c_e7 = self.fin_e7(z_new)
        c_e8 = self.fin_e8(z_new)
        c_e9 = self.fin_e9(z_new)
        c = self.fin_old(z_new)

        # print("z.shape",z.shape)
        # print("z_new shape",z_new.shape)
        # print("intensity error:", c_inten.shape)
        # print("output:", c.shape)
        # print("c_e1:", c_e1.shape)
        # print("c_e2:", c_e2.shape)
        # print("c_e3:", c_e3.shape)
        # print("c_e4:", c_e4.shape)
        # print("c_e5:", c_e5.shape)
        # print("c_e6:", c_e6.shape)
        # print("c_e7:", c_e7.shape)
        # print("c_e8:", c_e8.shape)
        # print("c_e9:", c_e9.shape)
        # print("logits.shape",logits.shape)


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

        return output,c_inten,c_e1,c_e2,c_e3,c_e4,c_e5,c_e6,c_e7,c_e8,c_e9


    def cross_entropy_loss(self, logits, labels):

        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        #lab,txt,rag,img,name,per,iro,alli,ana,inv,meta,puns,sat,hyp= train_batch
        lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6,e7,e8,e9= train_batch
        #logit_offen,a,b,c,d,e,f,g,h,i,logit_inten_target= self.forward(txt,img,rag)

        lab = train_batch[lab].unsqueeze(1)
        #print(lab)
        txt = train_batch[txt]
        rag = train_batch[rag]
        img = train_batch[img]
        name= train_batch[name]
        intensity = train_batch[intensity].unsqueeze(1)
        e1 = train_batch[e1].unsqueeze(1)
        e2 = train_batch[e2].unsqueeze(1)
        e3 = train_batch[e3].unsqueeze(1)
        e4 = train_batch[e4].unsqueeze(1)
        e5 = train_batch[e5].unsqueeze(1)
        e6 = train_batch[e6].unsqueeze(1)
        e7 = train_batch[e7].unsqueeze(1)
        e8 = train_batch[e8].unsqueeze(1)
        e9 = train_batch[e9].unsqueeze(1)

        lab = F.one_hot(lab, num_classes=2)
        intensity = torch.abs(intensity)
        intensity = F.one_hot(intensity, num_classes=6)  # Assuming you have 6 classes
        e1 = F.one_hot(e1,num_classes = 2)
        e2 = F.one_hot(e2,num_classes = 2)
        e3 = F.one_hot(e3,num_classes = 2)
        e4 = F.one_hot(e4,num_classes = 2)
        e5 = F.one_hot(e5,num_classes = 2)
        e6 = F.one_hot(e6,num_classes = 2)
        e7 = F.one_hot(e7,num_classes = 2)
        e8 = F.one_hot(e8,num_classes = 2)
        e9 = F.one_hot(e9,num_classes = 2)

        lab = lab.squeeze(dim=1)
        intensity = intensity.squeeze(dim=1)
        e1 = e1.squeeze(dim=1)
        e2 = e2.squeeze(dim=1)
        e3 = e3.squeeze(dim=1)
        e4 = e4.squeeze(dim=1)
        e5 = e5.squeeze(dim=1)
        e6 = e6.squeeze(dim=1)
        e7 = e7.squeeze(dim=1)
        e8 = e8.squeeze(dim=1)
        e9 = e9.squeeze(dim=1)



        logit_offen,logit_inten_target,a,b,c,d,e,f,g,h,i= self.forward(txt,img,rag)

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

        loss = loss1 + loss4 + loss5 + loss6 + loss7 + loss8 +loss9 + loss10 +loss11 +loss12 + loss17

        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_idx):
        #lab,txt,rag,img,name,per,iro,alli,ana,inv,meta,puns,sat,hyp = val_batch
        lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6,e7,e8,e9= val_batch
        lab = val_batch[lab].unsqueeze(1)
        #print(lab)
        txt = val_batch[txt]
        rag = val_batch[rag]
        img = val_batch[img]
        name = val_batch[name]
        intensity = val_batch[intensity].unsqueeze(1)
        e1 = val_batch[e1].unsqueeze(1)
        e2 = val_batch[e2].unsqueeze(1)
        e3 = val_batch[e3].unsqueeze(1)
        e4 = val_batch[e4].unsqueeze(1)
        e5 = val_batch[e5].unsqueeze(1)
        e6 = val_batch[e6].unsqueeze(1)
        e7 = val_batch[e7].unsqueeze(1)
        e8 = val_batch[e8].unsqueeze(1)
        e9 = val_batch[e9].unsqueeze(1)

        lab = F.one_hot(lab, num_classes=2)

        intensity = torch.abs(intensity)
        intensity = F.one_hot(intensity, num_classes=6)
        e1 = F.one_hot(e1,num_classes = 2)
        e2 = F.one_hot(e2,num_classes = 2)
        e3 = F.one_hot(e3,num_classes = 2)
        e4 = F.one_hot(e4,num_classes = 2)
        e5 = F.one_hot(e5,num_classes = 2)
        e6 = F.one_hot(e6,num_classes = 2)
        e7 = F.one_hot(e7,num_classes = 2)
        e8 = F.one_hot(e8,num_classes = 2)
        e9 = F.one_hot(e9,num_classes = 2)
        lab = lab.squeeze(dim=1)


        intensity = intensity.squeeze(dim = 1)
        e1 = e1.squeeze(dim=1)
        e2 = e2.squeeze(dim=1)
        e3 = e3.squeeze(dim=1)
        e4 = e4.squeeze(dim=1)
        e5 = e5.squeeze(dim=1)
        e6 = e6.squeeze(dim=1)
        e7 = e7.squeeze(dim=1)
        e8 = e8.squeeze(dim=1)
        e9 = e9.squeeze(dim=1)

        logits,inten,a,b,c,d,e,f,g,h,i = self.forward(txt,img,rag)

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
        lab,txt,rag,img,name,intensity,e1,e2,e3,e4,e5,e6,e7,e8,e9= batch
        lab = batch[lab].unsqueeze(1)
        #print(lab)
        txt = batch[txt]
        rag = batch[rag]
        img = batch[img]
        name = batch[name]
        intensity = batch[intensity].unsqueeze(1)
        e1 = batch[e1].unsqueeze(1)
        e2 = batch[e2].unsqueeze(1)
        e3 = batch[e3].unsqueeze(1)
        e4 = batch[e4].unsqueeze(1)
        e5 = batch[e5].unsqueeze(1)
        e6 = batch[e6].unsqueeze(1)
        e7 = batch[e7].unsqueeze(1)
        e8 = batch[e8].unsqueeze(1)
        e9 = batch[e9].unsqueeze(1)
        lab = F.one_hot(lab, num_classes=2)
        intensity = F.one_hot(intensity, num_classes=6)
        e1 = F.one_hot(e1,num_classes = 2)
        e2 = F.one_hot(e2,num_classes = 2)
        e3 = F.one_hot(e3,num_classes = 2)
        e4 = F.one_hot(e4,num_classes = 2)
        e5 = F.one_hot(e5,num_classes = 2)
        e6 = F.one_hot(e6,num_classes = 2)
        e7 = F.one_hot(e7,num_classes = 2)
        e8 = F.one_hot(e8,num_classes = 2)
        e9 = F.one_hot(e9,num_classes = 2)
        lab = lab.squeeze(dim=1)
        intensity = intensity.squeeze(dim=1)
        e1 = e1.squeeze(dim=1)
        e2 = e2.squeeze(dim=1)
        e3 = e3.squeeze(dim=1)
        e4 = e4.squeeze(dim=1)
        e5 = e5.squeeze(dim=1)
        e6 = e6.squeeze(dim=1)
        e7 = e7.squeeze(dim=1)
        e8 = e8.squeeze(dim=1)
        e9 = e9.squeeze(dim=1)

        logits,inten,a,b,c,d,e,f,g,h,i= self.forward(txt,img,rag)

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
        # OPTIONAL
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for out in self.test_step_outputs:
          outs.append(out['test_acc'])
          outs2.append(out['test_f1_score'])
        self.log('test_acc', sum(outs)/len(outs))
        self.log('test_f1_score', sum(outs2)/len(outs2))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
      # optimizer = torch.optim.Adam(self.parameters(), lr=3e-2)
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

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
      return DataLoader(self.hm_train, batch_size=10, drop_last=True)

    def val_dataloader(self):
      return DataLoader(self.hm_val, batch_size=10, drop_last=True)

    def test_dataloader(self):
      return DataLoader(self.hm_test, batch_size=10, drop_last=True)

  data_module = HmDataModule()
  checkpoint_callback = ModelCheckpoint(
      monitor='val_acc_all_offn',
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
  seed_everything(42, workers=True)
  hm_model = Classifier()
  gpus=1
  #if torch.cuda.is_available():gpus=0
  trainer = pl.Trainer(deterministic=True,max_epochs=20,precision=16,callbacks=all_callbacks)
  trainer.fit(hm_model, data_module)
