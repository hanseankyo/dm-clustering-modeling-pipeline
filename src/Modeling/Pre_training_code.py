def main():
    # -*- coding: utf-8 -*-

    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = BASE_DIR / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR
    from modeling_common import FCNetwork, check_correct, eGFR_cal, mkMBP, seed_everything

    import numpy as np
    import pandas as pd
    import torch
    import os
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import roc_auc_score

    def log_paths(title, paths):
        print(title)
        for p in paths:
            print(f"  {p}")

    log_paths("Input files:", [DATA_DIR / "Only_clinical.csv"])
    log_paths("Output dirs:", [OUTPUT_DIR, MODEL_DIR])

    clinical_only_raw = pd.read_csv(DATA_DIR / "Only_clinical.csv")

    import torch
    import numpy as np

    # Function for DNN performance
    def model_performance_DNN(real,pred,cutoff):

        pred_label = np.where(pred>cutoff, 1, 0) # relation to sigmoid activation function
        #print(pred_label)
        #print(real)
        accuracy, sensitivity, specificity=check_correct(pred_label,real)
        auc = roc_auc_score(real, pred)

        df_hypo=pd.DataFrame(pred) # prediction probability
        df_hypo.columns=['hypothesis 1']

        df_pred=pd.DataFrame(pred_label) # zero or one 
        df_pred.columns=['prediction']

        df_y=pd.DataFrame(real)
        df_y.columns=['y']

        pred_result=pd.concat([df_y,df_hypo, df_pred],axis=1)

        return accuracy ,sensitivity, specificity ,auc

    def validation(data,model,epoch,keyword, cutoff=0.5):
        x=data['x']
        y=data['y']

        tensor_x=torch.from_numpy(x).float().to(device)
        tensor_y=torch.from_numpy(y).float().to(device).view(-1,1)

        output=model(tensor_x)
        logit_output=torch.sigmoid(output)
        val_loss=criterion(output,tensor_y)
        acc,sen,spe,auc=model_performance_DNN(y,logit_output.cpu().detach().numpy(),cutoff)
        print(f'{epoch} {keyword}_loss:{val_loss:.4f} {keyword}_acc:{acc:.4f} {keyword}_sen:{sen:.4f} {keyword}_spe:{spe:.4f} {keyword}_auc:{auc:.4f} ')

        return val_loss, acc, sen, spe, auc

    class GenerateData(Dataset):
        def __init__(self,dataset):
            #dataset:dict
            self.x=torch.from_numpy(dataset['x']).float()
            self.y=torch.from_numpy(dataset['y']).float()
            self.len=dataset['x'].shape[0]

        def __getitem__(self,idx):
            data={'x':self.x[idx],'y':self.y[idx]}
            return data
        def __len__(self):
            return self.len

    eGFR_data = []
    for sample in range(len(clinical_only_raw)):
        eGFR_data.append(eGFR_cal(clinical_only_raw.loc[sample,"Cr"], clinical_only_raw.loc[sample,"age"], clinical_only_raw.loc[sample,"F"]))
    clinical_only_raw['Cr'] = eGFR_data
    clinical_only_raw = clinical_only_raw.rename(columns = {'Cr':'eGFR'})

    drop_cols=['sample','area']
    clinical_only_raw=clinical_only_raw.drop(columns=drop_cols)

    clinical_only_raw=clinical_only_raw.dropna()

    print(clinical_only_raw.shape)
    print(clinical_only_raw['progress_DM'].value_counts())

    cli_x,cli_y=clinical_only_raw.drop(columns=['progress_DM']),clinical_only_raw['progress_DM']

    cli_x['Tg']=np.log(cli_x['Tg'])
    cli_x = mkMBP(cli_x)

    ptr_x,pts_x,ptr_y,pts_y=train_test_split(cli_x,cli_y,test_size=0.2,stratify=cli_y,random_state=123)

    p_scaler=MinMaxScaler()
    nor_ptr_x=p_scaler.fit_transform(ptr_x)
    nor_pts_x=p_scaler.transform(pts_x)

    ptr_y.value_counts()

    pts_y.value_counts()

    ptr_set={'x':nor_ptr_x,'y':ptr_y.to_numpy()}
    train_set=GenerateData(ptr_set)

    class_counts = ptr_y.value_counts().to_list()
    num_samples = sum(class_counts)
    labels = ptr_y.to_list()

    #클래스별 가중치 부여
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))] 

    # 해당 데이터의 label에 해당되는 가중치
    weights = [class_weights[labels[i]] for i in range(int(num_samples))] #해당 레이블마다의 가중치 비율
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    g = torch.Generator()
    g.manual_seed(123)

    tr_loader=DataLoader(train_set,batch_size=512, generator=g, sampler=sampler)#,shuffle=True)

    val_set={'x':nor_pts_x,'y':pts_y.to_numpy()}

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    print(nor_ptr_x.shape)
    print(nor_pts_x.shape)

    FEATURE_NUM=ptr_x.shape[1]
    EPOCH=5000
    EARLY=5000

    seed_everything(123)
    net=FCNetwork(FEATURE_NUM)
    learningrate = 1e-02 # learning rate
    optimizer=optim.AdamW(net.parameters(),lr=learningrate,weight_decay=0.01)
    net.to(device)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=100)
    criterion=nn.BCEWithLogitsLoss()

    save_path = MODEL_DIR / 'Pretrain'
    save_path.mkdir(parents=True, exist_ok=True)

    val_P = pd.DataFrame()
    early_count = 0
    early_vsen = 0
    early_auc = 0
    for epoch in range(EPOCH):
        net.train()
        batch_tr_loss=[]
        batch_tr_acc=[]
        batch_tr_sen=[]
        batch_tr_spe=[]
        batch_tr_auc=[]

        for i,data in enumerate(tr_loader):
            x=data['x'].to(device)
            y=data['y'].to(device).view(-1,1)

            optimizer.zero_grad()
            #print(x)
            output=net(x)
            #print(output)
            logit_output=torch.sigmoid(output)
            #print(logit_output)
            acc,sen,spe,auc=model_performance_DNN(y.cpu().detach().numpy(),logit_output.cpu().detach().numpy(),0.5)
            batch_tr_acc.append(acc)
            batch_tr_sen.append(sen)
            batch_tr_spe.append(spe)
            batch_tr_auc.append(auc)

            loss=criterion(output,y)
            loss.backward()
            optimizer.step()
            batch_tr_loss.append(loss.item())
            print('Epoch: '+str(epoch+1)+' ['+'='*i+'-'*(len(tr_loader)-i-1)+']', end='\r')
        print()

        tr_loss=np.mean(batch_tr_loss)
        tr_acc=np.mean(batch_tr_acc)
        tr_sen=np.mean(batch_tr_sen)
        tr_spe=np.mean(batch_tr_spe)
        tr_auc=np.mean(batch_tr_auc)
        net.eval()    
        print(f'tr_loss:{tr_loss:.4f} tr_acc:{tr_acc:.4f} tr_sen:{tr_sen:.4f} tr_spe:{tr_spe:.4f} tr_auc:{tr_auc:.4f} ')
        vloss, vacc, vsen, vspe, vauc = validation(val_set,net,epoch,'VAL')    
        net.train()
        if early_count == EARLY:
            print('early stopping!')
            val_p['name'] = Ep + '_' + file_name.name
            val_P = pd.concat([val_P,val_p])
            print(f"[OUTPUT] {save_path / 'Validation_Performance.csv'}")
            val_P.to_csv(save_path / 'Validation_Performance.csv', index=False)
            break
        if early_auc < vauc: # AUC 기준 모델 저장
            #EARLY = 50
            print('Epoch: '+str(epoch)+' save model by validation loss\n')
            Ep = 'Ep-'+str(epoch+1)
            # 모델 저장 이름 설정
            file_name = save_path / f'best_model_{epoch}.h5'
            print(f"[OUTPUT] {file_name}")
            torch.save(net.state_dict(), file_name)
            val_p = pd.DataFrame({"name":[file_name.name],
                                 "acc":[vacc],
                                 "sen":[vsen],
                                 "spe":[vspe],
                                 "auc":[vauc]})
            early_point = vloss
            early_vsen = vsen
            early_count = 0     
            early_auc = vauc
        else:
            print(' ')
            early_count = early_count + 1
            scheduler.step(vloss)


if __name__ == '__main__':
    main()
