def main():
    # -*- coding: utf-8 -*-
    # Auto-generated from Pre_training_code.ipynb

    # %% [cell 1]
    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = BASE_DIR / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR

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

    # %% [cell 2]
    clinical_only_raw = pd.read_csv(DATA_DIR / "Only_clinical.csv")

    # %% [cell 3]
    import torch
    import random
    import numpy as np


    def seed_everything(seed):
        torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
        torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed) #numpy를 사용할 경우 고정
        random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

    # %% [cell 4]
    def eGFR_cal(Cr,age, sex):
        if sex == 1 :# 여성
            K = 0.7
            alpha = -0.241 
            last = 1.012
        else:# 남성
            K = 0.9
            alpha = -0.302 
            last = 1

        eGFR = 142*((min(Cr/K,1))**alpha)*((max(Cr/K,1))**(-1.2))*((0.9938)**age)*last
        return np.round(eGFR,0)

    # %% [cell 5]
    def mkMBP(cli):
        cli['MBP'] = (2*cli['dia'] + cli['sys'])/3
        cli = cli.drop(columns = ['dia','sys'])
        return cli

    # %% [cell 6]
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

    # %% [cell 7]
    # Function for confusion metrics
    def check_correct(predict, y):     #Using def enables us to customize our own functions. 
        result = {}
        result['True-Positive'] = 0
        result['True-Negative'] = 0
        result['False-Negative'] = 0
        result['False-Positive'] = 0

        for i in range(len(predict)) :
            if predict[i] == y[i] :
                if y[i] == 0 :
                    result['True-Negative'] += 1
                else :
                    result['True-Positive'] += 1
            else :
                if y[i] == 0 :
                    result['False-Positive'] += 1
                else :
                    result['False-Negative'] += 1

        try:
            accuracy=(result['True-Positive']+result['True-Negative'])/len(y)  # Accuracy = correct predictions / all predictions 
            sensitivity=result['True-Positive']/(result['True-Positive']+result['False-Negative']) # TP / TP + FN
            specificity=result['True-Negative']/(result['True-Negative']+result['False-Positive']) # TN / TN + FP
        except ZeroDivisionError:
            print('0 divisionerror')

        return accuracy, sensitivity, specificity

    # %% [cell 8]
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

    # %% [cell 9]
    class FCNetwork(nn.Module):
        def __init__(self,FEATURE_NUM):
            super(FCNetwork, self).__init__()
            self.pre_part=nn.Sequential(
                nn.Linear(FEATURE_NUM, FEATURE_NUM),
                nn.BatchNorm1d(FEATURE_NUM),
                nn.ELU(),
                nn.Dropout(),
                nn.Linear(FEATURE_NUM, FEATURE_NUM),
                nn.BatchNorm1d(FEATURE_NUM),
                nn.ELU(),
                nn.Dropout(),
                nn.Linear(FEATURE_NUM, FEATURE_NUM),
                nn.BatchNorm1d(FEATURE_NUM),
                nn.ELU(),
                nn.Dropout(),
                nn.Linear(FEATURE_NUM, FEATURE_NUM),
                nn.BatchNorm1d(FEATURE_NUM),
                nn.ELU(),
                nn.Dropout(),  

            )
            self.fc4 = nn.Linear(FEATURE_NUM, 1)

        def forward(self, x):
            x=self.pre_part(x)
            out=self.fc4(x)
            return out

    # %% [cell 10]
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

    # %% [cell 11]
    eGFR_data = []
    for sample in range(len(clinical_only_raw)):
        eGFR_data.append(eGFR_cal(clinical_only_raw.loc[sample,"Cr"], clinical_only_raw.loc[sample,"age"], clinical_only_raw.loc[sample,"F"]))
    clinical_only_raw['Cr'] = eGFR_data
    clinical_only_raw = clinical_only_raw.rename(columns = {'Cr':'eGFR'})

    # %% [cell 12]
    drop_cols=['sample','area']
    clinical_only_raw=clinical_only_raw.drop(columns=drop_cols)

    # %% [cell 13]
    clinical_only_raw=clinical_only_raw.dropna()

    # %% [cell 14]
    print(clinical_only_raw.shape)
    print(clinical_only_raw['progress_DM'].value_counts())

    # %% [cell 15]
    cli_x,cli_y=clinical_only_raw.drop(columns=['progress_DM']),clinical_only_raw['progress_DM']

    # %% [cell 16]
    cli_x['Tg']=np.log(cli_x['Tg'])
    cli_x = mkMBP(cli_x)

    # %% [cell 17]
    ptr_x,pts_x,ptr_y,pts_y=train_test_split(cli_x,cli_y,test_size=0.2,stratify=cli_y,random_state=123)

    # %% [cell 18]
    p_scaler=MinMaxScaler()
    nor_ptr_x=p_scaler.fit_transform(ptr_x)
    nor_pts_x=p_scaler.transform(pts_x)

    # %% [cell 19]
    ptr_y.value_counts()

    # %% [cell 20]
    pts_y.value_counts()

    # %% [cell 21]
    ptr_set={'x':nor_ptr_x,'y':ptr_y.to_numpy()}
    train_set=GenerateData(ptr_set)

    # %% [cell 22]
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

    # %% [cell 23]
    val_set={'x':nor_pts_x,'y':pts_y.to_numpy()}

    # %% [cell 24]
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # %% [cell 25]
    print(nor_ptr_x.shape)
    print(nor_pts_x.shape)

    # %% [cell 26]
    FEATURE_NUM=ptr_x.shape[1]
    EPOCH=5000
    EARLY=5000

    # %% [cell 27]
    seed_everything(123)
    net=FCNetwork(FEATURE_NUM)
    learningrate = 1e-02 # learning rate
    optimizer=optim.AdamW(net.parameters(),lr=learningrate,weight_decay=0.01)
    net.to(device)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=100)
    criterion=nn.BCEWithLogitsLoss()

    # %% [cell 28]
    save_path = MODEL_DIR / 'Pretrain'
    save_path.mkdir(parents=True, exist_ok=True)

    # %% [cell 29]
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

    # %% [cell 30]


if __name__ == '__main__':
    main()
