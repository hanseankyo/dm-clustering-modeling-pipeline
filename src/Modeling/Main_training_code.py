def main():
    # -*- coding: utf-8 -*-

    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = BASE_DIR / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from config import (
        DATA_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
        SNP_FEATURE_NUM as SNP_INPUT_LEN,
        PRE_FEATURE_NUM,
        TRANSFER_EARLY,
        TRANSFER_WARMUP_EPOCHS,
        TRANSFER_TOTAL_EPOCHS,
        TRANSFER_BATCH_SIZE_LIST,
        TRANSFER_RANDOM_STATE_LIST,
        TRANSFER_LEARNING_RATE_LIST,
        TRANSFER_SNP_NUM_LIST,
        TRANSFER_TOTAL_NUM_LIST,
        TRANSFER_POSITIVE_WEIGHT_LIST,
        TRANSFER_DROPOUT_NUM_LIST,
    )
    from modeling_common import FCNetwork, check_correct, mkMBP, mk_eGFR_data, seed_everything

    import numpy as np
    import pandas as pd
    import torch
    import os
    import re
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F

    def log_paths(title, paths):
        print(title)
        for p in paths:
            print(f"  {p}")

    log_paths("Input files:", [DATA_DIR / "Train.csv", DATA_DIR / "Validation.csv", DATA_DIR / "Test.csv"])
    log_paths("Output dirs:", [OUTPUT_DIR, MODEL_DIR])

    def latest_pretrain_path(pretrain_dir):
        pretrain_dir = Path(pretrain_dir)
        candidates = list(pretrain_dir.glob('best_model_*.h5'))
        if not candidates:
            raise FileNotFoundError(f'No pretrain models found in {pretrain_dir}')
        def epoch_num(p):
            m = re.search(r'best_model_(\d+)\.h5$', p.name)
            return int(m.group(1)) if m else -1
        return max(candidates, key=epoch_num)

    from IPython.display import clear_output
    import torchtuples as tt
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    import torch
    import numpy as np

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    class GenerateData(Dataset):
        def __init__(self,dataset):
            #dataset:dict
            self.snp=torch.from_numpy(dataset['snp']).float()
            self.cli=torch.from_numpy(dataset['cli']).float()
            self.y=torch.from_numpy(dataset['y']).float()
            self.len=dataset['snp'].shape[0]

        def __getitem__(self,idx):
            data={'snp':self.snp[idx],'cli':self.cli[idx],'y':self.y[idx]}
            return data
        def __len__(self):
            return self.len

    def validation_func(data,model,criterion,epoch,keyword, cutoff=0.5):
        snp=data['snp']
        cli=data['cli']
        y=data['y']

        tensor_snp=torch.from_numpy(snp).float().to(device)
        tensor_cli=torch.from_numpy(cli).float().to(device)
        tensor_y=torch.from_numpy(y).float().to(device).view(-1,1)

        output=model(tensor_snp, tensor_cli)
        logit_output=torch.sigmoid(output)
        val_loss=criterion(output,tensor_y)
        acc,sen,spe,auc=model_performance_DNN(y,logit_output.cpu().detach().numpy(),cutoff)

        return val_loss, acc, sen, spe, auc

    def _to_1d_np(x):
        """torch/tensor/list 등 무엇이 와도 1D numpy로 안전 변환"""
        if 'torch' in str(type(x)):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x

    def _sigmoid(z):
        # 수치 안정화된 시그모이드
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def model_performance_DNN(real, pred, cutoff=0.5, eps=1e-7, verbose=False):
        """
        real: (N,) 라벨 {0,1}
        pred: (N,) 확률 또는 로짓(자동 감지)
        cutoff: 양성 판정 임계값 (확률 기준)
        eps: 확률 클리핑 범위
        """
        # 1) 타입/형태 정리
        real = _to_1d_np(real)
        pred = _to_1d_np(pred)

        # 2) pred가 확률인지(logit인지) 자동 감지 → 필요시 sigmoid
        #   - 정상 확률이면 [0,1] 범위. 범위를 벗어나면 로짓으로 간주.
        if (pred.min() < 0.0) or (pred.max() > 1.0):
            pred = _sigmoid(pred)

        # 3) NaN/Inf 제거 및 클리핑
        mask = np.isfinite(real) & np.isfinite(pred)
        if verbose and not mask.all():
            print(f"[WARN] Dropped {np.size(mask)-np.sum(mask)} NaN/Inf samples")
        real = real[mask]
        pred = np.clip(pred[mask], eps, 1.0 - eps)

        # 4) 이진 라벨 보장(혹시 float로 들어오면 0/1로 정리)
        #    - 라벨에 NaN 있었으면 위에서 제거됨
        real = (real > 0.5).astype(int)

        # 5) 예측 라벨 계산
        pred_label = (pred >= cutoff).astype(int)
        if verbose:
            vc = pd.Series(pred_label).value_counts().to_dict()
            print(f"[INFO] pred_label counts: {vc}")

        # 6) 사용자 정의 정확도/민감도/특이도
        accuracy, sensitivity, specificity = check_correct(pred_label, real)

        # 7) AUC (단일 클래스면 NaN)
        unique_classes = np.unique(real)
        if unique_classes.size < 2:
            auc = np.nan
            if verbose:
                print("[WARN] Only one class present in y_true; AUC is undefined → np.nan")
        else:
            auc = roc_auc_score(real, pred)

        # (선택) 분석용 DF가 필요하면 아래 주석 해제해서 반환값에 추가하세요.
        # df = pd.DataFrame({"y": real, "prob": pred, "pred": pred_label})
        # return accuracy, sensitivity, specificity, auc, df

        return accuracy, sensitivity, specificity, auc

    pre_model=FCNetwork(PRE_FEATURE_NUM).to(device)
    pretraining_path = latest_pretrain_path(MODEL_DIR / 'Pretrain')
    print(f"[INPUT] {pretraining_path}")
    pre_model.load_state_dict(torch.load(pretraining_path, weights_only=True))
    pre_model=pre_model.pre_part

    from torch.optim.lr_scheduler import _LRScheduler

    #warm-up schedular
    class WarmUpLR(_LRScheduler):
        def __init__(self,optimizer,warmup_epochs, total_epochs, last_epochs=-1):
            self.warmup_epochs=warmup_epochs
            self.total_epochs=total_epochs
            super(WarmUpLR,self).__init__(optimizer,last_epochs)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                warmup_factor = float(self.last_epoch) / float(self.warmup_epochs)
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
            else:
                remaining_epochs = self.total_epochs - self.warmup_epochs
                regular_schedule = [(base_lr - (base_lr * float(self.last_epoch - self.warmup_epochs) / float(remaining_epochs))) for base_lr in self.base_lrs]
                return regular_schedule


    class Net1(nn.Module):
        def __init__(self, SNP_FEATURE_NUM, CLI_FEATURE_NUM, T_FEATURE_NUM,pre_model):
            super(Net1,self).__init__()

            self.ac=self.activation(acfunc)

            Total_linear_block_lst=[]

            Total_bn_block_lst=[]

            self.snp=nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5),
                nn.BatchNorm1d(16),
                #nn.ELU(),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=3,stride=1),
                nn.Dropout(drop_num),
                nn.Conv1d(16, 32, kernel_size=5),
                nn.BatchNorm1d(32),
                #nn.ELU(),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=3,stride=1),
                nn.Dropout(drop_num),
                nn.Conv1d(32, 32, kernel_size=3),
                nn.BatchNorm1d(32),
                #nn.ELU(),            
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=3,stride=1),
                nn.Dropout(drop_num),            
                nn.Conv1d(32, 64, kernel_size=3),
                nn.BatchNorm1d(64),
                #nn.ELU(),            
                nn.LeakyReLU(),
                nn.AdaptiveMaxPool1d(output_size=1),

                nn.Flatten(),
            )
            for i in range(0,Total_num-1,1):
                Total_linear_block_lst.append(nn.Linear(Total_fe_lst[i],Total_fe_lst[i+1]))

            for i in range(1,Total_num,1):
                Total_bn_block_lst.append(nn.BatchNorm1d(Total_fe_lst[i]))

            self.Total_linear_lst=nn.ModuleList(Total_linear_block_lst)

            self.Total_bn_lst=nn.ModuleList(Total_bn_block_lst)

            self.dropout = nn.Dropout(drop_num)

            self.cli_layer=pre_model

            self.last=nn.Linear(Total_fe_lst[-1],1)

        def forward(self,snp, cli):
            snp=snp.view([-1,1,SNP_INPUT_LEN])
            snp=self.snp(snp)

            cli_output=self.cli_layer(cli)

            total = torch.cat((snp, cli_output), dim=1)

            for i,(l,bn) in enumerate(zip(self.Total_linear_lst,self.Total_bn_lst)):
                total=l(total)
                total=bn(total)
                total=self.ac(total)
                total=self.dropout(total)

            out=self.last(total)
            return out

        def activation(self,func):
            if func=='elu':
                return nn.ELU()
            elif func=='leaky_relu':
                return nn.LeakyReLU()
            elif func=='gelu':
                return nn.GELU()
            elif func=='swish':
                return nn.SiLU()
            elif func=='soft':
                return F.softplus
            elif func=='mish':
                return nn.Mish()

    def load_data(file_path,idx,scaler):
        tr_path=os.path.join(file_path,f'Train.csv')
        val_path=os.path.join(file_path,f'Validation.csv')
        test_path=os.path.join(file_path,'Test.csv')
        train_raw=pd.read_csv(tr_path)
        validation_raw=pd.read_csv(val_path)
        test_raw=pd.read_csv(test_path)    

        train_raw = train_raw.dropna().reset_index(drop=True)
        validation_raw = validation_raw.dropna().reset_index(drop=True)
        test_raw = test_raw.dropna().reset_index(drop=True)

        train_raw = mkMBP(train_raw)
        validation_raw = mkMBP(validation_raw)
        test_raw = mkMBP(test_raw)

        train_raw['Tg']=np.log(train_raw['Tg'])
        validation_raw['Tg']=np.log(validation_raw['Tg'])
        test_raw['Tg']=np.log(test_raw['Tg'])

        drop_cols=['sample','area']

        train_raw=train_raw.drop(columns=drop_cols)
        validation_raw=validation_raw.drop(columns=drop_cols)
        test_raw=test_raw.drop(columns=drop_cols)

        train_x,train_y=train_raw.drop(columns=['progress_DM']),train_raw['progress_DM']
        validation_x,validation_y=validation_raw.drop(columns=['progress_DM']),validation_raw['progress_DM']
        test_x,test_y=test_raw.drop(columns=['progress_DM']),test_raw['progress_DM']

        train_x['Cr'] = mk_eGFR_data(train_x)
        validation_x['Cr'] = mk_eGFR_data(validation_x)
        test_x['Cr'] = mk_eGFR_data(test_x)

        train_x[train_x.columns]=scaler.fit_transform(train_x)
        train_x = train_x.rename(columns = {'Cr':'eGFR'})
        validation_x[validation_x.columns]=scaler.transform(validation_x)
        validation_x = validation_x.rename(columns = {'Cr':'eGFR'})
        test_x[test_x.columns]=scaler.transform(test_x)
        test_x = test_x.rename(columns = {'Cr':'eGFR'})

        return (train_x,train_y),(validation_x,validation_y),(test_x,test_y)

    def train_func(net,optimizer1,optimizer2,warmup_lr_scheduler,step_lr_scheduler,criterion,warmup_epochs,total_epochs,ptr_set,tr_loader,val_data,test_data,data_set_num,val_results_df,test_results_df,file_name):
        early_count = 0
        early_auc = 0 
        epoch_tr_loss=[]
        epoch_tr_auc=[]
        epoch_tr_sen=[]
        epoch_tr_spe=[]   

        for epoch in range(total_epochs):
            net.train()
            batch_tr_loss=[]
            batch_tr_acc=[]
            batch_tr_sen=[]
            batch_tr_spe=[]
            batch_tr_auc=[]
            for i,data in enumerate(tr_loader):
                snp_data=data['snp'].to(device)
                cli_data=data['cli'].to(device)
                y=data['y'].to(device).view(-1,1)

                if epoch<warmup_epochs:
                    optimizer1.zero_grad()

                    output=net(snp_data, cli_data)
                    logit_output=torch.sigmoid(output)
                    acc,sen,spe,auc=model_performance_DNN(y.cpu().detach().numpy(),logit_output.cpu().detach().numpy(),0.5)
                    batch_tr_acc.append(acc)
                    batch_tr_sen.append(sen)
                    batch_tr_spe.append(spe)
                    batch_tr_auc.append(auc)

                    loss=criterion(output,y)
                    loss.backward()
                    optimizer1.step()
                    batch_tr_loss.append(loss.item())

                else:
                    optimizer2.zero_grad()

                    output=net(snp_data, cli_data)
                    logit_output=torch.sigmoid(output)
                    acc,sen,spe,auc=model_performance_DNN(y.cpu().detach().numpy(),logit_output.cpu().detach().numpy(),0.5)
                    batch_tr_acc.append(acc)
                    batch_tr_sen.append(sen)
                    batch_tr_spe.append(spe)
                    batch_tr_auc.append(auc)

                    loss=criterion(output,y)
                    loss.backward()
                    optimizer2.step()
                    batch_tr_loss.append(loss.item())


                if epoch<warmup_epochs:
                    warmup_lr_scheduler.step()
                else:
                    step_lr_scheduler.step(loss.item())
                writer.add_scalar('MAINTAIN_training loss',
                                loss.item(),
                                epoch + i)


            net.eval()

            tr_loss=np.mean(batch_tr_loss)
            tr_acc=np.mean(batch_tr_acc)
            tr_sen=np.mean(batch_tr_sen)
            tr_spe=np.mean(batch_tr_spe)
            tr_auc=np.mean(batch_tr_auc)
            epoch_tr_loss.append(tr_loss)
            epoch_tr_auc.append(tr_auc)
            epoch_tr_sen.append(tr_sen)
            epoch_tr_spe.append(tr_spe)

            print(f'tr_loss:{tr_loss:.4f} tr_acc:{tr_acc:.4f} tr_sen:{tr_sen:.4f} tr_spe:{tr_spe:.4f} tr_auc:{tr_auc:.4f} ')

            early_auc,early_count,stop_flag,val_results_df,test_results_df = val_test_func(net,val_data,test_data,epoch,early_count,
                                                                                           early_auc,val_results_df,test_results_df,data_set_num,
                                                                                           criterion,optimizer2,file_name)

            if stop_flag==True:
                return val_results_df,test_results_df
            else:
                continue

    def val_test_func(net,val_data,test_data,epoch,early_count,early_auc,val_results_df,test_results_df,data_set_num,criterion,optimizer,file_name):
        net.eval()
        EARLY = TRANSFER_EARLY
        print(data_set_num)
        vloss, vacc, vsen, vspe, vauc = validation_func(val_data,net,criterion,epoch,'VAL')
        writer.add_scalar('MAINTAIN_validation loss',
                                vloss,
                                epoch)
        writer.add_scalar('MAINTAIN_validation acc',
                                vacc,
                                epoch)
        writer.add_scalar('MAINTAIN_validation sen',
                                vsen,
                                epoch)
        writer.add_scalar('MAINTAIN_validation spe',
                                vspe,
                                epoch)
        writer.add_scalar('MAINTAIN_validation auc',
                                vauc,
                                epoch)
        print(f'val_loss:{vloss:.4f} val_acc:{vacc:.4f} val_sen:{vsen:.4f} val_spe:{vspe:.4f} val_auc:{vauc:.4f} ')
        tloss, tacc, tsen, tspe, tauc = validation_func(test_data,net,criterion,epoch,'TEST')
        print(f'ts_loss:{tloss:.4f} ts_acc:{tacc:.4f} ts_sen:{tsen:.4f} ts_spe:{tspe:.4f} ts_auc:{tauc:.4f} ')
        current_lr=optimizer.param_groups[0]['lr']
        print(f'E{epoch} : {current_lr}')
        stop_flag=False
        if epoch%300==0:
            print()
            clear_output(wait=True)
        if early_count == EARLY:
            print('early stopping!')
            stop_flag=True
            return early_auc,early_count,stop_flag,val_results_df,test_results_df
        if early_auc < vauc: # AUC 기준 모델 저장
            print(f'early auc {early_auc}')
            print(f'early cnt {early_count}')
            val_results_df.loc[data_set_num]=[vloss.item(),vacc,vsen,vspe,vauc]
            test_results_df.loc[data_set_num]=[tloss.item(),tacc,tsen,tspe,tauc]

            print('Epoch: '+str(epoch)+' save model by validation loss\n')
            Ep = 'Ep-'+str(epoch+1)
            # 모델 저장 이름 설정
            print(f'[OUTPUT] {file_name}')
            torch.save(net.state_dict(), file_name)

            early_count = 0     
            early_auc = vauc
            return early_auc,early_count,stop_flag,val_results_df,test_results_df
        else:
            print(' ')
            early_count = early_count + 1
            return early_auc,early_count,stop_flag,val_results_df,test_results_df

    def linspace(start, end, n):
        print(end)
        step = (end - start) / (n - 1)
        return [start + int(np.round(i * step)) for i in range(n)]

    save_path = MODEL_DIR / 'Transfer_learning'
    save_path.mkdir(parents=True, exist_ok=True)

    from itertools import product

    batch_size_list = TRANSFER_BATCH_SIZE_LIST
    random_state_list = TRANSFER_RANDOM_STATE_LIST
    learning_rate_list = TRANSFER_LEARNING_RATE_LIST
    SNP_num_list = TRANSFER_SNP_NUM_LIST
    Total_num_list = TRANSFER_TOTAL_NUM_LIST
    positive_weight_list = TRANSFER_POSITIVE_WEIGHT_LIST
    dropout_num_list = TRANSFER_DROPOUT_NUM_LIST
    activation_function_list = ['elu']


    combinations = list(product(batch_size_list,random_state_list,learning_rate_list,SNP_num_list,
                                Total_num_list,positive_weight_list,dropout_num_list,activation_function_list))

    print(len(combinations))

    val_results_df=[]
    test_results_df=[]
    warmup_epochs=TRANSFER_WARMUP_EPOCHS
    total_epochs=TRANSFER_TOTAL_EPOCHS
    used_columns = ['M','F', 'age', 'waist', 'bmi', 'MBP', 'DM_FH', 'htndiag', 'lipdiag', 'exercise',
           'drink', 'smoke', 'FBS', 'Hba1c', 'WBC', 'Tg', 'ALT', 'eGFR']
    for data_set_num in range(1):#range(100):
        print(f'Data set {data_set_num} start')
        file_path = DATA_DIR
        scaler=MinMaxScaler()
        train,validation,test=load_data(file_path,data_set_num,scaler)
        snp_col = list(train[0].columns[~train[0].columns.isin(used_columns)])
        tr_snp_x,tr_cli_x=train[0][snp_col],train[0][used_columns]
        observed_snp_feature_num = tr_snp_x.shape[1]
        if observed_snp_feature_num != SNP_INPUT_LEN:
            raise ValueError(
                f"SNP feature count mismatch: expected {SNP_INPUT_LEN}, got {observed_snp_feature_num}"
            )

        val_snp_x,val_cli_x=validation[0][snp_col],validation[0][used_columns]
        ts_snp_x,ts_cli_x=test[0][snp_col],test[0][used_columns]
        CLI_FEATURE_NUM=tr_cli_x.shape[1]

        tr_set={'snp':tr_snp_x.to_numpy(),'cli':tr_cli_x.to_numpy(),'y':train[1].to_numpy()}
        val_data={'snp':val_snp_x.to_numpy(),'cli':val_cli_x.to_numpy(),'y':validation[1].to_numpy()}
        test_data={'snp':ts_snp_x.to_numpy(),'cli':ts_cli_x.to_numpy(),'y':test[1].to_numpy()}
        train_set=GenerateData(tr_set)

        class_counts = train[1].value_counts().to_list()
        num_samples = sum(class_counts)
        labels = train[1].to_list()
        pos_weight=class_counts[0]/class_counts[1]
        #클래스별 가중치 부여
        class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))] 

        # 해당 데이터의 label에 해당되는 가중치
        weights = [class_weights[labels[i]] for i in range(int(num_samples))] #해당 레이블마다의 가중치 비율
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

        for hypernum in combinations:
            val_results_df_1=pd.DataFrame(columns=['loss','acc','sen','spe','auc'])
            test_results_df_1=pd.DataFrame(columns=['loss','acc','sen','spe','auc'])
            SNP_num=hypernum[3]
            Total_num=hypernum[4]
            drop_num = hypernum[6]
            acfunc = hypernum[7]

            SNP_fe_num=64
            Cli_fe_num=CLI_FEATURE_NUM
            Total_fe_num=SNP_fe_num+int(Cli_fe_num*1.0)

            Total_step=int(Total_fe_num/Total_num)

            Total_fe_lst=linspace(1, Total_fe_num, Total_num+1)
            Total_fe_lst.reverse()
            Total_fe_lst.pop()

            weighted_pos_weight=hypernum[5]

            pre_model=FCNetwork(PRE_FEATURE_NUM).to(device)
            pre_model.load_state_dict(torch.load(pretraining_path, weights_only=True))
            pre_model=pre_model.pre_part
            BATCH_SIZE=hypernum[0]
            random_seed=int(hypernum[1])
            lr_rate=hypernum[2]
            g = torch.Generator()
            g.manual_seed(123)
            tr_loader=DataLoader(train_set,batch_size=BATCH_SIZE, generator=g, sampler=sampler)
            file_name = save_path / f'best_model_elu_dataset-{data_set_num}_W-{weighted_pos_weight}_Batch-{BATCH_SIZE}_lr-{lr_rate}_layer-{SNP_num}-{Total_num}_dropout-{drop_num}_random-{random_seed}_act-{acfunc}.h5'
            print('start modeling')

            warm_up_steps=len(tr_loader)*warmup_epochs
            total_up_steps=len(tr_loader)*total_epochs

            seed_everything(seed=random_seed)
            net=Net1(SNP_fe_num, Cli_fe_num, Total_fe_num, pre_model)
            net.to(device)
            optimizer1=optim.AdamW(net.parameters(),lr=1e-06,weight_decay=0.0001)
            warmup_lr_scheduler=WarmUpLR(optimizer1,warm_up_steps,total_up_steps)
            optimizer2=optim.AdamW(net.parameters(),lr=lr_rate,weight_decay=0.001)
            step_lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer2,'min',patience=20, factor=0.5)
            criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weighted_pos_weight))
            val_results_df_1,test_results_df_1=train_func(net,optimizer1,optimizer2,warmup_lr_scheduler,step_lr_scheduler,criterion,warmup_epochs,total_epochs,tr_set,tr_loader,val_data,test_data,data_set_num,val_results_df_1,test_results_df_1,file_name)
            writer.flush()
            val_results_df.append(val_results_df_1)
            test_results_df.append(test_results_df_1)
            writer.close()
    val_results_df = pd.concat(val_results_df)
    test_results_df = pd.concat(test_results_df)
    print(f"[OUTPUT] {save_path / 'validation_performance.csv'}")
    val_results_df.to_csv(save_path / "validation_performance.csv", index=False)
    print(f"[OUTPUT] {save_path / 'test_performance.csv'}")
    test_results_df.to_csv(save_path / "test_performance.csv", index=False)


if __name__ == '__main__':
    main()
