def main():
    # -*- coding: utf-8 -*-

    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = BASE_DIR / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, SNP_FEATURE_NUM as SNP_INPUT_LEN
    from modeling_common import FCNetwork, check_correct, mkMBP, mk_eGFR_data, seed_everything

    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    from torch.utils.data import Dataset,DataLoader
    from torch import autograd
    from torch.autograd import Variable
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    from datetime import datetime
    import datetime as dt
    import glob
    import os
    import re
    import argparse
    import time

    def log_paths(title, paths):
        print(title)
        for p in paths:
            print(f"  {p}")

    log_paths("Input files:", [
        DATA_DIR / "Train.csv",
        DATA_DIR / "Validation.csv",
        DATA_DIR / "Test.csv",
        DATA_DIR / "Only_clinical.csv",
        DATA_DIR / "SHAP_total_koges_sample.csv",
        DATA_DIR / "SHAP_total_koges.csv",
        DATA_DIR / "sample_area_mapping.csv",
    ])
    log_paths("Output dirs:", [OUTPUT_DIR, MODEL_DIR])

    seed_everything(42)

    def latest_pretrain_path(pretrain_dir):
        pretrain_dir = Path(pretrain_dir)
        candidates = list(pretrain_dir.glob('best_model_*.h5'))
        if not candidates:
            raise FileNotFoundError(f'No pretrain models found in {pretrain_dir}')
        def epoch_num(p):
            m = re.search(r'best_model_(\d+)\.h5$', p.name)
            return int(m.group(1)) if m else -1
        return max(candidates, key=epoch_num)

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    print(f"[INFO] device: {device}")

    def validation_func(data,model,criterion, cutoff=0.5):
        snp=data['snp']
        cli=data['cli']
        y=data['y']

        tensor_snp=torch.from_numpy(snp).float().to(device)
        tensor_cli=torch.from_numpy(cli).float().to(device)
        tensor_y=torch.from_numpy(y).float().to(device).view(-1,1)

        output=model(tensor_snp, tensor_cli)
        logit_output=torch.sigmoid(output)
        val_loss=criterion(output,tensor_y)
        acc,sen,spe,auc,f1,f2=model_performance_DNN(y,logit_output.cpu().detach().numpy(),cutoff)

        val_loss = val_loss.cpu().detach().numpy()

        return val_loss, acc, sen, spe, auc,f1,f2, logit_output.cpu().detach().numpy()

    # Function for DNN performance
    def model_performance_DNN(real,pred,cutoff):

        pred_label = np.where(pred>cutoff, 1, 0) # relation to sigmoid activation function
        f1 = f1_score(real, pred_label)
        f2 = fbeta_score(real, pred_label, beta=2)
        accuracy, sensitivity, specificity=check_correct(pred_label,real)
        auc = roc_auc_score(real, pred)

        df_hypo=pd.DataFrame(pred) # prediction probability
        df_hypo.columns=['hypothesis 1']

        df_pred=pd.DataFrame(pred_label) # zero or one 
        df_pred.columns=['prediction']

        df_y=pd.DataFrame(real)
        df_y.columns=['y']

        pred_result=pd.concat([df_y,df_hypo, df_pred],axis=1)

        return accuracy, sensitivity, specificity, auc, f1, f2

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

    def linspace(start, end, n):
        print(end)
        step = (end - start) / (n - 1)
        return [start + int(np.round(i * step)) for i in range(n)]

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

    PRE_FEATURE_NUM=18
    pre_model=FCNetwork(PRE_FEATURE_NUM)
    pretraining_path = latest_pretrain_path(MODEL_DIR / 'Pretrain')
    print(f"[INPUT] {pretraining_path}")

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

        sample = pd.concat([train_raw[['sample']],validation_raw[['sample']],test_raw[['sample']]])
        trsample, valsample, tssample = train_raw[['sample','area']],validation_raw[['sample','area']],test_raw[['sample','area']]

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

        return (train_x,train_y),(validation_x,validation_y),(test_x,test_y),sample,scaler, trsample, valsample, tssample

    def load_data2(file_path,idx,scaler):
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

        sample = pd.concat([train_raw[['sample']],validation_raw[['sample']],test_raw[['sample']]])

        train_raw=train_raw.drop(columns=drop_cols)
        validation_raw=validation_raw.drop(columns=drop_cols)
        test_raw=test_raw.drop(columns=drop_cols)

        train_x,train_y=train_raw.drop(columns=['progress_DM']),train_raw['progress_DM']
        validation_x,validation_y=validation_raw.drop(columns=['progress_DM']),validation_raw['progress_DM']
        test_x,test_y=test_raw.drop(columns=['progress_DM']),test_raw['progress_DM']

        train_x['Cr'] = mk_eGFR_data(train_x)
        validation_x['Cr'] = mk_eGFR_data(validation_x)
        test_x['Cr'] = mk_eGFR_data(test_x)

        train_x[train_x.columns]=scaler.transform(train_x)
        train_x = train_x.rename(columns = {'Cr':'eGFR'})
        validation_x[validation_x.columns]=scaler.transform(validation_x)
        validation_x = validation_x.rename(columns = {'Cr':'eGFR'})
        test_x[test_x.columns]=scaler.transform(test_x)
        test_x = test_x.rename(columns = {'Cr':'eGFR'})

        return (train_x,train_y),(validation_x,validation_y),(test_x,test_y),sample

    from itertools import product

    batch_size_list = [1024]
    random_state_list = [123]
    learning_rate_list = [1e-02]
    SNP_num_list = [3]
    Total_num_list = [3]
    positive_weight_list = [1.7]
    dropout_num_list = [0.5]
    activation_function_list = ['elu']


    combinations = list(product(batch_size_list,random_state_list,learning_rate_list,SNP_num_list,
                                Total_num_list,positive_weight_list,dropout_num_list,activation_function_list))

    print(len(combinations))

    # 디렉터리 경로 지정
    directory_path = MODEL_DIR / 'Transfer_learning'

    matching_files = glob.glob(str(directory_path / 'best_model_elu_*'))

    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
    val_perf = []
    test_perf = []
    used_columns = ['M','F', 'age', 'waist', 'bmi', 'MBP', 'DM_FH', 'htndiag', 'lipdiag', 'exercise',
           'drink', 'smoke', 'FBS', 'Hba1c', 'WBC', 'Tg', 'ALT', 'eGFR']
    for model_num in [1]: 
        file_path = DATA_DIR
        scaler=MinMaxScaler()
        train,validation,test, sample, scaler, trsample, valsample, tssample = load_data(file_path,model_num,scaler)
        snp_col = list(train[0].columns[~train[0].columns.isin(used_columns)])
        tr_snp_x,tr_cli_x=train[0][snp_col],train[0][used_columns]
        val_snp_x,val_cli_x=validation[0][snp_col],validation[0][used_columns]
        ts_snp_x,ts_cli_x=test[0][snp_col],test[0][used_columns]
        observed_snp_feature_num = tr_snp_x.shape[1]
        if observed_snp_feature_num != SNP_INPUT_LEN:
            raise ValueError(
                f"SNP feature count mismatch: expected {SNP_INPUT_LEN}, got {observed_snp_feature_num}"
            )
        CLI_FEATURE_NUM=tr_cli_x.shape[1]


        for hypernum in matching_files:
            print(hypernum)
            SNP_num=int(hypernum.split('layer-')[-1].split('-')[0])
            Total_num=int(hypernum.split('layer-')[-1].split('-')[1].split('_')[0])
            acfunc = hypernum.split('-')[-1].split('.')[0]
            drop_num = float(hypernum.split('-')[-3].split('_')[0])
            SNP_fe_num=64
            Cli_fe_num=CLI_FEATURE_NUM
            Total_fe_num=64+int(Cli_fe_num*1.0)

            Total_step=int(Total_fe_num/Total_num)

            Total_fe_lst=linspace(1, Total_fe_num, Total_num+1)
            Total_fe_lst.reverse()
            Total_fe_lst.pop()

            weight_match = re.search(r'W-([0-9]*\.?[0-9]+)', Path(hypernum).name)
            if not weight_match:
                raise ValueError(f"Unable to parse pos_weight from checkpoint name: {hypernum}")
            weighted_pos_weight = float(weight_match.group(1))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weighted_pos_weight))

            pre_model=FCNetwork(PRE_FEATURE_NUM).to(device)
            pre_model.load_state_dict(torch.load(pretraining_path, weights_only=True))
            pre_model=pre_model.pre_part

            models = Net1(SNP_fe_num, Cli_fe_num, Total_fe_num, pre_model).to(device)
            models.load_state_dict(torch.load(hypernum, weights_only=True))
            models.eval()

            tr_set={'snp':tr_snp_x.to_numpy(),'cli':tr_cli_x.to_numpy(),'y':train[1].to_numpy()}
            val_data={'snp':val_snp_x.to_numpy(),'cli':val_cli_x.to_numpy(),'y':validation[1].to_numpy()}
            test_data={'snp':ts_snp_x.to_numpy(),'cli':ts_cli_x.to_numpy(),'y':test[1].to_numpy()}

            trloss, tracc, trsen, trspe, trauc, trf1, trf2, troutput = validation_func(tr_set,models,criterion,cutoff=0.5)
            vloss, vacc, vsen, vspe, vauc, vf1, vf2, voutput = validation_func(val_data,models,criterion,cutoff=0.5)
            tloss, tacc, tsen, tspe, tauc, tf1, tf2, toutput= validation_func(test_data,models,criterion,cutoff=0.5)

            val_perf.append([hypernum,vloss, vacc, vsen, vspe, vauc, vf1, vf2])
            test_perf.append([hypernum,tloss, tacc, tsen, tspe, tauc, tf1, tf2])

            print(model_num, 'vauc: ',vauc)

    val_df = pd.DataFrame(val_perf,columns = ['model','loss','acc','sen','spe','auc','f1','f2'])
    test_df = pd.DataFrame(test_perf,columns = ['model','loss','acc','sen','spe','auc','f1','f2'])

    val_df.sort_values(by='auc',ascending=False)

    test_df.sort_values(by='auc',ascending=False)

    val_num = 0
    val_df.iloc[[val_num],:]

    tr3 = pd.read_csv(DATA_DIR / "Train.csv")
    val3 = pd.read_csv(DATA_DIR / "Validation.csv")
    ts3 = pd.read_csv(DATA_DIR / "Test.csv")
    total3 = pd.concat([tr3,val3,ts3]).reset_index(drop=True)

    total3 = mkMBP(total3)
    total3['Tg']=np.log(total3['Tg'])
    total3['Cr'] = mk_eGFR_data(total3)

    total3 = total3[total3['area'] != 'SN'].reset_index(drop=True)

    len(total3['sample'].drop_duplicates())

    total3['progress_DM'].value_counts()

    file_path = DATA_DIR

    train,validation,test, sample = load_data2(file_path,model_num,scaler)
    snp_col = list(train[0].columns[~train[0].columns.isin(used_columns)])
    tr_snp_x,tr_cli_x=train[0][snp_col],train[0][used_columns]
    val_snp_x,val_cli_x=validation[0][snp_col],validation[0][used_columns]
    ts_snp_x,ts_cli_x=test[0][snp_col],test[0][used_columns]

    torch_data_tr = torch.from_numpy(tr_snp_x.values).to(device).float()
    torch_data_tr2 = torch.from_numpy(tr_cli_x.values).to(device).float()

    torch_data_val = torch.from_numpy(val_snp_x.values).to(device).float()
    torch_data_val2 = torch.from_numpy(val_cli_x.values).to(device).float()

    torch_data_ts = torch.from_numpy(ts_snp_x.values).to(device).float()
    torch_data_ts2 = torch.from_numpy(ts_cli_x.values).to(device).float()

    snp = pd.concat([tr_snp_x,val_snp_x,ts_snp_x]).reset_index(drop=True)
    cli = pd.concat([tr_cli_x,val_cli_x,ts_cli_x]).reset_index(drop=True)

    torch_data_total = torch.from_numpy(snp.values).to(device).float()
    torch_data_total2 = torch.from_numpy(cli.values).to(device).float()

    import shap
    explainer_shap_total = shap.GradientExplainer(models, [torch_data_tr, torch_data_tr2])
    explainer_shap_ts = shap.GradientExplainer(models, [torch_data_tr, torch_data_tr2])

    # start = time.time()

    shap_total_path = OUTPUT_DIR / "SHAP_total_koges.csv"
    shap_total_sample_path = OUTPUT_DIR / "SHAP_total_koges_sample.csv"

    if shap_total_path.exists() and shap_total_sample_path.exists():
        print(f"[SKIP] Found existing SHAP outputs:")
        print(f"  {shap_total_path}")
        print(f"  {shap_total_sample_path}")
        total_shap = pd.read_csv(shap_total_sample_path)
    else:
        start = time.time()
        shap_values_total_data = explainer_shap_total.shap_values([torch_data_total, torch_data_total2])
        print(time.time() - start)

        shap_values_total_snp = pd.DataFrame(shap_values_total_data[0])
        shap_values_total_cli = pd.DataFrame(shap_values_total_data[1])
        shap_values_total_snp.columns = snp.columns
        shap_values_total_cli.columns = cli.columns

        total_shap = pd.concat([shap_values_total_snp,shap_values_total_cli],axis=1)
        print(f"[OUTPUT] {shap_total_path}")
        total_shap.to_csv(shap_total_path, index=False)
        total_shap['sample'] = sample['sample'].values
        print(f"[OUTPUT] {shap_total_sample_path}")
        total_shap.to_csv(shap_total_sample_path, index=False)

    total_shap = pd.read_csv(shap_total_sample_path)

    raw_data_path = DATA_DIR / "sample_area_mapping.csv"
    if not raw_data_path.exists():
        print(f"[WARN] Missing file: {raw_data_path}")
        print("[WARN] Skipping ANAS merge/export section.")
        raw_data0319 = None
    else:
        raw_data0319 = pd.read_csv(raw_data_path)

    if raw_data0319 is not None:
        raw_data0319 = raw_data0319[['dist_id','area']]
        raw_data0319 = raw_data0319.rename(columns = {'dist_id':'sample'})

        ANAS_shap = total_shap.merge(raw_data0319)

        ANAS_shap = ANAS_shap[(ANAS_shap['area'] == 'AS')|(ANAS_shap['area'] == 'AN')].reset_index(drop=True)

        print(f"[OUTPUT] {OUTPUT_DIR / 'SHAP_total_ANAS.csv'}")
        ANAS_shap.drop(columns = ['area']).to_csv(OUTPUT_DIR / "SHAP_total_ANAS.csv",index=False)

    tr3 = pd.read_csv(DATA_DIR / "Train.csv")
    val3 = pd.read_csv(DATA_DIR / "Validation.csv")
    ts3 = pd.read_csv(DATA_DIR / "Test.csv")
    total3 = pd.concat([tr3,val3,ts3]).reset_index(drop=True)

    total3 = mkMBP(total3)
    total3['Tg']=np.log(total3['Tg'])
    total3['Cr'] = mk_eGFR_data(total3)

    file_path = DATA_DIR / 'SNUH'

    train,validation,test, sample = load_data2(file_path,model_num,scaler)
    snp_col = list(train[0].columns[~train[0].columns.isin(used_columns)])
    tr_snp_x,tr_cli_x=train[0][snp_col],train[0][used_columns]
    val_snp_x,val_cli_x=validation[0][snp_col],validation[0][used_columns]
    ts_snp_x,ts_cli_x=test[0][snp_col],test[0][used_columns]

    torch_data_tr = torch.from_numpy(tr_snp_x.values).to(device).float()
    torch_data_tr2 = torch.from_numpy(tr_cli_x.values).to(device).float()

    torch_data_val = torch.from_numpy(val_snp_x.values).to(device).float()
    torch_data_val2 = torch.from_numpy(val_cli_x.values).to(device).float()

    torch_data_ts = torch.from_numpy(ts_snp_x.values).to(device).float()
    torch_data_ts2 = torch.from_numpy(ts_cli_x.values).to(device).float()

    snp = pd.concat([tr_snp_x,val_snp_x,ts_snp_x]).reset_index(drop=True)
    cli = pd.concat([tr_cli_x,val_cli_x,ts_cli_x]).reset_index(drop=True)

    torch_data_total = torch.from_numpy(snp.values).to(device).float()
    torch_data_total2 = torch.from_numpy(cli.values).to(device).float()

    import shap
    explainer_shap_total = shap.GradientExplainer(models, [torch_data_tr, torch_data_tr2])
    explainer_shap_ts = shap.GradientExplainer(models, [torch_data_tr, torch_data_tr2])

    shap_snu_path = OUTPUT_DIR / "SHAP_total_SNUH.csv"
    shap_snu_sample_path = OUTPUT_DIR / "SHAP_total_SNUH_sample.csv"

    if shap_snu_path.exists() and shap_snu_sample_path.exists():
        print(f"[SKIP] Found existing SHAP outputs:")
        print(f"  {shap_snu_path}")
        print(f"  {shap_snu_sample_path}")
    else:
        start = time.time()
        shap_values_total_data = explainer_shap_total.shap_values([torch_data_total, torch_data_total2])
        print(time.time() - start)

        shap_values_total_snp = pd.DataFrame(shap_values_total_data[0])
        shap_values_total_cli = pd.DataFrame(shap_values_total_data[1])
        shap_values_total_snp.columns = snp.columns
        shap_values_total_cli.columns = cli.columns

        total_shap = pd.concat([shap_values_total_snp,shap_values_total_cli],axis=1)
        print(f"[OUTPUT] {shap_snu_path}")
        total_shap.to_csv(shap_snu_path, index=False)
        total_shap['sample'] = sample['sample'].values
        print(f"[OUTPUT] {shap_snu_sample_path}")
        total_shap.to_csv(shap_snu_sample_path, index=False)

    # (Plotting section removed)


if __name__ == '__main__':
    main()
