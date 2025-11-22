import pandas as pd
import numpy as np
from ML_rank import Config
import xgboost as xgb
import os
import pickle
class MutantPrior:
    def __init__(self, select_ratio,file_path):
        self.select_ratio=select_ratio
        self.file_path=file_path
        self.selected_mutant_name=set()
        self.constantColumns=Config.constantColumns

    def process(self):
        data = pd.read_csv(self.file_path)
        data = self.fix_feat_loss_func(data)
        df=self.dataPreprocess(data)
        X_test, mut_info_df = self.split_target(df)
        dtest = xgb.DMatrix(X_test)
        # Load XGBoost model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        best_model_dir = os.path.join(current_dir,'best_model', 'best_model_rank', 'xgb_30.pkl')
        with open(best_model_dir, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(dtest)
        res_df = self.concat_df(y_pred, mut_info_df)
        res_df = res_df.sort_values(by='y_pred', ascending=False)
        self.df_to_mutant_set(res_df)

    def dataPreprocess(self,data):
        # Handle missing values
        df_filled = data.fillna(0)
        # Handle outliers
        df_filled = self.deal_abnormal_value(df_filled)
        # Remove constant columns
        df_filled = df_filled.drop(columns=self.constantColumns, errors='ignore')
        return df_filled


    def fix_feat_loss_func(self,data):
        value_to_find = 'ListWrapper([<function mean_squared_error at 0x0000023F51871310>])'
        value_to_find2 = 'CategoricalCrossentropy'
        value_to_find3 = 'MeanSquaredError'
        value_to_find4 = 'BinaryCrossentropy'
        data.loc[data["loss_func"] == value_to_find, "loss_func"] = 'mean_squared_error'
        data.loc[data["loss_func"] == value_to_find2, "loss_func"] = 'categorical_crossentropy'
        data.loc[data["loss_func"] == value_to_find3, "loss_func"] = 'mean_squared_error'
        data.loc[data["loss_func"] == value_to_find4, "loss_func"] = 'mean_squared_error'
        return data

    def deal_abnormal_value(self,df):
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Use clip to limit values within the float32 range
        df = df.replace([np.inf, -np.inf], [float32_max, float32_min])  # Replace positive and negative infinity
        df[numeric_cols] = df[numeric_cols].clip(lower=float32_min, upper=float32_max)
        return df

    def split_target(self,df):
        mut_info_df = df[['model_name', 'layer_id', 'neuron_idx', 'mutant_oper']]
        df = pd.get_dummies(df)
        X = df.drop(columns=['model_name'])
        return X, mut_info_df

    def concat_df(self,y_pred, mut_info_df):
        arr_df = pd.DataFrame({'y_pred': y_pred})
        result = pd.concat([mut_info_df, arr_df], axis=1)
        return result

    def df_to_mutant_set(self,df):
        top_k_rows = int(len(df) * self.select_ratio)
        for _, row in df.iloc[:top_k_rows].iterrows():
            layer_str=str(row['layer_id'])
            neuron_str=str(row['neuron_idx'])
            mutant_oper_str=row['mutant_oper']
            mutant_name_str='L'+layer_str+'-N'+neuron_str+'-'+mutant_oper_str+'model.h5'
            self.selected_mutant_name.add(mutant_name_str)

    def getMutantSet(self):
        return self.selected_mutant_name


