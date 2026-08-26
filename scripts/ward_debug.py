import pickle
import pandas as pd
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_P=ROOT/'models'/'lightgbm_model.pkl'
DATA_P=ROOT/'data'/'processed'/'trustloop'/'model_ready.csv'
model=pickle.load(open(MODEL_P,'rb'))
df=pd.read_csv(DATA_P)
# payload same as test
payload={'case_id':'TEST-WARD-001','customer_id':'CUST-WARD-001','order_id':'ORD-WARD-001','age':30,'account_age_days':200,'customer_segment':'Gold','country':'US','platform':'Mobile App','device_type':'iPhone','payment_method':'Credit Card','product_category':'Clothing','avg_order_value_usd':80.0,'is_high_value_item':0,'discount_used':1,'days_to_return':10,'return_reason':'Too Small','shipping_carrier':'FedEx','multiple_accounts_flag':0,'wishlist_to_cart_time_hrs':0.5,'customer_return_count_prior':4,'returns_last_30d_prior':2,'returns_last_90d_prior':5,'total_returns_lifetime_prior':9,'order_date':'2026-06-20','return_date':'2026-06-30','item_condition':'new','refund_amount':80.0}
# build features like ml_feature_builder
from datetime import datetime
def parse_date(v):
    if v is None: return None
    try:
        return pd.to_datetime(v)
    except:
        return None

def date_features(prefix, v):
    dt=parse_date(v)
    if dt is None:
        return {f"{prefix}_year":0,f"{prefix}_month":0,f"{prefix}_day":0,f"{prefix}_dayofweek":0,f"{prefix}_dayofyear":0,f"{prefix}_is_weekend":0}
    return {f"{prefix}_year":int(dt.year),f"{prefix}_month":int(dt.month),f"{prefix}_day":int(dt.day),f"{prefix}_dayofweek":int(dt.weekday()),f"{prefix}_dayofyear":int(dt.timetuple().tm_yday),f"{prefix}_is_weekend":int(dt.weekday()>=5)}

features={}
# defaults
defaults={"age":0,"account_age_days":0,"customer_segment":"unknown","country":"unknown","platform":"unknown","device_type":"unknown","payment_method":"unknown","product_category":"unknown","avg_order_value_usd":0.0,"is_high_value_item":0,"discount_used":0,"days_to_return":0.0,"return_reason":"unknown","shipping_carrier":"unknown","multiple_accounts_flag":0,"wishlist_to_cart_time_hrs":0.0,"customer_return_count_prior":0,"returns_last_30d_prior":0,"returns_last_90d_prior":0,"total_returns_lifetime_prior":0}
for k,defv in defaults.items():
    v=payload.get(k,defv)
    if v is None: v=defv
    features[k]=v
features.update(date_features('order_date', payload.get('order_date')))
features.update(date_features('return_date', payload.get('return_date')))
# calculated days
od=parse_date(payload.get('order_date'))
rd=parse_date(payload.get('return_date'))
if od is not None and rd is not None:
    features['calculated_days_to_return']=(rd-od).total_seconds()/86400.0
else:
    features['calculated_days_to_return']=float(features.get('days_to_return',0.0))
# build DF
MODEL_FEATURES = [
    "age","account_age_days","customer_segment","country","platform","device_type","payment_method","product_category","avg_order_value_usd","is_high_value_item","discount_used","days_to_return","return_reason","shipping_carrier","multiple_accounts_flag","wishlist_to_cart_time_hrs","customer_return_count_prior","returns_last_30d_prior","returns_last_90d_prior","total_returns_lifetime_prior","order_date_year","order_date_month","order_date_day","order_date_dayofweek","order_date_dayofyear","order_date_is_weekend","return_date_year","return_date_month","return_date_day","return_date_dayofweek","return_date_dayofyear","return_date_is_weekend","calculated_days_to_return"]
Xpd=pd.DataFrame([[features[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
# set categorical columns to category with categories from training (derive from df train portion)
cat_cols=["country","customer_segment","device_type","payment_method","platform","product_category","return_reason","shipping_carrier"]
# build training mappings
df_sorted = df.sort_values(by=['return_date', 'order_date']).reset_index(drop=True)
train_df = df_sorted.iloc[:int(len(df_sorted)*0.7)]
mappings = {}
for c in cat_cols:
    if c in train_df.columns:
        vals = train_df[c].fillna('unknown').astype(str).tolist()
        seen = []
        uniq = []
        for v in vals:
            if v not in seen:
                seen.append(v)
                uniq.append(v)
        mappings[c] = uniq
# apply categories
for c in cat_cols:
    if c in Xpd.columns:
        Xpd[c] = Xpd[c].astype(str)
        if c in mappings:
            Xpd[c] = pd.Categorical(Xpd[c], categories=mappings[c])
# coerce numerics
for col in Xpd.columns:
    if col not in cat_cols:
        Xpd[col] = pd.to_numeric(Xpd[col], errors='coerce')
print('Xpd dtypes')
print(Xpd.dtypes)
print('Xpd values')
print(Xpd.to_dict(orient='records'))
# predict
print('model predict_proba for payload')
print(model.predict_proba(Xpd))
print('model predict for payload')
print(model.predict(Xpd))
# compare numeric features to ward examples
ward = df[df['abuse_label'] == 3].copy()
for c in ['order_date', 'return_date']:
    s_dt = pd.to_datetime(ward[c], errors='coerce')
    ward[f"{c}_year"] = s_dt.dt.year
    ward[f"{c}_month"] = s_dt.dt.month
    ward[f"{c}_day"] = s_dt.dt.day
    ward[f"{c}_dayofweek"] = s_dt.dt.dayofweek
    ward[f"{c}_dayofyear"] = s_dt.dt.dayofyear
    ward[f"{c}_is_weekend"] = (s_dt.dt.dayofweek >= 5).astype(int)
if 'order_date' in ward.columns and 'return_date' in ward.columns:
    ward['calculated_days_to_return'] = (pd.to_datetime(ward['return_date']) - pd.to_datetime(ward['order_date'])).dt.total_seconds() / 86400.0
ward=ward.drop(columns=[c for c in ['order_date','return_date'] if c in ward.columns])
cmp_feats=['age','account_age_days','is_high_value_item','customer_return_count_prior','total_returns_lifetime_prior','calculated_days_to_return']
wnum=ward[cmp_feats].fillna(0).astype(float)
pnum=Xpd[[c for c in MODEL_FEATURES if c in cmp_feats]].astype(float).iloc[0]
dists=((wnum-pnum)**2).sum(axis=1)
nearest=dists.nsmallest(5).index.tolist()
print('nearest ward indices', nearest)
print('nearest ward numeric', wnum.loc[nearest].to_dict(orient='records'))
