
# V2 app (Segmentasyon + Churn + Kampanya)
import pandas as pd, numpy as np, streamlit as st
from datetime import timedelta
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:
    LogisticRegression=None

st.set_page_config(page_title="V2", layout="wide")
st.title("V2 — Segmentasyon + Churn + Kampanya")

with st.sidebar:
    page = st.radio("Bölüm", ["📤 Veri Yükle (CRM)","👥 Segmentasyon (RFM)","📉 Churn Skoru","🎯 Kampanya Önerileri"])

@st.cache_data
def load_csv(f): return pd.read_csv(f)

def rfm_table(orders_df, now=None):
    now = now or pd.to_datetime(orders_df["date"]).max() + pd.Timedelta(days=1)
    tmp = orders_df.copy(); tmp["date"]=pd.to_datetime(tmp["date"])
    g = tmp.groupby("customer_id").agg(R=("date", lambda s:(now-s.max()).days), F=("date","count"), M=("order_value","sum")).reset_index()
    g["R_q"]=pd.qcut(g["R"],5,labels=[5,4,3,2,1]).astype(int)
    g["F_q"]=pd.qcut(g["F"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    g["M_q"]=pd.qcut(g["M"],5,labels=[1,2,3,4,5]).astype(int)
    g["RFM_score"]=g["R_q"]*100+g["F_q"]*10+g["M_q"]
    return g

def label_churn(orders_df, window_days=60):
    last_date=pd.to_datetime(orders_df["date"]).max()
    cutoff=last_date-pd.Timedelta(days=window_days)
    lp=orders_df.groupby("customer_id")["date"].max().reset_index()
    lp["date"]=pd.to_datetime(lp["date"]); lp["churned"]=(lp["date"]<cutoff).astype(int)
    return lp[["customer_id","churned"]]

def simple_churn_model(feat):
    X=feat[["R","F","M"]].copy(); y=feat["churned"]
    if LogisticRegression is None or y.nunique()<2 or len(X)<50:
        r=(X["R"]-X["R"].min())/(X["R"].max()-X["R"].min()+1e-9)
        f=(X["F"]-X["F"].min())/(X["F"].max()-X["F"].min()+1e-9)
        m=(X["M"]-X["M"].min())/(X["M"].max()-X["M"].min()+1e-9)
        p=0.65*r+0.2*(1-f)+0.15*(1-m)
        return p.clip(0.01,0.99).values
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(X); Xs=sc.transform(X)
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(max_iter=500).fit(Xs,y)
    return clf.predict_proba(Xs)[:,1]

def choose_campaign(row):
    p=row["churn_prob"]; m=row["M"]
    if p>0.8 and m>800: return "KAMP15E"
    if p>0.7: return "CASH50"
    if p>0.6: return "KARGO"
    return "KAMP10"

if page=="📤 Veri Yükle (CRM)":
    c1,c2=st.columns(2)
    with c1:
        f1=st.file_uploader("customers.csv",type=["csv"],key="c")
        f2=st.file_uploader("orders_history.csv",type=["csv"],key="o")
    with c2:
        f3=st.file_uploader("interactions.csv",type=["csv"],key="i")
        f4=st.file_uploader("campaigns.csv",type=["csv"],key="p")
    if f1 and f2 and f3 and f4:
        st.session_state["customers"]=load_csv(f1)
        st.session_state["orders"]=load_csv(f2)
        st.session_state["interactions"]=load_csv(f3)
        st.session_state["campaigns"]=load_csv(f4)
        st.success("Veriler yüklendi.")
        st.dataframe(st.session_state["customers"].head())
        st.dataframe(st.session_state["orders"].head())

if page=="👥 Segmentasyon (RFM)":
    if "orders" not in st.session_state: st.warning("Önce CRM dosyalarını yükleyin.")
    else:
        rfm=rfm_table(st.session_state["orders"]); st.session_state["rfm"]=rfm
        st.dataframe(rfm.head(30))
        st.download_button("⬇️ RFM (CSV)", data=rfm.to_csv(index=False), file_name="rfm_scores.csv")

if page=="📉 Churn Skoru":
    if "orders" not in st.session_state: st.warning("Önce CRM dosyalarını yükleyin.")
    else:
        rfm=st.session_state.get("rfm") or rfm_table(st.session_state["orders"])
        y=label_churn(st.session_state["orders"],60)
        feat=rfm.merge(y,on="customer_id",how="left").fillna({"churned":0})
        feat["churn_prob"]=simple_churn_model(feat)
        st.session_state["churn_features"]=feat
        st.dataframe(feat.sort_values("churn_prob",ascending=False).head(30))
        st.download_button("⬇️ Churn (CSV)", data=feat.to_csv(index=False), file_name="churn_scores.csv")

if page=="🎯 Kampanya Önerileri":
    if "churn_features" not in st.session_state or "campaigns" not in st.session_state:
        st.warning("Önce CRM dosyalarını yükleyin ve churn skorlarını hesaplayın.")
    else:
        feat=st.session_state["churn_features"].copy()
        feat["campaign_id"]=feat.apply(choose_campaign,axis=1)
        out=feat[["customer_id","R","F","M","churn_prob","campaign_id"]].sort_values("churn_prob",ascending=False).head(200)
        st.dataframe(out.head(30))
        st.download_button("⬇️ Kampanya Atamaları (CSV)", data=out.to_csv(index=False), file_name="campaign_assignments.csv")
