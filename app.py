import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Space Debris XAI", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:2rem}
.title{text-align:center;font-size:42px;font-weight:800}
.subtitle{text-align:center;color:#777;font-size:18px;margin-bottom:30px}
.card{padding:18px;border-radius:14px;border:1px solid #ddd;margin-bottom:12px}
.result{text-align:center;padding:25px;border-radius:14px;border:1px solid #ddd}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n=10000
    distance=np.random.uniform(1,1000,n)
    velocity=np.random.uniform(.1,15,n)
    size=np.random.uniform(1,20,n)
    risk=(1/distance)*100 + velocity*2 + size*3
    norm=(risk-risk.min())/(risk.max()-risk.min())
    y=np.where(norm<=np.percentile(norm,33.33),0,
       np.where(norm<=np.percentile(norm,66.67),1,2))
    X=pd.DataFrame({"distance_km":distance,"relative_velocity_kms":velocity,"combined_size":size})
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.22,random_state=42,stratify=y)
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
    model=RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42)
    model.fit(Xtr,ytr)
    return model,scaler,accuracy_score(yte,model.predict(Xte))

model,scaler,accuracy=train_model()

st.markdown('<div class="title">🛰️ Space Debris Risk Prediction</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable AI system for satellite conjunction risk analysis</div>',unsafe_allow_html=True)

st.sidebar.header("⚙️ Prediction Settings")
distance=st.sidebar.slider("📏 Distance between objects (km)",1.0,1000.0,100.0,1.0)
velocity=st.sidebar.slider("⚡ Relative velocity (km/s)",.1,15.0,5.0,.1)
size=st.sidebar.slider("📦 Combined object size",1.0,20.0,5.0,.1)
run=st.sidebar.button("🚀 Analyze Risk",use_container_width=True)

a,b,c,d=st.columns(4)
for col,title,value in [(a,"MODEL","Random Forest"),(b,"MODEL ACCURACY",f"{accuracy*100:.2f}%"),(c,"FEATURES","3"),(d,"CLASSES","3")]:
    with col:
        st.markdown(f'<div class="card"><small>{title}</small><h3>{value}</h3></div>',unsafe_allow_html=True)

if run:
    inp=pd.DataFrame({"distance_km":[distance],"relative_velocity_kms":[velocity],"combined_size":[size]})
    p=model.predict(scaler.transform(inp))[0]
    probs=model.predict_proba(scaler.transform(inp))[0]
    names={0:"Low",1:"Medium",2:"High"}
    risk=names[p]
    emoji={"Low":"🟢","Medium":"🟡","High":"🔴"}[risk]
    conf=probs[p]*100

    st.markdown(f'<div class="result"><h1>{emoji} {risk} Risk</h1><h3>Prediction confidence: {conf:.2f}%</h3></div>',unsafe_allow_html=True)
    left,right=st.columns(2)

    with left:
        st.subheader("📋 Conjunction Parameters")
        st.dataframe(pd.DataFrame({
            "Parameter":["Distance","Relative Velocity","Combined Size"],
            "Value":[f"{distance:.2f} km",f"{velocity:.2f} km/s",f"{size:.2f}"]
        }),use_container_width=True,hide_index=True)

    with right:
        st.subheader("📊 Risk Probabilities")
        st.bar_chart(pd.DataFrame({"Probability":probs},index=["Low","Medium","High"]))

    st.divider()
    st.header("🔍 Explainable AI")
    importance=pd.DataFrame({
        "Feature":["Distance","Relative Velocity","Combined Size"],
        "Importance":model.feature_importances_
    }).sort_values("Importance",ascending=False)
    x,y=st.columns(2)
    with x: st.dataframe(importance,use_container_width=True,hide_index=True)
    with y: st.bar_chart(importance.set_index("Feature"))

else:
    st.info("👈 Enter the parameters in the sidebar and click Analyze Risk.")
    st.header("🛰️ About the Project")
    st.write("This Streamlit application demonstrates an Explainable AI workflow for estimating space-debris conjunction risk using distance, relative velocity, and combined object size.")

st.markdown("---")
st.caption("🛰️ Space Debris Risk Prediction & Explainable AI Project")
