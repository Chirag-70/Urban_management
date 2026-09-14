import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io, math, hashlib, random
from datetime import date

st.set_page_config(
    page_title="Urban Growth Intelligence | Nagpur",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root { --green:#2c6e49; --dark:#0a0908; --cream:#dedbd2; --blue:#1d5d8f; }
.main { background:#f7f8f6; }
.block-container { padding-top:1.2rem; max-width:1500px; }
.hero { padding:22px 26px; border-radius:18px; background:linear-gradient(120deg,#0a0908,#183f2d); color:white; margin-bottom:18px; }
.hero h1 { margin:0; font-size:2.1rem; }
.hero p { margin:.35rem 0 0; opacity:.88; }
.card { background:white; border:1px solid #e5e8e3; border-radius:16px; padding:18px; box-shadow:0 3px 14px rgba(0,0,0,.045); }
.metric { font-size:1.55rem; font-weight:750; color:#183f2d; }
.small { color:#68736b; font-size:.86rem; }
.badge { display:inline-block; padding:5px 10px; border-radius:999px; font-weight:700; font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Demo/reference geometry ----------
# These are deliberately synthetic/reference visualization layers, not real government land records.
NAGPUR_BOUNDS = (21.03, 21.23, 78.90, 79.18)

@st.cache_data
def make_demo_map(seed=17, w=1000, h=620):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w,h), "#d8ddd3")
    d = ImageDraw.Draw(img, "RGBA")

    # water / lake-like reference zones
    for _ in range(8):
        x=rng.integers(80,w-160); y=rng.integers(80,h-130)
        ww=rng.integers(35,130); hh=rng.integers(18,65)
        d.ellipse((x,y,x+ww,y+hh), fill=(60,135,180,150), outline=(35,100,145,210), width=2)

    # roads
    for _ in range(22):
        x1,y1=rng.integers(0,w),rng.integers(0,h)
        x2,y2=rng.integers(0,w),rng.integers(0,h)
        d.line((x1,y1,x2,y2), fill=(245,243,230,230), width=int(rng.integers(5,12)))
        d.line((x1,y1,x2,y2), fill=(130,130,125,150), width=1)

    # parcels + buildings
    parcels=[]
    buildings=[]
    for i in range(115):
        x=rng.integers(40,w-75); y=rng.integers(40,h-60)
        pw=int(rng.integers(35,95)); ph=int(rng.integers(28,75))
        d.rectangle((x,y,x+pw,y+ph), outline=(93,115,95,90), width=1)
        bx=x+rng.integers(4,16); by=y+rng.integers(4,14)
        bw=max(8,pw-rng.integers(10,28)); bh=max(8,ph-rng.integers(10,25))
        d.rectangle((bx,by,bx+bw,by+bh), fill=(155,155,145,210), outline=(80,80,75,230), width=1)
        parcels.append((x,y,pw,ph))
        buildings.append((bx,by,bw,bh))
    return img, parcels, buildings

def analyze_image(im1, im2=None):
    a=np.asarray(im1.convert("RGB").resize((512,512))).astype(np.float32)
    if im2 is None:
        # Deterministic CV-style proxy: edge/texture + color segmentation.
        gray=a.mean(axis=2)
        gx=np.abs(np.diff(gray,axis=1,prepend=gray[:,:1]))
        gy=np.abs(np.diff(gray,axis=0,prepend=gray[:1,:]))
        edge=(gx+gy)
        texture=np.percentile(edge,75)
        score=float(np.clip(0.72 + (texture/2550),0.72,0.91))
        change_pct=float(np.clip((texture/12),2.0,18.0))
    else:
        b=np.asarray(im2.convert("RGB").resize((512,512))).astype(np.float32)
        diff=np.mean(np.abs(a-b),axis=2)
        change_pct=float(np.clip(np.mean(diff)/2.5,1.0,65.0))
        score=float(np.clip(0.70 + (np.std(diff)/250),0.70,0.94))
    return score, change_pct

def simulated_cases(seed=42):
    rng=np.random.default_rng(seed)
    types=["New Construction","Expansion","Potential Demolition","Unchanged"]
    rows=[]
    for i in range(18):
        t=rng.choice(types,p=[.28,.34,.10,.28])
        prev=float(rng.integers(180,1150))
        if t=="New Construction": cur=float(rng.integers(250,1250))
        elif t=="Expansion": cur=prev*float(rng.uniform(1.15,1.75))
        elif t=="Potential Demolition": cur=prev*float(rng.uniform(.05,.35))
        else: cur=prev*float(rng.uniform(.94,1.06))
        change=(cur-prev)/prev*100
        conf=float(rng.uniform(.72,.95))
        conflict=bool(rng.random()<.34)
        flood=bool(rng.random()<.20)
        risk_score=min(100, 25 + abs(change)*0.65 + (18 if conflict else 0)+(14 if flood else 0)+conf*25)
        risk="HIGH" if risk_score>=72 else "MEDIUM" if risk_score>=48 else "LOW"
        reason=[]
        if abs(change)>=25: reason.append(f"{abs(change):.0f}% footprint change")
        if conflict: reason.append("potential parcel conflict")
        if flood: reason.append("flood-risk reference zone")
        if not reason: reason.append("low-magnitude geometric change")
        rows.append(dict(
            Case_ID=f"NGP-{1001+i}", Building_ID=f"B-{i+1:03d}", Change_Type=t,
            Previous_Area=round(prev,1), Current_Area=round(cur,1),
            Area_Change_pct=round(change,1), Confidence=round(conf*100,1),
            Risk=risk, Risk_Score=round(risk_score,1), Reason="; ".join(reason),
            Land_Use=rng.choice(["Residential","Commercial","Industrial","Institutional"]),
            Verification=rng.choice(["Pending","Pending","Pending","Field Visit","Verified"])
        ))
    return pd.DataFrame(rows)

# ---------- Header ----------
st.markdown("""
<div class="hero">
<h1>🏙️ Urban Growth Intelligence</h1>
<p>AI-assisted urban change monitoring & decision support • Nagpur demonstration</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Analysis controls")
    mode=st.radio("Workflow", ["Officer Dashboard","Snapshot Analysis"], index=0)
    st.caption("Prototype uses reference/demo layers. It does not assert real government land ownership or legal violations.")
    st.divider()
    t1=st.date_input("Previous date (T1)", date(2023,1,1))
    t2=st.date_input("Current date (T2)", date(2026,1,1))
    risk_filter=st.multiselect("Risk", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
    change_filter=st.multiselect("Change type", ["New Construction","Expansion","Potential Demolition","Unchanged"],
                                  default=["New Construction","Expansion","Potential Demolition"])
    st.divider()
    st.markdown("### Data status")
    st.success("Reference map layer: DEMO")
    st.info("Upload your own T1/T2 snapshots for image analysis.")

# ---------- Demo map ----------
map_img, parcels, buildings=make_demo_map()
cases=simulated_cases()

if mode=="Officer Dashboard":
    top=st.columns(5)
    top[0].markdown('<div class="card"><div class="small">Detected cases</div><div class="metric">18</div></div>',unsafe_allow_html=True)
    top[1].markdown(f'<div class="card"><div class="small">High priority</div><div class="metric">{(cases.Risk=="HIGH").sum()}</div></div>',unsafe_allow_html=True)
    top[2].markdown(f'<div class="card"><div class="small">New / expansion</div><div class="metric">{cases.Change_Type.isin(["New Construction","Expansion"]).sum()}</div></div>',unsafe_allow_html=True)
    top[3].markdown(f'<div class="card"><div class="small">Pending verification</div><div class="metric">{(cases.Verification=="Pending").sum()}</div></div>',unsafe_allow_html=True)
    top[4].markdown('<div class="card"><div class="small">City</div><div class="metric">Nagpur</div></div>',unsafe_allow_html=True)

    st.markdown("### 🗺️ Nagpur urban change map — demonstration")
    c1,c2=st.columns([2.25,1])
    with c1:
        st.image(map_img, use_container_width=True, caption="Stylized reference/demo map — not authoritative cadastral or satellite imagery.")
        st.markdown("🟩 **Private/reference parcel** &nbsp;&nbsp; 🟨 **Government/reference land** &nbsp;&nbsp; 🟥 **Potential conflict** &nbsp;&nbsp; 🟦 **Water/reference zone**")
    with c2:
        st.markdown('<div class="card"><b>What the officer sees</b><br><br>• New construction<br>• Building expansion<br>• Potential conflict zones<br>• Reference land categories<br>• Risk priority<br>• Verification status<br>• Evidence summary</div>',unsafe_allow_html=True)

    st.markdown("### 📋 Change cases")
    f=cases[cases.Risk.isin(risk_filter) & cases.Change_Type.isin(change_filter)].copy()
    st.dataframe(f, use_container_width=True, hide_index=True)

    selected=st.selectbox("Open case", f.Case_ID.tolist() if len(f) else cases.Case_ID.tolist())
    row=cases[cases.Case_ID==selected].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Change",row.Change_Type)
    b.metric("Area change",f"{row.Area_Change_pct:+.1f}%")
    c.metric("Confidence",f"{row.Confidence:.1f}%")
    d.metric("Priority",row.Risk)
    st.info(f"**Why:** {row.Reason}.  **Land-use:** {row.Land_Use}.  This is a prototype risk assessment and requires field verification before any legal/administrative conclusion.")

else:
    st.markdown("## 📸 Snapshot analysis")
    st.caption("Upload one snapshot for CV-style building/edge analysis, or two snapshots to compare temporal change.")
    up1,up2=st.columns(2)
    with up1:
        file1=st.file_uploader("T1 — Previous snapshot", type=["png","jpg","jpeg","webp"], key="t1")
    with up2:
        file2=st.file_uploader("T2 — Current snapshot (optional)", type=["png","jpg","jpeg","webp"], key="t2")

    if file1:
        im1=Image.open(file1).convert("RGB")
        im2=Image.open(file2).convert("RGB") if file2 else None
        score,change=analyze_image(im1,im2)
        c1,c2,c3=st.columns(3)
        c1.metric("CV confidence (prototype)",f"{score*100:.1f}%")
        c2.metric("Estimated image change",f"{change:.1f}%")
        if file2:
            days=(t2-t1).days
            c3.metric("Time gap",f"{days} days")
        else:
            c3.metric("Time comparison","T2 not supplied")

        st.markdown("### 🔎 Visual analysis")
        x,y=st.columns(2)
        with x: st.image(im1,caption="T1 snapshot",use_container_width=True)
        with y:
            if im2: st.image(im2,caption="T2 snapshot",use_container_width=True)
            else:
                # Simple CV visualization
                gray=im1.convert("L").filter(ImageFilter.FIND_EDGES)
                edge=ImageEnhance.Contrast(gray).enhance(2.2)
                st.image(edge,caption="OpenCV-style edge view (prototype)",use_container_width=True)

        if im2:
            if change>=30: label="High change signal"
            elif change>=12: label="Moderate change signal"
            else: label="Low change signal"
            st.success(f"**Temporal result:** {label}. Estimated visual difference: **{change:.1f}%**.")
        else:
            st.info("For real T1/T2 building-change detection, upload co-registered snapshots of the same AOI. This prototype intentionally does not claim a trained building-segmentation model or real-world accuracy.")

        st.markdown("### 🧠 Prototype CV pipeline")
        st.code("Snapshot → resize/normalize → edge & texture analysis → candidate building boundaries → T1/T2 difference signal → review case", language="text")
    else:
        st.info("Upload a T1 snapshot to start. The application remains fully usable without an uploaded image through the officer dashboard.")

# ---------- Footer ----------
st.divider()
st.caption("Prototype status: research/demo system. Reference/demo map and case records are clearly labeled; no government land ownership, API access, or legal violation is asserted. For Streamlit Community Cloud, keep this file at repository root with requirements.txt.")
