# frontend/dashboard.py — Project Xceed Streamlit dashboard
# Access from Mac: http://<pi-ip>:8501
# Auto-refreshes every 2 seconds.

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# ── Config ───────────────────────────────────────────
API_BASE     = "http://localhost:8000"
REFRESH_SECS = 2

ALERT_CLASSES = {"no_belt", "clipped_behind", "decoy"}

CLASS_COLORS = {
    "proper_belt":   "#22c55e",
    "no_belt":       "#ef4444",
    "clipped_behind":"#f97316",
    "decoy":         "#eab308",
    "none":          "#6b7280",
}

CLASS_ICONS = {
    "proper_belt":   "✅",
    "no_belt":       "🔴",
    "clipped_behind":"⚠️",
    "decoy":         "🟡",
    "none":          "⬜",
}

# ── Page setup ───────────────────────────────────────
st.set_page_config(
    page_title="Project Xceed",
    page_icon="🚗",
    layout="wide",
    menu_items={}
)

# Hide sidebar, hamburger menu, and footer completely
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        footer {display: none;}
        #MainMenu {display: none;}
        .block-container {padding-top: 1.5rem;}
    </style>
""", unsafe_allow_html=True)

# ── Fetch helpers ─────────────────────────────────────
def fetch_status():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=1)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_violations(limit=50):
    try:
        r = requests.get(f"{API_BASE}/violations?limit={limit}", timeout=1)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def fetch_summary():
    try:
        r = requests.get(f"{API_BASE}/violations/summary", timeout=1)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def clear_log():
    try:
        requests.delete(f"{API_BASE}/violations", timeout=1)
    except Exception:
        pass

# ── Fetch data ───────────────────────────────────────
status     = fetch_status()
violations = fetch_violations()
summary    = fetch_summary()

# ── Header ───────────────────────────────────────────
st.title("🚗 Project Xceed — Seatbelt Detection")
st.caption("Real-time embedded AI · Raspberry Pi 5 · YOLOv8n ONNX")

st.divider()

# ── Row 1: Live state card + 4 class counters ────────
col_state, col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1, 1])

with col_state:
    if status:
        cls   = status.get("class_name", "none")
        conf  = status.get("confidence", 0.0)
        alert = status.get("alert", False)
        color = CLASS_COLORS.get(cls, "#6b7280")
        icon  = CLASS_ICONS.get(cls, "⬜")
        alert_text  = "🚨 ALERT ACTIVE" if alert else "✓ No Alert"
        alert_color = "#ef4444" if alert else "#22c55e"

        st.markdown(f"""
            <div style="
                background:{color}22;
                border:2px solid {color};
                border-radius:12px;
                padding:1rem 1.25rem;
            ">
                <div style="font-size:2rem;margin-bottom:4px">
                    {icon} <b style="color:{color}">{cls}</b>
                </div>
                <div style="font-size:1rem;color:#888">
                    Confidence: {conf:.2f}
                </div>
                <div style="font-size:1.1rem;color:{alert_color};font-weight:500;margin-top:6px">
                    {alert_text}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Backend not reachable — is detect.py running?")

with col_s1:
    st.metric("proper_belt", summary.get("proper_belt", 0))
with col_s2:
    val = summary.get("no_belt", 0)
    st.metric("no_belt", val,
              delta="⚠️" if val > 0 else None,
              delta_color="inverse")
with col_s3:
    val = summary.get("clipped_behind", 0)
    st.metric("clipped_behind", val,
              delta="⚠️" if val > 0 else None,
              delta_color="inverse")
with col_s4:
    st.metric("decoy", summary.get("decoy", 0))

st.divider()

# ── Row 2: Violation log + clear button ──────────────
col_log, col_ctrl = st.columns([5, 1])

with col_log:
    st.subheader("Violation log")
    if violations:
        df = pd.DataFrame(violations)
        df["status"] = df["alert"].apply(lambda a: "🔴 ALERT" if a else "—")
        df_display = df[["timestamp", "class_name", "confidence", "status"]].copy()
        df_display.columns = ["Timestamp", "Class", "Confidence", "Status"]
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=280)
    else:
        st.info("No events logged yet.")

with col_ctrl:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear log", use_container_width=True):
        clear_log()
        st.rerun()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"Refresh: {REFRESH_SECS}s")

# ── Auto-refresh ─────────────────────────────────────
time.sleep(REFRESH_SECS)
st.rerun()
