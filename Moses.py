import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from sklearn.linear_model import LinearRegression

# ===================== 1. 頁面設定 =====================
st.set_page_config(page_title="全方位績效分析", layout="wide")
st.title("📊 預測行為診斷與效能分析系統")

try:
    import openpyxl
except ImportError:
    st.error("❌ 缺少套件：openpyxl，請執行 pip install openpyxl")
    st.stop()

# ===================== 2. 工具函式 =====================
def parse_date_robust(v):
    if pd.isna(v): return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime)): return v.date()
    if isinstance(v, date): return v
    s = str(v).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try: return datetime.strptime(digits[:8], "%Y%m%d").date()
        except: pass
    if len(digits) >= 6:
        try: return datetime.strptime(digits[:6] + "01", "%Y%m%d").date()
        except: pass
    try: return pd.to_datetime(v).date()
    except: return pd.NaT

def parse_target_date_num(v):
    s = str(v).strip()
    if len(s) == 6:
        try: return datetime.strptime(s, "%Y%m").toordinal()
        except: pass
    return np.nan

def parse_date_num(v):
    if pd.isna(v): return np.nan
    s = str(v).strip(); digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try: return datetime.strptime(digits[:8], "%Y%m%d").toordinal()
        except: pass
    return np.nan

def int_to_yw(v):
    if pd.isna(v): return (np.nan, np.nan)
    v = int(v)
    return v // 100, v % 100

def weeks_diff(y1, w1, y2, w2):
    if pd.isna(y1) or pd.isna(w1) or pd.isna(y2) or pd.isna(w2): return np.nan
    try:
        d1 = date.fromisocalendar(int(y1), int(w1), 1)
        d2 = date.fromisocalendar(int(y2), int(w2), 1)
        return int((d2 - d1).days // 7)
    except: return np.nan

def iso_year_week(d):
    if pd.isna(d): return (np.nan, np.nan)
    iso = d.isocalendar()
    return (int(iso[0]), int(iso[1]))

def yw_to_int(y, w):
    if pd.isna(y) or pd.isna(w): return np.nan
    try: return int(f"{int(y)}{int(w):02d}")
    except: return np.nan

# ===================== 3. 核心運算 =====================

@st.cache_data
def process_data_final(file, threshold_x_error):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"讀取失敗: {e}")
        st.stop()
        
    dcp_col_name = "DCP定板" 
    if "DCP定版" in df.columns: dcp_col_name = "DCP定版"
    elif "DCP定板" in df.columns: dcp_col_name = "DCP定板"
        
    colmap = {
        "SALES_NAME": "SALES_NAME",
        "PRODUCT_NAME": "PRODUCT_NAME",
        "AREA_NAME": "AREA_NAME",
        "PP_GROUP_ID": "PP_GROUP_ID",
        "DONE_DATE": "DONE_YEAR_MON" if "DONE_YEAR_MON" in df.columns else "DONE_DATE",
        "BFC_QTY": "BFC_QTY",
        "SHIPMENT_QTY": "SHIPMENT_QTY",
        "DCP_DATE": dcp_col_name,
        "REQUEST_DATE": "REQUEST_DATE" 
    }
    
    dfn = pd.DataFrame()
    for t, s in colmap.items():
        if s in df.columns: dfn[t] = df[s]
        else: dfn[t] = np.nan
            
    dfn["BFC_QTY"] = pd.to_numeric(dfn["BFC_QTY"], errors="coerce").fillna(0)
    dfn["SHIPMENT_QTY"] = pd.to_numeric(dfn["SHIPMENT_QTY"], errors="coerce")
    dfn["DONE_DATE"] = dfn["DONE_DATE"].apply(parse_date_robust)
    dfn["DCP_DATE"] = dfn["DCP_DATE"].apply(parse_date_robust) 
    
    done_yw = dfn["DONE_DATE"].apply(iso_year_week)
    dfn["DONE_YW"] = [yw_to_int(y, w) for y, w in done_yw]
    
    if 'REQUEST_DATE' in df.columns:
        dfn['TARGET_DATE_NUM'] = df['REQUEST_DATE'].apply(parse_target_date_num)
        dfn['DONE_DATE_NUM'] = dfn["DONE_DATE"].apply(parse_date_num)
        dfn['WEEKS_BEFORE'] = (dfn['TARGET_DATE_NUM'] - dfn['DONE_DATE_NUM']) / 7
        dfn['BIAS_PCT'] = (dfn['BFC_QTY'] - dfn['SHIPMENT_QTY']) / dfn['SHIPMENT_QTY']
        dfn['BIAS_PCT'] = dfn['BIAS_PCT'].replace([np.inf, -np.inf], np.nan)
        dfn['ABS_PCT_ERR'] = dfn['BIAS_PCT'].abs()
        dfn['SINGLE_ACCURACY'] = (1 - dfn['ABS_PCT_ERR']).clip(0, 1)
        dfn['WB_BIN'] = dfn['WEEKS_BEFORE'].fillna(0).astype(int)
    
    df_cleaned = dfn[~dfn["SHIPMENT_QTY"].isna() & ~dfn["DONE_YW"].isna()].copy()
    df_cleaned.sort_values(["DONE_YW", "DONE_DATE"], inplace=True)
    
    GROUP_KEYS = ["SALES_NAME", "PRODUCT_NAME", "AREA_NAME", "PP_GROUP_ID", "REQUEST_DATE"]
    if df_cleaned["PP_GROUP_ID"].isna().all(): GROUP_KEYS.remove("PP_GROUP_ID")
    if df_cleaned["REQUEST_DATE"].isna().all(): GROUP_KEYS.remove("REQUEST_DATE")
    
    latest = df_cleaned.drop_duplicates(subset=GROUP_KEYS + ["DONE_DATE"], keep="last").copy()
    
    pred = latest["BFC_QTY"].astype(float)
    act = latest["SHIPMENT_QTY"].astype(float)
    latest["ABS_PCT_ERR"] = np.where(act==0, np.nan, (pred-act).abs()/act)
    latest["ACCURACY_SCORE"] = (1 - latest["ABS_PCT_ERR"]).clip(0, 1)

    def calc_metrics(g):
        g = g.sort_values("DONE_YW")
        if g.empty: return pd.Series()

        dcp_date = g.iloc[0]["DCP_DATE"] 
        g_calc = g[g["DONE_DATE"] >= dcp_date].copy() if pd.notna(dcp_date) else g.copy()
        
        if g_calc.empty:
             return pd.Series({
                "實際銷量": g.iloc[-1]["SHIPMENT_QTY"], "最終預測": g.iloc[-1]["BFC_QTY"], "初版預測": np.nan, "版本總數": 0,
                "反應速度X": np.nan, "最終準確率Y": 0.0, "加權準確率_供應鏈": 0.0, "PP分數": 0.0,
                "P1得分": 0.0, "P2得分": 0.0, "P3得分": 0.0, "DCP準確率": 0.0,
                "DCP後不準次數": 0, "DCP後版本數": 0, "DCP後不準率": 0.0,
                "樂觀偏誤": 0.0, "下修次數": 0, "無效微調次數": 0, "FVA": 0.0
            })

        start_yw = g_calc.iloc[0]["DONE_YW"]
        sy, sw = int_to_yw(start_yw)
        actual_qty = g_calc.iloc[-1]["SHIPMENT_QTY"]
        final_forecast = g_calc.iloc[-1]["BFC_QTY"]
        version_count = len(g_calc)
        
        bx = np.nan
        if actual_qty > 0:
            x_mask = g_calc["ABS_PCT_ERR"] <= threshold_x_error
            if x_mask.any():
                f = g_calc.loc[x_mask].iloc[0]
                fy, fw = int_to_yw(f["DONE_YW"])
                bx = weeks_diff(sy, sw, fy, fw)
        
        final_acc_pct = 0.0
        if actual_qty > 0:
            err = abs(final_forecast - actual_qty) / actual_qty
            final_acc_pct = max(0.0, 1.0 - err)

        weighted_score_proc = 0.0
        weighted_score_pp = 0.0
        score_breakdown = {"p1":0, "p2":0, "p3":0}
        
        if actual_qty > 0:
            scores = g_calc["ACCURACY_SCORE"].values
            p1 = scores[0:8]; p2 = scores[8:16]; p3 = scores[16:]
            s1 = np.mean(p1) if len(p1) > 0 else 0.0
            s2 = np.mean(p2) if len(p2) > 0 else 0.0
            s3 = np.mean(p3) if len(p3) > 0 else 0.0
            
            w1, w2, w3 = 0.7, 0.2, 0.1
            weighted_score_proc = (s1 * w1) + (s2 * w2) + (s3 * w3)
            wp1, wp2, wp3 = 0.2, 0.7, 0.1
            weighted_score_pp = (s1 * wp1) + (s2 * wp2) + (s3 * wp3)
            score_breakdown = {"p1":s1, "p2":s2, "p3":s3}

        dcp_acc = 0.0
        dcp_fcst = np.nan
        dcp_total_count = len(g_calc)
        
        dcp_ver = g[g["DONE_DATE"] == dcp_date]
        if not dcp_ver.empty:
            dcp_fcst = dcp_ver.iloc[0]["BFC_QTY"]
            if actual_qty > 0:
                dcp_acc = max(0.0, 1.0 - abs(dcp_fcst - actual_qty) / actual_qty)
        
        fails = g_calc[g_calc["ABS_PCT_ERR"] > threshold_x_error]
        inaccurate_count = len(fails)
        dcp_fail_rate = inaccurate_count / dcp_total_count if dcp_total_count > 0 else 0.0

        optimism_bias = g_calc["BIAS_PCT"].mean() if "BIAS_PCT" in g_calc.columns else 0.0
        
        qty_series = g_calc["BFC_QTY"].values
        downward_count = 0
        tinkering_count = 0
        
        if len(qty_series) > 1:
            diffs = np.diff(qty_series)
            p2_diffs = diffs[8:16] 
            if len(p2_diffs) > 0:
                downward_count = np.sum(p2_diffs < 0)
            
            p3_series = qty_series[16:]
            if len(p3_series) > 1:
                p3_pct_change = np.abs(np.diff(p3_series) / p3_series[:-1])
                valid_changes = p3_pct_change[~np.isnan(p3_pct_change) & ~np.isinf(p3_pct_change)]
                tinkering_count = np.sum((valid_changes > 0) & (valid_changes < 0.20))

        fva = final_acc_pct - dcp_acc

        return pd.Series({
            "實際銷量": actual_qty, "最終預測": final_forecast, "初版預測": dcp_fcst,
            "版本總數": version_count, "反應速度X": bx, "最終準確率Y": final_acc_pct,
            "加權準確率_供應鏈": weighted_score_proc, "PP分數": weighted_score_pp * 100,
            "P1得分": score_breakdown["p1"], "P2得分": score_breakdown["p2"], "P3得分": score_breakdown["p3"],
            "DCP準確率": dcp_acc, "DCP後不準次數": inaccurate_count, 
            "DCP後版本數": dcp_total_count, "DCP後不準率": dcp_fail_rate,
            "樂觀偏誤": optimism_bias, "下修次數": downward_count, 
            "無效微調次數": tinkering_count, "FVA": fva
        })

    summary = latest.groupby(GROUP_KEYS, dropna=False).apply(calc_metrics).reset_index()
    return summary, df_cleaned

# ===================== 4. 介面呈現 =====================

uploaded_file = st.file_uploader("📂 請上傳 Excel 檔案", type=['xlsx', 'csv'])

if uploaded_file is not None:
    st.sidebar.header("👁️ 分析視角")
    view_mode = st.sidebar.radio("請選擇模式", [
        "1. 供應鏈視角 (反應速度 vs 分數) - 防禦性寬估預警", 
        "2. PP視角 (反應速度 vs PP分數) - 預測變動與下修成效", 
        "3. DCP專案視角 (DCP準度 vs 後續差異) - 初版品質分析",
        "4. 👑 管理層總覽 (全盤健康度 & 缺貨風險預警)",
        "5. 📖 學術辭典與公式透視 (白盒計算拆解)" 
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 參數設定")
    
    cutoff_x = st.sidebar.slider("反應速度門檻 (週)", 4, 20, 8)
    cutoff_weighted = st.sidebar.slider("分數門檻 (%)", 50, 100, 75) / 100.0
    cutoff_y = st.sidebar.slider("最終準確率門檻 (%)", 50, 100, 80) / 100.0
    thr_error = st.sidebar.number_input("誤差認定 (0.2=20%)", 0.05, 0.5, 0.2)

    with st.spinner('🚀 運算中...'):
        df_res, df_raw = process_data_final(uploaded_file, thr_error)
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 篩選資料")
    
    all_areas = sorted(df_res["AREA_NAME"].astype(str).unique())
    sel_area = st.sidebar.multiselect("地區", all_areas)
    df_s1 = df_res[df_res["AREA_NAME"].isin(sel_area)] if sel_area else df_res
    
    avail_sales = sorted(df_s1["SALES_NAME"].astype(str).unique())
    sel_sales = st.sidebar.multiselect("業務", avail_sales)
    df_s2 = df_s1[df_s1["SALES_NAME"].isin(sel_sales)] if sel_sales else df_s1
    
    if "REQUEST_DATE" in df_res.columns:
        all_req_dates = sorted(df_s2["REQUEST_DATE"].astype(str).unique())
        sel_req_date = st.sidebar.multiselect("目標月份 (REQUEST_DATE)", all_req_dates)
        if sel_req_date:
            df_s2 = df_s2[df_s2["REQUEST_DATE"].astype(str).isin(sel_req_date)]

    if "PP_GROUP_ID" in df_res.columns:
        avail_groups = sorted(df_s2["PP_GROUP_ID"].astype(str).unique())
        sel_group = st.sidebar.multiselect("產品群組", avail_groups)
        df_s3 = df_s2[df_s2["PP_GROUP_ID"].isin(sel_group)] if sel_group else df_s2
    else: df_s3 = df_s2
        
    avail_prod = sorted(df_s3["PRODUCT_NAME"].astype(str).unique())
    sel_prod = st.sidebar.multiselect("產品", avail_prod)
    
    df_show = (df_s3[df_s3["PRODUCT_NAME"].isin(sel_prod)] if sel_prod else df_s3).copy()

    # ================= 視角邏輯與圖表 =================
    
    max_x = max(df_show["反應速度X"].max(), 20)
    if pd.isna(max_x): max_x=20
    df_show["Plot_X"] = df_show["反應速度X"].fillna(max_x + 2)

    chart_key = f"main_chart_{view_mode[:2]}" 
    fig = None
    fig_top = None
    fig_bad = None
    
    if view_mode.startswith("1"):
        st.subheader("⚖️ 供應鏈視角：反應速度 vs 供應鏈加權分數")
        st.info(f"目標：↖️ 左上角。協助識別前期預測是否有「防禦性寬估」現象（達標定義：誤差在 ±{thr_error*100:.0f}% 內）。")
        
        def get_quad_sc(row):
            x_speed = row["反應速度X"]
            y_score = row["加權準確率_供應鏈"] 
            x_pass = (pd.notna(x_speed) and x_speed <= cutoff_x)
            y_pass = (y_score >= cutoff_weighted)
            
            if x_pass and y_pass: return "1. 模範生 (快且高分)"
            elif not x_pass and y_pass: return "2. 穩重型 (慢但高分)"
            elif x_pass and not y_pass: return "3. 衝動型 (快但低分)"
            else: return "4. 待改善 (慢且低分)"
            
        df_show["象限"] = df_show.apply(get_quad_sc, axis=1)
        
        fig = px.scatter(
            df_show, x="Plot_X", y="加權準確率_供應鏈", color="象限",
            custom_data=["SALES_NAME", "PRODUCT_NAME", "REQUEST_DATE"],
            hover_data={"SALES_NAME":True, "PRODUCT_NAME":True, "REQUEST_DATE":True, "實際銷量":":,.0f", "樂觀偏誤":":.1%"},
            color_discrete_map={
                "1. 模範生 (快且高分)": "green", 
                "2. 穩重型 (慢但高分)": "orange",
                "3. 衝動型 (快但低分)": "purple",
                "4. 待改善 (慢且低分)": "red"
            },
            labels={"Plot_X": "達標反應速度 (週)", "加權準確率_供應鏈": "供應鏈指標得分 (早期權重高)"}
        )
        
        fig.add_vline(x=cutoff_x, line_dash="dash", line_color="red")
        fig.add_hline(y=cutoff_weighted, line_dash="dash", line_color="red")
        fig.update_xaxes(range=[-1, max_x+5], title="達標反應速度 (週) -> 越左越快")
        fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="供應鏈指標得分 -> 越上越佳")

    elif view_mode.startswith("2"):
        st.subheader("🏭 PP視角：反應速度 vs PP分數")
        st.info("目標：↖️ 左上角。評估中期下修的貢獻度，以及排產前夕預測變動的穩定性。")
        
        def get_quad_pp(row):
            x_speed = row["反應速度X"]
            y_score = row["PP分數"] / 100.0
            x_pass = (pd.notna(x_speed) and x_speed <= cutoff_x)
            y_pass = (y_score >= cutoff_weighted)
            
            if x_pass and y_pass: return "1. 模範生 (快且高分)"
            elif not x_pass and y_pass: return "2. 穩重型 (慢但高分)"
            elif x_pass and not y_pass: return "3. 衝動型 (快但低分)"
            else: return "4. 待改善 (慢且低分)"
            
        df_show["象限"] = df_show.apply(get_quad_pp, axis=1)
        
        fig = px.scatter(
            df_show, x="Plot_X", y="PP分數", color="象限",
            custom_data=["SALES_NAME", "PRODUCT_NAME", "REQUEST_DATE"],
            hover_data={"SALES_NAME":True, "PRODUCT_NAME":True, "REQUEST_DATE":True, "實際銷量":":,.0f", "FVA":":.1%"},
            color_discrete_map={
                "1. 模範生 (快且高分)": "green", 
                "2. 穩重型 (慢但高分)": "orange",
                "3. 衝動型 (快但低分)": "purple",
                "4. 待改善 (慢且低分)": "red"
            },
            labels={"Plot_X": "達標反應速度 (週)", "PP分數": "PP指標得分 (0-100)"}
        )
        
        fig.add_vline(x=cutoff_x, line_dash="dash", line_color="red")
        fig.add_hline(y=cutoff_weighted*100, line_dash="dash", line_color="red")
        fig.update_xaxes(range=[-1, max_x+5], title="達標反應速度 (週) -> 越左越快")
        fig.update_yaxes(range=[0, 105], title="PP指標得分 -> 越上越佳")

    elif view_mode.startswith("3"):
        st.subheader("🎯 DCP專案視角：不準次數 (柱狀) & DCP準度 (折線)")
        st.info("紅柱較低、藍線較高為佳。量化 DCP 定板後的預測變動頻率與初版品質。")

        df_show["Item"] = df_show["SALES_NAME"] + " - " + df_show["PRODUCT_NAME"] + " (" + df_show["REQUEST_DATE"].astype(str) + ")"
        df_sorted = df_show.sort_values("DCP後不準次數", ascending=False).copy()
        
        custom_data_list = df_sorted[["SALES_NAME", "PRODUCT_NAME", "REQUEST_DATE"]].values.tolist()

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_sorted["Item"],
            y=df_sorted["DCP後不準次數"],
            name="超出誤差範圍次數",
            marker_color="rgb(255, 100, 100)", 
            yaxis="y",
            customdata=custom_data_list,
            hovertemplate="<b>%{x}</b><br>超出誤差範圍次數: %{y:,}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=df_sorted["Item"],
            y=df_sorted["DCP準確率"],
            name="DCP初版準確率",
            mode="lines+markers",
            marker=dict(size=8, color="blue"),
            line=dict(width=3, color="blue"),
            yaxis="y2",
            customdata=custom_data_list,
            hovertemplate="DCP初版準確率: %{y:.1%}<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis=dict(title="業務 - 產品 - 目標月", tickangle=-45, categoryorder='total descending'),
            yaxis=dict(title="超出誤差範圍次數 (紅柱)", showgrid=True, zeroline=True, side="left", tickformat=","),
            yaxis2=dict(title="DCP初版準確率 (藍線)", overlaying="y", side="right", range=[0, 1.1], tickformat=".0%", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=600
        )

    elif view_mode.startswith("4"):
        st.subheader("👑 管理層總覽：全盤健康度、缺貨風險與優先檢討清單")
        st.info("此視角專為管理層設計：快速掌握整體預測偏誤分佈，點擊下方圖表可動態切換檢視風險來源。")

        def get_top10_status(row):
            if row["最終準確率Y"] >= cutoff_y and row["反應速度X"] <= cutoff_x: return "🟢 表現優良"
            elif row["最終準確率Y"] < cutoff_y: return "🔴 需優先關注" 
            else: return "🟡 反應偏慢"
            
        def get_bias_status(row):
            act = row["實際銷量"]
            pred = row["最終預測"]
            if act == 0 or pd.isna(act) or pd.isna(pred): return "未知"
            err = (pred - act) / act
            if err > thr_error: return "🟡 嚴重寬估 (呆滯風險)"
            elif err < -thr_error: return "🔴 嚴重低估 (缺貨風險)"
            else: return f"🟢 準確達標 (誤差 ±{thr_error*100:.0f}% 內)"
        
        if not df_show.empty:
            df_show["健康度"] = df_show.apply(get_top10_status, axis=1)
            df_show["偏誤狀態"] = df_show.apply(get_bias_status, axis=1)

        # ---------------- ★★★ 互動式偏誤狀態長條圖 (Row 1) ★★★ ----------------
        st.markdown("### 🎯 整體預測偏誤分佈 (點擊左圖動態切換右表)")
        c_bias_chart, c_bias_table = st.columns([1, 1.8])

        sel_bias = None
        with c_bias_chart:
            if not df_show.empty:
                pie_data = df_show["偏誤狀態"].value_counts().reset_index()
                pie_data.columns = ["偏誤狀態", "品項數"]
                pie_data["品項數_文字"] = pie_data["品項數"].apply(lambda x: f"{x:,.0f}")
                
                pie_data["排序"] = pie_data["偏誤狀態"].apply(
                    lambda x: 1 if "達標" in x else (3 if "寬估" in x else (2 if "低估" in x else 4))
                )
                pie_data = pie_data.sort_values("排序", ascending=False) 

                fig_bias = px.bar(
                    pie_data, x="品項數", y="偏誤狀態", color="偏誤狀態",
                    text="品項數_文字", orientation='h',
                    color_discrete_map={
                        f"🟢 準確達標 (誤差 ±{thr_error*100:.0f}% 內)": "#2ecc71", 
                        "🟡 嚴重寬估 (呆滯風險)": "#f1c40f", 
                        "🔴 嚴重低估 (缺貨風險)": "#e74c3c", 
                        "未知": "#95a5a6"
                    }
                )
                fig_bias.update_traces(textposition='inside', textfont=dict(color='white', size=16))
                fig_bias.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300, yaxis_title="")
                fig_bias.update_xaxes(tickformat=",")
                
                sel_bias = st.plotly_chart(fig_bias, use_container_width=True, on_select="rerun", key="bias_bar_chart")

        with c_bias_table:
            clicked_category = "🔴 嚴重低估 (缺貨風險)" 
            if sel_bias and len(sel_bias.get("selection", {}).get("points", [])) > 0:
                clicked_category = sel_bias["selection"]["points"][0]["y"]
                
            st.markdown(f"#### {clicked_category} 清單")
            st.caption("篩選條件：依「實際銷量」由大到小排序 Top 10")
            
            df_bias_list = df_show[df_show["偏誤狀態"] == clicked_category].sort_values(by="實際銷量", ascending=False).head(10)
            
            if df_bias_list.empty:
                st.success(f"🎉 太棒了！目前無 {clicked_category} 的品項。")
            else:
                # ★★★ 加入三大指標 ★★★
                disp_bias = df_bias_list[["SALES_NAME", "PRODUCT_NAME", "最終預測", "實際銷量", "最終準確率Y", "加權準確率_供應鏈", "PP分數"]].copy()
                disp_bias.rename(columns={
                    "最終預測": "最後一版預測",
                    "最終準確率Y": "最終準確率",
                    "加權準確率_供應鏈": "供應鏈指標得分",
                    "PP分數": "PP指標得分"
                }, inplace=True)
                
                if "低估" in clicked_category:
                    disp_bias["落差"] = disp_bias["實際銷量"] - disp_bias["最後一版預測"]
                    col_name = "🚨 短缺量預估"
                elif "寬估" in clicked_category:
                    disp_bias["落差"] = disp_bias["最後一版預測"] - disp_bias["實際銷量"]
                    col_name = "📦 呆滯量預估"
                else:
                    disp_bias["落差"] = (disp_bias["最後一版預測"] - disp_bias["實際銷量"]).abs()
                    col_name = "微小誤差量"
                    
                disp_bias.rename(columns={"落差": col_name}, inplace=True)
                
                disp_bias["最後一版預測"] = disp_bias["最後一版預測"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
                disp_bias["實際銷量"] = disp_bias["實際銷量"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
                disp_bias[col_name] = disp_bias[col_name].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
                
                disp_bias["最終準確率"] = disp_bias["最終準確率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_bias["供應鏈指標得分"] = disp_bias["供應鏈指標得分"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_bias["PP指標得分"] = disp_bias["PP指標得分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NaN")
                
                disp_bias = disp_bias[["SALES_NAME", "PRODUCT_NAME", "最後一版預測", "實際銷量", col_name, "最終準確率", "供應鏈指標得分", "PP指標得分"]]
                disp_bias.index = range(1, len(disp_bias) + 1)
                st.dataframe(disp_bias, use_container_width=True)

        st.markdown("---")

        # ---------------- 排版：傳統 Top 10 與 戰犯清單 (Row 2) ----------------
        df_top10 = df_show.sort_values(by="實際銷量", ascending=False).head(10).copy()
        bad_mask = ((df_show["反應速度X"] > cutoff_x) | (df_show["反應速度X"].isna())) & (df_show["最終準確率Y"] < cutoff_y)
        df_bad = df_show[bad_mask].sort_values(by="實際銷量", ascending=False).head(10).copy()

        st.markdown("### 🏆 主力品項與高風險預警 (Top 10)")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("#### 🏆 銷量 Top 10 主力清單")
            st.caption("篩選條件：依「實際銷量」由大到小排序之首 10 大品項")
            if not df_top10.empty:
                disp_top10 = df_top10[["SALES_NAME", "PRODUCT_NAME", "實際銷量", "最終準確率Y", "加權準確率_供應鏈", "PP分數", "健康度"]].copy()
                disp_top10.rename(columns={"最終準確率Y": "最終準確率", "加權準確率_供應鏈": "供應鏈指標得分", "PP分數": "PP指標得分"}, inplace=True)
                disp_top10["實際銷量"] = disp_top10["實際銷量"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
                disp_top10["最終準確率"] = disp_top10["最終準確率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_top10["供應鏈指標得分"] = disp_top10["供應鏈指標得分"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_top10["PP指標得分"] = disp_top10["PP指標得分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NaN")
                disp_top10.index = range(1, len(disp_top10) + 1)
                st.dataframe(disp_top10, use_container_width=True)

        with col_t2:
            st.markdown("#### 🚨 高風險預警：優先檢討清單")
            st.caption(f"篩選條件：反應速度 > {cutoff_x}週 且 最終準確率 < {cutoff_y*100:.0f}%")
            if df_bad.empty:
                st.success("🎉 目前系統篩選條件下，無顯著的高風險品項，預測狀況整體良好。")
            else:
                disp_bad = df_bad[["SALES_NAME", "PRODUCT_NAME", "實際銷量", "最終準確率Y", "加權準確率_供應鏈", "PP分數"]].copy()
                disp_bad.rename(columns={"最終準確率Y": "最終準確率", "加權準確率_供應鏈": "供應鏈指標得分", "PP分數": "PP指標得分"}, inplace=True)
                disp_bad["實際銷量"] = disp_bad["實際銷量"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
                disp_bad["最終準確率"] = disp_bad["最終準確率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_bad["供應鏈指標得分"] = disp_bad["供應鏈指標得分"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
                disp_bad["PP指標得分"] = disp_bad["PP指標得分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NaN")
                disp_bad.index = range(1, len(disp_bad) + 1)
                st.dataframe(disp_bad, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 視覺化對比：橫向長條圖 (點擊圖表可查看下方詳細診斷報告)")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### 🟢 主力品項健康度")
            if not df_top10.empty:
                df_top10_plot = df_top10.copy()
                df_top10_plot["Item"] = df_top10_plot.apply(lambda r: f"{r['SALES_NAME']}-{r['PRODUCT_NAME']} ({r['REQUEST_DATE']})", axis=1)
                df_top10_plot['Bar_Text'] = df_top10_plot.apply(
                    lambda r: f"準度:{r['最終準確率Y']:.0%} | 速度:{r['反應速度X']:.0f}週" if pd.notna(r['反應速度X']) else f"準度:{r['最終準確率Y']:.0%} | 速度:未達標", axis=1
                )
                df_top10_plot = df_top10_plot.sort_values("實際銷量", ascending=True) 
                
                fig_top = px.bar(
                    df_top10_plot, x="實際銷量", y="Item", color="健康度",
                    text="Bar_Text",
                    custom_data=["SALES_NAME", "PRODUCT_NAME", "REQUEST_DATE"],
                    hover_data={"SALES_NAME":True, "REQUEST_DATE":True, "實際銷量":":,.0f", "反應速度X":True, "FVA":":.1%", "Item":False, "Bar_Text":False},
                    color_discrete_map={"🟢 表現優良": "green", "🟡 反應偏慢": "orange", "🔴 需優先關注": "red"},
                    orientation='h'
                )
                fig_top.update_xaxes(tickformat=",")
                fig_top.update_yaxes(categoryorder='total ascending', title="")
                fig_top.update_traces(textposition='inside', textfont=dict(color='white'))
                fig_top.update_layout(xaxis_title="實際銷量", height=450, margin=dict(l=0, r=0, t=30, b=0))
        
        with col_c2:
            st.markdown("#### 🔴 待改善品項分佈")
            if not df_bad.empty:
                df_bad_plot = df_bad.copy()
                df_bad_plot["Item"] = df_bad_plot.apply(lambda r: f"{r['SALES_NAME']}-{r['PRODUCT_NAME']} ({r['REQUEST_DATE']})", axis=1)
                df_bad_plot["狀態"] = "🔴 需優先介入"
                df_bad_plot['Bar_Text'] = df_bad_plot.apply(
                    lambda r: f"準度:{r['最終準確率Y']:.0%} | 速度:{r['反應速度X']:.0f}週" if pd.notna(r['反應速度X']) else f"準度:{r['最終準確率Y']:.0%} | 速度:未達標", axis=1
                )
                df_bad_plot = df_bad_plot.sort_values("實際銷量", ascending=True)
                
                fig_bad = px.bar(
                    df_bad_plot, x="實際銷量", y="Item", color="狀態",
                    text="Bar_Text",
                    custom_data=["SALES_NAME", "PRODUCT_NAME", "REQUEST_DATE"],
                    hover_data={"SALES_NAME":True, "REQUEST_DATE":True, "實際銷量":":,.0f", "反應速度X":True, "FVA":":.1%", "Item":False, "Bar_Text":False},
                    color_discrete_map={"🔴 需優先介入": "red"},
                    orientation='h'
                )
                fig_bad.update_xaxes(tickformat=",")
                fig_bad.update_yaxes(categoryorder='total ascending', title="")
                fig_bad.update_traces(textposition='inside', textfont=dict(color='white'))
                fig_bad.update_layout(xaxis_title="實際銷量", height=450, margin=dict(l=0, r=0, t=30, b=0))

    # ================= ★★★ 模式 5：學術辭典與公式透視 ★★★ =================
    elif view_mode.startswith("5"):
        st.subheader("📖 學術辭典與公式透視 (變數定義與計算白盒)")
        st.info("此視角仿造學術期刊的「變數定義表」。在了解各項指標定義後，您可於下方選擇單一品項，系統將自動把該品項的真實數據套入公式，進行「白盒（White-box）」拆解說明。")

        st.markdown("### 📚 核心變數定義表 (Variable Definitions)")
        var_data = [
            {"變數名稱 (Variable)": "最終準確率 (Final Accuracy)", "學術/管理定義": "預測最終版本與實際銷量的吻合程度", "計算邏輯": "Max(0, 1 - (|最終預測 - 實際銷量| / 實際銷量))"},
            {"變數名稱 (Variable)": "供應鏈指標得分 (Supply Chain Score)", "學術/管理定義": "評估前期備料品質的加權得分 (Phase 1 權重達 70%)", "計算邏輯": "Phase1準確率×70% + Phase2×20% + Phase3×10%"},
            {"變數名稱 (Variable)": "PP指標得分 (PP Score)", "學術/管理定義": "評估中期產能排程品質的加權得分 (Phase 2 權重達 70%)", "計算邏輯": "Phase1準確率×20% + Phase2×70% + Phase3×10%"},
            {"變數名稱 (Variable)": "FVA 預測附加價值", "學術/管理定義": "人工介入修改系統初版預測後，所提升的準確度", "計算邏輯": "最終準確率 - DCP初版準確率"},
            {"變數名稱 (Variable)": "樂觀偏誤 (Optimism Bias)", "學術/管理定義": "衡量預測是否常態性高估或低估，正值為高估(寬估)，負值為低估", "計算邏輯": "各版本 [(預測 - 實際銷量) / 實際銷量] 的平均值"},
            {"變數名稱 (Variable)": "達標反應速度 (Reaction Speed)", "學術/管理定義": "預測數字首次進入合理誤差範圍 (依左側設定) 所耗費的時間", "計算邏輯": "首次達標週次 - DCP定板週次 (若從未達標則記為空值)"},
            {"變數名稱 (Variable)": "Phase 2 下修次數", "學術/管理定義": "在中期備料階段，主動將過度膨脹的數字向下修正的次數", "計算邏輯": "Phase 2 期間，本版數字 < 前版數字的總次數"},
            {"變數名稱 (Variable)": "Phase 3 預測變動 (Tinkering)", "學術/管理定義": "臨近交期(產線已排定)仍進行小幅度修改的\次數", "計算邏輯": "Phase 3 期間，變動幅度 >0% 且 <20% 的總次數"}
        ]
        st.table(pd.DataFrame(var_data))

        st.markdown("---")
        
        st.markdown("### 🧮 白盒計算透視機 (Formula Breakdown)")
        if df_show.empty:
            st.warning("目前篩選條件下無資料，請調整左側篩選器。")
        else:
            df_show_copy = df_show.copy()
            df_show_copy["Item_Key"] = df_show_copy.apply(
                lambda r: f"{r['SALES_NAME']} - {r['PRODUCT_NAME']} (目標月: {r['REQUEST_DATE']})", axis=1
            )
            
            selected_item = st.selectbox("🎯 請選擇一筆品項，親眼見證它的計算拆解過程：", df_show_copy["Item_Key"].unique())
            
            if selected_item:
                row = df_show_copy[df_show_copy["Item_Key"] == selected_item].iloc[0]
                
                act = row["實際銷量"]
                pred_final = row["最終預測"]
                pred_initial = row["初版預測"]
                acc_final = row["最終準確率Y"]
                acc_dcp = row["DCP準確率"]
                fva = row["FVA"]
                bias = row["樂觀偏誤"]
                
                # 取得各階段準確率
                p1 = row["P1得分"]
                p2 = row["P2得分"]
                p3 = row["P3得分"]
                sc_score = row["加權準確率_供應鏈"]
                pp_score = row["PP分數"]
                
                st.markdown(f"#### 正在解析： `{selected_item}`")
                st.markdown(f"**核心基礎數據**：實際銷量 = **`{act:,.0f}`** | 初版預測 (DCP) = **`{pred_initial:,.0f}`** | 最終預測 = **`{pred_final:,.0f}`**")
                
                c_math1, c_math2 = st.columns(2)
                
                with c_math1:
                    with st.expander("1️⃣ 最終準確率 (Final Accuracy) 拆解", expanded=True):
                        st.latex(r"Accuracy = \max\left(0, 1 - \frac{|Forecast_{final} - Actual|}{Actual}\right)")
                        if act > 0:
                            err_abs = abs(pred_final - act)
                            st.write(f"**套入真實數據**：")
                            st.write(f"$1 - \\frac{{|{pred_final:,.0f} - {act:,.0f}|}}{{{act:,.0f}}} = 1 - \\frac{{{err_abs:,.0f}}}{{{act:,.0f}}}$")
                            st.write(f"**計算結果**： **`{acc_final:.1%}`**")
                        else:
                            st.write("實際銷量為 0，無法計算分母。")

                    with st.expander("2️⃣ 供應鏈指標與 PP 指標 拆解", expanded=True):
                        st.latex(r"Supply Chain Score = P1 \times 70\% + P2 \times 20\% + P3 \times 10\%")
                        st.latex(r"PP Score = (P1 \times 20\% + P2 \times 70\% + P3 \times 10\%) \times 100")
                        st.write(f"**各階段準確率**： P1=`{p1:.1%}`, P2=`{p2:.1%}`, P3=`{p3:.1%}`")
                        st.write(f"**供應鏈指標計算結果**： **`{sc_score:.1%}`**")
                        st.write(f"**PP 指標計算結果**： **`{pp_score:.1f}`** 分")

                with c_math2:
                    with st.expander("3️⃣ FVA 預測附加價值 拆解", expanded=True):
                        st.latex(r"FVA = Accuracy_{final} - Accuracy_{initial}")
                        st.write(f"**套入真實數據**： `{acc_final:.1%}` (最終準確率) - `{acc_dcp:.1%}` (初版準確率)")
                        st.write(f"**計算結果**： **`{fva:.1%}`**")
                        if fva > 0: st.info("解讀：正值代表業務的人工修改，成功讓最終數字比電腦初版更準確。")
                        else: st.warning("解讀：非正值代表人工修改『沒有幫上忙』或甚至『幫倒忙』。")

                    with st.expander("4️⃣ 樂觀偏誤 (Optimism Bias) 拆解", expanded=True):
                        st.latex(r"Bias = \frac{1}{n} \sum \frac{Forecast_{i} - Actual}{Actual}")
                        st.write(f"**套入真實數據**： 系統將各版本的 `(預測 - 實際) / 實際` 誤差率相加後取平均。")
                        st.write(f"**計算結果**： **`{bias:.1%}`**")
                        if bias > 0.15: st.error("解讀：偏誤大於 15%，證明該業務有常態性『防禦性寬估(多報)』的習慣。")
                        elif bias < -0.15: st.error("解讀：偏誤小於 -15%，證明該業務有常態性『嚴重低估(少報)』的習慣。")
                        else: st.success("解讀：無明顯方向性偏誤。")

    # ================= 顯示圖表與鑽取連動 (模式1~4共用) =================
    selection = None

    if not view_mode.startswith("5"):
        if view_mode.startswith("4"):
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                if fig_top:
                    sel1 = st.plotly_chart(fig_top, use_container_width=True, on_select="rerun", key=f"{chart_key}_top")
                    if sel1 and len(sel1.get("selection", {}).get("points", [])) > 0:
                        selection = sel1
            with c_chart2:
                if fig_bad:
                    sel2 = st.plotly_chart(fig_bad, use_container_width=True, on_select="rerun", key=f"{chart_key}_bad")
                    if sel2 and len(sel2.get("selection", {}).get("points", [])) > 0:
                        selection = sel2
        else:
            if fig:
                selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
                
        if selection and len(selection.get("selection", {}).get("points", [])) > 0:
            try:
                point_data = selection["selection"]["points"][0]
                sel_sales = None
                sel_prod = None
                sel_req_date = None
                
                if "customdata" in point_data:
                    cdata = point_data["customdata"]
                    if len(cdata) >= 2:
                        sel_sales = cdata[0]
                        sel_prod = cdata[1]
                    if len(cdata) >= 3:
                        sel_req_date = cdata[2]
                
                if sel_sales and sel_prod:
                    mask = (df_res["SALES_NAME"] == sel_sales) & (df_res["PRODUCT_NAME"] == sel_prod)
                    
                    if sel_req_date is not None and str(sel_req_date) not in ["nan", "NaT", "None"]:
                        mask = mask & (df_res["REQUEST_DATE"].astype(str) == str(sel_req_date))
                        
                    target_row = df_res[mask]
                    
                    if not target_row.empty:
                        selected_row = target_row.iloc[0]
                        sel_actual = selected_row["實際銷量"]
                        
                        st.markdown("---")
                        st.subheader(f"🔍 深度診斷與優化建議報告：{sel_sales} - {sel_prod} ({sel_req_date})")
                        
                        # ================= ★★★ 詳盡白話文版 AI 診斷重點報告 ★★★ =================
                        bias = selected_row["樂觀偏誤"]
                        fva = selected_row["FVA"]
                        tinkering = selected_row["無效微調次數"]
                        speed = selected_row["反應速度X"]
                        final_acc = selected_row["最終準確率Y"]
                        proc_score = selected_row["加權準確率_供應鏈"]
                        pp_score = selected_row["PP分數"]
                        downward_count = selected_row["下修次數"]
                        
                        speed_str = f"第 {speed:.0f} 週" if pd.notna(speed) else "至出貨前皆未"
                        err_txt = f"±{thr_error*100:.0f}%"
                        
                        report_type = "info"
                        s1, s2 = "", ""
                        
                        if bias < -0.15 and fva <= 0: 
                            report_type = "error"
                            s1 = f"歷史數據顯示，該品項預測有明顯的「嚴重低估」現象（平均偏誤約 {bias:.1%}）。耗費了 {speed_str} 才將數字修正至達標區間（誤差 {err_txt} 內）。進一步檢視，中期（Phase 2）下修了 {downward_count:.0f} 次，臨近出貨（Phase 3）又變動了 {tinkering:.0f} 次，但整體人工介入並未帶來正向價值（FVA 為 {fva:.1%}）。最終準確率為 {final_acc:.1%}，供應鏈與PP指標得分分別為 {proc_score:.1%} 與 {pp_score:.1f} 分。"
                            s2 = "低估預測是供應鏈中最危險的行為之一，直接導致備料不足與潛在缺貨。反覆的微調也顯示需求掌握度不佳。請務必優先介入此品項，確保前端業務能更透明地反映真實市場拉貨動能，減少無效的系統變動。"
                        elif bias > 0.15 and fva <= 0:
                            report_type = "warning" 
                            s1 = f"該品項前期的預測有「防禦性寬估」的傾向（平均偏誤約 {bias:.1%}）。從歷程來看，雖然經過 {downward_count:.0f} 次 Phase 2 下修與 {tinkering:.0f} 次 Phase 3 微調，但花費了約 {speed_str} 才達標。整體人工修正效益有限（FVA 為 {fva:.1%}）。最終準確率達到 {final_acc:.1%}，但受前期高估影響，供應鏈指標得分僅 {proc_score:.1%}，PP得分為 {pp_score:.1f} 分。"
                            s2 = "前期持續偏高且達標速度慢，會大幅增加資材備料的資金積壓風險 (呆滯風險)。過多的微調也徒增管理成本。建議在前期備料階段 (Phase 1) 就與業務端對焦，適度擠出預測水分。"
                        elif tinkering > 2:
                            report_type = "warning"
                            s1 = f"在臨近交期的關鍵期（Phase 3），系統觀察到高達 {tinkering:.0f} 次的小幅變動。雖然中期有 {downward_count:.0f} 次下修，且最終準確率達 {final_acc:.1%}，但花費 {speed_str} 才達標。臨近出貨前的頻繁微調對提升準確度實質幫助有限（FVA 為 {fva:.1%}），供應鏈與PP指標得分為 {proc_score:.1%} 與 {pp_score:.1f} 分。"
                            s2 = "臨近交期的反覆變動極易引發排程系統重算與應變成本，產生排產雜訊。建議若無重大市場異動，應維持預測穩定性，或考慮鎖定出貨前夕的系統調整權限。"
                        elif pd.notna(speed) and speed > 8:
                            report_type = "warning"
                            s1 = f"預測修正反應偏慢，DCP 定板後歷經 {speed_str} 才首度校正至達標區間。期間累積了 {downward_count:.0f} 次 Phase 2 下修與 {tinkering:.0f} 次 Phase 3 微調。因遲遲未給出正確數字，導致供應鏈指標得分較低（{proc_score:.1%}）。儘管最終準確率拉升至 {final_acc:.1%}（PP得分 {pp_score:.1f} 分），但預測附加價值 (FVA) 僅為 {fva:.1%}。"
                            s2 = "正確需求數字較晚反映，會迫使工廠承擔高昂急單與緊急調料成本。建議將終端情報更早回饋至系統，提升初期反應速度。"
                        elif fva > 0.1 and final_acc > 0.8:
                            report_type = "success"
                            s1 = f"這是值得參考的優質預測案例！該品項在 {speed_str} 便迅速達標，期間進行了 {downward_count:.0f} 次有效的 Phase 2 下修，且 Phase 3 變動僅 {tinkering:.0f} 次。優良的操作帶來高達 {fva:.1%} 的 FVA 附加價值，最終準確率 {final_acc:.1%}，供應鏈與PP得分更高達 {proc_score:.1%} 與 {pp_score:.1f} 分。"
                            s2 = "展現了極高的市場敏銳度與精準的下修決策。此穩定且反應迅速的模式為產銷協調提供了極佳基礎，建議將此預測邏輯列為團隊學習標竿。"
                        else:
                            s1 = f"預測表現整體平穩。歷經 {speed_str} 進入達標區間，伴隨 {downward_count:.0f} 次 Phase 2 下修與 {tinkering:.0f} 次 Phase 3 微調。最終準確率為 {final_acc:.1%}。人工微調結果與系統基準表現相當（FVA 為 {fva:.1%}），供應鏈與PP得分為 {proc_score:.1%} 及 {pp_score:.1f} 分。"
                            s2 = "由於需求相對穩定，人工反覆介入的額外效益較不明顯。建議未來針對此類品項，可評估逐步導入「系統輔助預測 (Baseline)」作為基礎，降低人工作業負擔。"
                        
                        if report_type == "error":
                            st.error(f"💡 **系統客觀診斷與建議**\n\n1. **數據現象觀察**：{s1}\n2. **潛針對影響與優化方向**：{s2}")
                        elif report_type == "warning":
                            st.warning(f"💡 **系統客觀診斷與建議**\n\n1. **數據現象觀察**：{s1}\n2. **潛在影響與優化方向**：{s2}")
                        elif report_type == "success":
                            st.success(f"💡 **系統客觀診斷與建議**\n\n1. **數據現象觀察**：{s1}\n2. **正面效益與後續運用**：{s2}")
                        else:
                            st.info(f"💡 **系統客觀診斷與建議**\n\n1. **數據現象觀察**：{s1}\n2. **資源優化建議**：{s2}")
                        
                        st.write("") 
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            mask_hist = (df_raw["SALES_NAME"] == sel_sales) & (df_raw["PRODUCT_NAME"] == sel_prod)
                            
                            if sel_req_date is not None and str(sel_req_date) not in ["nan", "NaT", "None"]:
                                mask_hist = mask_hist & (df_raw["REQUEST_DATE"].astype(str) == str(sel_req_date))
                                
                            history_df = df_raw[mask_hist].copy()
                            
                            dcp_date_val = history_df.iloc[0]["DCP_DATE"] if (not history_df.empty and "DCP_DATE" in history_df.columns) else None
                            
                            if pd.notna(dcp_date_val):
                                history_df = history_df[history_df["DONE_DATE"] >= dcp_date_val].copy()
                            
                            history_df["DONE_DATE_STR"] = history_df["DONE_DATE"].apply(
                                lambda x: x.strftime("%Y/%m/%d") if pd.notna(x) else ""
                            )
                            
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Scatter(
                                x=history_df["DONE_DATE_STR"], y=[sel_actual]*len(history_df),
                                mode='lines', name='實際出貨量', line=dict(color='green', dash='dash')
                            ))
                            fig_hist.add_trace(go.Scatter(
                                x=history_df["DONE_DATE_STR"], y=history_df["BFC_QTY"],
                                mode='lines+markers', name='人為調整預測', line=dict(color='orange', width=3)
                            ))
                            
                            dcp_fcst_val = selected_row["初版預測"]
                            if pd.notna(dcp_fcst_val):
                                fig_hist.add_trace(go.Scatter(
                                    x=history_df["DONE_DATE_STR"], y=[dcp_fcst_val]*len(history_df),
                                    mode='lines', name='系統基準(不作修改)', line=dict(color='purple', dash='dot', width=2)
                                ))
                            
                            if pd.notna(dcp_date_val):
                                try:
                                    if isinstance(dcp_date_val, date):
                                        dcp_fmt = dcp_date_val.strftime("%Y/%m/%d")
                                        fig_hist.add_vline(x=dcp_fmt, line_dash="dot", line_color="blue", annotation_text="DCP起點")
                                except: pass

                            fig_hist.update_layout(
                                title="預測演變軌跡 vs. 系統基準對照", 
                                xaxis_title="預測更新日期", 
                                yaxis_title="數量",
                                yaxis=dict(tickformat=","),
                                xaxis=dict(tickangle=-45),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_drilldown")
                            
                        with col2:
                            st.markdown("### 🧬 指標深度解析")
                            
                            fva_color = "normal" if fva >= 0 else "inverse"
                            fva_desc = "👍 具備正向修正價值" if fva > 0 else "📉 調整未達顯著效益" if fva < 0 else "持平"
                            st.metric("FVA 預測附加價值", f"{fva:.1%}", delta=fva_desc, delta_color=fva_color)
                            
                            if bias > 0.2: st.warning(f"🟡 防禦性寬估傾向 (平均高出 {bias:.1%})") 
                            elif bias < -0.2: st.error(f"🔴 嚴重低估傾向 (平均低於 {abs(bias):.1%})") 
                            else: st.success(f"⚖️ 無顯著方向性偏誤 (Bias: {bias:.1%})")
                            
                            st.divider()
                            c_sub1, c_sub2 = st.columns(2)
                            
                            with c_sub1:
                                st.metric("Phase 2 下修次數", f"{selected_row['下修次數']:.0f} 次")
                                st.caption("中期有效收斂次數")
                            with c_sub2:
                                st.metric("Phase 3 預測變動", f"{selected_row['無效微調次數']:.0f} 次")
                                st.caption("臨近出貨小幅調整次數")

                            st.divider()
                            st.markdown("### 📊 綜合效能成績")
                            c_k1, c_k2 = st.columns(2)
                            c_k1.metric("達標反應速度", f"{selected_row['反應速度X']:.0f} 週" if pd.notna(selected_row['反應速度X']) else "未達標")
                            c_k2.metric("最終準確率", f"{selected_row['最終準確率Y']:.1%}")
                            c_k1.metric("供應鏈指標得分", f"{selected_row['加權準確率_供應鏈']:.1%}")
                            c_k2.metric("PP指標得分", f"{selected_row['PP分數']:.1f}")

                else:
                    st.warning("⚠️ 無法讀取該點資料，可能資料已被過濾，請嘗試點擊其他長條圖。")
            except Exception as e:
                st.error(f"發生未預期的錯誤: {e}")
        else:
            st.info("👆 點擊上方圖表柱狀區塊，即可解鎖『預測演變軌跡』與『系統客觀診斷報告』")

    st.markdown("---")
    st.subheader("📋 診斷資料明細匯出")
    df_table = df_show.drop(columns=["Plot_X", "象限", "最終狀態", "BIAS_PCT", "ABS_PCT_ERR", "SINGLE_ACCURACY", "WB_BIN", "ACCURACY_SCORE", "DCP象限", "Item", "Item_Key", "健康度", "偏誤狀態"], errors="ignore").copy()
    
    for c in ["實際銷量", "最終預測", "初版預測", "BFC_QTY", "SHIPMENT_QTY"]:
        if c in df_table.columns: df_table[c] = df_table[c].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
        
    for c in ["最終準確率Y", "加權準確率_供應鏈", "命中率", "平均偏差", "DCP準確率", "DCP後不準率", "樂觀偏誤", "FVA"]:
        if c in df_table.columns: df_table[c] = df_table[c].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "NaN")
    if "PP分數" in df_table.columns:
        df_table["PP分數"] = df_table["PP分數"].apply(lambda x: f"{x:.1f}")
        
    st.dataframe(df_table, use_container_width=True)

else:
    st.info("請上傳 Excel 檔案")
