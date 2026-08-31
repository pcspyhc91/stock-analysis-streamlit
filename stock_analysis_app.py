# 檔名：202608_2330new_streamlit_股票選單_自訂日期_欄位指定版.py
# 執行方式：
#   streamlit run 202608_2330new_streamlit.py
#
# 本程式使用「本地 NotoSansTC-Regular.ttf」。
# 請將 NotoSansTC-Regular.ttf 與本 .py 放在同一個資料夾。

from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import mplfinance.original_flavor as mpf


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="2330 股市技術分析互動網站",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Local Noto Sans TC Font
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FONT_PATH = BASE_DIR / "NotoSansTC-Regular.ttf"


def setup_local_font():
    """
    同時設定：
    1. Matplotlib 圖表中文字型
    2. Streamlit 網頁中文字型
    """
    if not FONT_PATH.exists():
        st.error(
            "找不到本地字型檔：NotoSansTC-Regular.ttf\n\n"
            "請將 `NotoSansTC-Regular.ttf` 與本程式放在同一個資料夾後重新執行。"
        )
        st.stop()

    # Matplotlib 使用本地字型
    fm.fontManager.addfont(str(FONT_PATH))
    font_prop = fm.FontProperties(fname=str(FONT_PATH))
    font_name = font_prop.get_name()

    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["axes.unicode_minus"] = False

    # Streamlit / Browser 使用本地字型
    # 以 base64 內嵌，部署時不依賴作業系統是否安裝該字型
    import base64

    font_base64 = base64.b64encode(FONT_PATH.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Noto Sans TC Local';
            src: url(data:font/ttf;base64,{font_base64}) format('truetype');
            font-weight: 400;
            font-style: normal;
        }}

        html, body, [class*="css"], [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], button, input, textarea, select {{
            font-family: 'Noto Sans TC Local', sans-serif !important;
        }}

        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            font-family: 'Noto Sans TC Local', sans-serif !important;
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.65rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return font_name


FONT_NAME = setup_local_font()


# ============================================================
# RSI
# ============================================================

def calculate_yahoo_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# Download + Technical Indicators
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(stock_id, start_date_text, end_date_text, warmup_days):
    # 使用者自訂觀測起始日與觀測結束日
    observation_start = datetime.strptime(start_date_text, "%Y-%m-%d").date()
    observation_end = datetime.strptime(end_date_text, "%Y-%m-%d").date()

    # 額外預熱資料，供 SMA / MACD / RSI 等技術指標計算
    warmup_start = observation_start - timedelta(days=warmup_days)

    # yfinance 的 end 為不含當日（exclusive），因此 +1 天，
    # 才能讓使用者選擇的「觀測結束日」被納入下載範圍。
    download_end = observation_end + timedelta(days=1)

    df = yf.download(
        stock_id,
        start=warmup_start,
        end=download_end,
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(), observation_start, observation_end, warmup_start

    # yfinance 若回傳 MultiIndex，展平欄位名稱
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}

    if not required_columns.issubset(df.columns):
        return pd.DataFrame(), observation_start, observation_end, warmup_start

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------
    df["middle_band"] = df["SMA_20"]
    df["std_dev"] = df["Close"].rolling(window=20).std()

    df["upper_band"] = (
        df["middle_band"]
        + (df["std_dev"] * 2)
    )

    df["lower_band"] = (
        df["middle_band"]
        - (df["std_dev"] * 2)
    )

    # --------------------------------------------------------
    # RSV / K / D / J
    # --------------------------------------------------------
    n = 9

    low_min = df["Low"].rolling(window=n).min()
    high_max = df["High"].rolling(window=n).max()

    denominator = (high_max - low_min).replace(0, np.nan)

    df["RSV"] = (
        (df["Close"] - low_min)
        / denominator
    ) * 100

    df["K"] = df["RSV"].ewm(
        alpha=1/3,
        adjust=False
    ).mean()

    df["D"] = df["K"].ewm(
        alpha=1/3,
        adjust=False
    ).mean()

    # 保留原始程式公式
    df["J"] = (
        3 * df["D"]
        - 2 * df["K"]
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------
    df["OBV"] = np.where(
        df["Close"] > df["Close"].shift(1),
        df["Volume"],
        -df["Volume"]
    )

    df["OBV"] = df["OBV"].cumsum()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------
    fast_period = 12
    slow_period = 26
    signal_period = 9

    df["EMA12"] = df["Close"].ewm(
        span=fast_period,
        adjust=False
    ).mean()

    df["EMA26"] = df["Close"].ewm(
        span=slow_period,
        adjust=False
    ).mean()

    df["DIF"] = (
        df["EMA12"]
        - df["EMA26"]
    )

    df["MACD"] = df["DIF"].ewm(
        span=signal_period,
        adjust=False
    ).mean()

    df["MACD Histogram"] = (
        df["DIF"]
        - df["MACD"]
    )

    # --------------------------------------------------------
    # RSI5 / RSI10
    # --------------------------------------------------------
    df["RSI5"] = calculate_yahoo_rsi(
        df["Close"],
        period=5
    )

    df["RSI10"] = calculate_yahoo_rsi(
        df["Close"],
        period=10
    )

    # --------------------------------------------------------
    # BIAS10 / BIAS20 / B10-B20
    # --------------------------------------------------------
    df["BIAS10"] = (
        (df["Close"] - df["SMA_10"])
        / df["SMA_10"]
    ) * 100

    df["BIAS20"] = (
        (df["Close"] - df["SMA_20"])
        / df["SMA_20"]
    ) * 100

    df["B10-B20"] = (
        df["BIAS10"]
        - df["BIAS20"]
    )

    # 正式切取使用者選擇的觀測期間
    # datetime.date 明確轉成 pd.Timestamp，避免 Pandas4Warning
    df = df.loc[
        pd.Timestamp(observation_start):pd.Timestamp(observation_end),
        :
    ].copy()

    return df, observation_start, observation_end, warmup_start


# ============================================================
# Matplotlib Chart
# ============================================================

def create_technical_chart(df, stock_id, tick_interval):
    plot_df = df.copy()

    # Matplotlib / mplfinance 原始風格使用連續整數 X 軸
    plot_df["Date_Label"] = plot_df.index.strftime("%y-%m-%d")
    plot_df = plot_df.reset_index(drop=True)

    x = np.arange(len(plot_df))
    tick_interval = max(1, int(tick_interval))
    x_ticks_pos = x[::tick_interval]
    x_ticks_labels = plot_df["Date_Label"].iloc[::tick_interval]

    fig = plt.figure(
        figsize=(14, 12),
        layout="constrained"
    )

    # ========================================================
    # 1. K線 + SMA + Bollinger Bands
    # ========================================================

    ax1 = fig.add_subplot(8, 1, (1, 3))

    mpf.candlestick2_ochl(
        ax1,
        plot_df["Open"],
        plot_df["Close"],
        plot_df["High"],
        plot_df["Low"],
        width=0.8,
        colorup="r",
        colordown="g",
        alpha=1
    )

    ax1.plot(
        x,
        plot_df["SMA_5"],
        label="5日均線",
        alpha=0.9,
        color="cyan",
        lw=0.7
    )

    ax1.plot(
        x,
        plot_df["SMA_10"],
        label="10日均線",
        alpha=0.9,
        color="purple",
        lw=0.7
    )

    ax1.plot(
        x,
        plot_df["SMA_20"],
        label="20日均線",
        alpha=0.9,
        color="orange",
        lw=0.7
    )

    ax1.plot(
        x,
        plot_df["upper_band"],
        label="upperband",
        alpha=0.9,
        color="green",
        ls=":"
    )

    ax1.plot(
        x,
        plot_df["lower_band"],
        label="lowerband",
        alpha=0.9,
        color="green",
        ls=":"
    )

    ax1.set_xticks(x_ticks_pos)
    ax1.set_xticklabels([])
    ax1.legend(loc=0, fontsize=8)

    ax1.set_title(
        f"{stock_id} 股市技術分析互動圖",
        fontsize=18
    )

    # ========================================================
    # 2. OBV + Volume
    # ========================================================

    ax2 = fig.add_subplot(8, 1, 4)

    conditions = [
        plot_df["Close"] > plot_df["Close"].shift(1),
        plot_df["Close"] < plot_df["Close"].shift(1)
    ]

    choices = ["r", "g"]

    volume_colors = np.select(
        conditions,
        choices,
        default="gray"
    )

    ax2.plot(
        x,
        plot_df["OBV"],
        color="purple",
        linestyle="--",
        label="OBV"
    )

    ax2.legend(loc=1, fontsize=8)
    ax2.set_xticks(x_ticks_pos)
    ax2.set_xticklabels([])

    ax2_1 = ax2.twinx()

    ax2_1.bar(
        x,
        plot_df["Volume"],
        color=volume_colors,
        width=0.8,
        alpha=0.8
    )

    red_patch = mpatches.Patch(
        color="red",
        label="紅色漲"
    )

    green_patch = mpatches.Patch(
        color="green",
        label="綠色跌"
    )

    gray_patch = mpatches.Patch(
        color="gray",
        label="灰持平"
    )

    ax2_1.legend(
        handles=[
            red_patch,
            green_patch,
            gray_patch
        ],
        loc=2,
        title="交易量",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.5
    )

    # ========================================================
    # 3. KDJ
    # ========================================================

    ax3 = fig.add_subplot(8, 1, 5)

    ax3.plot(
        x,
        plot_df["K"],
        label="K line",
        color="cyan",
        lw=0.7
    )

    ax3.plot(
        x,
        plot_df["D"],
        label="D line",
        color="purple",
        lw=0.7
    )

    ax3.plot(
        x,
        plot_df["J"],
        label="J line",
        linestyle="--",
        color="orange"
    )

    ax3.set_xticks(x_ticks_pos)
    ax3.set_xticklabels([])
    ax3.legend(loc=0, fontsize=8)

    # ========================================================
    # 4. MACD
    # ========================================================

    ax4 = fig.add_subplot(8, 1, 6)

    ax4.plot(
        x,
        plot_df["DIF"],
        label="DIF9",
        color="purple",
        lw=0.8
    )

    ax4.plot(
        x,
        plot_df["MACD"],
        label="MACD",
        color="skyblue",
        lw=0.8
    )

    macd_colors = np.where(
        plot_df["MACD Histogram"] >= 0,
        "r",
        "g"
    )

    ax4.bar(
        x,
        plot_df["MACD Histogram"],
        color=macd_colors,
        alpha=0.8
    )

    ax4.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0
    )

    ax4.set_xticks(x_ticks_pos)
    ax4.set_xticklabels([])

    # 保留原始程式的 MACD Y 軸設定
    ax4.set_ylim(-100, 100)

    macd_red_patch = mpatches.Patch(
        color="red",
        label="MACD多頭"
    )

    macd_green_patch = mpatches.Patch(
        color="green",
        label="MACD空頭"
    )

    handles, labels = ax4.get_legend_handles_labels()

    handles.extend([
        macd_red_patch,
        macd_green_patch
    ])

    ax4.legend(
        handles=handles,
        loc=2,
        fontsize=8,
        framealpha=0.5
    )

    # ========================================================
    # 5. RSI
    # ========================================================

    ax5 = fig.add_subplot(8, 1, 7)

    ax5.plot(
        x,
        plot_df["RSI5"],
        label="RSI5",
        color="cyan",
        lw=0.7
    )

    ax5.plot(
        x,
        plot_df["RSI10"],
        label="RSI10",
        color="purple",
        lw=0.7
    )

    ax5.set_xticks(x_ticks_pos)
    ax5.set_xticklabels([])
    ax5.set_ylim(0, 100)

    ax5.axhline(
        70,
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5
    )

    ax5.axhline(
        30,
        color="green",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5
    )

    ax5.legend(loc=2, fontsize=8)

    # ========================================================
    # 6. BIAS
    # ========================================================

    ax6 = fig.add_subplot(8, 1, 8)

    ax6.plot(
        x,
        plot_df["BIAS10"],
        label="BIAS10",
        color="cyan",
        lw=0.7
    )

    ax6.plot(
        x,
        plot_df["BIAS20"],
        label="BIAS20",
        color="purple",
        lw=0.7
    )

    bias_colors = np.where(
        plot_df["B10-B20"] >= 0,
        "r",
        "g"
    )

    ax6.bar(
        x,
        plot_df["B10-B20"],
        color=bias_colors,
        alpha=0.8
    )

    ax6.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.0
    )

    max_bias = max(
        float(plot_df["B10-B20"].max()),
        15
    )

    min_bias = min(
        float(plot_df["B10-B20"].min()),
        -15
    )

    ax6.set_ylim(
        min_bias * 1.1,
        max_bias * 1.1
    )

    bias_red_patch = mpatches.Patch(
        color="red",
        label="BIAS正強"
    )

    bias_green_patch = mpatches.Patch(
        color="green",
        label="BIAS負弱"
    )

    handles, labels = ax6.get_legend_handles_labels()

    handles.extend([
        bias_red_patch,
        bias_green_patch
    ])

    ax6.set_xticks(x_ticks_pos)
    ax6.set_xticklabels(
        x_ticks_labels,
        rotation=45,
        ha="right",
        fontsize=8
    )

    ax6.legend(
        handles=handles,
        loc=2,
        fontsize=8,
        framealpha=0.5
    )

    return fig


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("📌 查詢設定")

    # 股票代碼選單
    stock_options = {
        "元大台灣50 (0050.TW)": "0050.TW",
        "台積電 (2330.TW)": "2330.TW",
        "鴻海 (2317.TW)": "2317.TW",
        "聯發科 (2454.TW)": "2454.TW",
        "廣達 (2382.TW)": "2382.TW",
        "台達電 (2308.TW)": "2308.TW",
        "中華電 (2412.TW)": "2412.TW",
        "富邦金 (2881.TW)": "2881.TW",
        "國泰金 (2882.TW)": "2882.TW",
        "兆豐金 (2886.TW)": "2886.TW",
        "長榮 (2603.TW)": "2603.TW",
        "陽明 (2609.TW)": "2609.TW",
    }

    selected_stock = st.selectbox(
        "選擇股票代碼",
        options=list(stock_options.keys()),
        index=1,
        help="請從清單中選擇股票，系統會自動帶入 Yahoo Finance 股票代碼。"
    )

    stock_id = stock_options[selected_stock]

    # --------------------------------------------------------
    # 自訂觀測期間
    # --------------------------------------------------------
    default_end_date = datetime.today().date()
    default_start_date = default_end_date - timedelta(days=180)

    observation_start = st.date_input(
        "觀測起始日",
        value=default_start_date,
        help="選擇技術分析圖表的開始日期。"
    )

    observation_end = st.date_input(
        "觀測結束日",
        value=default_end_date,
        help="選擇技術分析圖表的結束日期。"
    )

    # 日期合理性檢查
    if observation_start > observation_end:
        st.error("觀測起始日不可晚於觀測結束日。")
        st.stop()

    warmup_days = st.slider(
        "指標預熱天數",
        min_value=30,
        max_value=120,
        value=60,
        step=10
    )

    tick_interval = st.slider(
        "X 軸日期間隔",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )

    show_data = st.checkbox(
        "顯示技術指標資料表",
        value=False
    )

    if st.button(
        "🔄 重新整理資料",
        use_container_width=True
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "字型：本地 NotoSansTC-Regular.ttf"
    )


# ============================================================
# Main Page
# ============================================================

st.title("📈 台股技術分析互動網站")
st.caption(
    "K線 × SMA × Bollinger Bands × OBV × KDJ × MACD × RSI × BIAS"
)

if not stock_id:
    st.warning("請選擇股票代碼。")
    st.stop()


with st.spinner(f"正在下載 {stock_id} 股票資料並計算技術指標..."):

    df, formal_start, formal_end, warmup_start = load_stock_data(
        stock_id=stock_id,
        start_date_text=observation_start.strftime("%Y-%m-%d"),
        end_date_text=observation_end.strftime("%Y-%m-%d"),
        warmup_days=warmup_days
    )


if df.empty:
    st.error(
        f"無法取得 `{stock_id}` 的股票資料。"
        "請檢查股票代號、日期或網路連線。"
    )
    st.stop()


# ============================================================
# Latest Metrics
# ============================================================

latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) >= 2 else latest

close_value = float(latest["Close"])
close_change = close_value - float(previous["Close"])
close_pct = (
    close_change / float(previous["Close"]) * 100
    if float(previous["Close"]) != 0
    else 0.0
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "最新收盤價",
    f"{close_value:,.2f}",
    f"{close_change:+,.2f} ({close_pct:+.2f}%)"
)

m2.metric(
    "成交量",
    f"{float(latest['Volume']):,.0f}"
)

m3.metric(
    "RSI5",
    f"{float(latest['RSI5']):.2f}"
)

m4.metric(
    "MACD Histogram",
    f"{float(latest['MACD Histogram']):.2f}"
)

m5.metric(
    "BIAS10",
    f"{float(latest['BIAS10']):.2f}%"
)


# ============================================================
# Period Information
# ============================================================

st.info(
    f"股票：{stock_id} ｜ "
    f"觀測期間：{formal_start.strftime('%Y-%m-%d')} ～ {formal_end.strftime('%Y-%m-%d')} ｜ "
    f"預熱起始日：{warmup_start.strftime('%Y-%m-%d')} ｜ "
    f"實際交易資料：{len(df):,} 筆"
)


# ============================================================
# Technical Chart
# ============================================================

fig = create_technical_chart(
    df=df,
    stock_id=stock_id,
    tick_interval=tick_interval
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# Latest Technical Signal Summary
# ============================================================

st.subheader("📊 最新技術指標摘要")

c1, c2 = st.columns(2)

with c1:
    if latest["RSI5"] >= 70:
        st.warning("RSI5：位於 70 以上（超買區）")
    elif latest["RSI5"] <= 30:
        st.success("RSI5：位於 30 以下（超賣區）")
    else:
        st.info("RSI5：位於 30～70 中性區間")

    if latest["MACD Histogram"] >= 0:
        st.error("MACD Histogram：多頭區（紅柱）")
    else:
        st.success("MACD Histogram：空頭區（綠柱）")

with c2:
    if latest["B10-B20"] >= 0:
        st.error("BIAS：BIAS正強（紅柱）")
    else:
        st.success("BIAS：BIAS負弱（綠柱）")

    if latest["Close"] >= latest["SMA_20"]:
        st.info("收盤價：位於 20 日均線之上")
    else:
        st.info("收盤價：位於 20 日均線之下")


# ============================================================
# Data Table
# ============================================================

if show_data:
    st.subheader("🧾 技術指標資料表")

    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "upper_band",
        "lower_band",
        "OBV",
        "K",
        "D",
        "J",
        "DIF",
        "MACD",
        "MACD Histogram",
        "RSI5",
        "RSI10",
        "BIAS10",
        "BIAS20",
        "B10-B20",
    ]

    table_df = df[display_columns].copy()
    table_df.index = table_df.index.strftime("%Y-%m-%d")
    table_df.index.name = "日期"

    # --------------------------------------------------------
    # Data Table 欄位名稱：English Term（繁體中文）
    # --------------------------------------------------------
    display_column_names = {
        "Open": "開盤價",
        "High": "最高價",
        "Low": "最低價",
        "Close": "收盤價",
        "Volume": "成交量",
        "SMA_5": "SMA_5",
        "SMA_10": "SMA_10",
        "SMA_20": "SMA_20",
        "upper_band": "Upper Band",
        "lower_band": "Lower Band",
        "OBV": "OBV",
        "K": "K值",
        "D": "D值",
        "J": "J值",
        "DIF": "DIF",
        "MACD": "MACD",
        "MACD Histogram": "MACD柱狀圖",
        "RSI5": "RSI5日指標",
        "RSI10": "RSI10日指標",
        "BIAS10": "BIAS10日乖離率",
        "BIAS20": "BIAS20日乖離率",
        "B10-B20": "B10-B20差值",
    }

    table_df = table_df.rename(columns=display_column_names)

    st.dataframe(
        table_df.sort_index(ascending=False),
        use_container_width=True,
        height=500
    )

    csv_data = table_df.to_csv(
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ 下載技術指標 CSV",
        data=csv_data,
        file_name=f"{stock_id}_technical_indicators.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# Footer
# ============================================================

st.caption(
    "資料來源：Yahoo Finance（yfinance）。"
    "技術指標僅供教學與資料分析展示，不構成投資建議。"
)
