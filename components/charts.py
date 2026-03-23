import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def candlestick_with_indicators(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03
    )

    # Chandeliers japonais
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Prix",
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff6b6b"
    ), row=1, col=1)

    # Moyennes mobiles
    for col, color in [("MA20","#f5a623"), ("MA50","#6c8ebf"), ("MA200","white")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(color=color, width=1.2)
            ), row=1, col=1)

    # Bandes de Bollinger
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            line=dict(color="rgba(108,142,191,0.4)", dash="dot"),
            name="BB+", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            line=dict(color="rgba(108,142,191,0.4)", dash="dot"),
            name="BB−", fill="tonexty",
            fillcolor="rgba(108,142,191,0.05)", showlegend=False
        ), row=1, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#c084fc", width=1.5)
        ), row=2, col=1)
        for level, color in [(70,"#ff6b6b"), (30,"#00d4aa")]:
            fig.add_hline(
                y=level, line_dash="dot",
                line_color=color, opacity=0.5, row=2, col=1
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig