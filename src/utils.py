"""
Traffic Sign Recognition System - Utilities & Visualization Helpers
Handles base64 conversion, thumbnail generation, custom CSS styling, and Plotly charts.
"""

import os
import io
import base64
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

def image_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Converts a numpy RGB image array to base64 string for HTML embedding."""
    if img_rgb is None or img_rgb.size == 0:
        return ""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", bgr)
    return base64.b64encode(buffer).decode("utf-8")

def get_dark_navy_css() -> str:
    """Returns custom CSS styles for the modern Dark Navy-Blue Road-Themed Dashboard."""
    return """
    <style>
        /* Main background & fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        .stApp {
            background-color: #0B1120;
            color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }
        
        /* Road themed subtle pattern overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 0%, rgba(30, 58, 138, 0.25) 0%, rgba(11, 17, 32, 0) 70%),
                        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
            pointer-events: none;
            z-index: 0;
        }
        
        /* Modern Sidebar */
        [data-testid="stSidebar"] {
            background-color: #070D18 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        /* Glassmorphism Stat Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 22px 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
            transition: transform 0.25s ease, border-color 0.25s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(59, 130, 246, 0.5);
        }
        .metric-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3B82F6, #10B981);
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #F8FAFC;
            margin: 6px 0 2px 0;
            font-family: 'Inter', sans-serif;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
        }
        .metric-icon {
            font-size: 1.8rem;
            margin-bottom: 6px;
        }

        /* Category Showcase Badges */
        .cat-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 14px;
            text-align: center;
            transition: all 0.2s ease;
        }
        .cat-card:hover {
            border-color: #3B82F6;
            background: rgba(30, 58, 138, 0.2);
            transform: scale(1.02);
        }
        .cat-icon {
            font-size: 2rem;
            margin-bottom: 8px;
        }
        .cat-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #F1F5F9;
        }
        .cat-desc {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-top: 4px;
        }

        /* Result chips */
        .sign-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-speed { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-stop { background: rgba(220, 38, 38, 0.25); color: #F87171; border: 1px solid rgba(220, 38, 38, 0.5); }
        .badge-prohibit { background: rgba(249, 115, 22, 0.2); color: #FDBA74; border: 1px solid rgba(249, 115, 22, 0.4); }
        .badge-turn { background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-warn { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }

        /* Custom buttons styling */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: linear-gradient(180deg, #1E3A8A 0%, #1E293B 100%);
            color: #F8FAFC;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            border-color: #3B82F6;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
            color: #FFFFFF;
        }
        
        /* Highlight header */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FFFFFF, #93C5FD, #60A5FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            max-width: 780px;
            line-height: 1.5;
        }
        
        /* Code / HUD monospace */
        .hud-mono {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #38BDF8;
        }
    </style>
    """

def create_category_pie_chart(category_counts: Dict[str, int]):
    """Generates an aesthetic donut chart of detections grouped by category."""
    if not category_counts:
        return None
        
    labels = list(category_counts.keys())
    values = list(category_counts.values())
    
    colors = ['#EF4444', '#3B82F6', '#F59E0B', '#10B981', '#8B5CF6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0B1120', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter'),
        margin=dict(t=30, b=30, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        showlegend=True,
        height=320
    )
    return fig

def create_confidence_bar_chart(df: pd.DataFrame):
    """Generates a bar chart showing top detected signs with confidence averages."""
    if df.empty or "sign_name" not in df.columns:
        return None
        
    top_signs = df.groupby("sign_name").agg({
        "confidence": "mean",
        "id": "count"
    }).rename(columns={"id": "count"}).sort_values(by="count", ascending=False).head(7).reset_index()
    
    fig = px.bar(
        top_signs,
        x="count",
        y="sign_name",
        orientation="h",
        color="confidence",
        color_continuous_scale="Viridis",
        labels={"count": "Times Detected", "sign_name": "Traffic Sign", "confidence": "Avg Conf %"},
        title=""
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter'),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=320
    )
    return fig

def create_confidence_distribution_chart(df: pd.DataFrame):
    """Generates a histogram of detection confidence scores."""
    if df.empty or "confidence" not in df.columns:
        return None
        
    fig = px.histogram(
        df,
        x="confidence",
        nbins=15,
        color_discrete_sequence=['#3B82F6'],
        labels={"confidence": "Confidence Score (%)", "count": "Frequency"},
        title=""
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter'),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        height=300
    )
    return fig
