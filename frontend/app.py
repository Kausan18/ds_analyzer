import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DS Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "access_token": None,
    "user_id": None,
    "user_email": None,
    "username": None,
    "page": "login",
    "report": None,
    "session_id": None,
    "chat_history": [],
    "active_tab": 0,
    "auth_mode": "login",        # "login" | "signup"
    "show_signup_prompt": False,
    "theme": "dark",             # "dark" | "light"
    "overview_section": None,    # which section to open from overview
    "analyzing": False,          # guard: prevent double-submit on file upload
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Theme CSS ─────────────────────────────────────────────────────────────────
def inject_theme():
    dark = st.session_state["theme"] == "dark"

    bg_primary   = "#0f1117" if dark else "#f5f7fa"
    bg_secondary = "#1a1d27" if dark else "#ffffff"
    bg_card      = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.03)"
    bg_card_hover= "rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.06)"
    border_color = "rgba(255,255,255,0.10)" if dark else "rgba(0,0,0,0.10)"
    text_primary = "#f0f2f6" if dark else "#0f1117"
    text_muted   = "#8b95a5" if dark else "#6b7280"
    accent       = "#6366f1"
    accent2      = "#8b5cf6"
    success      = "#10b981"
    warning      = "#f59e0b"
    danger       = "#ef4444"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {{
        --bg-primary:   {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card:      {bg_card};
        --bg-card-hover:{bg_card_hover};
        --border:       {border_color};
        --text:         {text_primary};
        --text-muted:   {text_muted};
        --accent:       {accent};
        --accent2:      {accent2};
        --success:      {success};
        --warning:      {warning};
        --danger:       {danger};
    }}

    /* ── Global ── */
    html, body {{
        font-family: 'DM Sans', sans-serif !important;
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}
    .stApp {{
        background-color: {bg_primary} !important;
    }}
    .main, .main > div, .block-container {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}
    .main .block-container {{
        padding: 1.5rem 2.5rem 3rem;
        max-width: 1400px;
    }}
    /* All text elements */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif !important;
    }}
    p, span, label, li, td, th {{
        color: {text_primary} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    /* Streamlit specific text containers */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: {text_primary} !important;
    }}
    .stMarkdown, .stText, .stCaption {{
        color: {text_primary} !important;
    }}
    [data-testid="stCaptionContainer"] {{
        color: {text_muted} !important;
    }}
    /* Widget labels */
    .stSelectbox label, .stMultiSelect label,
    .stTextInput label, .stFileUploader label,
    .stRadio label, .stCheckbox label {{
        color: {text_primary} !important;
    }}
    /* Dataframe */
    [data-testid="stDataFrame"] {{
        background: var(--bg-card) !important;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        transition: border-color .2s;
    }}
    [data-testid="stMetric"]:hover {{ border-color: var(--accent); }}
    [data-testid="stMetricValue"] {{ color: var(--text) !important; font-weight: 700; font-size: 1.6rem; }}
    [data-testid="stMetricLabel"] {{ color: var(--text-muted) !important; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all .2s ease !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-card) !important;
        color: var(--text) !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        font-size: 0.85rem !important;
        line-height: 1.3 !important;
        padding: 0.4rem 0.75rem !important;
    }}
    .stButton > button:hover {{
        border-color: var(--accent) !important;
        background: var(--bg-card-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99,102,241,.25) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        border: none !important;
        color: #fff !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: .9;
        box-shadow: 0 4px 20px rgba(99,102,241,.45) !important;
    }}

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea textarea {{
        background: {"#1e2130" if dark else "#ffffff"} !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: {"#f0f2f6" if dark else "#0f1117"} !important;
        caret-color: {"#f0f2f6" if dark else "#0f1117"} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    .stTextInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {{
        color: {"rgba(255,255,255,0.35)" if dark else "rgba(0,0,0,0.35)"} !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
    }}

    /* ── Radio pills ── */
    .stRadio > div {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        background: transparent !important;
    }}
    .stRadio > div > label {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        padding: 6px 16px !important;
        font-size: .85rem !important;
        cursor: pointer;
        transition: all .2s;
        color: var(--text-muted) !important;
        white-space: nowrap;
    }}
    .stRadio > div > label:has(input:checked) {{
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        border-color: transparent !important;
        color: #fff !important;
    }}
    .stRadio > div > label:hover {{
        border-color: var(--accent) !important;
        color: var(--text) !important;
    }}
    /* hide radio circles */
    .stRadio > div > label > div:first-child {{
        display: none !important;
    }}

    /* ── Chat messages (AI Assistant) ── */
    [data-testid="stChatMessage"] {{
        background: {"rgba(30,33,50,0.85)" if dark else "rgba(240,242,246,0.85)"} !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: {text_primary} !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div {{
        color: {text_primary} !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: transparent !important;
        color: {text_primary} !important;
    }}
    /* Chat input box */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] textarea {{
        background: {"#1e2130" if dark else "#ffffff"} !important;
        color: {text_primary} !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--text-muted) !important;
    }}

    /* ── Divider ── */
    hr {{ border-color: var(--border) !important; margin: 1.2rem 0; }}

    /* ── Tabs (login page) ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--bg-card);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: var(--text-muted) !important;
        background: transparent !important;
        font-weight: 500;
        padding: 8px 24px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #fff !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    /* ── Alerts ── */
    .stSuccess, .stInfo, .stWarning, .stError {{
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

    /* ── Glass card helper ── */
    .glass-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px;
        backdrop-filter: blur(16px);
        transition: border-color .25s, box-shadow .25s;
    }}
    .glass-card:hover {{
        border-color: var(--accent);
        box-shadow: 0 8px 32px rgba(99,102,241,.12);
    }}

    /* ── Category widget cards ── */
    .cat-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 22px 20px 18px;
        text-align: center;
        cursor: pointer;
        transition: all .25s;
        backdrop-filter: blur(10px);
    }}
    .cat-card:hover {{
        border-color: var(--accent);
        box-shadow: 0 6px 24px rgba(99,102,241,.18);
        transform: translateY(-2px);
    }}
    .cat-card .icon {{ font-size: 2rem; margin-bottom: 8px; }}
    .cat-card .title {{ font-weight: 600; font-size: .95rem; color: var(--text); margin-bottom: 4px; }}
    .cat-card .desc {{ font-size: .78rem; color: var(--text-muted); }}

    /* ── Nav bar ── */
    .ds-nav {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 10px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }}
    .ds-nav .brand {{
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .ds-nav .breadcrumb {{
        font-size: .82rem;
        color: var(--text-muted);
    }}
    .ds-nav .breadcrumb span {{ color: var(--accent); }}
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api(method, path, auth=True, **kwargs):
    url = f"{BACKEND}{path}"
    try:
        headers = kwargs.pop("headers", {})
        if auth and st.session_state.get("access_token"):
            headers["Authorization"] = f"Bearer {st.session_state['access_token']}"
        res = getattr(requests, method)(url, timeout=300, headers=headers, **kwargs)
        return res
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend. Is uvicorn running?")
        return None


def logout():
    for key in ["authenticated", "access_token", "user_id", "user_email", "username",
                "report", "session_id", "chat_history", "active_tab",
                "auth_mode", "show_signup_prompt", "overview_section"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["page"] = "login"
    st.session_state["authenticated"] = False
    st.session_state["theme"] = st.session_state.get("theme", "dark")
    st.rerun()


def go(page, overview_section=None):
    st.session_state["page"] = page
    st.session_state["active_tab"] = 0
    if overview_section is not None:
        st.session_state["overview_section"] = overview_section
    st.rerun()


def theme_toggle():
    """Render ☀️/🌙 button inline."""
    icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
    if st.button(icon, key="theme_btn", help="Toggle light/dark mode"):
        st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
        st.rerun()


def nav_bar(breadcrumb_items=None, show_new_analysis=False):
    """Top navigation bar. breadcrumb_items = list of (label, page_key or None)."""
    cols = st.columns([4, 1, 1, 1, 1])
    with cols[0]:
        items = breadcrumb_items or [("DS Analyzer", None)]
        crumb_parts = []
        for label, page_key in items:
            if page_key is None:
                # Current page — bold accent
                crumb_parts.append(
                    f"<span style='color:var(--text);font-weight:600'>{label}</span>"
                )
            else:
                # Navigable crumb — muted
                crumb_parts.append(
                    f"<span style='color:var(--text-muted)'>{label}</span>"
                )
        crumb_html = " <span style='color:var(--text-muted)'>›</span> ".join(crumb_parts)
        st.markdown(
            f"<div style='padding-top:8px;font-size:.88rem;line-height:1'>🔬 {crumb_html}</div>",
            unsafe_allow_html=True
        )
    btn_col = 1
    if show_new_analysis:
        with cols[btn_col]:
            if st.button("➕ New", use_container_width=True, key="nav_new"):
                st.session_state["report"] = None
                st.session_state["session_id"] = None
                st.session_state["chat_history"] = []
                st.session_state["active_tab"] = 0
                go("analysis")
        btn_col += 1
    with cols[btn_col]:
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            go("dashboard")
        btn_col += 1
    with cols[btn_col]:
        theme_toggle()
        btn_col += 1
    with cols[btn_col]:
        if st.button("Logout", use_container_width=True, key="nav_logout"):
            logout()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN / SIGNUP
# ════════════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1.5rem;'>
            <div style='font-size:2.8rem; margin-bottom:.5rem'>🔬</div>
            <h2 style='margin:0; font-size:1.9rem; font-weight:700;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text;'>DS Analyzer</h2>
            <p style='color: var(--text-muted); margin:.4rem 0 0; font-size:.95rem;'>
                AI-powered EDA & data quality analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Use radio for programmatic tab switching
        mode = st.radio("", ["Login", "Sign Up"],
                        index=0 if st.session_state["auth_mode"] == "login" else 1,
                        horizontal=True, key="auth_mode_radio",
                        label_visibility="collapsed")
        st.session_state["auth_mode"] = "login" if mode == "Login" else "signup"
        st.markdown("")

        if st.session_state["auth_mode"] == "login":
            # ── Login form ──
            with st.container():
                email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="login_pass",
                                         placeholder="••••••••")

                if st.button("Login", type="primary", use_container_width=True, key="login_btn"):
                    if not email or not password:
                        st.error("Please enter email and password.")
                    else:
                        res = api("post", "/auth/login", auth=False,
                                  json={"email": email, "password": password})
                        if res and res.status_code == 200:
                            data = res.json()
                            st.session_state["authenticated"] = True
                            st.session_state["access_token"] = data["access_token"]
                            st.session_state["user_id"] = data["user_id"]
                            st.session_state["user_email"] = data["email"]
                            st.session_state["username"] = data.get(
                                "username", data["email"].split("@")[0])
                            st.session_state["show_signup_prompt"] = False
                            go("dashboard")
                        else:
                            detail = res.json().get("detail", "Login failed.") if res else "No response."
                            st.error(detail)
                            # Show sign-up prompt if user not found
                            if res and any(kw in detail.lower()
                                           for kw in ["not found", "invalid", "no user",
                                                      "wrong", "incorrect", "user"]):
                                st.session_state["show_signup_prompt"] = True

                if st.session_state.get("show_signup_prompt"):
                    st.info("Don't have an account yet?")
                    if st.button("➜ Create an account", use_container_width=True,
                                 key="goto_signup"):
                        st.session_state["auth_mode"] = "signup"
                        st.session_state["show_signup_prompt"] = False
                        st.rerun()

        else:
            # ── Sign-up form ──
            with st.container():
                username_s = st.text_input("Username", key="signup_username",
                                           placeholder="coolname42")
                email_s = st.text_input("Email", key="signup_email",
                                        placeholder="you@example.com")
                password_s = st.text_input("Password (min 6 chars)", type="password",
                                           key="signup_pass", placeholder="••••••••")
                password_s2 = st.text_input("Confirm Password", type="password",
                                            key="signup_pass2", placeholder="••••••••")

                if st.button("Create Account", type="primary", use_container_width=True,
                             key="signup_btn"):
                    if not username_s or not email_s or not password_s:
                        st.error("Please fill all fields.")
                    elif password_s != password_s2:
                        st.error("Passwords do not match.")
                    elif len(password_s) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif len(username_s) < 2:
                        st.error("Username must be at least 2 characters.")
                    else:
                        res = api("post", "/auth/signup", auth=False,
                                  json={"email": email_s, "password": password_s,
                                        "username": username_s})
                        if res and res.status_code == 200:
                            st.success(
                                "Account created! Please check your email to confirm, then log in.")
                            st.session_state["auth_mode"] = "login"
                            st.rerun()
                        else:
                            detail = res.json().get("detail", "Signup failed.") if res else "No response."
                            st.error(detail)

                st.markdown("")
                st.markdown("<p style='text-align:center; color:var(--text-muted); font-size:.85rem'>"
                            "Already have an account?</p>", unsafe_allow_html=True)
                if st.button("➜ Sign in instead", use_container_width=True,
                             key="goto_login"):
                    st.session_state["auth_mode"] = "login"
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    nav_bar([("DS Analyzer", None), ("Dashboard", None)])

    username = st.session_state.get("username") or st.session_state.get(
        "user_email", "").split("@")[0]

    st.markdown(f"""
    <div style='margin-bottom: 2rem;'>
        <h2 style='margin:0; font-size:1.8rem; font-weight:700;'>
            Welcome back, {username} 👋
        </h2>
        <p style='color: var(--text-muted); margin:.3rem 0 0;'>
            What would you like to do today?
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="glass-card" style="text-align:center; min-height:180px;">
            <div style="font-size:3rem; margin-bottom:12px">📊</div>
            <h3 style="margin:0 0 8px; font-size:1.15rem">New Analysis</h3>
            <p style="color:var(--text-muted); font-size:.88rem; margin:0">
                Upload a CSV or Excel dataset and get a full automated EDA report with AI insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Start New Analysis", type="primary", use_container_width=True,
                     key="dash_new"):
            st.session_state["report"] = None
            st.session_state["session_id"] = None
            st.session_state["chat_history"] = []
            st.session_state["active_tab"] = 0
            go("analysis")

    with c2:
        st.markdown("""
        <div class="glass-card" style="text-align:center; min-height:180px;">
            <div style="font-size:3rem; margin-bottom:12px">📁</div>
            <h3 style="margin:0 0 8px; font-size:1.15rem">Past Analyses</h3>
            <p style="color:var(--text-muted); font-size:.88rem; margin:0">
                View and revisit your previously saved EDA reports and AI conversations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("View Past Analyses", use_container_width=True, key="dash_past"):
            go("past")

    st.divider()
    st.markdown("### Recent Analyses")

    res = api("get", f"/store/list/{st.session_state['user_id']}")
    if res and res.status_code == 200:
        analyses = res.json().get("analyses", [])
        if analyses:
            for item in analyses[:5]:
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    with col_a:
                        st.markdown(f"**{item['filename']}**")
                    with col_b:
                        created = item["created_at"][:19].replace("T", " ")
                        st.caption(created)
                    with col_c:
                        if st.button("Open", key=f"open_{item['session_id']}",
                                     use_container_width=True):
                            with st.spinner("Loading..."):
                                load_past_analysis(item["session_id"])
                    st.divider()
        else:
            st.info("No analyses yet. Start your first one above.")
    else:
        st.info("Could not load recent analyses.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PAST ANALYSES
# ════════════════════════════════════════════════════════════════════════════
def load_past_analysis(session_id: str):
    res = api("get", f"/store/load/{session_id}/{st.session_state['user_id']}")
    if res and res.status_code == 200:
        data = res.json()
        st.session_state["report"] = data["report"]
        st.session_state["session_id"] = data["session_id"]
        st.session_state["chat_history"] = []
        st.session_state["active_tab"] = 0
        st.session_state["overview_section"] = None
        api("post", "/api/reembed",
            json={"session_id": data["session_id"], "report": data["report"]})
        go("overview")
    else:
        st.error("Could not load analysis.")


def page_past():
    nav_bar([("DS Analyzer", None), ("Dashboard", "dashboard"), ("Past Analyses", None)])

    st.markdown("## Past Analyses")
    st.divider()

    res = api("get", f"/store/list/{st.session_state['user_id']}")
    if not res or res.status_code != 200:
        st.error("Could not load analyses.")
        return

    analyses = res.json().get("analyses", [])
    if not analyses:
        st.info("No saved analyses found.")
        return

    for item in analyses:
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            with c1:
                st.markdown(f"**{item['filename']}**")
                st.caption(f"Session: {item['session_id']}")
            with c2:
                created = item["created_at"][:19].replace("T", " ")
                st.caption(f"Created: {created}")
            with c3:
                if st.button("Open", key=f"past_open_{item['session_id']}",
                             use_container_width=True):
                    with st.spinner("Loading..."):
                        load_past_analysis(item["session_id"])
            with c4:
                if st.button("Delete", key=f"del_{item['session_id']}",
                             use_container_width=True):
                    del_res = api(
                        "delete",
                        f"/store/delete/{item['session_id']}/{st.session_state['user_id']}"
                    )
                    if del_res and del_res.status_code == 200:
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
            st.divider()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS UPLOAD
# ════════════════════════════════════════════════════════════════════════════
def page_analysis():
    nav_bar([("DS Analyzer", None), ("Dashboard", "dashboard"), ("New Analysis", None)])

    st.markdown("## Upload Dataset")
    st.markdown("<p style='color:var(--text-muted)'>Upload a CSV or Excel file to begin your EDA.</p>",
                unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Choose your dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    # Guard: prevent multiple analysis runs on Streamlit reruns
    if uploaded_file and not st.session_state.get("analyzing"):
        st.session_state["analyzing"] = True
        progress = st.progress(0, text="Reading file...")
        try:
            progress.progress(20, text="Sending to analysis engine...")
            res = api("post", "/api/analyze",
                      files={"file": (uploaded_file.name,
                                      uploaded_file.getvalue(),
                                      uploaded_file.type)})
            progress.progress(80, text="Processing results...")

            if res and res.status_code == 200:
                data = res.json()
                st.session_state["report"] = data["report"]
                st.session_state["session_id"] = data["session_id"]
                st.session_state["chat_history"] = []
                st.session_state["overview_section"] = None
                st.session_state["analyzing"] = False

                api("post", "/store/save", json={
                    "session_id": data["session_id"],
                    "filename": uploaded_file.name,
                    "report": data["report"]
                })

                progress.progress(100, text="Done!")
                if data["report"].get("sampled"):
                    st.warning(
                        f"Large dataset — {data['report']['total_rows_original']:,} rows. "
                        f"Analysis on 50,000-row sample."
                    )
                st.success("Analysis complete and saved!")
                go("overview")
            else:
                st.session_state["analyzing"] = False
                progress.empty()
                detail = res.json().get("detail", "Unknown") if res else "No response"
                st.error(f"Error: {detail}")
        except Exception as e:
            st.session_state["analyzing"] = False
            progress.empty()
            st.error(str(e))
    elif uploaded_file and st.session_state.get("analyzing"):
        st.info("Analysis in progress, please wait...")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW  (hub with category widgets)
# ════════════════════════════════════════════════════════════════════════════
CATEGORY_WIDGETS = [
    {"key": "missing_values",    "icon": "🕳️",  "title": "Missing Values",    "desc": "Detect & visualize nulls"},
    {"key": "outliers",          "icon": "⚡",  "title": "Outliers",           "desc": "Z-score anomaly detection"},
    {"key": "distributions",     "icon": "📈",  "title": "Distributions",      "desc": "Column histograms & skew"},
    {"key": "class_imbalance",   "icon": "⚖️",  "title": "Class Imbalance",   "desc": "Target class balance check"},
    {"key": "correlations",      "icon": "🔗",  "title": "Correlations",       "desc": "Feature correlation heatmap"},
    {"key": "feature_importance","icon": "🎯",  "title": "Feature Importance", "desc": "Random forest importance"},
    {"key": "column_stats",      "icon": "📋",  "title": "Column Stats",       "desc": "Descriptive statistics"},
    {"key": "recommendations",   "icon": "💡",  "title": "Recommendations",    "desc": "Actionable data quality tips"},
    {"key": "ai_assistant",      "icon": "🤖",  "title": "AI Assistant",       "desc": "Chat with your data"},
    {"key": "preview",           "icon": "👁️",  "title": "Data Preview",       "desc": "Raw data & type overview"},
]


def page_overview():
    report = st.session_state.get("report")
    if not report:
        go("dashboard")
        return

    nav_bar(
        [("DS Analyzer", None), ("Dashboard", "dashboard"), ("Overview", None)],
        show_new_analysis=True
    )

    st.markdown("## Analysis Overview")

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Rows", f"{report['shape']['rows']:,}")
    with c2:
        st.metric("Columns", report["shape"]["cols"])
    with c3:
        st.metric("Missing Cols", len(report["missing"]))
    with c4:
        st.metric("Duplicates", report["duplicates"]["count"])
    with c5:
        st.metric("Outlier Cols", len(report["outliers"]))
    with c6:
        fi = report.get("feature_importance", {})
        st.metric("Target Col", fi.get("target_column", "N/A") if fi.get("available") else "N/A")

    st.divider()

    # ── PDF report download ───────────────────────────────────────────────────
    session_id = st.session_state.get("session_id")
    with st.expander("📄 Generate PDF Report", expanded=False):
        st.markdown("Download a full PDF of this EDA report.")
        if st.button("Generate PDF", type="primary", key="ov_pdf"):
            with st.spinner("Generating PDF..."):
                pdf_res = api("get", f"/api/download/{session_id}")
            if pdf_res and pdf_res.status_code == 200:
                st.download_button(
                    "⬇️ Download PDF", data=pdf_res.content,
                    file_name=f"eda_report_{session_id}.pdf",
                    mime="application/pdf",
                    key="ov_pdf_dl"
                )
            else:
                st.error("Failed to generate PDF.")

    st.markdown("### Explore Categories")
    st.markdown("<p style='color:var(--text-muted); margin-top:-10px; font-size:.88rem'>"
                "Click any card to dive deeper into that analysis.</p>",
                unsafe_allow_html=True)
    st.markdown("")

    # ── Category widget grid (2 rows × 5 cols) ────────────────────────────────
    rows = [CATEGORY_WIDGETS[:5], CATEGORY_WIDGETS[5:]]
    for row in rows:
        cols = st.columns(5, gap="small")
        for col, widget in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="cat-card" id="cat_{widget['key']}">
                    <div class="icon">{widget['icon']}</div>
                    <div class="title">{widget['title']}</div>
                    <div class="desc">{widget['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Open →", key=f"cat_{widget['key']}_btn",
                             use_container_width=True):
                    go("detail", overview_section=widget["key"])
        st.markdown("")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DETAIL  (individual analysis section)
# ════════════════════════════════════════════════════════════════════════════
def page_detail():
    report = st.session_state.get("report")
    if not report:
        go("dashboard")
        return

    session_id = st.session_state.get("session_id")
    section = st.session_state.get("overview_section", "preview")

    # Build breadcrumb title
    section_meta = {w["key"]: w for w in CATEGORY_WIDGETS}
    section_title = section_meta.get(section, {}).get("title", section.replace("_", " ").title())

    nav_bar(
        [("DS Analyzer", None), ("Dashboard", "dashboard"),
         ("Overview", "overview"), (section_title, None)],
        show_new_analysis=True
    )

    # ── Section navigation pills ──────────────────────────────────────────────
    tab_keys = [w["key"] for w in CATEGORY_WIDGETS]
    tab_labels = [f"{w['icon']} {w['title']}" for w in CATEGORY_WIDGETS]
    current_idx = tab_keys.index(section) if section in tab_keys else 0

    selected_label = st.radio(
        label="", options=tab_labels,
        index=current_idx,
        horizontal=True, key="detail_tab",
        label_visibility="collapsed"
    )
    selected_key = tab_keys[tab_labels.index(selected_label)]
    if selected_key != section:
        st.session_state["overview_section"] = selected_key
        st.rerun()

    st.divider()

    back_col, _ = st.columns([1, 6])
    with back_col:
        if st.button("← Back to Overview", key="back_overview"):
            go("overview")

    st.markdown("")

    # ── PREVIEW ───────────────────────────────────────────────────────────────
    if section == "preview":
        st.subheader("Dataset Preview")
        preview = report.get("dataset_preview", {})
        if preview:
            preview_df = pd.DataFrame(preview["rows"], columns=preview["columns"])
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"First 10 rows of {report['shape']['rows']:,} total rows.")
        dtype_df = pd.DataFrame([
            {"Column": col, "Type": dtype,
             "Category": "Numeric" if "int" in dtype or "float" in dtype
             else "Categorical" if dtype == "object" else "Other"}
            for col, dtype in report["dtypes"].items()
        ])
        fig = px.pie(dtype_df, names="Category",
                     title="Column Type Distribution", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dtype_df, use_container_width=True)

    # ── MISSING VALUES ────────────────────────────────────────────────────────
    elif section == "missing_values":
        st.subheader("Missing Values")
        if report["missing"]:
            miss_df = pd.DataFrame([
                {"Column": k, "Missing Count": v["count"], "Missing %": v["percent"]}
                for k, v in report["missing"].items()
            ]).sort_values("Missing %", ascending=False)
            fig = px.bar(miss_df, x="Column", y="Missing %", color="Missing %",
                         color_continuous_scale="Reds",
                         title="Missing Value % per Column", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df, use_container_width=True)
        else:
            st.success("✅ No missing values found.")

    # ── OUTLIERS ──────────────────────────────────────────────────────────────
    elif section == "outliers":
        st.subheader("Outliers (Z-score > 3)")
        if report["outliers"]:
            out_df = pd.DataFrame([
                {"Column": k, "Outlier Count": v["count"], "Outlier %": v["percent"]}
                for k, v in report["outliers"].items()
            ]).sort_values("Outlier %", ascending=False)
            fig = px.bar(out_df, x="Column", y="Outlier %", color="Outlier %",
                         color_continuous_scale="Oranges",
                         title="Outlier % per Column", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out_df, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected.")

    # ── DISTRIBUTIONS ─────────────────────────────────────────────────────────
    elif section == "distributions":
        st.subheader("Column Distributions")
        distributions = report.get("distributions", {})
        if distributions:
            cols_list = list(distributions.keys())
            selected_cols = st.multiselect(
                "Select columns", options=cols_list, default=cols_list[:6])
            if not selected_cols:
                selected_cols = cols_list[:6]

            for row_i in range(0, len(selected_cols), 2):
                cols = st.columns(2)
                for col_i in range(2):
                    idx = row_i + col_i
                    if idx >= len(selected_cols):
                        break
                    col_name = selected_cols[idx]
                    d = distributions[col_name]
                    s = report["column_stats"].get(col_name, {})
                    with cols[col_i]:
                        fig = gobj.Figure()
                        fig.add_trace(gobj.Bar(
                            x=d["bin_centers"], y=d["counts"],
                            marker_color="#6366f1"
                        ))
                        if "mean" in s:
                            fig.add_vline(
                                x=s["mean"], line_dash="dash", line_color="#ef4444",
                                annotation_text=f"Mean: {s['mean']}"
                            )
                        fig.update_layout(
                            title=col_name, template="plotly_white",
                            height=280, showlegend=False,
                            margin=dict(t=40, b=30, l=30, r=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if "skewness" in s:
                            skew = s["skewness"]
                            if abs(skew) > 1:
                                st.warning(f"Highly skewed: {skew}")
                            elif abs(skew) > 0.5:
                                st.info(f"Moderately skewed: {skew}")
                            else:
                                st.success(f"Normal-ish: {skew}")
        else:
            st.info("No numeric columns found.")

    # ── CLASS IMBALANCE ───────────────────────────────────────────────────────
    elif section == "class_imbalance":
        st.subheader("Class Imbalance")
        if report["class_imbalance"]:
            for col, info in report["class_imbalance"].items():
                badge = "🔴 Imbalanced" if info["is_imbalanced"] else "🟢 Acceptable"
                st.markdown(f"**{col}** — Ratio: `{info['imbalance_ratio']}:1`  {badge}")
                dist_df = pd.DataFrame(
                    list(info["distribution"].items()), columns=["Class", "Count"])
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.pie(dist_df, names="Class", values="Count",
                                 title=f"Distribution of {col}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(dist_df, x="Class", y="Count",
                                 title=f"Counts — {col}", template="plotly_white",
                                 color="Count", color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)
                st.divider()
        else:
            st.success("✅ No class imbalance issues found.")

    # ── CORRELATIONS ──────────────────────────────────────────────────────────
    elif section == "correlations":
        st.subheader("Correlation Matrix")
        matrix = report["correlations"].get("matrix", {})
        if matrix:
            corr_df = pd.DataFrame(matrix)
            fig = px.imshow(
                corr_df, text_auto=True, color_continuous_scale="RdBu_r",
                title="Feature Correlation Heatmap", zmin=-1, zmax=1,
                template="plotly_white"
            )
            fig.update_layout(height=max(400, len(corr_df.columns) * 40))
            st.plotly_chart(fig, use_container_width=True)
            high = report["correlations"].get("high_correlations", [])
            if high:
                st.warning(f"**{len(high)} highly correlated pairs (|r| > 0.8):**")
                st.dataframe(pd.DataFrame(high), use_container_width=True)
            else:
                st.success("✅ No high correlations found.")
        else:
            st.info("Not enough numeric columns.")

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────
    elif section == "feature_importance":
        st.subheader("Feature Importance")
        fi = report.get("feature_importance", {})
        if fi.get("available"):
            st.info(f"Target: **{fi['target_column']}** | Model: Random Forest {fi['model_type'].title()}")
            fi_df = pd.DataFrame(fi["features"])
            fi_df["signal"] = fi_df["importance"].apply(
                lambda x: "High" if x > 0.1 else ("Medium" if x > 0.03 else "Low"))
            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                color="signal",
                color_discrete_map={"High": "#10b981", "Medium": "#f59e0b", "Low": "#ef4444"},
                title=f"Feature Importance → '{fi['target_column']}'",
                template="plotly_white"
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=max(400, len(fi_df) * 30)
            )
            st.plotly_chart(fig, use_container_width=True)
            fi_df["recommendation"] = fi_df["importance"].apply(
                lambda x: "Keep — high signal" if x > 0.1
                else ("Keep — moderate" if x > 0.03 else "Consider dropping"))
            st.dataframe(fi_df[["feature", "importance", "recommendation"]],
                         use_container_width=True)
        else:
            st.warning(f"Not available: {fi.get('reason', 'unknown')}")

    # ── COLUMN STATS ──────────────────────────────────────────────────────────
    elif section == "column_stats":
        st.subheader("Column Statistics")
        numeric_rows, cat_rows = [], []
        for col, s in report["column_stats"].items():
            if "mean" in s:
                numeric_rows.append({
                    "Column": col, "Mean": s["mean"], "Median": s["median"],
                    "Std": s["std"], "Min": s["min"], "Max": s["max"],
                    "Skewness": s["skewness"], "Q25": s.get("q25", ""),
                    "Q75": s.get("q75", ""), "Unique": s["unique"]
                })
            else:
                top = ", ".join([f"{k}({v})"
                                 for k, v in list(s.get("top_values", {}).items())[:3]])
                cat_rows.append({"Column": col, "Unique": s["unique"], "Top Values": top})
        if numeric_rows:
            st.markdown("**Numeric Columns**")
            st.dataframe(pd.DataFrame(numeric_rows), use_container_width=True)
            skew_df = pd.DataFrame(numeric_rows)[["Column", "Skewness"]]
            fig = px.bar(skew_df, x="Column", y="Skewness", color="Skewness",
                         color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                         title="Skewness per Column", template="plotly_white")
            fig.add_hline(y=1, line_dash="dash", line_color="orange")
            fig.add_hline(y=-1, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
        if cat_rows:
            st.markdown("**Categorical Columns**")
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    elif section == "recommendations":
        st.subheader("Actionable Recommendations")
        if report["recommendations"]:
            for i, rec in enumerate(report["recommendations"]):
                if rec.startswith("DROP"):
                    st.error(f"🗑️ **{i+1}.** {rec}")
                elif any(w in rec for w in ["IMPUTE", "Consider", "HIGH CORR"]):
                    st.warning(f"⚠️ **{i+1}.** {rec}")
                elif "skewed" in rec.lower():
                    st.info(f"📊 **{i+1}.** {rec}")
                elif "important" in rec.lower():
                    st.success(f"🎯 **{i+1}.** {rec}")
                else:
                    st.info(f"💡 **{i+1}.** {rec}")
        else:
            st.success("✅ Dataset looks clean!")

    # ── AI ASSISTANT ──────────────────────────────────────────────────────────
    elif section == "ai_assistant":
        st.subheader("Ask the AI about your dataset")
        st.caption("The AI has full context of your EDA report.")

        if not st.session_state["chat_history"]:
            st.markdown("**Suggested questions:**")
            suggestions = [
                "Summarize all issues in this dataset",
                "Which features should I drop?",
                "Explain the correlation heatmap",
                "Which columns need transformation?",
                "What is the most important feature?",
            ]
            cols = st.columns(len(suggestions))
            for i, s in enumerate(suggestions):
                with cols[i]:
                    if st.button(s, key=f"sugg_{i}", use_container_width=True):
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": s})
                        st.rerun()
            st.divider()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # If last message is from user with no assistant reply yet, fetch answer now
        history = st.session_state["chat_history"]
        if history and history[-1]["role"] == "user":
            pending_q = history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = api("post", "/api/chat",
                              json={"session_id": session_id, "question": pending_q,
                                    "history": history})
                    answer = res.json()["answer"] if res and res.status_code == 200 else "Error reaching AI."
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer})
            st.rerun()

        question = st.chat_input("Ask anything about your dataset...")
        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
inject_theme()

if not st.session_state["authenticated"]:
    page_login()
else:
    page = st.session_state["page"]
    if page == "dashboard":
        page_dashboard()
    elif page == "analysis":
        page_analysis()
    elif page == "overview":
        if st.session_state.get("report"):
            page_overview()
        else:
            go("analysis")
    elif page == "detail":
        if st.session_state.get("report"):
            page_detail()
        else:
            go("analysis")
    elif page == "past":
        page_past()
    else:
        page_login()
        