import streamlit as st
import base64


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ctdiheader():
    image_path = "images/RL1ENG_Header_Logo.png"  # relative path (important)

    img_base64 = img_to_base64(image_path)

    st.markdown(
    f"""
    <style>

    /* ----------- GLOBAL RESET ----------- */
    .stApp {{
        margin: 0;
        padding: 0;
    }}

    /* ----------- SIDEBAR WIDTH ----------- */
    section[data-testid="stSidebar"] {{
        width: 220px !important;
    }}

    section[data-testid="stSidebar"] > div {{
        width: 220px !important;
    }}

    /* ----------- MAIN CONTENT PADDING ----------- */
    [data-testid="stMainBlockContainer"] {{
        padding-top: 1.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ----------- REMOVE EXTRA TOP GAP ----------- */
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ----------- HEADER IMAGE ----------- */
    .header-div {{
        width: 100%;
        margin: 0;
        padding: 0;
        line-height: 0;
    }}

    .header-div img {{
        width: 100%;
        height: auto;
        display: block;
    }}

    </style>

    <div class="header-div">
        <img src="data:image/png;base64,{img_base64}">
    </div>
    """,
    unsafe_allow_html=True
)