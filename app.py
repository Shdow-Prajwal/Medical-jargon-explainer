import requests
import streamlit as st

st.set_page_config(page_title="Document Intelligence System", layout="wide")
st.title("Document Intelligence System")
st.caption("Upload a PDF, then ask questions about it. Everything runs locally.")

UPLOAD_URL = "http://127.0.0.1:8000/upload"
QUERY_URL = "http://127.0.0.1:8000/query"

with st.sidebar:
    st.header("Pipeline Settings")
    dpi_setting = st.slider("Render DPI", min_value=150, max_value=600, value=300, step=50)
    top_k = st.slider("Pages to read per question", min_value=1, max_value=3, value=2)

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None and st.button("Process Document", type="primary"):
    with st.spinner("Rendering and indexing pages..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            response = requests.post(
                UPLOAD_URL, files=files, params={"dpi": dpi_setting}, timeout=600
            )
            if response.status_code == 200:
                st.session_state["ingest"] = response.json()
                st.session_state.pop("answer", None)
            else:
                st.error(f"Backend API Error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection refused: start the backend with `python main.py` first.")

if "ingest" in st.session_state:
    data = st.session_state["ingest"]
    st.success(f"Indexed {data['total_pages']} page(s) of '{data['doc_id']}' at {data['dpi']} DPI.")

    cols = st.columns(min(data["total_pages"], 3))
    for idx, page_info in enumerate(data["pages"]):
        with cols[idx % 3]:
            st.subheader(f"Page {page_info['page']}")
            st.image(page_info["path"])
            st.caption(page_info["resolution"])

    st.divider()
    question = st.text_input("Ask a question about this document")
    if st.button("Ask") and question:
        with st.spinner("Searching pages and reading them with the VLM..."):
            try:
                r = requests.post(
                    QUERY_URL,
                    json={"doc_id": data["doc_id"], "question": question, "k": top_k},
                    timeout=300,
                )
                if r.status_code == 200:
                    st.session_state["answer"] = r.json()
                else:
                    st.error(f"Backend API Error ({r.status_code}): {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection refused: is the backend running?")

    if "answer" in st.session_state:
        ans = st.session_state["answer"]
        st.caption(f"Pages read: {ans['pages_used']}")
        st.write(ans["result"]["answer"])

        findings = ans["result"]["findings"]
        if findings:
            cols_order = ["name", "value", "unit", "reference_range", "status",
                          "what_it_measures", "page"]
            st.dataframe([{c: f.get(c) for c in cols_order} for f in findings])