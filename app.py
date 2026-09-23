import requests
import streamlit as st
import time

st.set_page_config(page_title="Document Intelligence System", layout="wide")
st.title("Document Intelligence System")
st.caption("Upload a PDF, then ask questions about it. Everything runs locally.")

UPLOAD_URL = "http://127.0.0.1:8000/upload"
QUERY_URL = "http://127.0.0.1:8000/query"
TERMS_URL = "http://127.0.0.1:8000/terms"

with st.sidebar:
    st.header("Pipeline Settings")
    dpi_setting = st.slider("Render DPI", min_value=150, max_value=600, value=300, step=50)
    top_k = st.slider("Pages to read per question", min_value=1, max_value=3, value=2)

def fetch_term_index(doc_id, max_wait=120, poll_interval=3):
    """Poll for term index until ready or timeout."""
    start = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while time.time() - start < max_wait:
        elapsed = int(time.time() - start)
        progress = min(elapsed / max_wait, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Building medical term index... ({elapsed}s)")
        
        try:
            r = requests.get(f"{TERMS_URL}/{doc_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                terms = data.get("terms", {})
                # Filter only terms with definitions
                terms_with_def = {k: v for k, v in terms.items() if v.get("definition")}
                progress_bar.progress(1.0)
                status_text.text(f"Term index ready: {len(terms_with_def)} terms found")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                return terms_with_def
            elif r.status_code == 404:
                # Not ready yet, continue polling
                pass
            else:
                st.warning(f"Term index error: {r.status_code}")
                break
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(poll_interval)
    
    progress_bar.empty()
    status_text.empty()
    st.warning("Term index generation timed out. Check backend logs.")
    return {}

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
    
    # Medical Term Index section
    st.subheader("📋 Medical Term Index")
    if "term_index" not in st.session_state or st.session_state.get("term_index_doc_id") != data["doc_id"]:
        with st.spinner("Extracting medical terms..."):
            terms = fetch_term_index(data["doc_id"])
            st.session_state["term_index"] = terms
            st.session_state["term_index_doc_id"] = data["doc_id"]
    else:
        terms = st.session_state["term_index"]
    
    if terms:
        # Build table rows
        term_rows = []
        for term_name, term_data in terms.items():
            term_rows.append({
                "Term": term_name,
                "Definition": term_data.get("definition", ""),
                "Pages": ", ".join(map(str, term_data.get("pages", []))),
                "Example Value": term_data.get("example_value", ""),
                "Unit": term_data.get("unit", ""),
                "Reference Range": term_data.get("reference_range", ""),
            })
        st.dataframe(term_rows, use_container_width=True, hide_index=True)
        st.caption(f"Total terms: {len(term_rows)}")
    else:
        st.info("No medical terms with definitions found yet. The index may still be generating.")
    
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