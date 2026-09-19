from pathlib import Path
import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retriever import index_page, search
from vlm import extract

app = FastAPI(title="Document Intelligence Service")

OUTPUT_DIR = Path("./extracted_pages")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


@app.post("/upload")
def upload_and_convert_pdf(
    file: UploadFile = File(...),
    dpi: int = Query(default=300, ge=72, le=600),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    pdf_bytes = file.file.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF stream: {e}")

    doc_id = Path(file.filename).stem
    saved_pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        image_name = f"{doc_id}_page_{page_num + 1}.png"
        image_path = OUTPUT_DIR / image_name
        pix.save(str(image_path))

        try:
            index_page(doc_id, page_num + 1, page.get_text(), str(image_path.resolve()))
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Indexing failed. Is Ollama running and is nomic-embed-text pulled? ({e})",
            )

        saved_pages.append({
            "page": page_num + 1,
            "filename": image_name,
            "resolution": f"{pix.width}x{pix.height}px",
            "path": str(image_path.resolve()),
            "url": f"http://127.0.0.1:8000/static/{image_name}",
        })

    doc.close()

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename,
        "total_pages": len(saved_pages),
        "dpi": dpi,
        "pages": saved_pages,
    }


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    k: int = 2


@app.post("/query")
def query_document(req: QueryRequest):
    pages = search(req.question, req.doc_id, k=req.k)
    if not pages:
        raise HTTPException(status_code=404, detail="No indexed pages for this document.")
    try:
        result = extract(req.question, pages)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"VLM call failed: {e}")
    return {"pages_used": [p["page"] for p in pages], "result": result.model_dump()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)