import io, ollama
from PIL import Image
from pydantic import BaseModel

class Finding(BaseModel):
    name: str
    value: str
    unit: str | None = None
    flag: str | None = None          # "high" / "low" / "normal" / None
    page: int
    bbox: list[int]                  # see coordinate warning below

class Extraction(BaseModel):
    findings: list[Finding]

def _shrink(path, max_side=1400):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()

def extract(question, pages):        # pages = output of search()
    msgs = [{
        "role": "user",
        "content": f"Question: {question}\nPages are in order: {[p['page'] for p in pages]}.\n"
                   "Only report values visible in the images. If not visible, return no finding.",
        "images": [_shrink(p["image_path"]) for p in pages],
    }]
    r = ollama.chat(model="qwen3-vl:8b", messages=msgs,
                    format=Extraction.model_json_schema(),
                    options={"num_ctx": 8192, "temperature": 0},
                    keep_alive="10m")
    return Extraction.model_validate_json(r["message"]["content"])