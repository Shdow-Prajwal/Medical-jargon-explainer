import json
import logging
from pathlib import Path
from typing import List, Dict
import structlog

from vlm import extract_terms_from_image, explain
from dictionary_service import bulk_add
from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()
settings = get_settings()

IMAGES_DIR = Path("./extracted_pages")


def main():
    image_files = list(IMAGES_DIR.glob("*.png"))
    logger.info("populate.start", image_count=len(image_files))
    
    all_terms = {}  # name -> category
    
    for img_path in image_files:
        logger.info("populate.processing", image=img_path.name)
        terms = extract_terms_from_image(str(img_path))
        for t in terms:
            name = t["name"]
            if name.lower() not in [k.lower() for k in all_terms]:
                all_terms[name] = t["category"]
    
    logger.info("populate.unique_terms", count=len(all_terms))
    
    if not all_terms:
        logger.warning("populate.no_terms_found")
        return
    
    # Get definitions via hybrid explain (dict + VLM fallback)
    # Create mock findings to reuse explain()
    from vlm import Finding
    mock_findings = [Finding(name=name, value="", unit="", reference_range="", page=0) for name in all_terms.keys()]
    explanations = explain(mock_findings)
    
    # Build records for bulk insert
    records = []
    for name, category in all_terms.items():
        definition = explanations.get(name, "")
        if not definition:
            definition = f"Medical term: {name} (category: {category})"
        records.append((name, category, definition, ""))
    
    logger.info("populate.inserting", count=len(records))
    inserted = bulk_add(records)
    logger.info("populate.completed", inserted=inserted, total=len(records))


if __name__ == "__main__":
    main()