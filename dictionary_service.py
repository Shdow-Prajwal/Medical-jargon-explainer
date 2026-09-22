import sqlite3

DB_FILE = "medical_terms.db"

def lookup_terms(extracted_terms: list[str]) -> dict:
    """
    Takes a list of extracted keywords/abbreviations (e.g. ['eGFR', 'Anterolisthesis'])
    and performs a batch lookup against the local SQLite database.
    """
    if not extracted_terms:
        return {}

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Build case-insensitive query string
    placeholders = ",".join(["?"] * len(extracted_terms))
    query = f"""
        SELECT term, category, definition, clinical_significance 
        FROM medical_terms 
        WHERE LOWER(term) IN ({','.join(['LOWER(?)'] * len(extracted_terms))})
    """

    cursor.execute(query, extracted_terms)
    rows = cursor.fetchall()
    conn.close()

    results = {}
    found_terms = set()

    for row in rows:
        term, category, definition, significance = row
        results[term] = {
            "category": category,
            "definition": definition,
            "significance": significance,
            "source": "Local Medical DB"
        }
        found_terms.add(term.lower())

    # Identify missing terms for fallback processing (e.g., sending to Qwen3-VL)
    missing_terms = [t for t in extracted_terms if t.lower() not in found_terms]

    return {
        "found": results,
        "missing": missing_terms
    }

# Example Execution Test
if __name__ == "__main__":
    test_terms = ["eGFR", "Anterolisthesis", "UnknownMarkerXYZ"]
    response = lookup_terms(test_terms)
    
    print("\n--- MATCHED TERMS (Instant SQL Lookup) ---")
    for term, data in response["found"].items():
        print(f"[{data['category']}] {term}: {data['definition']}")

    print("\n--- MISSING TERMS (Route to Qwen-VL Fallback) ---")
    print(response["missing"])