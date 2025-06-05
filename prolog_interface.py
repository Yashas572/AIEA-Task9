from pyswip import Prolog

class PrologInterface:
    def __init__(self, kb_path: str):
        self.prolog = Prolog()
        try:
            self.prolog.consult(kb_path)
            print(f"[DEBUG] Loaded KB from: {kb_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load KB: {e}")

    def query(self, prolog_query: str):
        try:
            # Ensure the query ends with a period and has no '?-'
            clean_query = prolog_query.strip()
            if clean_query.startswith("?-"):
                clean_query = clean_query[2:].strip()
            if not clean_query.endswith("."):
                clean_query += "."

            print(f"[DEBUG] Running Query: {clean_query}")
            results = list(self.prolog.query(clean_query))

            if results:
                return results
            return "No result."
        except Exception as e:
            return f"[ERROR] {str(e)}"
