try:
    from rapidfuzz import fuzz

    def sim(a, b):
        """0..1 similarity using RapidFuzz, smarter for single words."""
        a = str(a or "").strip().lower()
        b = str(b or "").strip().lower()
        if not a or not b:
            return 0.0

        # If either side has spaces, use token_set_ratio (good for multi-word phrases)
        if " " in a or " " in b:
            return fuzz.token_set_ratio(a, b) / 100.0

        # For single tokens like 'tamilnadu' vs 'tamil nadu', use char-level ratio
        return fuzz.ratio(a, b) / 100.0

except Exception:  # fallback stays as-is
    import difflib

    def sim(a, b):
        """0..1 similarity using difflib."""
        return difflib.SequenceMatcher(
            None, str(a or "").lower(), str(b or "").lower()
        ).ratio()
