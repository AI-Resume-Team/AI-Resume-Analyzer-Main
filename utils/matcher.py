from utils.preprocessing import clean_text

def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compare resume with job description using TF-IDF cosine similarity.
    Returns a percentage score (0–100), rounded to 2 decimals.
    """
    if not resume_text or not jd_text:
        return 0.0

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

        similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]
        return round(float(similarity) * 100, 2)

    except ImportError:
        # Fallback: simple Jaccard similarity
        resume_words = set(cleaned_resume.split())
        jd_words = set(cleaned_jd.split())

        if not resume_words or not jd_words:
            return 0.0

        common_words = resume_words & jd_words
        all_words = resume_words | jd_words

        return round(len(common_words) / len(all_words) * 100, 2)


def compute_resume_quality_score(
    sections_found: dict,
    skills: list,
    resume_text: str,
    grammar_errors: int = 0,
) -> dict:
    """
    Calculate an overall resume quality score (0–100) with a breakdown.

    Scoring:
    - Sections           : 30 pts max
    - Skills             : 25 pts max
    - Word count         : 20 pts max
    - Grammar            : 15 pts max
    - Contact info       : 10 pts max
    """
    total_score = 0
    breakdown = {}

    # Sections (30 pts)
    key_sections = ["Education", "Experience", "Skills", "Projects", "Summary", "Certifications"]
    section_points = sum(5 for section in key_sections if sections_found.get(section, False))
    total_score += section_points
    breakdown["Sections Detected"] = f"{section_points}/30"

    # Skills (25 pts)
    skill_points = min(len(skills), 25)
    total_score += skill_points
    breakdown["Skills Found"] = f"{skill_points}/25"

    # Word count (20 pts)
    word_count = len(resume_text.split())

    if word_count >= 400:
        wc_points = 20
    elif word_count >= 250:
        wc_points = 14
    elif word_count >= 100:
        wc_points = 7
    else:
        wc_points = 2

    total_score += wc_points
    breakdown["Word Count Adequacy"] = f"{wc_points}/20"

    # Grammar (15 pts)
    grammar_points = max(0, 15 - grammar_errors * 2)
    total_score += grammar_points
    breakdown["Grammar Quality"] = f"{grammar_points}/15"

    # Contact info (10 pts)
    import re

    has_email = bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", resume_text))
    has_phone = bool(re.search(r"\+?\d[\d\s\-().]{7,}\d", resume_text))

    contact_points = (5 if has_email else 0) + (5 if has_phone else 0)
    total_score += contact_points
    breakdown["Contact Information"] = f"{contact_points}/10"

    return {
        "total": min(total_score, 100),
        "breakdown": breakdown,
    }


def generate_suggestions(
    sections_found: dict,
    skills: list,
    missing_skills: list,
    match_score: float,
    quality_score: int,
    word_count: int,
) -> list:
    """
    Generate practical suggestions to improve the resume.
    """
    suggestions = []

    # Missing sections
    missing_sections = [section for section, present in sections_found.items() if not present]
    if missing_sections:
        suggestions.append(
            f"Add missing resume sections: {', '.join(missing_sections)}."
        )

    # Skills improvement
    if len(skills) < 8:
        suggestions.append(
            "Include more skills — aim for at least 10 relevant technical and soft skills."
        )

    if missing_skills:
        suggestions.append(
            f"Consider adding these skills from the job description: {', '.join(missing_skills[:5])}."
        )

    # Match score feedback
    if match_score < 40:
        suggestions.append(
            "Low match with the job description. Try aligning your summary, skills, and experience with JD keywords."
        )
    elif match_score < 65:
        suggestions.append(
            "Moderate match score. Improve alignment by incorporating more JD-specific keywords."
        )

    # Word count feedback
    if word_count < 250:
        suggestions.append(
            "Resume is too short. Expand your experience, projects, or summary."
        )
    elif word_count > 900:
        suggestions.append(
            "Resume is quite long. Keep it concise — ideally 400–700 words."
        )

    # Overall quality
    if quality_score < 50:
        suggestions.append(
            "Overall quality is low. Improve completeness, grammar, and contact details."
        )

    if not suggestions:
        suggestions.append(
            "Strong resume overall. Fine-tune keywords and quantify achievements for better impact."
        )

    return suggestions


def check_grammar(text: str) -> tuple:
    """
    Check grammar using language_tool_python.
    Returns (error_count, error_messages).
    Gracefully falls back if the library isn't available.
    """
    try:
        import language_tool_python

        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text[:3000])  # limit for performance

        errors = [f"• {match.ruleId}: {match.message}" for match in matches[:10]]
        return len(matches), errors

    except Exception:
        return 0, ["Grammar checking unavailable (install language_tool_python)."]
