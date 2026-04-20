from utils.preprocessing import clean_text


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between resume and job description using TF-IDF.
    Returns a percentage score rounded to 2 decimal places (0–100).
    """
    if not resume_text or not jd_text:
        return 0.0

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(float(score) * 100, 2)

    except ImportError:
        # Fallback: Jaccard similarity on word sets
        set_r = set(cleaned_resume.split())
        set_j = set(cleaned_jd.split())
        if not set_r or not set_j:
            return 0.0
        intersection = set_r & set_j
        union = set_r | set_j
        return round(len(intersection) / len(union) * 100, 2)


def compute_resume_quality_score(
    sections_found: dict,
    skills: list,
    resume_text: str,
    grammar_errors: int = 0,
) -> dict:
    """
    Compute a holistic resume quality score (0–100) and return a breakdown.

    Scoring rubric
    --------------
    Sections present      : up to 30 pts  (5 pts each for 6 key sections)
    Skill count           : up to 25 pts  (1 pt per skill, capped at 25)
    Word count adequacy   : up to 20 pts
    Grammar               : up to 15 pts  (deduct per error)
    Contact info present  : up to 10 pts
    """
    score = 0
    breakdown = {}

    # 1. Sections (30 pts)
    key_sections = ["Education", "Experience", "Skills", "Projects", "Summary", "Certifications"]
    section_score = sum(5 for s in key_sections if sections_found.get(s, False))
    score += section_score
    breakdown["Sections Detected"] = f"{section_score}/30"

    # 2. Skill count (25 pts)
    skill_score = min(len(skills), 25)
    score += skill_score
    breakdown["Skills Found"] = f"{skill_score}/25"

    # 3. Word count (20 pts)
    word_count = len(resume_text.split())
    if word_count >= 400:
        wc_score = 20
    elif word_count >= 250:
        wc_score = 14
    elif word_count >= 100:
        wc_score = 7
    else:
        wc_score = 2
    score += wc_score
    breakdown["Word Count Adequacy"] = f"{wc_score}/20"

    # 4. Grammar (15 pts)
    grammar_score = max(0, 15 - grammar_errors * 2)
    score += grammar_score
    breakdown["Grammar Quality"] = f"{grammar_score}/15"

    # 5. Contact info (10 pts) – simple heuristic
    import re
    has_email = bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", resume_text))
    has_phone = bool(re.search(r"\+?\d[\d\s\-().]{7,}\d", resume_text))
    contact_score = (5 if has_email else 0) + (5 if has_phone else 0)
    score += contact_score
    breakdown["Contact Information"] = f"{contact_score}/10"

    return {
        "total": min(score, 100),
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
    Generate actionable improvement suggestions based on analysis results.
    Returns a list of suggestion strings.
    """
    suggestions = []

    # Sections
    missing_sections = [s for s, v in sections_found.items() if not v]
    if missing_sections:
        suggestions.append(
            f"Add missing resume sections: {', '.join(missing_sections)}."
        )

    # Skills
    if len(skills) < 8:
        suggestions.append(
            "List more technical and soft skills — aim for at least 10 relevant skills."
        )

    if missing_skills:
        top_missing = missing_skills[:5]
        suggestions.append(
            f"Consider adding these skills from the job description: {', '.join(top_missing)}."
        )

    # Match score
    if match_score < 40:
        suggestions.append(
            "Your resume has a low match with the job description. "
            "Tailor your summary, skills, and experience to mirror the JD's language."
        )
    elif match_score < 65:
        suggestions.append(
            "Moderate match score. Strengthen alignment by using more keywords from the job description."
        )

    # Word count
    if word_count < 250:
        suggestions.append(
            "Your resume seems too short. Expand your experience, projects, or summary sections."
        )
    elif word_count > 900:
        suggestions.append(
            "Your resume may be too long. Keep it concise — ideally 400–700 words for most roles."
        )

    # Quality
    if quality_score < 50:
        suggestions.append(
            "Overall quality score is low. Focus on completeness, contact info, and grammar."
        )

    if not suggestions:
        suggestions.append(
            "Great resume! Fine-tune keywords and quantify achievements for maximum impact."
        )

    return suggestions


def check_grammar(text: str) -> tuple:
    """
    Use language_tool_python to check grammar.
    Returns (error_count, list_of_error_messages).
    Falls back gracefully if the library isn't installed.
    """
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text[:3000])  # limit to first 3000 chars for speed
        errors = [f"• {m.ruleId}: {m.message}" for m in matches[:10]]
        return len(matches), errors
    except Exception:
        return 0, ["Grammar checking unavailable (install language_tool_python)."]
