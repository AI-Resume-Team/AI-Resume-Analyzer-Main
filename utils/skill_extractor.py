import os
import csv

# ---------------------------------------------------------------------------
# Load skills dataset
# ---------------------------------------------------------------------------

_SKILLS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "skills.csv")


def _load_skills() -> dict:
    """
    Returns a dict of  { skill_lowercase: category }.
    Falls back to an empty dict if the CSV is missing.
    """
    skills = {}
    try:
        with open(_SKILLS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill = row.get("skill", "").strip().lower()
                category = row.get("category", "General").strip()
                if skill:
                    skills[skill] = category
    except FileNotFoundError:
        pass
    return skills


_SKILLS_DB: dict = _load_skills()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> list:
    """
    Scan *text* for known skills (multi-word phrases first, then single words).
    Returns a sorted list of matched skill strings (original casing from DB).
    """
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    # Sort by length descending so longer phrases are matched before substrings
    for skill in sorted(_SKILLS_DB.keys(), key=len, reverse=True):
        # Use word-boundary-aware search
        import re
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)


def get_skill_categories(skills: list) -> dict:
    """
    Map each skill to its category.
    Returns { skill: category }.
    """
    return {skill: _SKILLS_DB.get(skill, "General") for skill in skills}


def get_missing_skills(resume_skills: list, jd_skills: list) -> list:
    """
    Return skills present in the job description but absent from the resume.
    """
    resume_set = set(s.lower() for s in resume_skills)
    return [s for s in jd_skills if s.lower() not in resume_set]


def predict_job_role(skills: list) -> list:
    """
    Very lightweight role predictor based on skill-category frequency.
    Returns up to 3 predicted roles sorted by match strength.
    """
    if not skills:
        return ["General Role"]

    categories = get_skill_categories(skills)
    freq: dict = {}
    for cat in categories.values():
        freq[cat] = freq.get(cat, 0) + 1

    # Map category → job role label
    role_map = {
        "Programming":     "Software Developer",
        "Web Development": "Full Stack / Web Developer",
        "AI/ML":           "Machine Learning Engineer / Data Scientist",
        "Data Science":    "Data Analyst / Data Scientist",
        "Database":        "Database Administrator / Backend Developer",
        "Cloud & DevOps":  "DevOps / Cloud Engineer",
        "Mobile":          "Mobile Application Developer",
        "Testing":         "QA / Test Engineer",
        "Tools":           "Technical Project Manager",
        "Soft Skills":     "Project Manager / Team Lead",
    }

    sorted_cats = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    roles = []
    for cat, _ in sorted_cats[:3]:
        role = role_map.get(cat, f"{cat} Specialist")
        if role not in roles:
            roles.append(role)

    return roles if roles else ["General Software Role"]
