import os
from flask import Flask, render_template, request, redirect, url_for, flash

from utils.resume_parser import parse_resume, extract_email, extract_phone, extract_name, detect_sections
from utils.skill_extractor import extract_skills, get_missing_skills, predict_job_role, get_skill_categories
from utils.matcher import compute_match_score, compute_resume_quality_score, generate_suggestions, check_grammar

app = Flask(__name__)
app.secret_key = "ai-resume-analyzer-secret-key"

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── 1. Validate inputs ────────────────────────────────────────────────
    if "resume" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["resume"]
    jd_text = request.form.get("job_description", "").strip()

    if file.filename == "":
        flash("No file selected. Please upload a PDF or DOCX resume.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only PDF and DOCX are supported.", "error")
        return redirect(url_for("index"))

    # ── 2. Parse resume ───────────────────────────────────────────────────
    resume_text = parse_resume(file.stream, file.filename)

    if not resume_text or resume_text.startswith("Error"):
        flash(f"Could not extract text from resume: {resume_text}", "error")
        return redirect(url_for("index"))

    # ── 3. Extract metadata ───────────────────────────────────────────────
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    sections = detect_sections(resume_text)

    # ── 4. Skill extraction ───────────────────────────────────────────────
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text) if jd_text else []
    missing_skills = get_missing_skills(resume_skills, jd_skills)
    skill_categories = get_skill_categories(resume_skills)

    # ── 5. Role prediction ────────────────────────────────────────────────
    predicted_roles = predict_job_role(resume_skills)

    # ── 6. Match score ────────────────────────────────────────────────────
    match_score = compute_match_score(resume_text, jd_text) if jd_text else 0.0

    # ── 7. Grammar check ─────────────────────────────────────────────────
    grammar_error_count, grammar_errors = check_grammar(resume_text)

    # ── 8. Quality score ──────────────────────────────────────────────────
    quality_data = compute_resume_quality_score(
        sections_found=sections,
        skills=resume_skills,
        resume_text=resume_text,
        grammar_errors=grammar_error_count,
    )

    # ── 9. Suggestions ────────────────────────────────────────────────────
    suggestions = generate_suggestions(
        sections_found=sections,
        skills=resume_skills,
        missing_skills=missing_skills,
        match_score=match_score,
        quality_score=quality_data["total"],
        word_count=len(resume_text.split()),
    )

    # ── 10. Render results ────────────────────────────────────────────────
    return render_template(
        "result.html",
        name=name,
        email=email,
        phone=phone,
        sections=sections,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        missing_skills=missing_skills,
        skill_categories=skill_categories,
        predicted_roles=predicted_roles,
        match_score=match_score,
        grammar_error_count=grammar_error_count,
        grammar_errors=grammar_errors,
        quality_score=quality_data["total"],
        quality_breakdown=quality_data["breakdown"],
        suggestions=suggestions,
        word_count=len(resume_text.split()),
        jd_provided=bool(jd_text),
    )


if __name__ == "__main__":
    app.run(debug=True)
