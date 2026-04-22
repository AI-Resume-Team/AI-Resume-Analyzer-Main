import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from utils.resume_parser import parse_resume, extract_email, extract_phone, extract_name, detect_sections
from utils.skill_extractor import extract_skills, get_missing_skills, predict_job_role, get_skill_categories
from utils.matcher import compute_match_score, compute_resume_quality_score, generate_suggestions, check_grammar

app = Flask(__name__)
app.secret_key = "ai-resume-analyzer-secret-key-2026"

# ── Simple in-memory user store (replace with DB in production) ──
# Format: { "email": { "name": ..., "password_hash": ... } }
USERS: dict = {}

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            flash("Please log in to access the analyzer.", "info")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_email" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        name      = request.form.get("name", "").strip()
        email     = request.form.get("email", "").strip().lower()
        password  = request.form.get("password", "")
        confirm   = request.form.get("confirm_password", "")

        # Validation
        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("signup.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        if email in USERS:
            flash("An account with this email already exists.", "error")
            return render_template("signup.html")

        # Create user
        USERS[email] = {
            "name": name,
            "password_hash": generate_password_hash(password),
        }

        session["user_email"] = email
        session["user_name"]  = name
        flash(f"Welcome, {name}! Your account has been created.", "success")
        return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = USERS.get(email)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session["user_email"] = email
        session["user_name"]  = user["name"]
        flash(f"Welcome back, {user['name']}!", "success")
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════
#  MAIN ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html", user_name=session.get("user_name", ""))


@app.route("/analyze", methods=["POST"])
@login_required
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
        user_name=session.get("user_name", ""),
    )


if __name__ == "__main__":
    app.run(debug=True)
