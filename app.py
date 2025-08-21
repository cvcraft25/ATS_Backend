from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


# =========================
# Data Models
# =========================
class EducationItem(BaseModel):
    dates: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    gpa: Optional[str] = None
    institute: Optional[str] = None
    location: Optional[str] = None


class ExperienceItem(BaseModel):
    company: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[str] = None
    position: Optional[str] = None


class PaperItem(BaseModel):
    conference: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None
    title: Optional[str] = None


class ProjectItem(BaseModel):
    name: Optional[str] = None
    tools: Optional[str] = None
    link: Optional[str] = None
    description: Optional[List[str]] = None


class SchoolItem(BaseModel):
    board: Optional[str] = None
    dates: Optional[str] = None
    location: Optional[str] = None
    percentage: Optional[str] = None
    schoolName: Optional[str] = None


class Profile(BaseModel):
    address: Optional[str] = None
    email: Optional[str] = None
    github: Optional[str] = None
    linkedin: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None


class Skills(BaseModel):
    languages: Optional[List[str]] = None
    technologies: Optional[List[str]] = None
    tools: Optional[List[str]] = None


class ResumePayload(BaseModel):
    userId: Optional[str] = None
    coursework: Optional[List[str]] = None
    education: Optional[List[EducationItem]] = None
    email: Optional[str] = None
    experience: Optional[List[ExperienceItem]] = None
    extracurricular: Optional[List[str]] = None
    higherSecondarySchool: Optional[SchoolItem] = None
    name: Optional[str] = None
    papers: Optional[List[PaperItem]] = None
    profile: Optional[Profile] = None
    projects: Optional[List[ProjectItem]] = None
    secondarySchool: Optional[SchoolItem] = None
    skills: Optional[Skills] = None
    summary: Optional[str] = None
    jobDescription: str = Field(..., description="Full JD text")


# =========================
# Helpers
# =========================
def build_resume_text(data: ResumePayload) -> str:
    parts: List[str] = []
    if data.name:
        parts.append(f"Name: {data.name}")
    if data.email:
        parts.append(f"Email: {data.email}")
    if data.profile:
        p = data.profile
        contact_bits = []
        if p.phone: contact_bits.append(f"Phone: {p.phone}")
        if p.linkedin: contact_bits.append(f"LinkedIn: {p.linkedin}")
        if p.github: contact_bits.append(f"GitHub: {p.github}")
        if p.address: contact_bits.append(f"Address: {p.address}")
        if contact_bits:
            parts.append("Profile: " + "; ".join(contact_bits))
    if data.summary:
        parts.append(f"Summary: {data.summary}")
    if data.skills:
        s = []
        if data.skills.languages:
            s.append("Languages: " + ", ".join(data.skills.languages))
        if data.skills.technologies:
            s.append("Technologies: " + ", ".join(data.skills.technologies))
        if data.skills.tools:
            s.append("Tools: " + ", ".join(data.skills.tools))
        if s:
            parts.append("Skills: " + " | ".join(s))
    if data.coursework:
        parts.append("Coursework: " + ", ".join(data.coursework))
    if data.education:
        edu_lines = []
        for e in data.education:
            bits = [v for v in [e.degree, e.field, e.institute, e.location, e.dates, ("GPA " + e.gpa) if e.gpa else None] if v]
            if bits:
                edu_lines.append(" - " + ", ".join(bits))
        if edu_lines:
            parts.append("Education:\n" + "\n".join(edu_lines))
    if data.experience:
        exp_lines = []
        for x in data.experience:
            bits = [v for v in [x.position, x.company, x.duration, x.location if hasattr(x, 'location') else None] if v]
            line = " - " + ", ".join(bits) if bits else " - Experience"
            if x.description:
                line += f" | {x.description}"
            exp_lines.append(line)
        if exp_lines:
            parts.append("Experience:\n" + "\n".join(exp_lines))
    if data.projects:
        proj_lines = []
        for pr in data.projects:
            line = " - " + ", ".join([v for v in [pr.name, pr.tools, pr.link] if v])
            if pr.description:
                line += " | Desc: " + "; ".join(pr.description)
            proj_lines.append(line)
        if proj_lines:
            parts.append("Projects:\n" + "\n".join(proj_lines))
    if data.papers:
        paper_lines = []
        for pa in data.papers:
            bits = [v for v in [pa.title, pa.conference, pa.link] if v]
            line = " - " + ", ".join(bits)
            if pa.description:
                line += f" | {pa.description}"
            paper_lines.append(line)
        if paper_lines:
            parts.append("Papers:\n" + "\n".join(paper_lines))
    if data.extracurricular:
        parts.append("Extracurricular: " + ", ".join(data.extracurricular))
    if data.higherSecondarySchool:
        hs = data.higherSecondarySchool
        bits = [v for v in [hs.schoolName, hs.board, hs.location, hs.dates, hs.percentage] if v]
        if bits:
            parts.append("Higher Secondary: " + ", ".join(bits))
    if data.secondarySchool:
        ss = data.secondarySchool
        bits = [v for v in [ss.schoolName, ss.board, ss.location, ss.dates, ss.percentage] if v]
        if bits:
            parts.append("Secondary School: " + ", ".join(bits))

    return "\n".join(parts)


def gemini_generate(prompt: str, resume_text: str, job_desc: str) -> str:
    try:
        resp = model.generate_content([prompt, "\n[RESUME]\n" + resume_text, "\n[JOB_DESCRIPTION]\n" + job_desc])
        return resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


# =========================
# Prompts (preserve original behavior)
# =========================
PROMPT_RESUME_EVAL = (
    "You are an experienced Technical Human Resource Manager. Review the resume against the job description and "
    "return a concise, point-wise report in Markdown with headings and bullet points. Emphasize critical keywords.\n\n"
    "FORMAT STRICTLY AS:\n"
    "## Resume Evaluation: {CANDIDATE_NAME if available}\n\n"
    "**Position:** React Native Developer (or inferred role)\n\n"
    "**Overall Alignment (1-2 lines):** <short summary>\n\n"
    "### Strengths\n"
    "- <bullet> Include bolded keywords, e.g., **React Native**, **iOS/Android**, **TypeScript**, **Redux**, **Jest/Detox**, **CI/CD**\n"
    "- <bullet> Quantified impact where applicable\n\n"
    "### Weaknesses / Gaps\n"
    "- <bullet> Keep crisp; highlight missing keywords or limited experience areas\n\n"
    "### Keywords\n"
    "- Present 10-20 most relevant keywords as a compact comma-separated list with bolded terms, e.g., **React Native**, **Redux Toolkit**, **GraphQL**, **Firebase**, **Fastlane**, **EAS**, **Xcode**, **Android Studio**\n\n"
    "### Recommendations\n"
    "- <bullet> Actionable next steps (skills to learn, metrics to add, architecture patterns to mention)\n\n"
    "### Overall Recommendation\n"
    "- <one line> Proceed / Consider / Hold with brief rationale."
)

PROMPT_SKILL_IMPROVE = (
    "You're an expert career advisor. Based on the resume and job description, "
    "suggest the top 5 skills to improve or learn. Include concrete technologies or certifications."
)

PROMPT_PERCENTAGE = (
    "You are an ATS (Applicant Tracking System) scanner. Evaluate the resume against the job description. "
    "First provide the overall match percentage. Then list missing important keywords. "
    "Finally provide a concise final thought."
)


# =========================
# FastAPI App & Endpoints
# =========================
app = FastAPI(title="Resume Screening API", version="1.0.0")

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/ats/evaluate")
def evaluate_resume(payload: ResumePayload) -> Dict[str, Any]:
    resume_text = build_resume_text(payload)
    output = gemini_generate(PROMPT_RESUME_EVAL, resume_text, payload.jobDescription)
    return {"userId": payload.userId, "analysis": output}


@app.post("/api/v1/ats/improve")
def improve_skills(payload: ResumePayload) -> Dict[str, Any]:
    resume_text = build_resume_text(payload)
    output = gemini_generate(PROMPT_SKILL_IMPROVE, resume_text, payload.jobDescription)
    return {"userId": payload.userId, "suggestions": output}


@app.post("/api/v1/ats/match")
def percentage_match(payload: ResumePayload) -> Dict[str, Any]:
    resume_text = build_resume_text(payload)
    output = gemini_generate(PROMPT_PERCENTAGE, resume_text, payload.jobDescription)
    return {"userId": payload.userId, "result": output}
