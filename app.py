# app.py
import streamlit as st
import time
import random
import io

# PDF generation (ReportLab)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Your question bank file
from questions_bank import QUESTION_BANK

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def pick_questions(bank, category, num_questions):
    pool = bank.get(category, [])
    if not pool:
        return []
    if num_questions >= len(pool):
        selected = pool.copy()
        random.shuffle(selected)
    else:
        selected = random.sample(pool, num_questions)
    return selected

def evaluate(questions, answers):
    score = 0
    results = []
    for i, q in enumerate(questions):
        selected = answers.get(i, "Not Answered")
        correct = q["answer"]
        is_correct = (selected == correct)

        if is_correct:
            score += 1

        results.append({
            "index": i,
            "question": q["question"],
            "options": q["options"],
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })
    return score, results

def text_wrap(text, width_chars=90):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def create_results_pdf(title, participant_name, results, score, total):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    margin = 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 24
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Participant: {participant_name or 'Anonymous'}")
    y -= 18
    c.drawString(margin, y, f"Score: {score} / {total}")
    y -= 20
    c.line(margin, y, width - margin, y)
    y -= 20

    for r in results:
        if y < 120:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)

        c.setFont("Helvetica-Bold", 11)
        for ln in text_wrap(f"{r['index']+1}. {r['question']}"):
            c.drawString(margin, y, ln)
            y -= 14

        c.setFont("Helvetica", 10)
        for opt in r["options"]:
            c.drawString(margin + 10, y, f"- {opt}")
            y -= 12

        c.drawString(margin + 10, y, f"Your answer: {r['selected']}")
        y -= 12
        c.drawString(margin + 10, y, f"Correct answer: {r['correct']}")
        y -= 12

        if r["explanation"]:
            c.setFont("Helvetica-Oblique", 9)
            for ln in text_wrap("Explanation: " + r["explanation"]):
                c.drawString(margin + 10, y, ln)
                y -= 12
            c.setFont("Helvetica", 10)

        y -= 10

    c.save()
    buffer.seek(0)
    return buffer

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Modular AI Quiz", layout="wide")

# Initialize session
for key, value in {
    "state": "start",
    "questions": [],
    "answers": {},
    "index": 0,
    "submitted": False,
    "quiz_results": []
}.items():
    st.session_state.setdefault(key, value)

# Title
st.title("📚 Modular AI Quiz — Beginner (4 domains)")

# Domain & settings
domains = list(QUESTION_BANK.keys())
col1, col2 = st.columns([2, 1])

with col1:
    selected_domain = st.selectbox("Choose domain:", domains)
with col2:
    max_q = len(QUESTION_BANK[selected_domain])
    avail = [n for n in [5, 10, 15, 20, 25, 30] if n <= max_q]
    num_questions = st.selectbox("Number of questions:", avail)

name = st.text_input("Your name (optional):")

# Start quiz
if st.button("Start Quiz"):
    st.session_state.questions = pick_questions(QUESTION_BANK, selected_domain, num_questions)
    st.session_state.answers = {}
    st.session_state.index = 0
    st.session_state.submitted = False
    st.session_state.state = "running"
    st.rerun()

if st.session_state.state == "start":
    st.info("Pick a domain and number of questions, then click Start Quiz.")
    st.stop()

# ----------------------------------------------------
# Progress Bar (NO TIMER ANYMORE)
# ----------------------------------------------------
questions = st.session_state.questions
q_total = len(questions)
answered = len([v for v in st.session_state.answers.values() if v])

st.progress(int((answered / q_total) * 100))
st.write(f"Progress: **{answered} / {q_total} answered**")
st.write("---")

# ----------------------------------------------------
# Question View
# ----------------------------------------------------
cur = st.session_state.index

if not st.session_state.submitted and cur < q_total:
    q = questions[cur]

    st.subheader(f"Question {cur+1} of {q_total}")
    st.write(q["question"])

    key = f"q_{cur}"

    def choose():
        st.session_state.answers[cur] = st.session_state.get(key)
        st.session_state.index = min(cur + 1, q_total)

    st.radio("Select your answer:", q["options"], key=key, index=None, on_change=choose)

    if st.button("Submit Now"):
        st.session_state.submitted = True

if not st.session_state.submitted and st.session_state.index >= q_total:
    st.session_state.submitted = True

# ----------------------------------------------------
# Results View
# ----------------------------------------------------
if st.session_state.submitted:
    score, results = evaluate(questions, st.session_state.answers)
    st.session_state.quiz_results = results

    st.success(f"Quiz completed — Score: {score} / {q_total}")

    for r in results:
        st.write("---")
        st.write(f"### {r['index']+1}. {r['question']}")

        st.write(f"**Your answer:** {r['selected']}")    
        st.write(f"**Correct answer:** {r['correct']}")

        if r["explanation"]:
            st.info(f"Explanation: {r['explanation']}")

        st.write("**Options:**")
        for opt in r["options"]:
            mark = "✓" if opt == r["correct"] else "-"
            st.write(f"{mark} {opt}")

    # PDF download
    pdf_buf = create_results_pdf(f"{selected_domain} Quiz Results", name, results, score, q_total)
    st.download_button(
        label="📥 Download Results as PDF",
        data=pdf_buf.getvalue(),
        file_name="quiz_results.pdf",
        mime="application/pdf"
    )

    if st.button("Restart Quiz"):
        for k in ["questions", "answers", "index", "submitted", "state", "quiz_results"]:
            st.session_state.pop(k, None)
        st.session_state.state = "start"
        st.rerun()

