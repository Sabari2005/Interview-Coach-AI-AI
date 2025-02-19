from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import cv2
import base64
import uvicorn
from groq import Groq
import pymupdf  
import ast
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
# Initialize Groq client for Llama
client = Groq(api_key="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")  # Replace with your Groq API key

def analyze_confidence_with_llama(user_response):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an expert in analyzing spoken responses for confidence levels. "
                           f"Given the following response from a user, assess their confidence:\n"
                           f"Response: '{user_response}'\n"
                           "Evaluate based on clarity, assertiveness, use of filler words, and tone strength. "
                           "Provide a confidence score (0-100) and give constructive feedback on how they can improve. "
                           "Output format:\n"
                           "- Confidence Score: (numeric value out of 100)\n"
                           "- Feedback: (brief analysis and improvement tips)"
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()
def evaluate_resume_with_llama(resume_text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an expert in resume evaluation for interview preparation. "
                           f"Analyze the following resume and extract key details:\n"
                           f"Resume:\n{resume_text}\n\n"
                           "Identify and list the following:\n"
                           "1️⃣ **Key Skills**: Extract technical and soft skills relevant to the job market.\n"
                           "2️⃣ **Knowledge Areas**: Highlight important domains of expertise.\n"
                           "3️⃣ **Work Experience**: Summarize past roles, responsibilities, and achievements.\n"
                           "4️⃣ **Areas for Improvement**: Provide suggestions for enhancing the resume.\n\n"
                           "Format the output as:\n"
                           "- **Key Skills**: (List of skills)\n"
                           "- **Knowledge Areas**: (Relevant domains)\n"
                           "- **Work Experience**: (Summary of roles & achievements)\n"
                           "- **Suggestions for Improvement**: (Actionable tips to refine the resume)"
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()
def generate_interview_questions(resume_text, job_role):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an HR interviewer preparing for a job interview. "
                           f"Based on the candidate's resume and the job role, generate tailored interview questions.\n\n"
                           f"🔹 **Job Role**: {job_role}\n"
                           f"🔹 **Candidate's Resume**:\n{resume_text}\n\n"
                           "Ask structured questions that assess the following aspects:\n"
                           "1️⃣ **Technical Skills**: Questions to evaluate technical proficiency related to the job.\n"
                           "2️⃣ **Problem-Solving Ability**: Scenario-based or case study questions.\n"
                           "3️⃣ **Past Experience & Achievements**: Questions about the candidate’s previous roles.\n"
                           "4️⃣ **Behavioral & Soft Skills**: Situational questions based on teamwork, leadership, and communication.\n"
                           "5️⃣ **Culture Fit**: Questions to understand alignment with company values.\n\n"
                           "you should give the questions in python list, each question as elements without inner list enclosed by square bracket."
                                       }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()
def evaluate_interview_answer(hr_question, user_answer, job_role):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an expert HR interviewer evaluating a candidate's response. "
                           f"Assess the following answer based on relevance, clarity, confidence, and completeness.\n\n"
                           f"🔹 **Job Role**: {job_role}\n"
                           f"🔹 **Interview Question**: {hr_question}\n"
                           f"🔹 **Candidate's Answer**: {user_answer}\n\n"
                           "Evaluate the answer using the following criteria:\n"
                           " 1️ **Relevance**: Does the response correctly address the question?\n"
                           " 2️ **Clarity & Structure**: Is the answer well-structured and easy to understand?\n"
                           " 3️ **Confidence & Communication**: Does the response reflect confidence and professionalism?\n"
                           " 4️ **Depth & Examples**: Does the candidate provide detailed explanations or examples?\n\n"
                           "Provide a structured output with:\n"
                           "- **Score**: (Numeric value out of 100)\n"
                           "- **Strengths**: (What was done well?)\n"
                           "- **Areas for Improvement**: (How can the candidate improve?)"
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)  
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

pdf_path = "Resume.pdf"
pdf_text = extract_text_from_pdf(pdf_path)



resume_text=pdf_text
job_role="Full Stack Developer"
a2=evaluate_resume_with_llama(resume_text)
a3=generate_interview_questions(resume_text, job_role)







app = FastAPI()
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins (e.g., ["http://127.0.0.1:5500"]) for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static files for styles, scripts, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    """Stream video frames with emotion detection via WebSocket."""
    await websocket.accept()

    # Open the default camera
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze the frame for emotion detection
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']

            # Overlay detected emotion on the frame
            cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)

            # Convert the frame to base64
            frame_data = base64.b64encode(buffer).decode("utf-8")

            # Send the frame and emotion data to the client
            await websocket.send_json({"frame": frame_data, "emotion": dominant_emotion})
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        await websocket.close()


current_question_index = 0
text = a3
start_index = text.find("[")
end_index = text.rfind("]") + 1
list_str = text[start_index:end_index]


question_list = ast.literal_eval(list_str)
@app.post("/get-question")
async def get_question():
    global current_question_index

    if current_question_index >= len(question_list):
        return {"message": "All questions completed!"}
    # global question
    question = question_list[current_question_index]
    current_question_index += 1

    return {"question": question}

@app.post("/submit-answer")
async def submit_answer(request: Request):
    global current_question_index
    data = await request.json()
    user_answer = data.get("answer")
    question_index = data.get("question_index")  # Get the question index from the client

    if not user_answer:
        return {"message": "No answer provided!"}

    # Process the user's answer
    question = question_list[question_index]
    print(f"Question: {question}")
    print(f"Answer: {user_answer}")

    # Analyze the confidence and evaluate the answer
    a1= analyze_confidence_with_llama(user_answer)
    a4 = evaluate_interview_answer(question, user_answer, job_role)
    # a1 = f"Confidence analysis for: {user_answer}"  # Mocking analyze_confidence_with_llama
    # a4 = f"Evaluation for: {user_answer} (job role: {job_role})"  # Mocking evaluate_interview_answer

    return {
        "analysis": a1,
        "evaluation": a4,
        "next_question": current_question_index < len(question_list),
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)