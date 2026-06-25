"""
AI Interview Preparation Agent — LangChain three-agent pipeline version.

Architecture: three chained agents
  1. Analysis Agent          -> extracts key themes from the CV + job description
  2. Generation Agent        -> writes a focused, professional interview question
  3. Quality-Control Agent   -> validates the question and prevents hallucinations

The three agents are connected with a SequentialChain.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# The API key is read from an environment variable for security.
# Set it before running, e.g.:  export OPENAI_API_KEY="your_key_here"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# Define the model (GPT-4o) and parameters
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Stage 1: Analysis Agent — extract key themes from the resume
analysis_prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="נתח את קורות החיים: {resume} עבור משרת: {job_description}. זהה 2 נושאים קריטיים לבחינה.",
)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis")

# Stage 2: Generation Agent — write a focused interview question
question_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="בהתבסס על הניתוח הבא: {analysis}, נסח שאלת ראיון מאתגרת ומקצועית.",
)
question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="raw_question")

# Stage 3: Quality-Control Agent — validate relevance and prevent hallucinations
qc_prompt = PromptTemplate(
    input_variables=["raw_question"],
    template=(
        "בדוק את השאלה: '{raw_question}'. האם היא מקצועית ומנוסחת היטב? "
        "אם כן, החזר אותה. אם לא, תקן אותה."
    ),
)
qc_chain = LLMChain(llm=llm, prompt=qc_prompt, output_key="final_question")

# Build the full pipeline (SequentialChain)
overall_pipeline = SequentialChain(
    chains=[analysis_chain, question_chain, qc_chain],
    input_variables=["resume", "job_description"],
    output_variables=["analysis", "final_question"],
    verbose=True,
)


# Example run
if __name__ == "__main__":
    result = overall_pipeline.invoke({
        "resume": "סטודנט להנדסת תעשייה וניהול, ידע ב-Python, חסר ניסיון קודם.",
        "job_description": "אנליסט נתונים מתחיל.",
    })
    print(result["final_question"])
