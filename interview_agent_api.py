"""
AI Interview Preparation Agent — Direct OpenAI API version.

A conversational agent that prepares a candidate for a job interview:
it analyzes the candidate's resume, asks challenging questions,
and gives constructive feedback to improve their answers.
"""

import os
import openai

# The API key is read from an environment variable for security.
# Set it before running, e.g.:  export OPENAI_API_KEY="your_key_here"
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run_interview_agent(user_input, resume_context):
    # System prompt that defines the agent's behavior (as designed in the project)
    system_instruction = f"""
    אתה מאמן מקצועי להכנה לראיונות עבודה.
    עליך לנתח את קורות החיים של המועמד: {resume_context}
    תפקידך לשאול שאלות מאתגרות, לתת משוב בונה ולעזור למועמד לשפר את תשובותיו.
    הקפד על טון מקצועי, מעודד וממוקד במשרה המבוקשת.
    """

    try:
        # Send the request to OpenAI's GPT-4o model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=1.0,
        )

        # Extract the reply from the response
        agent_reply = response.choices[0].message.content
        return agent_reply

    except Exception as e:
        return f"שגיאה בהתחברות ל-API: {str(e)}"


# Example run
if __name__ == "__main__":
    resume = "סטודנט להנדסת תעשייה וניהול, שנה ד', ניסיון באופטימיזציה של תהליכים."
    user_msg = "אני מתראיין למשרת PM בסטארטאפ, תוכל לשאול אותי שאלה על התמודדות עם משבר בצוות?"

    print("--- תשובת הסוכן (דרך API) ---")
    print(run_interview_agent(user_msg, resume))
