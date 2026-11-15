import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()  # reads GROQ_API_KEY from a local .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise SystemExit("ERROR: GROQ_API_KEY is not set. Add it to your .env file.")

try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512
    )

    response = llm.invoke([HumanMessage(content="Say 'API is working' in 5 words")])
    print("SUCCESS:", response.content)

except Exception as e:
    print("ERROR:", e)