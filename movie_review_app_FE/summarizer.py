import os
import traceback

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()  # reads GROQ_API_KEY from a local .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_CHUNK_CHARS = 2000  # limit sent to the model per chunk


def get_llm_llama():
    """Initialize Groq LLM"""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set. Add it to your .env file.")
        return None
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=512
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None


def split_text(text, chunk_size=1500):
    """Split large text into chunks"""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Split text into {len(chunks)} chunks")
    return chunks


def summarize_chunk(llm, chunk, chunk_num=1):
    """Summarize individual chunk"""
    if not llm:
        return f"Unable to summarize chunk {chunk_num}: LLM not initialized"

    try:
        limited_chunk = chunk[:MAX_CHUNK_CHARS]  # limit chunk length
        prompt = f"""
        Summarize the following movie reviews into a short paragraph. Keep it concise and informative (max 100 words).

        Reviews:
        {limited_chunk}
        """

        print(f"Summarizing chunk {chunk_num}...")
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"Chunk {chunk_num} summarized successfully")
        return response.content
    except Exception as e:
        print(f"Error summarizing chunk {chunk_num}: {e}")
        print(traceback.format_exc())
        return f"Error summarizing chunk {chunk_num}: {str(e)}"


def summarize_reviews(full_text):
    """Generate final summary from all reviews"""
    print(f"summarize_reviews called with text length: {len(full_text)}")

    if not full_text or full_text.strip() == "":
        print("No text to summarize")
        return "No reviews to summarize."

    try:
        llm = get_llm_llama()
        if not llm:
            print("Failed to initialize LLM")
            return "Error: Could not initialize the summarization model. Please check your API key."

        chunks = split_text(full_text)

        if not chunks:
            print("No chunks created")
            return "Unable to process the reviews."

        print(f"Processing {len(chunks)} chunks...")
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            summary = summarize_chunk(llm, chunk, i)
            chunk_summaries.append(summary)

        print(f"All chunks summarized. Combining {len(chunk_summaries)} summaries...")

        # If only one chunk, return that summary directly
        if len(chunk_summaries) == 1:
            print("Returning single chunk summary")
            return chunk_summaries[0]

        # Combine multiple chunks
        final_prompt = f"""
        Combine the following summaries into ONE clear and concise paragraph summary of the movie reviews.
        Do NOT use bullet points or headings. Make it flow naturally. Keep it under 150 words.
        No Preamble

        Summaries to combine:
        {' '.join(chunk_summaries)}
        """

        print("Generating final combined summary...")
        final_summary = llm.invoke([HumanMessage(content=final_prompt)])
        print("Final summary generated successfully")
        return final_summary.content

    except Exception as e:
        print(f"Summarization error: {e}")
        print(traceback.format_exc())
        return f"Error generating summary: {str(e)}"


def generate_dual_summary(positive_reviews, negative_reviews):
    """Generate separate summaries for positive and negative reviews"""
    print("=" * 50)
    print("Generate dual summary function called")
    print(f"Positive reviews count: {len(positive_reviews) if positive_reviews else 0}")
    print(f"Negative reviews count: {len(negative_reviews) if negative_reviews else 0}")
    print("=" * 50)

    summaries = {}

    # Process positive reviews
    if positive_reviews and len(positive_reviews) > 0:
        positive_text = "\n\n".join(positive_reviews)
        print(f"Positive reviews text length: {len(positive_text)}")
        print("Generating positive summary...")
        summaries['positive'] = summarize_reviews(positive_text)
        print(f"Positive summary generated: {summaries['positive'][:100]}...")
    else:
        print("No positive reviews available")
        summaries['positive'] = "No positive reviews available."

    # Process negative reviews
    if negative_reviews and len(negative_reviews) > 0:
        negative_text = "\n\n".join(negative_reviews)
        print(f"Negative reviews text length: {len(negative_text)}")
        print("Generating negative summary...")
        summaries['negative'] = summarize_reviews(negative_text)
        print(f"Negative summary generated: {summaries['negative'][:100]}...")
    else:
        print("No negative reviews available")
        summaries['negative'] = "No negative reviews available."

    print("=" * 50)
    print("Dual summary generation completed")
    print("=" * 50)

    return summaries