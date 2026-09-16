from dotenv import load_dotenv
from flask import render_template, Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

# Reload environment variables from .env
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with Groq base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

def get_ai_response(user_content, system_content="Act like a helpful personal assistant.", temperature=0.7):
    """Safely calls AI model using standard chat completions with responses fallback."""
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=512
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        # Fallback to responses endpoint
        resp = client.responses.create(
            model="openai/gpt-oss-20b",
            input=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_output_tokens=512
        )
        return resp.output_text.strip()

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form.get("question") or request.form.get("query")
        if not question or not question.strip():
            return jsonify({"response": "Please enter a valid query."}), 400

        answer = get_ai_response(
            user_content=question.strip(),
            system_content="Act like a helpful personal assistant.",
            temperature=0.7
        )
        return jsonify({"response": answer}), 200
    except Exception as e:
        print(f"Error in /ask: {e}")
        return jsonify({"response": f"Error: Unable to get response ({str(e)})"}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        email_text = request.form.get("email") or request.form.get("question")
        if not email_text or not email_text.strip():
            return jsonify({"response": "Please enter valid email text to summarize."}), 400

        prompt = f"Summarize the following email in 2-3 sentences:\n\n{email_text.strip()}"
        summary = get_ai_response(
            user_content=prompt,
            system_content="Act like an expert email assistant.",
            temperature=0.3
        )
        return jsonify({"response": summary}), 200
    except Exception as e:
        print(f"Error in /summarize: {e}")
        return jsonify({"response": f"Error: Unable to summarize ({str(e)})"}), 500

if __name__ == "__main__":
    app.run(debug=True)
