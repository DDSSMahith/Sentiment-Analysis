from flask import Flask, request, render_template_string
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Mini Tech Challenge</title>
  </head>
  <body>
    <h2>Enter Customer Call Transcript</h2>
    <form method="POST">
      <textarea name="transcript" rows="6" cols="60"></textarea><br><br>
      <button type="submit">Analyze</button>
    </form>
    {% if transcript %}
      <h3>Original Transcript:</h3>
      <p>{{ transcript }}</p>
      <h3>Summary:</h3>
      <p>{{ summary }}</p>
      <h3>Sentiment:</h3>
      <p>{{ sentiment }}</p>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    transcript, summary, sentiment = None, None, None

    if request.method == "POST":
        transcript = request.form["transcript"]

        
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Transcript: {transcript}\n\n1. Summarize in 2-3 sentences.\n2. Identify sentiment (positive, neutral, negative)."}
            ]
        )

        
        result = response.choices[0].message.content.strip()

        
        parts = result.split("Sentiment:")
        summary = parts[0].replace("Summary:", "").strip()
        sentiment = parts[1].strip() if len(parts) > 1 else "Unknown"

        
        df = pd.DataFrame([[transcript, summary, sentiment]], columns=["Transcript", "Summary", "Sentiment"])
        df.to_csv("call_analysis.csv", mode="a", header=not os.path.exists("call_analysis.csv"), index=False)

    return render_template_string(HTML_TEMPLATE, transcript=transcript, summary=summary, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
