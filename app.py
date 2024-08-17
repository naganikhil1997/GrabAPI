from flask import Flask, request, jsonify
import os
import google.generativeai as genai
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get API key from environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",  
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/generate_recipe_info', methods=['POST'])
def generate_recipe_info():
    user_input = request.json.get('user_input')

    chat_session = model.start_chat(history=[])

    response = chat_session.send_message(f"""
    You are an expert recipe assistant. Given the following request, provide detailed information about the top 10 recipes, including:

    1. The cost estimation of each item 
    2. The calories for each item.
    3. The overall budget for the recipes.
    4. The quantity of ingredients required for each recipe.
    5. Write the short summary of the problem and your suggested resolutions.
                                         

    Request: {user_input}

  If the input is a greeting like "hi" or "hello", respond appropriately with a greeting message. Otherwise, provide the response in a structured format with headings, and include all relevant details. Ensure that cost and calorie details are included in a descriptive model & table format.
    """)

    logging.debug(f"Raw response: {response.text}")

    try:
        response_json = json.loads(response.text)
        return jsonify(response_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse response as JSON", "details": response.text}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
