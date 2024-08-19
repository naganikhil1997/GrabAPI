from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

app = Flask(__name__)

def create_model(api_key: str):
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            description="Generated response based on user request about food items or dishes",
            properties={
                "message": content.Schema(
                    type=content.Type.STRING,
                    description="Personalized message or greeting",
                ),
                "description": content.Schema(
                    type=content.Type.STRING,
                    description="Description or summary of the user's query and generated information",
                ),
                "dishes": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "dish_name": content.Schema(
                                type=content.Type.STRING,
                                description="Name of the dish",
                            ),
                            "description": content.Schema(
                                type=content.Type.STRING,
                                description="Detailed description of the dish",
                            ),
                            "recipe_steps": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    description="Step-by-step instructions for making the dish",
                                ),
                            ),
                            "ingredients": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    description="Ingredients required for the dish",
                                ),
                            ),
                            "cost": content.Schema(
                                type=content.Type.STRING,
                                description="Estimated cost to make the dish",
                            ),
                            "calories": content.Schema(
                                type=content.Type.STRING,
                                description="Caloric content of the dish",
                            ),
                            "servings": content.Schema(
                                type=content.Type.STRING,
                                description="Number of servings the dish provides",
                            ),
                        },
                    ),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    return model

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({'error': 'API key not found'}), 500

    model = create_model(api_key)
    
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Generate a detailed response with a personalized message based on the following request: {user_query}. Include a greeting or well-wish for the occasion and then provide detailed information such as dish name, description, step-by-step recipe, ingredients, cost, calories, and number of servings for each dish. The response should be in JSON format.",
                ],
            },
        ]
    )
    
    response = chat_session.send_message(user_query)
    return jsonify(response.text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
