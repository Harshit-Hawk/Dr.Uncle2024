from flask import Flask, request, jsonify, render_template_string
import requests
import json
import warnings
import time

# Suppress Flask development server warnings
warnings.filterwarnings('ignore', category=UserWarning, module='flask')

app = Flask(__name__)

# --- Prediction Function using Gemini API ---
def predict_disease_with_gemini(symptoms_input):
    """
    Predicts a disease using the Gemini API based on a conversational prompt,
    providing a diagnosis, description, precautions, and home remedies in the user's language.

    Args:
        symptoms_input (str): A string of symptoms provided by the user.

    Returns:
        dict: A dictionary containing the predicted disease, description,
              precautions, and home remedies. Returns an error dictionary
              if the API call fails or no data is found.
    """
    prompt = f"""
    You are a friendly and professional medical assistant. A user has described their symptoms. Please respond in a conversational tone and use the same language they used (including Hinglish). Provide a likely disease, a brief description, precautions, and home remedies.

    Symptoms provided: {symptoms_input}

    Your response MUST be in a structured JSON format. Do not include any text outside of the JSON object. The JSON should have the following keys:
    - "predicted_disease": The most likely disease based on the symptoms.
    - "description": A concise, professional description of the disease.
    - "precautions": An array of 3 to 4 strings, each being a clear, actionable precaution.
    - "home_remedies": An array of 3 to 4 strings, each being a simple home remedy.
    - "language": The language of the response, e.g., "English", "Hindi", or "Hinglish".

    If the symptoms are too vague or do not match any known diseases, respond with a "No prediction" result in the user's language.
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    api_key = "AIzaSyAKV_8qYBNV_Z84S3I-p1NhpY-6n8kZVoc"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    retries = 0
    max_retries = 3
    delay = 1

    while retries < max_retries:
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()

            result_text = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
            parsed_json = json.loads(result_text)
            return parsed_json

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}. Retrying in {delay} seconds...")
            retries += 1
            time.sleep(delay)
            delay *= 2

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from API response: {e}. Response was: {result_text}")
            return {"predicted_disease": "Error", "description": "Failed to parse API response. The model may have returned an invalid format.", "precautions": [], "home_remedies": [], "language": "English"}

    return {"predicted_disease": "Error", "description": "Failed to connect to the prediction service after multiple retries.", "precautions": [], "home_remedies": [], "language": "English"}


# Main Chat Page Route
@app.route('/')
def chat_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dr. Uncle - Chat</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            body { font-family: 'Inter', sans-serif; background-color: #0b1a2a; }
            .chat-container { height: calc(100vh - 120px); overflow-y: auto; scroll-behavior: smooth; padding-top: 1rem; }
            .message-bubble { max-width: 80%; }
        </style>
    </head>
    <body class="flex flex-col min-h-screen">
        <header class="bg-[#122839] text-white p-4 md:p-6 flex items-center justify-center shadow-lg">
            <div class="flex items-center space-x-2">
                <svg class="w-8 h-8 text-blue-400" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h2v-6h-2v6zm0-8h2V7h-2v2z"/>
                </svg>
                <span class="text-2xl font-bold">Dr. Uncle</span>
            </div>
        </header>
        <main class="flex-grow p-4 md:p-8 flex flex-col items-center">
            <div class="w-full max-w-2xl bg-[#122839] rounded-xl p-6 shadow-xl flex flex-col h-[80vh] md:h-auto">
                <div id="chat-messages" class="chat-container flex-grow space-y-4 p-4 rounded-xl bg-gray-800 text-gray-100 mb-4">
                    <div class="flex items-start space-x-2">
                        <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm">AI</div>
                        <div class="bg-[#1e3247] p-4 rounded-xl shadow-sm message-bubble">
                            <p class="text-gray-200">Hello! I'm Dr. Uncle, your friendly medical assistant. Please tell me your symptoms, and I'll do my best to help you.</p>
                        </div>
                    </div>
                </div>
                <div class="flex space-x-2 items-center">
                    <input type="text" id="symptoms-input" class="flex-grow p-3 rounded-xl border border-gray-600 bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Send a message...">
                    <button id="voice-input-button" class="bg-gray-600 hover:bg-blue-600 text-white font-semibold p-3 rounded-xl transition duration-300 ease-in-out shadow-md">
                        <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3.53-2.64 6.45-6.29 6.91V21h-1.02v-3.09c-3.65-.46-6.3-3.38-6.3-6.91h-2c0 4.19 3.35 7.63 7.6 8.12V22h3.4v-2.88c4.25-.49 7.6-3.93 7.6-8.12h-2z"/>
                        </svg>
                    </button>
                    <button id="send-button" class="bg-blue-600 hover:bg-blue-700 text-white font-semibold p-3 rounded-xl transition duration-300 ease-in-out shadow-md">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </button>
                </div>
            </div>
        </main>
        <script>
            const chatMessages = document.getElementById('chat-messages');
            const symptomsInput = document.getElementById('symptoms-input');
            const sendButton = document.getElementById('send-button');
            const voiceInputButton = document.getElementById('voice-input-button');
            let isListening = false;

            function scrollToBottom() {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            function addUserMessage(message) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('flex', 'justify-end');
                messageElement.innerHTML = `<div class="bg-blue-600 text-white p-4 rounded-xl shadow-sm message-bubble"><p>${message}</p></div>`;
                chatMessages.appendChild(messageElement);
                scrollToBottom();
            }
            function addAIMessage(data) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('flex', 'items-start', 'space-x-2');
                let responseHtml = `<div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm">AI</div><div class="bg-[#1e3247] text-gray-200 p-4 rounded-xl shadow-sm message-bubble">${data.predicted_disease !== 'Error' ? `<h3 class="font-bold text-lg mb-2 text-blue-400">${data.predicted_disease}</h3><p class="mb-4">${data.description}</p><h4 class="font-semibold mb-2">Precautions:</h4><ul class="list-disc list-inside space-y-1 text-sm text-gray-300">${data.precautions.map(p => `<li>${p}</li>`).join('')}</ul><h4 class="font-semibold mt-4 mb-2">Home Remedies:</h4><ul class="list-disc list-inside space-y-1 text-sm text-gray-300">${data.home_remedies.map(r => `<li>${r}</li>`).join('')}</ul><div class="mt-4 pt-4 border-t border-gray-600 text-xs text-gray-400 italic">Disclaimer: This is for informational purposes only and not a substitute for professional medical advice. Always consult with a qualified healthcare professional.</div>` : `<p class="text-red-400 font-semibold">${data.description}</p>`}</div>`;
                messageElement.innerHTML = responseHtml;
                chatMessages.appendChild(messageElement);
                scrollToBottom();
            }
            function addLoadingIndicator() {
                const loadingElement = document.createElement('div');
                loadingElement.id = 'loading-indicator';
                loadingElement.classList.add('flex', 'items-start', 'space-x-2');
                loadingElement.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm">AI</div>
                    <div class="bg-[#1e3247] text-gray-200 p-4 rounded-xl shadow-sm message-bubble animate-pulse">
                        <p>Dr. Uncle is thinking...</p>
                    </div>
                `;
                chatMessages.appendChild(loadingElement);
                scrollToBottom();
            }
            function removeLoadingIndicator() {
                const loadingElement = document.getElementById('loading-indicator');
                if (loadingElement) {
                    loadingElement.remove();
                }
            }
            async function sendMessage() {
                const symptoms = symptomsInput.value.trim();
                if (symptoms === "") return;
                addUserMessage(symptoms);
                symptomsInput.value = '';
                addLoadingIndicator();
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ symptoms: symptoms })
                    });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const result = await response.json();
                    removeLoadingIndicator();
                    addAIMessage(result);
                } catch (error) {
                    console.error('Error:', error);
                    removeLoadingIndicator();
                    addAIMessage({ predicted_disease: 'Error', description: 'Could not connect to the service. Please try again.', precautions: [], home_remedies: [] });
                }
            }

            // Voice Input Logic
            if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';

                voiceInputButton.addEventListener('click', () => {
                    if (isListening) {
                        recognition.stop();
                    } else {
                        recognition.start();
                    }
                });

                recognition.addEventListener('start', () => {
                    isListening = true;
                    voiceInputButton.classList.add('bg-red-600', 'animate-pulse');
                    voiceInputButton.classList.remove('bg-gray-600', 'hover:bg-blue-600');
                    symptomsInput.placeholder = 'Speak now...';
                });

                recognition.addEventListener('result', (event) => {
                    const transcript = event.results[0][0].transcript;
                    symptomsInput.value = transcript;
                });

                recognition.addEventListener('end', () => {
                    isListening = false;
                    voiceInputButton.classList.remove('bg-red-600', 'animate-pulse');
                    voiceInputButton.classList.add('bg-gray-600', 'hover:bg-blue-600');
                    symptomsInput.placeholder = 'Send a message...';
                    if (symptomsInput.value.trim() !== '') {
                        sendMessage();
                    }
                });

                recognition.addEventListener('error', (event) => {
                    console.error('Speech recognition error:', event.error);
                    isListening = false;
                    voiceInputButton.classList.remove('bg-red-600', 'animate-pulse');
                    voiceInputButton.classList.add('bg-gray-600', 'hover:bg-blue-600');
                    symptomsInput.placeholder = 'Send a message...';
                    // You can add a visual error message to the user here
                });

            } else {
                console.warn('Speech Recognition is not supported in this browser.');
                voiceInputButton.style.display = 'none'; // Hide the button if not supported
            }

            sendButton.addEventListener('click', sendMessage);
            symptomsInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

# This route handles the AI prediction request from the front end.
@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms')
    result = predict_disease_with_gemini(symptoms)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
