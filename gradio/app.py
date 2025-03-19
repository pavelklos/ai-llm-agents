import gradio as gr
import openai
from fastapi import FastAPI

MODEL_GPT = 'gpt-4o-mini'
QUESTION = "Short article about city Mount Maunganui (New Zealand) and Antigua (Guatemala), max. 200 words, in Czech language"

# Initialize FastAPI app
app = FastAPI()

# Function to query OpenAI's API
def ask_llm(api_key: str, user_question: str) -> str:
    if not api_key:
        return "Please enter your OpenAI API key."
    if not user_question:
        return "Please enter a question."
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model=MODEL_GPT,
            messages=[{"role": "user", "content": user_question}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to reset inputs
def reset_fields():
    return "", ""

# Define Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="OpenAI API Key", type="password", placeholder="Enter your API Key")
            model = gr.Textbox(
                label="OpenAI Model", value=MODEL_GPT, interactive=False)
        with gr.Column(scale=2):
            user_question = gr.Textbox(
                label="Your Question", placeholder="Ask something...", value=QUESTION)
            ai_answer = gr.Textbox(label="AI Answer", interactive=False)
    with gr.Row():
        reset_btn = gr.Button("Reset")
        submit_btn = gr.Button("Ask AI")
    submit_btn.click(ask_llm, inputs=[
                     api_key, user_question], outputs=ai_answer)
    reset_btn.click(reset_fields, inputs=[], outputs=[
                    user_question, ai_answer])

# Run if this file is executed
if __name__ == "__main__":
    ui = demo
    ui.launch()

# Define Gradio interface
# demo = gr.Interface(
#     fn=ask_llm,
#     inputs=[
#         gr.Textbox(label="OpenAI API Key", type="password"),
#         gr.Textbox(label="Your Question", placeholder="Ask something..."),
#     ],
#     outputs=gr.Textbox(label="AI Answer", interactive=False),
#     allow_flagging="never"
# )
