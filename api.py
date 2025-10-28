from dotenv import load_dotenv
import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=api_key)


@app.post("/inf")
async def analyze_image(
    file: UploadFile = File(...), dietary_preference: str = Form(None)
):
    contents = await file.read()

    prompt_text = "Analyze this food label image. Identify who it is suitable for (eg., people with iron deficiency) and who it isn't suitable for (e.g., lactose intolerant people). Also, provide a breakdown of nutrient percentages (carbohydrates, proteins, fats, etc.)."
    if dietary_preference:
        prompt_text += (
            f" Consider the following dietary preferences: {dietary_preference}."
        )
    prompt_text += " Do not write the whole process, just give me the final answer."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.content_type};base64,{base64.b64encode(contents).decode()}"
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=1024,
    )
    response = chat_completion.choices[0].message.content
    return {"analysis": response}
