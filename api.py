from dotenv import load_dotenv
import os
import base64
from fastapi import FastAPI, File, UploadFile
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()
client = Groq(api_key=api_key)


@app.post("/inf")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this food label image. Identify who it is suitable for (e.g., people with iron deficiency) and who it isn't suitable for (e.g., lactose intolerant people). Respond in JSON format: {'suitable_for': ['condition1', 'condition2'], 'not_suitable_for': ['condition3', 'condition4']}",
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
