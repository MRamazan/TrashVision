import os
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="TrashVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


@app.get("/")
async def root():
    return {"status": "TrashVision API is running", "version": "1.0.0"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), city: str = Form(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 20MB.")

    media_type = image.content_type
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    scout_prompt = (
        "Analyze this image and identify the waste material. "
        "Return a single valid JSON object with these exact keys: "
        "material (the primary material category: plastic, glass, metal, paper, cardboard, organic, electronic, textile, hazardous, ceramic, rubber, or mixed), "
        "item (specific item name, e.g. plastic bottle), "
        "confidence (high, medium, or low), "
        "description (one sentence describing what you see). "
        "Return only the JSON object with no markdown, no code fences, no extra text."
    )

    scout_response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": scout_prompt},
                ],
            }
        ],
        max_tokens=400,
    )

    classification = extract_json(scout_response.choices[0].message.content)
    if not classification:
        raise HTTPException(status_code=500, detail="Visual classification failed. Please try a clearer image.")

    disposal_prompt = f"""
You are a waste management expert. Given this identified waste item:
{json.dumps(classification)}

And the user's city: {city}

Provide specific disposal instructions and environmental impact data for this city.
Return a single valid JSON object with these exact keys:
- disposal_method: string, clear instructions on how to dispose in {city}
- bin_color: string, exact color name (blue, green, yellow, red, black, grey, orange, brown, purple, white)
- bin_label: string, what locals call this bin (e.g. "Blue Recycling Bin", "General Waste Bin")
- preparation: array of up to 4 strings, steps to prepare item before disposal
- years_to_decompose: string, e.g. "450" or "20-30" or "Indefinite"
- co2_emissions_kg: number, estimated CO2 equivalent in kg if landfilled (use 0.1 for very small items)
- ocean_risk: string, one of: None, Low, Medium, High, Very High
- soil_risk: string, one of: None, Low, Medium, High, Very High
- wildlife_risk: string, one of: None, Low, Medium, High, Very High
- microplastic_risk: string, one of: None, Low, Medium, High, Very High
- recycling_benefit: string, one sentence on what happens if properly recycled
- energy_saved_percent: number, percentage of energy saved recycling vs virgin production (0-100)
- local_tip: string, one specific recycling tip for {city} residents

Return only the JSON object with no markdown, no code fences, no extra text.
"""

    disposal_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": disposal_prompt}],
        max_tokens=900,
    )

    impact_data = extract_json(disposal_response.choices[0].message.content)
    if not impact_data:
        raise HTTPException(status_code=500, detail="Disposal data generation failed. Please try again.")

    return {**classification, **impact_data, "city": city}
