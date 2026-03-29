import os
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from groq import Groq, APIStatusError, APIConnectionError

app = FastAPI()

_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=_groq_api_key)


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
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


@app.get("/api/analyze")
async def health():
    return {"status": "TrashVision API ready"}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), location: str = Form(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await image.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 15 MB.")

    media_type = image.content_type
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    scout_prompt = (
        "Analyze this image and identify the waste material. "
        "Return one valid JSON object — no markdown, no code fences — with these keys: "
        "material (one of: plastic, glass, metal, paper, cardboard, organic, electronic, textile, hazardous, ceramic, rubber, mixed), "
        "item (specific item name, e.g. plastic bottle), "
        "confidence (high | medium | low), "
        "description (one sentence about what you see)."
    )

    try:
        scout_resp = client.chat.completions.create(
            # Fixed: Groq uses this ID, not "meta-llama/llama-4-scout-17b-16e-instruct"
            model="llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                        {"type": "text", "text": scout_prompt},
                    ],
                }
            ],
            max_tokens=400,
        )
    except APIStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Image classification failed: {e.status_code} {e.message}",
        )
    except APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Groq API: {e}")

    classification = extract_json(scout_resp.choices[0].message.content)
    if not classification:
        raise HTTPException(status_code=500, detail="Could not classify the image. Please try a clearer photo.")

    disposal_prompt = f"""
You are a waste management expert with deep knowledge of local recycling laws worldwide.

Identified waste: {json.dumps(classification)}
User location: {location}

Return one valid JSON object — no markdown, no code fences — with these exact keys:
- disposal_method: string, specific instructions for {location}
- bin_color: string, exact color (blue | green | yellow | red | black | grey | orange | brown | purple | white)
- bin_label: string, local name for the bin (e.g. "Yellow Lid Recycling Bin" for Australia, "Blauer Container" hint for Germany)
- preparation: array of up to 4 strings, steps to prepare item before disposal
- years_to_decompose: string (e.g. "450", "20-30", "Indefinite", "<1")
- co2_emissions_kg: number, estimated CO2 equivalent kg if landfilled
- ocean_risk: one of: None | Low | Medium | High | Very High
- soil_risk: one of: None | Low | Medium | High | Very High
- wildlife_risk: one of: None | Low | Medium | High | Very High
- microplastic_risk: one of: None | Low | Medium | High | Very High
- recycling_benefit: string, one sentence on environmental benefit of recycling this item
- energy_saved_percent: number 0-100, energy saved recycling vs virgin production
- local_tip: string, one specific actionable tip for residents of {location}
"""

    try:
        llama_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": disposal_prompt}],
            max_tokens=900,
        )
    except APIStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Disposal data generation failed: {e.status_code} {e.message}",
        )
    except APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Groq API: {e}")

    impact = extract_json(llama_resp.choices[0].message.content)
    if not impact:
        raise HTTPException(status_code=500, detail="Could not generate disposal data. Please try again.")

    return {**classification, **impact, "location": location}


handler = app
