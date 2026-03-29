# TrashVision

AI-powered waste identification and localized disposal guidance.

Upload a photo of any waste item, enter your location, and get instant guidance on how to dispose of it correctly, including bin type, preparation steps, environmental impact, and a local tip.

**Live at:** [[[your-url-here](https://trash-vision.vercel.app/)]](https://trash-vision.vercel.app/)

## Features

- Waste material classification from a photo
- Location-specific disposal instructions
- Environmental impact breakdown (CO2, decomposition, ocean/soil/wildlife/microplastic risk)
- Energy savings estimate from recycling
- No account required. Free to use.

## Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **AI:** Groq API — Llama 4 Scout (vision), Llama 3.3 70B (disposal data)
- **Deployment:** Vercel

## Run locally

```bash
git clone https://github.com/your-username/TrashVision
cd TrashVision
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn api.analyze:app --reload
```

## Environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |

