# How to run

- install requirements.txt
- ensure python version: Python 3.11.9
- cuda + torch build: 2.5.1+cu121 12.1

- Open two terminals
- Activate venv: ` .\.venv\Scripts\activate`
- Run backend: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
- Run frontend: `npm run dev`
