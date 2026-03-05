# Running the Project

## Requirements

- Python **3.11.9**
- Node.js (recommended: **v18+**)
- CUDA compatible GPU (optional but recommended)

### PyTorch Build

- **torch:** `2.5.1+cu121`
- **CUDA:** `12.1`

---

# 1. Clone the Repository

```bash
git clone https://github.com/MauvG/MultiviewMaterialTransfer.git
cd MultiviewMaterialTransfer
```

---

# 2. Create a Python Virtual Environment

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.\.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

# 3. Install Dependencies

- Install torch first (important)

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
```

```bash
pip install -r requirements.txt
```

---

# 4. Install Frontend Dependencies

Navigate to the frontend folder and install Node packages:

```bash
cd frontend
npm install
cd ..
```

---

# 5. Download the Required Model

The **SEVA model weights** are not included in the repository.

1. Download **`model.safetensors`** from:
   https://huggingface.co/stabilityai/stable-virtual-camera/tree/main

2. Create a folder named `models` in the project root:

```bash
mkdir models
```

3. Move the downloaded file into this folder and rename it:

```
models/seva_model.safetensors
```

The final structure should look like:

```
MultiviewMaterialTransfer/
│
├── models/
│   └── seva_model.safetensors
├── backend/
├── frontend/
├── MultiviewMaterialTransfer/
└── ZImageTurbo/
```

---

# 6. Run the Application

Open **two terminals**.

### Terminal 1 — Backend (FastAPI)

Activate the virtual environment if not already active:

```bash
.\.venv\Scripts\activate
```

Run the backend server:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will start at:

```
http://localhost:8000
```

---

### Terminal 2 — Frontend (Vite)

Navigate to the frontend folder:

```bash
cd frontend
```

Start the development server:

```bash
npm run dev
```

Frontend will start at:

```
http://localhost:5173
```

---

# Notes

- The backend requires the correct **PyTorch + CUDA build** to run efficiently on GPU.
- If CUDA is unavailable, the project will fall back to CPU (slower).
- Make sure the virtual environment is activated before running backend commands.
- The SEVA model file must exist at `models/seva_model.safetensors` before starting the backend.
