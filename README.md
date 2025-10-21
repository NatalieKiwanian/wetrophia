# Right Care, Right Time – AI-Powered Subspecialty Triage for Women's Health


An intelligent clinical triage assistant that automatically converses with patients, collects key medical information, determines urgency, matches the most appropriate OB/GYN subspecialty, and generates appointment summaries **with transparent RAG-based references** from the textbook *Telephone Triage for Obstetrics and Gynecology*.

Try the system at http://rightcareai.com/

## Key Features

- **Multi-turn patient dialogue**
  - Collects patient name, DOB, contact info, menstrual and pregnancy details, symptoms, allergies, and insurance.
- **Intelligent slot-filling**
  - Dynamically decides which question to ask next based on missing information.
- **RAG-powered clinical reasoning**
  - Retrieves relevant pages from *Telephone Triage for Obstetrics and Gynecology* using FAISS + OpenAI Embeddings.
  - Displays referenced page numbers and excerpts alongside the triage report.
- **Automatic urgency & specialty classification**
  - Identifies whether the case is *Emergency / Urgent / Routine* and recommends the appropriate subspecialty.
- **Doctor schedule integration**
  - Reads an uploaded `.xlsx` schedule and assigns the earliest available physician automatically.
- **Dual-mode compatibility**
  - Works with both **Streamlit web chat** and **Twilio phone call** interfaces through a unified agent backend.


## System Architecture

```
Twilio (voice input/output)
Streamlit (text chat UI)
TriageAgent (core logic)
├── Slot extraction & next-question logic
├── RAG retrieval from obgyn_index/
├── LLM summarization & confirmation
└── Reference display (page numbers + snippets)
```


## Project Structure

```
projects/
├── obgyn_index/              # FAISS vector store for textbook embeddings
│   ├── index.faiss
│   └── index.pkl
├── app_chat.py               # Streamlit chat interface
├── main.py                   # Twilio voice interface (FastAPI + WebSocket)
├── schedule_loader.py        # Utility to read doctor schedule Excel
├── triage_agent.py           # Core multi-agent logic (slots + RAG + LLM)
├── .env                      # Contains OPENAI_API_KEY
└── README.md
```


## Quick Start

### Install dependencies
```bash
pip install streamlit openai langchain faiss-cpu python-dotenv twilio
```

### Prepare your `.env`

Create a file named `.env` in the root folder:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### Launch the Streamlit App

```bash
streamlit run app_chat.py
```

Then open the displayed local URL (e.g., `http://localhost:8501`).

### Upload your doctor schedule

Upload an `.xlsx` file in the sidebar before starting the triage.

## RAG Integration

The **RAG pipeline** retrieves the most relevant passages from the OB/GYN telephone triage textbook to support model reasoning.
These references are displayed at the end of the summary.

Example output:

```
Triage Summary
- Patient: Alice Chen
- Complaint: Light spotting at 32 weeks
- Urgency: Urgent
- Specialty: Maternal-Fetal Medicine

References from Handbook:
Page 132: Patients presenting with spotting after 30 weeks should be assessed for placental causes...
Page 135: If bleeding is associated with pain, immediate evaluation is recommended...
```

## Optional: Twilio Voice Interface

To enable phone-based triage:

```bash
python main.py
```

* This spins up a FastAPI WebSocket server compatible with **Twilio Media Streams**.
* Patients can talk to the same AI logic through a phone call.
* The system uses real-time speech detection and synthesis.

## Technologies Used

| Component               | Description                         |
| ----------------------- | ----------------------------------- |
| **OpenAI GPT-4o-mini**  | LLM reasoning & response generation |
| **LangChain + FAISS**   | RAG pipeline for textbook retrieval |
| **Streamlit**           | Web chat frontend                   |
| **Twilio Realtime API** | Phone call streaming interface      |
| **dotenv**              | Secure API key management           |

