# -*- coding: utf-8 -*-
"""
triage_agent.py  —  Unified agent with:
- Robust slot extraction (OpenAI tool/function-calling)
- Full question set (insurance, menstrual_cycle, last_period, pregnancy_week, allergies, contact)
- English name parsing (prefix or bare name)
- Enhanced triage (subspecialty + confidence + red flags)
- Capacity-aware doctor selection (specialty + insurance + schedule + patient availability)
- Optional RAG citation (FAISS vectorstore auto-fallback)
- Emergency detection with immediate ER referral
"""

import os, re, json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI

# ---------- Env ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Optional RAG (safe fallback) ----------
USE_RAG = True
qa_chain = None
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain

    def init_rag():
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            db = FAISS.load_local("obgyn_index", embeddings, allow_dangerous_deserialization=True)
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )
            print("✅ RAG index loaded (obgyn_index)")
            return chain
        except Exception as e:
            print("⚠️ RAG disabled:", e)
            return None

    qa_chain = init_rag()
except Exception as e:
    USE_RAG = False
    print("⚠️ RAG modules unavailable:", e)

# ===========================
# Slot Schema
# ===========================
SLOTS: Dict[str, Any] = {
    "emergency_check": None,
    "name": None,
    "symptom": None,
    "dob": None,
    "age": None,
    "insurance": None,
    "menstrual_cycle": None,  # days
    "last_period": None,      # YYYY-MM-DD
    "pregnancy_week": None,   # number or "NA"
    "allergies": None,
    "contact": None,
}

# ===========================
# Subspecialties
# ===========================
SUBSPECIALTIES = {
    "maternal_fetal": "Maternal-Fetal Medicine (High-Risk Pregnancy)",
    "urogynecology": "Urogynecology & Pelvic Reconstructive Medicine",
    "gynecologic_oncology": "Gynecologic Oncology",
    "reproductive_endo": "Reproductive Endocrinology & Infertility",
    "minimally_invasive": "Complex/Minimally Invasive Gynecologic Surgery",
    "general_obgyn": "General OB/GYN",
    "emergency": "Emergency OB/GYN",
}

# ===========================
# Utilities
# ===========================
def _cleanse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        return json.loads(m.group(0)) if m else {}

def _is_skip_token(s: str) -> bool:
    return s.strip().lower() in {"none", "na", "n/a", "skip", "no", "not applicable"}

def _validate_dob(dob: str) -> bool:
    try:
        datetime.strptime(dob, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def calc_age(dob: str) -> Optional[int]:
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d").date()
        today = datetime.today().date()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return None

def _parse_pregnancy_from_text(text: str) -> Optional[str]:
    t = text.lower().strip()
    if re.search(r"\bnot\s+pregnant\b", t):
        return "NA"
    m = re.search(r"(\d{1,2})\s*(weeks?|w)\b", t)
    if m:
        return str(int(m.group(1)))
    return None

# -------- English name parsing (prefix + bare) --------
NAME_PREFIX_RE = re.compile(
    r"(?:my name is|i am|i'm|this is|it'?s)\s+"
    r"([A-Za-z][A-Za-z\-']+(?:\s+(?:[A-Za-z]\.|[A-Za-z][A-Za-z\-']+)){1,3})\b",
    re.I,
)
NAME_BARE_RE = re.compile(
    r"[A-Za-z][A-Za-z\-']+(?:\s+(?:[A-Za-z]\.|[A-Za-z][A-Za-z\-']+)){1,3}\b"
)

def parse_full_name_en(text: str) -> Optional[str]:
    t = text.strip()
    if any(x in t for x in ["@", "http://", "https://"]) or re.search(r"\d", t) or len(t) > 60:
        return None
    m = NAME_PREFIX_RE.search(t)
    if m:
        candidate = m.group(1)
        parts = re.split(r"\s+", candidate)
        norm = [(p.upper() if re.fullmatch(r"[A-Za-z]\.", p) else p[:1].upper() + p[1:].lower()) for p in parts]
        return " ".join(norm)
    if NAME_BARE_RE.fullmatch(t):
        parts = re.split(r"\s+", t)
        norm = [(p.upper() if re.fullmatch(r"[A-Za-z]\.", p) else p[:1].upper() + p[1:].lower()) for p in parts]
        return " ".join(norm)
    return None

# ===========================
# Extraction (Function-Calling) - FIXED
# ===========================
def extract_slots(user_text: str, current: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use OpenAI function-calling to extract *missing* fields.
    Still keeps quick rules for emergency/DOB/name/contact/pregnancy phrases.
    """
    raw = user_text.strip()
    lower = raw.lower()

    # Lightweight noise filter
    if lower in {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "sure"}:
        return current

    if current.get("emergency_check") is None:
        if any(x in lower for x in ["yes", "y", "urgent", "definitely", "absolutely", "emergency", "critical", "immediate"]):
            current["emergency_check"] = "yes"
        elif any(x in lower for x in ["no", "n", "not", "nope", "fine", "okay", "non-urgent"]):
            current["emergency_check"] = "no"
        return current
    # # Emergency check
    # if current.get("emergency_check") is None:
    #     if lower in {"yes", "y", "urgent"} or "emergency" in lower:
    #         current["emergency_check"] = "yes"
    #     elif lower in {"no", "n", "non-urgent", "not emergency", "nope"}:
    #         current["emergency_check"] = "no"
    #     return current

    # DOB detection
    if not current.get("dob") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", lower):
        if _validate_dob(lower):
            current["dob"] = lower
            current["age"] = calc_age(lower)
        return current

    # Contact (phone/email)
    if not current.get("contact") and (re.search(r"\d{7,}", raw) or "@" in raw):
        current["contact"] = raw
        return current

    # English full name
    if not current.get("name"):
        name = parse_full_name_en(raw)
        if name:
            current["name"] = name
            return current

    # Quick pregnancy text
    if not current.get("pregnancy_week"):
        pg = _parse_pregnancy_from_text(raw)
        if pg:
            current["pregnancy_week"] = pg

    # FIX: Handle skip tokens for CURRENT question only
    if _is_skip_token(raw):
        # Find the first missing field and set it to NA/None
        missing_fields = [k for k, v in current.items() if not v and k != "emergency_check"]
        if missing_fields:
            first_missing = missing_fields[0]
            if first_missing in {"insurance", "menstrual_cycle", "last_period", "pregnancy_week"}:
                current[first_missing] = "NA"
            elif first_missing == "allergies":
                current[first_missing] = "None"
            return current

    # Function calling for the remaining missing keys
    missing = [k for k, v in current.items() if not v and k != "emergency_check"]
    if not missing:
        return current

    tools = [{
        "type": "function",
        "function": {
            "name": "extract_patient_info",
            "description": "Extract patient info from the message; only fill keys that are present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient full name (English)"},
                    "symptom": {"type": "string", "description": "Main complaint / symptom"},
                    "dob": {"type": "string", "pattern": "\\d{4}-\\d{2}-\\d{2}", "description": "DOB YYYY-MM-DD"},
                    "contact": {"type": "string", "description": "Phone or email"},
                    "insurance": {"type": "string", "description": "Insurance provider name"},
                    "menstrual_cycle": {"type": "string", "description": "Menstrual cycle length in days"},
                    "last_period": {"type": "string", "description": "Last menstrual period date YYYY-MM-DD"},
                    "pregnancy_week": {"type": "string", "description": "Pregnancy week number or 'NA'"},
                    "allergies": {"type": "string", "description": "Medication or food allergies"},
                },
                "additionalProperties": False
            }
        }
    }]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Only extract these missing fields: {', '.join(missing)}"},
                {"role": "user", "content": raw},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        tcalls = resp.choices[0].message.tool_calls or []
        if tcalls:
            args = _cleanse_json(tcalls[0].function.arguments)
            for k, v in args.items():
                if k in current and v and not current.get(k):
                    if k == "dob" and not _validate_dob(v):
                        continue
                    if k == "dob":
                        current["age"] = calc_age(v)
                    current[k] = str(v).strip()

    except Exception as e:
        print("Extraction error:", e)

    # If name已填而symptom未填，默认把本句当作症状，避免循环
    if current.get("name") and not current.get("symptom"):
        current["symptom"] = raw

    return current

# ===========================
# Question Flow
# ===========================
def next_question(slots: Dict[str, Any]) -> str:
    if not slots.get("emergency_check"):
        return "🚨 Is this an emergency requiring immediate care? (Yes/No)"
    if not slots.get("name"):
        return "👤 Please tell me your **full name**."
    if not slots.get("symptom"):
        return "💬 Please describe your **main symptom** or reason for the visit."
    if not slots.get("dob"):
        return "📅 What is your **date of birth**? (YYYY-MM-DD)"
    if not slots.get("insurance"):
        return "🏥 What is your **insurance provider**? (e.g., UnitedHealthcare/Aetna/Blue Cross, or type 'skip')"
    if not slots.get("menstrual_cycle"):
        return "📊 What is your usual **menstrual cycle length** in days? (e.g., 28; type 'NA' if not applicable)"
    if not slots.get("last_period"):
        return "📆 When was your **last menstrual period**? (YYYY-MM-DD, or type 'NA')"
    if not slots.get("pregnancy_week"):
        return "🤰 If applicable, how many **weeks pregnant** are you? (e.g., '12 weeks' or type 'NA')"
    if not slots.get("allergies"):
        return "⚠️ Do you have any **medication or food allergies**? (If none, type 'None')"
    if not slots.get("contact") or slots.get("contact", "").lower() in {"none", "na", "n/a"}:
        return "📱 Please provide your **contact information** (phone or email)."
    return ""

# ===========================
# Triage (enhanced)
# ===========================
def _detect_red_flags(symptom: str, pregnancy_week: str) -> List[str]:
    s = symptom.lower()
    flags = []
    EMERGENCY = {
        "heavy bleeding": "Severe hemorrhage",
        "hemorrhage": "Severe hemorrhage",
        "severe pain": "Severe abdominal pain",
        "chest pain": "Chest pain (possible PE)",
        "shortness of breath": "Respiratory distress",
        "can't breathe": "Respiratory distress",
        "difficulty breathing": "Respiratory distress",
        "fainting": "Syncope/loss of consciousness",
        "seizure": "Seizure activity",
        "severe headache": "Severe headache (preeclampsia)",
        "vision changes": "Visual disturbances (preeclampsia)",
        "blurred vision": "Visual disturbances (preeclampsia)",
    }
    for k, v in EMERGENCY.items():
        if k in s:
            flags.append(v)
    if pregnancy_week and pregnancy_week != "NA":
        try:
            week = int(pregnancy_week)
            if week > 20 and any(k in s for k in ["bleeding", "fluid", "contractions", "pain"]):
                flags.append("Possible preterm labor/complications")
        except Exception:
            pass
    return flags

def _fallback_triage(symptom: str, pregnancy_week: Optional[str], age: Optional[int]) -> Dict[str, Any]:
    s = (symptom or "").lower()
    if pregnancy_week and pregnancy_week != "NA":
        return {"urgency": "urgent", "subspecialty_code": "maternal_fetal",
                "subspecialty": SUBSPECIALTIES["maternal_fetal"],
                "confidence": 0.8, "reasoning": "Pregnancy-related complaint", "red_flags": []}
    if any(k in s for k in ["mass", "lump", "abnormal pap", "bleeding after menopause", "pelvic mass"]):
        return {"urgency": "urgent", "subspecialty_code": "gynecologic_oncology",
                "subspecialty": SUBSPECIALTIES["gynecologic_oncology"],
                "confidence": 0.75, "reasoning": "Suspicious findings", "red_flags": []}
    if any(k in s for k in ["incontinence", "prolapse", "leaking urine", "bladder"]):
        return {"urgency": "routine", "subspecialty_code": "urogynecology",
                "subspecialty": SUBSPECIALTIES["urogynecology"],
                "confidence": 0.85, "reasoning": "Pelvic floor disorder", "red_flags": []}
    if any(k in s for k in ["infertility", "can't get pregnant", "trying to conceive", "pcos"]):
        return {"urgency": "routine", "subspecialty_code": "reproductive_endo",
                "subspecialty": SUBSPECIALTIES["reproductive_endo"],
                "confidence": 0.8, "reasoning": "Reproductive endocrine issue", "red_flags": []}
    if any(k in s for k in ["fibroid", "endometriosis", "ovarian cyst", "heavy periods"]):
        return {"urgency": "routine", "subspecialty_code": "minimally_invasive",
                "subspecialty": SUBSPECIALTIES["minimally_invasive"],
                "confidence": 0.7, "reasoning": "Likely surgical condition", "red_flags": []}
    return {"urgency": "routine", "subspecialty_code": "general_obgyn",
            "subspecialty": SUBSPECIALTIES["general_obgyn"],
            "confidence": 0.6, "reasoning": "Routine care", "red_flags": []}

def enhanced_triage(slots: Dict[str, Any]) -> Dict[str, Any]:
    symptom = slots.get("symptom", "") or ""
    pregnancy_week = slots.get("pregnancy_week", "NA")
    age = slots.get("age")
    last_period = slots.get("last_period", "NA")

    red_flags = _detect_red_flags(symptom, pregnancy_week)
    if red_flags:
        return {"urgency": "emergency", "subspecialty_code": "emergency",
                "subspecialty": SUBSPECIALTIES["emergency"],
                "confidence": 1.0, "reasoning": "IMMEDIATE MEDICAL ATTENTION REQUIRED",
                "red_flags": red_flags}

    # Ask LLM to classify; fall back on rules
    prompt = f"""
Return ONLY JSON with keys: subspecialty_code, urgency ("routine"|"urgent"), confidence (0-1), reasoning.
Patient: age={age}, symptom={symptom}, pregnancy_week={pregnancy_week}, last_period={last_period}.
Subspecialties: {', '.join(SUBSPECIALTIES.keys())}.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert OB/GYN triage specialist. Return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content)
        code = data.get("subspecialty_code", "general_obgyn")
        return {
            "urgency": data.get("urgency", "routine"),
            "subspecialty_code": code,
            "subspecialty": SUBSPECIALTIES.get(code, SUBSPECIALTIES["general_obgyn"]),
            "confidence": float(data.get("confidence", 0.7)),
            "reasoning": data.get("reasoning", "Standard triage protocol"),
            "red_flags": [],
        }
    except Exception as e:
        print("Triage error:", e)
        return _fallback_triage(symptom, pregnancy_week, age)

# Back-compat legacy shims
def triage(symptom: str) -> str:
    return enhanced_triage({"symptom": symptom})["urgency"]

def pick_specialty(symptom: str) -> str:
    return enhanced_triage({"symptom": symptom})["subspecialty"]

# ===========================
# Capacity-Aware Doctor Selection
# ===========================
def _iter_schedule_slots(
    schedule: Dict[str, List[str]], start_date: datetime, days: int
) -> List[Tuple[str, str, datetime]]:
    """
    Expand a weekly schedule like {"Mon":["09:00","10:00"]} into concrete datetimes
    for the next N days starting from start_date.
    Returns list of (weekday, "HH:MM", dt)
    """
    out: List[Tuple[str, str, datetime]] = []
    day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    for i in range(days):
        d = start_date + timedelta(days=i)
        # Normalize to "Mon".."Sun"
        wd = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d.weekday()]
        for day, times in schedule.items():
            if day == wd:
                for t in times:
                    hh, mm = map(int, t.split(":"))
                    slot_dt = d.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    out.append((day, t, slot_dt))
    return sorted(out, key=lambda x: x[2])

def pick_doctor_advanced(
    availability: Dict[str, Any],
    subspecialty_code: str,
    urgency: str,
    insurance: Optional[str] = None,
    patient_windows: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    availability schema (example):
    {
      "doctors": [
        {
          "name": "Dr. Hannah Kim",
          "subspecialties": ["general_obgyn","maternal_fetal"],
          "insurances": ["aetna","uhc","bcbs"],
          "schedule": { "Mon": ["09:00","10:00"], "Wed": ["14:00"] }
        },
        ...
      ]
    }
    patient_windows: list of ISO strings the patient is available, e.g. ["2025-10-12T09:00","2025-10-13T14:00"]
    """

    # Search horizon depends on urgency
    horizon = 1 if urgency == "emergency" else (7 if urgency == "urgent" else 14)
    start = datetime.today()

    # Normalize insurance text
    ins_norm = (insurance or "").strip().lower()

    # Pre-compute patient preferred windows (as days)
    pref_days = set()
    if patient_windows:
        try:
            for iso in patient_windows:
                d = datetime.fromisoformat(iso).date()
                pref_days.add(d)
        except Exception:
            pass

    best: Optional[Dict[str, Any]] = None

    for doc in availability.get("doctors", []):
        # Specialty filter
        if subspecialty_code not in set(doc.get("subspecialties", [])):
            continue
        # Insurance filter (if provided)
        if ins_norm and ins_norm not in [x.lower() for x in doc.get("insurances", [])]:
            continue

        slots = _iter_schedule_slots(doc.get("schedule", {}), start, horizon)
        for day, time_str, slot_dt in slots:
            # If patient has preferred days, prioritize matching days
            prefer = (slot_dt.date() in pref_days) if pref_days else True
            cand = {
                "doctor_name": doc["name"],
                "subspecialty_code": subspecialty_code,
                "available_date": slot_dt.strftime("%Y-%m-%d"),
                "available_time": slot_dt.strftime("%H:%M"),
                "available_slots": [time_str],
                "wait_days": (slot_dt.date() - start.date()).days,
                "preferred_match": prefer,
            }
            # Choose earliest; break ties by preferred day
            if best is None:
                best = cand
            else:
                if cand["preferred_match"] and not best["preferred_match"]:
                    best = cand
                elif cand["preferred_match"] == best["preferred_match"]:
                    if slot_dt < datetime.strptime(
                        best["available_date"] + " " + best["available_time"], "%Y-%m-%d %H:%M"
                    ):
                        best = cand
            # For emergency, first qualifying slot is enough
            if urgency == "emergency":
                return best

    return best or {
        "doctor_name": "No Doctor Available",
        "available_date": "TBD",
        "available_time": "",
        "available_slots": [],
        "wait_days": -1,
        "preferred_match": False,
        "subspecialty_code": subspecialty_code,
    }

# Legacy wrapper for backward compatibility
def pick_doctor(availability: Dict[str, list]) -> str:
    res = pick_doctor_advanced({"doctors": [{"name": d, "subspecialties": ["general_obgyn"], "insurances": [], "schedule": {day:[ "09:00" ]}} for day, docs in availability.items() for d in docs]}, "general_obgyn", "routine")
    return res["doctor_name"]


# ===========================
# Get Available Doctors List (NEW)
# ===========================
def get_available_doctors_list(
    availability: Dict[str, Any],
    subspecialty_code: str,
    urgency: str,
    insurance: Optional[str] = None,
    days_ahead: int = 14
) -> List[Dict[str, Any]]:
    """
    Get list of all available doctors with their earliest appointment slots.
    Returns list of doctors sorted by earliest availability.
    
    Args:
        availability: Doctor availability data
        subspecialty_code: Required subspecialty code
        urgency: Urgency level ("routine", "urgent", "emergency")
        insurance: Patient's insurance provider
        days_ahead: Number of days to search ahead
        
    Returns:
        List of dicts containing doctor info and available slots
    """
    start = datetime.today()
    ins_norm = (insurance or "").strip().lower()
    
    available_doctors = []
    
    for doc in availability.get("doctors", []):
        # Specialty filter
        if subspecialty_code not in set(doc.get("subspecialties", [])):
            continue
        # Insurance filter (if provided and not NA)
        if ins_norm and ins_norm != "na" and ins_norm not in [x.lower() for x in doc.get("insurances", [])]:
            continue
        
        slots = _iter_schedule_slots(doc.get("schedule", {}), start, days_ahead)
        if slots:
            # Get first 5 available slots
            earliest_slots = []
            for day, time_str, slot_dt in slots[:5]:  # Show up to 5 slots per doctor
                earliest_slots.append({
                    "date": slot_dt.strftime("%Y-%m-%d"),
                    "day": day,
                    "time": slot_dt.strftime("%H:%M"),
                    "datetime": slot_dt
                })
            
            if earliest_slots:
                available_doctors.append({
                    "name": doc["name"],
                    "subspecialty": SUBSPECIALTIES.get(subspecialty_code, "General OB/GYN"),
                    "subspecialty_code": subspecialty_code,
                    "insurance_accepted": ins_norm in [x.lower() for x in doc.get("insurances", [])] if ins_norm and ins_norm != "na" else True,
                    "earliest_slot": earliest_slots[0],
                    "available_slots": earliest_slots,
                    "wait_days": (earliest_slots[0]["datetime"].date() - start.date()).days
                })
    
    # Sort by earliest availability (wait_days)
    available_doctors.sort(key=lambda x: x["wait_days"])
    
    return available_doctors

# ===========================
# Confirmation with optional RAG refs - UPDATED FOR EMERGENCY HANDLING
# ===========================
def confirmation(slots: Dict[str, Any], triage_result: Dict[str, Any], doctor_info: Dict[str, Any], rag_summary: str = "") -> str:
    """
    Generate confirmation message based on triage urgency.
    For emergency cases, directs to ER instead of scheduling appointment.
    """
    
    # EMERGENCY CASE: Direct to ER immediately
    if triage_result.get('urgency') == 'emergency':
        red_flags = triage_result.get('red_flags', [])
        red_flags_text = "\n".join([f"⚠️ {flag}" for flag in red_flags]) if red_flags else ""
        
        prompt = f"""
Generate an URGENT emergency notification for a patient who needs immediate medical attention.

Patient: {slots.get('name')}
Symptoms: {slots.get('symptom')}

RED FLAGS DETECTED:
{red_flags_text}

Clinical Assessment: {triage_result.get('reasoning', 'Emergency situation identified')}

Requirements:
1. Start with a clear, urgent tone
2. Explicitly tell them to go to the nearest ER or call 911 IMMEDIATELY
3. List the specific red flags/warning signs detected
4. Emphasize this is NOT something that can wait for an appointment
5. Be compassionate but firm about the urgency
6. Provide clear next steps (call 911 or go to ER)
7. Close professionally

Do NOT mention any doctor appointments or scheduling.
"""
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an urgent care triage coordinator. Your priority is patient safety."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print("Confirmation error:", e)
            # Fallback emergency message
            return f"""
🚨 **URGENT MEDICAL ATTENTION REQUIRED** 🚨

Dear {slots.get('name', 'Patient')},

Based on your symptoms ({slots.get('symptom', 'your reported condition')}), you require **IMMEDIATE EMERGENCY CARE**.

**⚠️ WARNING SIGNS DETECTED:**
{red_flags_text if red_flags else '⚠️ Symptoms require immediate evaluation'}

**PLEASE TAKE IMMEDIATE ACTION:**
1. 🚑 **Call 911** if symptoms are severe or worsening
2. 🏥 **Go to your nearest Emergency Room** immediately
3. 📞 Do NOT wait for a regular appointment

**This is NOT a condition that can wait.**

Your health and safety are paramount. Emergency departments are equipped to handle your situation right now.

If you have any questions while en route, you can contact our emergency line.

Stay safe,
Emergency Triage Coordinator
OB/GYN Clinic
"""
    
    # NON-EMERGENCY CASE: Regular appointment confirmation
    prompt = f"""
Generate a warm, professional appointment confirmation message.

Patient: {slots.get('name')}
Chief Complaint: {slots.get('symptom')}
Triage Assessment:
- Urgency: {triage_result['urgency']}
- Subspecialty: {triage_result['subspecialty']}
- Confidence: {triage_result['confidence']:.0%}
- Clinical note: {triage_result['reasoning']}

Appointment Details:
- Doctor: {doctor_info.get('doctor_name', 'TBD')}
- Date: {doctor_info.get('available_date', 'TBD')}
- Time: {doctor_info.get('available_time', '')}
- Wait time: {doctor_info.get('wait_days', 0)} day(s)

Requirements:
1. Thank them for providing information
2. Confirm the appointment details clearly
3. Mention the matched subspecialty and why it's appropriate
4. Include any preparation instructions if needed
5. Remind them to bring insurance card and ID
6. Provide cancellation/rescheduling information
7. Close warmly and professionally

{f"Note: Clinical guidance consulted from our medical handbook." if rag_summary else ""}

RAG context (for your reference only, don't quote directly): {rag_summary[:300]}
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a compassionate, professional OB/GYN clinic coordinator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print("Confirmation error:", e)
        # Fallback regular message
        return f"""
Dear {slots.get('name', 'Patient')},

Thank you for providing your information. Based on your symptoms, we have scheduled an appointment for you:

**Appointment Details:**
- 👨‍⚕️ Doctor: {doctor_info.get('doctor_name', 'To be assigned')}
- 📅 Date: {doctor_info.get('available_date', 'TBD')}
- 🕐 Time: {doctor_info.get('available_time', '')}
- 🏥 Specialty: {triage_result['subspecialty']}
- ⚡ Priority: {triage_result['urgency'].capitalize()}

Please bring your insurance card and a valid ID to your appointment.

If you need to reschedule or have any questions, please contact our office.

Best regards,
OB/GYN Clinic Coordinator
"""

# ===========================
# Agent class (state + RAG glue) - FIXED
# ===========================
class TriageAgent:
    def __init__(self):
        self.slots = dict(SLOTS)
        self.chat_history: List[Tuple[str, str]] = []

    def reset(self):
        self.slots = dict(SLOTS)
        self.chat_history = []

    def update(self, user_text: str) -> Dict[str, Any]:
        self.slots = extract_slots(user_text, self.slots)
        return self.slots

    def next_question(self) -> str:
        return next_question(self.slots)

    # FIX: Add the missing triage method
    def triage(self) -> Dict[str, Any]:
        """Perform enhanced triage on current slots"""
        return enhanced_triage(self.slots)

    def triage_and_confirm(self, availability: Dict[str, Any], patient_windows: Optional[List[str]] = None) -> Dict[str, Any]:
        # Triage
        t = enhanced_triage(self.slots)

        # Doctor selection (skip for emergency cases)
        if t.get('urgency') == 'emergency':
            doctor = {
                "doctor_name": "Emergency Department",
                "available_date": "IMMEDIATE",
                "available_time": "NOW",
                "available_slots": [],
                "wait_days": 0,
                "preferred_match": False,
                "subspecialty_code": "emergency",
            }
        else:
            doctor = pick_doctor_advanced(
                availability=availability,
                subspecialty_code=t["subspecialty_code"],
                urgency=t["urgency"],
                insurance=(self.slots.get("insurance") or ""),
                patient_windows=patient_windows,
            )

        # Optional RAG consult (skip for emergency cases)
        rag_answer, rag_refs = "", []
        if qa_chain and t.get('urgency') != 'emergency':
            q = self.slots.get("symptom", "")
            try:
                res = qa_chain({"question": q, "chat_history": self.chat_history})
                self.chat_history.append((q, res["answer"]))
                rag_answer = res["answer"]
                for src in res["source_documents"]:
                    page = src.metadata.get("page", "N/A")
                    snippet = src.page_content[:300].replace("\n", " ")
                    rag_refs.append((page, snippet))
            except Exception as e:
                print("RAG query error:", e)

        # Compose summary
        summary = confirmation(self.slots, t, doctor, rag_summary=rag_answer)

        return {
            "summary": summary,
            "triage": t,
            "doctor": doctor,
            "references": rag_refs,
        }
            