import pandas as pd
from typing import Dict, List, Any, Optional

WEEKDAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c:str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    has_wide_days = any(c[:3] in WEEKDAYS for c in df.columns)
    if has_wide_days:
        day_cols = [c for c in df.columns if str(c)[:3] in WEEKDAYS]
        key_cols = [c for c in df.columns if c not in day_cols]
        long = df[key_cols + day_cols].melt(id_vars=key_cols, value_vars=day_cols, var_name="Day", value_name="Value")
        long["Day"] = long["Day"].astype(str).str[:3]
        def to_bool(x):
            if pd.isna(x): return False
            s = str(x).strip().lower()
            return s in {"1","y","yes","true","available","avail","ok","✓"} or len(s) > 0 and s not in {"0","n","no","false","","na","nan"}
        long["Available"] = long["Value"].apply(to_bool)
        if "Doctor" not in long.columns:
            for k in ["Provider","Physician","MD","Name"]:
                if k in long.columns:
                    long = long.rename(columns={k:"Doctor"})
                    break
        if "Specialty" not in long.columns:
            long["Specialty"] = "OBGYN"
        return long[["Doctor","Specialty","Day","Available"]].dropna(subset=["Doctor"])
    # long format
    if {"Doctor","Specialty","Day"}.issubset(set(df.columns)):
        long = df.copy()
        long["Day"] = long["Day"].astype(str).str[:3]
        long["Available"] = True
        return long[["Doctor","Specialty","Day","Available"]].dropna(subset=["Doctor"])
    # fallback
    cols = list(df.columns)
    if len(cols) >= 3:
        df2 = df.rename(columns={cols[0]:"Doctor", cols[1]:"Specialty"})
        day_cols = cols[2:]
        long = df2.melt(id_vars=["Doctor","Specialty"], value_vars=day_cols, var_name="Day", value_name="Value")
        long["Day"] = long["Day"].astype(str).str[:3]
        long["Available"] = long["Value"].notna() & (long["Value"].astype(str).str.strip()!="")
        return long[["Doctor","Specialty","Day","Available"]].dropna(subset=["Doctor"])
    raise ValueError("Unrecognized schedule format.")


def get_default_doctors() -> Dict[str, Any]:
    """
    Returns default mock doctor data with 11 OB/GYN specialists.
    This data is used when no Excel file is uploaded.
    """
    return {
        "doctors": [
            {
                "name": "Dr. Alice Smith",
                "subspecialties": ["general_obgyn", "maternal_fetal"],
                "insurances": ["aetna", "uhc", "bcbs"],
                "schedule": {
                    "Mon": ["09:00", "10:00", "14:00", "15:00"],
                    "Tue": ["09:00", "10:00", "11:00"],
                    "Thu": ["09:00", "14:00", "15:00", "16:00"]
                }
            },
            {
                "name": "Dr. Brian Lee",
                "subspecialties": ["general_obgyn", "minimally_invasive"],
                "insurances": ["aetna", "cigna"],
                "schedule": {
                    "Tue": ["10:00", "11:00", "14:00"],
                    "Wed": ["09:00", "10:00", "14:00", "15:00"],
                    "Fri": ["09:00", "10:00", "11:00"]
                }
            },
            {
                "name": "Dr. Carol Chen",
                "subspecialties": ["general_obgyn", "urogynecology"],
                "insurances": ["aetna", "uhc", "medicare"],
                "schedule": {
                    "Mon": ["10:00", "11:00", "15:00"],
                    "Wed": ["09:00", "14:00", "15:00", "16:00"],
                    "Fri": ["09:00", "10:00", "14:00"]
                }
            },
            {
                "name": "Dr. David Patel",
                "subspecialties": ["maternal_fetal"],
                "insurances": ["aetna", "uhc", "bcbs", "cigna"],
                "schedule": {
                    "Mon": ["09:00", "10:00", "11:00", "14:00"],
                    "Tue": ["09:00", "14:00", "15:00"]
                }
            },
            {
                "name": "Dr. Emily Johnson",
                "subspecialties": ["gynecologic_oncology"],
                "insurances": ["aetna", "bcbs", "medicare"],
                "schedule": {
                    "Wed": ["09:00", "10:00", "14:00"],
                    "Thu": ["09:00", "10:00", "11:00", "14:00"]
                }
            },
            {
                "name": "Dr. Frank Garcia",
                "subspecialties": ["reproductive_endo"],
                "insurances": ["uhc", "cigna"],
                "schedule": {
                    "Wed": ["10:00", "11:00", "15:00", "16:00"]
                }
            },
            {
                "name": "Dr. Grace Wong",
                "subspecialties": ["general_obgyn", "minimally_invasive"],
                "insurances": ["aetna", "bcbs"],
                "schedule": {
                    "Thu": ["10:00", "11:00", "14:00", "15:00"]
                }
            },
            {
                "name": "Dr. Hannah Kim",
                "subspecialties": ["general_obgyn", "maternal_fetal"],
                "insurances": ["aetna", "uhc", "bcbs", "medicare"],
                "schedule": {
                    "Sat": ["09:00", "10:00", "11:00"]
                }
            },
            {
                "name": "Dr. Kevin Miller",
                "subspecialties": ["general_obgyn"],
                "insurances": ["uhc", "cigna", "bcbs"],
                "schedule": {
                    "Mon": ["14:00", "15:00", "16:00"],
                    "Fri": ["09:00", "10:00", "14:00"]
                }
            },
            {
                "name": "Dr. Linda Lopez",
                "subspecialties": ["urogynecology"],
                "insurances": ["aetna", "medicare"],
                "schedule": {
                    "Sat": ["09:00", "10:00"]
                }
            },
            {
                "name": "Dr. Michael Zhang",
                "subspecialties": ["general_obgyn", "reproductive_endo"],
                "insurances": ["aetna", "uhc", "cigna"],
                "schedule": {
                    "Tue": ["14:00", "15:00", "16:00"]
                }
            }
        ]
    }


def load_schedule(xlsx_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load doctor schedule and return structured data with subspecialties and schedules.
    
    Args:
        xlsx_path: Optional path to Excel file. If None or file doesn't exist, 
                   returns default mock data with 11 doctors.
    
    Returns:
        Dict with "doctors" list containing name, subspecialties, insurances, and schedule.
        Format: {
            "doctors": [
                {
                    "name": "Dr. Name",
                    "subspecialties": ["specialty1", "specialty2"],
                    "insurances": ["insurance1", "insurance2"],
                    "schedule": {"Mon": ["09:00", "10:00"], ...}
                },
                ...
            ]
        }
    """
    # If no path provided, return default mock data
    if xlsx_path is None:
        print("ℹ️  Using default doctor schedule (11 built-in doctors)")
        return get_default_doctors()
    
    # Try to load from Excel file
    try:
        import os
        if not os.path.exists(xlsx_path):
            print(f"⚠️  File not found: {xlsx_path}")
            print("ℹ️  Falling back to default doctor schedule")
            return get_default_doctors()
        
        # Attempt to read Excel file
        df = pd.read_excel(xlsx_path)
        print(f"✅ Excel file loaded: {xlsx_path}")
        
        # TODO: Implement Excel parsing logic here if needed
        # For now, still return default data even if file exists
        # You can implement custom Excel parsing logic here
        
        print("ℹ️  Using default data (Excel parsing not yet implemented)")
        return get_default_doctors()
        
    except Exception as e:
        print(f"❌ Error loading schedule from {xlsx_path}: {e}")
        print("ℹ️  Falling back to default doctor schedule")
        return get_default_doctors()