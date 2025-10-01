import re
import numpy as np
from numpy import nan 
from typing import Optional, Dict, Any
import pandas as pd

def isntFloat(var):
    try:
        float(var)
        return False
    except ValueError:
        return True

def clean_coadsorbent(coad_str):
    print(coad_str)
    if isntFloat(coad_str):
        ismM = bool(re.search(r"(mM|mmol\/dm3)", coad_str))
        isuM = bool(re.search(r"(uM|μM)", coad_str))
        isEquiv = bool(re.search(r"\sequiv\s", coad_str))
        isSaturated = bool(re.search(r"saturated", coad_str))

        element_lst = coad_str.split(' ')

        if isSaturated:
            co_conc = "saturated"
        else:
            raw_concentration = re.findall(r'\b[\d]*[.][\d]+|\b[\d]+' ,''.join(re.findall(r'(?<!\d|\.)\d+(?:\.\d+)?\s*?(?:mM|mmol\/dm3|uM|μM|M|equiv)(?!\w)', coad_str)))
            print('raw value', raw_concentration)
            if len(raw_concentration) == 0:
                co_conc = np.nan
            else:
                if ismM or isEquiv:
                    co_conc = float(raw_concentration[0])
                elif isuM:
                    print('reached here')
                    co_conc = float(raw_concentration[0])/1000
                else:
                    co_conc = float(raw_concentration[0])*1000


        if bool(re.search(r'CDCA', coad_str)):
            co_type = "CDCA"
        elif bool(re.search(r'HC-A1', coad_str)):
            co_type = "HC-A1"
        
        elif bool(re.search(r'DCA', coad_str)):
            co_type = "DCA"
        else:
            if element_lst[-1] == 'acid':
                co_type = element_lst[-2] + ' ' + element_lst[-1]
            else:
                co_type = element_lst[-1]
        
        if co_type == 'saturated' or co_type == 'film' or co_type == 'Saturated':
            co_type = np.nan
        print(ismM, isuM, isEquiv, co_type, co_conc)
    else:
        co_type = np.nan
        co_conc = np.nan   
    return co_type, co_conc

def concentration_zero(co_adsorb_type):
    if co_adsorb_type == np.nan:
        return 0 

def standardize_saturation(concentration):
    if (concentration == 'saturated') or (concentration >=1000):
        sat_con = 1000
    else:
        sat_con = concentration
    return sat_con

def clean_electrolyte(i):
    isIodide = bool(re.search(r"iodide|triiodide|DMPImI|DMPII|I2|LiI|OPV-AN-I|DMII|Solaronix|Iodolyte|HI-30|Iodine|iodine|I-|AN-50", str(i)))
    isSpiro = bool(re.search(r"Spiro-OMeTAD|MeOTAD|spiro|OmeTAD", str(i)))
    isCobalt = bool(re.search(r"cobalt|Co", str(i)))
    isBromide = bool(re.search(r"bromide|Br", str(i)))
    isDyesol = bool(re.search(r"Dyesol|dyesol|EL-HSE", str(i)))
    isDHS = bool(re.search(r"DHS-Z23", str(i)))
    isDMPIC = bool(re.search(r"DMPIC|DMPIDC", str(i)))
    isSolid = bool(re.search(r"solid", str(i)))
    isIsopropanol = bool(re.search(r"0.005 M isopropanol solution of H2PtCl6·6H2O", str(i)))
    isCu = bool(re.search(r"Cu(I)|CuI|CuII", str(i)))
    isGuanthio = bool(re.search(r"EMISCN|GuanThio, NMB", str(i)))
    isThiocyanate = bool(re.search(r"PyC6|Py2C6", str(i)))
    if isIodide:
        typ = 'Iodide_Triiodide'
    elif isSpiro:
        typ = 'Spiro-OMeTAD'
    elif isCobalt:
        typ = 'Co(II)_Co(III)'
    elif isBromide:
        typ = 'Bromide_Tribromide'
    elif isDyesol:
        typ = 'Dyesol Mixes'
    elif isDHS:
        typ = 'DHS-Z23, Heptachroma'
    elif isDMPIC:
        typ = 'DMPIC_DMPIDC'
    elif isSolid:
        typ = 'solid'
    elif isIsopropanol:
        typ = 'H2PtCl6·6H2O'
    elif isCu:
        typ = 'Cu(I)_Cu(II)'
    elif isGuanthio:
        typ = 'EMISCN, K(SeCN)3, GuanThio, NMB'
    elif isThiocyanate:
        typ = 'SCN-'
    else:
        typ = np.nan
    return typ

def _to_um(val_str: str, unit: str = "um") -> Optional[float]:
    """Convert number + unit to micrometers (µm)."""
    try:
        v = float(val_str)
    except Exception:
        return None
    u = unit.lower().replace(" ", "")
    if u == "mm" or u.startswith("millimeter"):
        return v * 1000.0
    # treat um/u m/uM/µm as micrometers
    return v

# ---------- compiled regexes for SEMICONDUCTOR ----------
UNIT_GROUP = r'(?:u\s*m|u?m|µm|mm)'

# strip nm tokens/ranges (keep rest)
RE_STRIP_NM = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*nm\b'
    r'(?:\s*(?:particles?|particle\s*size))?',
    re.IGNORECASE
)

# material type (first token)
RE_TYPE = re.compile(r'^\s*(?P<type>[A-Za-z][A-Za-z0-9_+\-]*)', re.IGNORECASE)

# --- edge case A: explicit pair with words
RE_PAIR_WITH_WORDS_FILM_FIRST = re.compile(
    rf'\(\s*'
    rf'(?P<film>\d+(?:\.\d+)?)\s*(?P<film_unit>{UNIT_GROUP})\s*transp\w*[^()+]*\+\s*'
    rf'(?P<scat>\d+(?:\.\d+)?)\s*(?P<scat_unit>{UNIT_GROUP})\s*scatt\w*(?:\s*layer)?'
    rf'\s*\)',
    re.IGNORECASE
)

RE_PAIR_WITH_WORDS_SCAT_FIRST = re.compile(
    rf'\(\s*'
    rf'(?P<scat>\d+(?:\.\d+)?)\s*(?P<scat_unit>{UNIT_GROUP})\s*scatt\w*(?:\s*layer)?[^()+]*\+\s*'
    rf'(?P<film>\d+(?:\.\d+)?)\s*(?P<film_unit>{UNIT_GROUP})\s*transp\w*'
    rf'\s*\)',
    re.IGNORECASE
)

# --- edge case B: shared-unit pair like "(12 + 4 um thick)" ---
RE_PAIR_SHARED_UNIT = re.compile(
    rf'\(\s*(?P<film>\d+(?:\.\d+)?)\s*\+\s*(?P<scat>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\b[^)]*\)',
    re.IGNORECASE
)

# Pair with single unit after second number (requires '+' or 'and'):
RE_PAIR_BOTH_IN_PARENS = re.compile(
    rf'\(\s*(?P<film>\d+(?:\.\d+)?)\s*(?:{UNIT_GROUP})?\s*(?:\+|and)\s*'
    rf'(?P<scat>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\s*\)',
    re.IGNORECASE
)

# Film range like "(16-18 um)" or "(16–18 um)"
RE_FILM_RANGE = re.compile(
    rf'\(\s*(?P<val1>\d+(?:\.\d+)?)\s*[-–]\s*(?P<val2>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\s*\)',
    re.IGNORECASE
)

# Scattering thickness (order matters: try TEXT_FIRST, then NUM_FIRST)
RE_SCAT_TEXT_FIRST = re.compile(
    rf'\bscatt\w*(?:\s*layer)?(?:(?!\d).){{0,40}}(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})',
    re.IGNORECASE
)
RE_SCAT_NUM_FIRST = re.compile(
    rf'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})(?:(?!\d).){{0,20}}\bscatt\w*(?:\s*layer)?',
    re.IGNORECASE
)

# Explicit film phrases
RE_FILM_IN_THICKNESS = re.compile(
    rf'\bfilm[s]?\b(?:(?!\d).){{0,80}}(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\s+in\s+thickness',
    re.IGNORECASE
)
RE_FILM_THICK = re.compile(
    rf'\bfilm[s]?\b(?:(?!\d).){{0,80}}(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\s*thick',
    re.IGNORECASE
)

# Fallbacks
RE_NUM_UNIT = re.compile(rf'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})', re.IGNORECASE)
RE_FILM_WORD = re.compile(r'\b(?:thin\s+film[s]?|film[s]?)\b', re.IGNORECASE)


def parse_semiconductor(s: str) -> Dict[str, Any]:
    out = {"type": None, "film_thickness_um": None, "scattering_thickness_um": None}
    if not isinstance(s, str) or not s.strip():
        return out

    txt = s.strip()

    # Type
    mtype = RE_TYPE.search(txt)
    if mtype:
        out["type"] = mtype.group("type")

    # Remove nm tokens/ranges; normalize spaces & soft hyphen
    cleaned = RE_STRIP_NM.sub("", txt)
    cleaned = cleaned.replace("\u00ad", "")  # remove soft hyphen (e.g., in "scat­tering")
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    # --- Film range: "(16-18 um)" → average; return early
    mrange = RE_FILM_RANGE.search(cleaned)
    if mrange:
        v1 = _to_um(mrange.group("val1"), mrange.group("unit"))
        v2 = _to_um(mrange.group("val2"), mrange.group("unit"))
        if v1 is not None and v2 is not None:
            out["film_thickness_um"] = (v1 + v2) / 2.0
        return out

    # --- Pair with words "(... transparent + ... scattering layer)" (either order)
    mwords =  RE_PAIR_WITH_WORDS_FILM_FIRST.search(cleaned) or RE_PAIR_WITH_WORDS_SCAT_FIRST.search(cleaned)
    if mwords:
        out["film_thickness_um"] = _to_um(mwords.group("film"), mwords.group("film_unit"))
        out["scattering_thickness_um"] = _to_um(mwords.group("scat"), mwords.group("scat_unit"))
        return out

    # --- Shared-unit pair "(12 + 4 um thick)"
    mshared = RE_PAIR_SHARED_UNIT.search(cleaned)
    if mshared:
        out["film_thickness_um"] = _to_um(mshared.group("film"), mshared.group("unit"))
        out["scattering_thickness_um"] = _to_um(mshared.group("scat"), mshared.group("unit"))
        return out

    # --- Simple pair "(film + scatter unit)" (unit applies to both)
    mpair = RE_PAIR_BOTH_IN_PARENS.search(cleaned)
    if mpair:
        unit = mpair.group("unit")
        out["film_thickness_um"] = _to_um(mpair.group("film"), unit)
        out["scattering_thickness_um"] = _to_um(mpair.group("scat"), unit)
        return out

    # --- Scattering first: prefer TEXT_FIRST then NUM_FIRST (tight windows)
    msc = RE_SCAT_TEXT_FIRST.search(cleaned) or RE_SCAT_NUM_FIRST.search(cleaned)
    scat_span = None
    if msc:
        out["scattering_thickness_um"] = _to_um(msc.group("val"), msc.group("unit"))
        scat_span = msc.span()

    # --- Film: explicit phrases
    mfilm = RE_FILM_IN_THICKNESS.search(cleaned) or RE_FILM_THICK.search(cleaned)
    if mfilm:
        out["film_thickness_um"] = _to_um(mfilm.group("val"), mfilm.group("unit"))
        return out

    # --- Film: fallback → first num+unit not inside scattering span, preferably after 'film'
    film_anchor = RE_FILM_WORD.search(cleaned)
    film_start = film_anchor.start() if film_anchor else -1
    for m in RE_NUM_UNIT.finditer(cleaned):
        if scat_span and (m.start() >= scat_span[0] and m.end() <= scat_span[1]):
            continue
        if film_start == -1 or m.start() >= film_start:
            out["film_thickness_um"] = _to_um(m.group("val"), m.group("unit"))
            break

    return out

def clean_cosensitizer(name):
    if pd.isna(name):
        return name
    # Remove patterns like "0.2 mM", "1 µM", "10 μM", "5 mM", etc.
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mM|uM|µM|μM|M)\b", "", str(name), flags=re.IGNORECASE)
    # Remove extra spaces and commas and Ruthenium
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r", Ruthenium", " ", cleaned).strip()
    isConc = bool(re.search('mol/cm3', cleaned))
    if cleaned == '-' or isConc:
        cleaned = np.nan
    return cleaned

_NUM = r'[-+]?\d+(?:[.,]\d+)?(?:e[+\-]?\d+)?'  # supports 1, 1.2, 1,2, 1e-3, 1.2e+3

def _to_float(x: str) -> float:
    return float(x.replace(',', '.'))

def parse_active_area(area_str):
    # Missing or placeholder
    if pd.isna(area_str):
        return np.nan
    s = str(area_str).strip()
    if s in {"", "-"}:
        return np.nan

    s_low = s.lower()

    # --- detect unit for later conversion ---
    unit = "cm2"  # default
    if re.search(r'\bmm\s*\^?\s*2|\bmm2|mm²', s_low):
        unit = "mm2"
    elif re.search(r'\bm\s*\^?\s*2|\bm2|m²', s_low):
        unit = "m2"
    elif re.search(r'\bcm\s*\^?\s*2|\bcm2|cm²', s_low):
        unit = "cm2"

    # --- clean obvious unit tokens and parentheses to ease number matching ---
    s_clean = re.sub(r'[\(\)]', ' ', s_low)
    s_clean = re.sub(r'\b(cm|mm|m)\s*\^?\s*2|cm²|mm²|m²|cm2|mm2|m2', ' ', s_clean)
    s_clean = re.sub(r'\s{2,}', ' ', s_clean).strip()

    # --- try range first: "a-b", "a–b", or "a/b" (keep only number tokens) ---
    m_range = re.search(rf'({_NUM})\s*[-–/]\s*({_NUM})', s_clean)
    if m_range:
        a = _to_float(m_range.group(1))
        b = _to_float(m_range.group(2))
        val = (a + b) / 2.0
    else:
        # single number: take the first numeric token
        m_single = re.search(rf'({_NUM})', s_clean)
        if not m_single:
            return np.nan
        val = _to_float(m_single.group(1))

    # --- unit conversion to cm^2 ---
    if unit == "mm2":
        val = val / 100.0        # 1 cm2 = 100 mm2
    elif unit == "m2":
        val = val * 10000.0      # 1 m2 = 10,000 cm2
    # else already cm²

    return val

def parse_exposure_time(s):
    if pd.isna(s) or str(s).strip() in ["-", ""]:
        return np.nan
    
    text = str(s).lower().strip()
    
    # Replace slashes with hyphen for ranges
    text = text.replace("/", "-")
    
    def parse_single_value(val):
        val = val.strip()
        # minutes to hours
        m = re.match(r"(\d+(?:\.\d+)?)\s*(min|minutes?)", val)
        if m:
            return float(m.group(1)) / 60.0
        # hours
        m = re.match(r"(\d+(?:\.\d+)?)\s*(h|hours?|hour)", val)
        if m:
            return float(m.group(1))
        return None
    
    # Handle multi-part exposures separated by "+"
    if "+" in text:
        parts = text.split("+")
        total = 0.0
        for part in parts:
            # Extract all numeric+unit patterns from each part
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(h|hours?|hour|min|minutes?)", part)
            for num, unit in matches:
                num = float(num)
                if unit.startswith("min"):
                    num /= 60.0
                total += num
        return total
    
    # Handle ranges (e.g., "16-17 hours" or "8-12 h")
    range_match = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(h|hours?|hour|min|minutes?)", text)
    if range_match:
        v1, v2, unit = range_match.groups()
        v1, v2 = float(v1), float(v2)
        if unit.startswith("min"):
            v1 /= 60.0
            v2 /= 60.0
        return (v1 + v2) / 2.0
    
    # Otherwise, try direct parsing
    val = parse_single_value(text)
    if val is not None:
        return val
    
    # If no direct match, look for any number-unit pairs and sum
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(h|hours?|hour|min|minutes?)", text)
    if matches:
        total = 0.0
        for num, unit in matches:
            num = float(num)
            if unit.startswith("min"):
                num /= 60.0
            total += num
        return total if total > 0 else np.nan
    
    return np.nan

AM15G_CANON = "AM 1.5G"

_mwcm2_re = re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*m[wW]\s*/\s*cm2\b')
_sun_pct_re = re.compile(r'(?P<pct>\d+(?:\.\d+)?)\s*%\s*sun\b', re.IGNORECASE)
_sun_frac_re = re.compile(r'(?P<frac>\d+(?:\.\d+)?)\s*sun\b', re.IGNORECASE)
_am_tag_re = re.compile(r'\bam\s*1\.?5\s*g?\b', re.IGNORECASE)

def _to_float(x: str) -> float:
    return float(x.replace(",", "."))

def parse_solar_simulator(cell, one_sun_mwcm2: float = 100.0):
    """
    Return irradiance as float (mW/cm²), normalized under AM 1.5G.
    If no irradiance found, returns np.nan.
    """
    if pd.isna(cell):
        return np.nan

    s = str(cell).strip()
    if s in {"", "-"}:
        return np.nan

    # Normalize separators/spacing
    t = s.replace(",", " ").replace("  ", " ").strip()

    # 1) Direct mW/cm2 anywhere in the string
    m = _mwcm2_re.search(t)
    if m:
        return _to_float(m.group("val"))

    # 2) Percent sun → mW/cm2
    m = _sun_pct_re.search(t)
    if m:
        return (_to_float(m.group("pct")) / 100.0) * one_sun_mwcm2

    # 3) Fractional sun (e.g., 0.66 sun, 0.095 sun) → mW/cm2
    m = _sun_frac_re.search(t)
    if m:
        return _to_float(m.group("frac")) * one_sun_mwcm2

    # 4) If it looks like an AM 1.5/G entry but no numeric irradiance is present → NaN
    if _am_tag_re.search(t):
        return np.nan

    # 5) Bare numeric mW/cm2 (e.g., "0.336 mW/cm2") already handled in (1).
    # If we reach here, nothing matched
    
    return np.nan

NA = 6.02214076e23  # Avogadro constant
UNIT_FACTORS = {
    "nmol/cm2": 1.0,
    "mol/cm2": 1e9,  # mol → nmol
    "mmol/cm3": None,  # Can't convert to cm² without thickness
    "nmol/mm3": None,  # Same issue as mmol/cm³
    "molecules/cm2": 1e9 / NA,  # molecules → nmol
    "mols/mm": None  # no direct surface conversion
}

def parse_dye_loading(cell):
    if pd.isna(cell):
        return np.nan
    
    s = str(cell).strip()
    if s in {"", "-"}:
        return np.nan
    
    # Replace multiple spaces and commas
    s = re.sub(r"[ ,]+", " ", s)
    
    # Handle sums like "95 nmol/cm2 + 52 nmol/cm2"
    parts = re.split(r"\s*\+\s*", s)
    total_nmol_cm2 = 0.0
    parsed_any = False
    
    for part in parts:
        # Scientific notation with optional units
        m = re.match(r"([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-zA-Z+/0-9.]+)", part)
        if not m:
            continue
        
        val = float(m.group(1))
        unit = m.group(2).lower()
        
        # Normalize weird spacing, e.g., 'nmol/cm2' vs 'nmol / cm2'
        unit = unit.replace(" ", "")
        
        # Match known units
        if "nmol/cm2" in unit:
            total_nmol_cm2 += val
            parsed_any = True
        elif "mol/cm2" in unit and not "nmol" in unit:
            total_nmol_cm2 += val * 1e9
            parsed_any = True
        elif "molecules/cm2" in unit:
            total_nmol_cm2 += val * (1e9 / NA)
            parsed_any = True
        # If mmol/cm³ or nmol/mm³, skip because we lack thickness
        elif "mmol/cm3" in unit or "nmol/mm3" in unit or "mols/mm" in unit:
            return np.nan
    
    return total_nmol_cm2 if parsed_any else np.nan


