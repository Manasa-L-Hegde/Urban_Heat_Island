"""AI advisor utilities for the Urban Heat Island dashboard.

Purpose:
- Build concise textual context from ward-level data and answer natural-language
    questions about heat risk using simple rules and optional LLM fallbacks.

Used by:
- `app/app.py` (chatbot widget), `src/heat_risk_advisor.py` can be used standalone
    in scripts to generate advisor responses.

Input → Process → Output:
- Input: ward-level DataFrame → Process: summarise top wards and apply rules/LLM →
- Output: human-readable advice strings.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

import pandas as pd
import numpy as np

QUESTION_HINTS = {
    "elderly": "elderly people",
    "old": "elderly people",
    "danger": "highest heat and vulnerability",
    "urgent": "urgent cooling measures",
    "tree": "tree planting first",
    "plant": "tree planting first",
    "cool": "cooling measures",
    "roof": "reflective roofing",
    "heat": "heat stress",
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column name from `candidates` present in `df`.

    Parameters:
    - df (pd.DataFrame): DataFrame to search.
    - candidates (list[str]): Candidate column names (case-insensitive).

    Returns:
    - str: The actual column name present in `df`.

    Raises KeyError when none of the candidates are found.
    """
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of these columns were found: {candidates}")


def _normalize_text(value: str) -> str:
    """Lowercase and remove non-alphanumeric characters for simple matching."""
    return re.sub(r"[^a-z0-9\s]", " ", str(value).lower())


def _extract_ward_name(df: pd.DataFrame, question: str) -> Optional[str]:
    """Try to find a ward name mentioned in `question` by simple substring match.

    Returns the ward name if found, otherwise `None`.
    """
    ward_col = _find_column(df, ["Ward Name"])
    question_text = _normalize_text(question)

    for ward_name in df[ward_col].dropna().astype(str).unique():
        if _normalize_text(ward_name) in question_text:
            return ward_name
    return None


def _explain_ward_risk(df: pd.DataFrame, ward_name: str) -> str:
    """Produce a short explanation for a single ward's heat risk drivers.

    Looks up heat, green cover, population (and building density where present)
    and returns human-readable observations and recommended quick actions.
    """
    ward_col = _find_column(df, ["Ward Name"])
    heat_col = _find_column(df, ["Heat_Index"])
    green_col = _find_column(df, ["Green Cover", "Green Cover %", "Green Cover (%)"])
    population_col = _find_column(df, ["Population Density", "Population Density (people/km²)"])

    building_col = None
    try:
        building_col = _find_column(df, ["Building Density", "Building Density (0-1)"])
    except KeyError:
        building_col = None

    ward_rows = df[df[ward_col].astype(str).str.lower() == str(ward_name).lower()]
    if ward_rows.empty:
        return "I could not find that ward in the dataset."

    ward_data = ward_rows.iloc[0]
    parts = [
        f"{ward_name} has a Heat_Index of {ward_data[heat_col]:.2f}, which indicates notable heat stress.",
        f"Its green cover is {ward_data[green_col]:.2f}, so limited vegetation is reducing local cooling.",
        f"Population density is {ward_data[population_col]:.0f}, which increases exposure during hot periods.",
    ]

    # Optionally include building density if available
    if building_col is not None and building_col in ward_data.index:
        parts.append(f"Building density is {ward_data[building_col]:.2f}, adding to retained heat in the ward.")

    parts.append("Priority actions here are shade, tree planting, cool roofs, and hydration support for residents.")
    return " ".join(parts)


def _top_ward_lines(df: pd.DataFrame, sort_column: str, value_label: str, top_n: int = 5) -> list[str]:
    """Return formatted lines for the top `top_n` wards by `sort_column`."""
    ward_col = _find_column(df, ["Ward Name"])
    subset = df.sort_values(sort_column, ascending=False).head(top_n)
    lines = []
    for _, row in subset.iterrows():
        ward_name = row[ward_col]
        value = row[sort_column]
        lines.append(f"- {ward_name}: {value_label} = {value:.2f}")
    return lines


def build_context(df: pd.DataFrame) -> str:
    """Build a compact textual city summary used as context for the advisor.

    The returned text includes top wards by heat, vulnerability, elderly share,
    and a short pattern note flagging wards with high density and low green cover.
    """
    heat_col = _find_column(df, ["Heat_Index"])
    vulnerability_col = _find_column(df, ["Vulnerability_Index"])
    elderly_col = _find_column(df, ["Elderly Population", "Elderly Population (%)", "Elderly_Population_Normalized"])
    population_col = _find_column(df, ["Population Density", "Population Density (people/km²)"])
    building_col = _find_column(df, ["Building Density", "Building Density (0-1)"])
    green_col = _find_column(df, ["Green Cover", "Green Cover (%)"])
    ward_col = _find_column(df, ["Ward Name"])

    top_heat = df.sort_values(heat_col, ascending=False).head(5)
    top_vulnerability = df.sort_values(vulnerability_col, ascending=False).head(5)
    top_elderly = df.sort_values(elderly_col, ascending=False).head(5)

    density_threshold = df[population_col].quantile(0.75)
    green_threshold = df[green_col].quantile(0.25)
    pattern_df = df[(df[population_col] >= density_threshold) & (df[green_col] <= green_threshold)]

    lines = [
        "City heat-risk context summary:",
        "",
        "Top 5 wards by Heat_Index:",
    ]
    for _, row in top_heat.iterrows():
        lines.append(
            f"- {row[ward_col]} | Heat_Index={row[heat_col]:.2f} | Vulnerability={row[vulnerability_col]:.2f} | Elderly={row[elderly_col]:.2f}"
        )

    lines.extend([
        "",
        "Top 5 wards by Vulnerability_Index:",
    ])
    for _, row in top_vulnerability.iterrows():
        lines.append(
            f"- {row[ward_col]} | Vulnerability={row[vulnerability_col]:.2f} | Heat_Index={row[heat_col]:.2f} | Elderly={row[elderly_col]:.2f}"
        )

    lines.extend([
        "",
        "Top 5 wards by Elderly Population:",
    ])
    for _, row in top_elderly.iterrows():
        lines.append(
            f"- {row[ward_col]} | Elderly={row[elderly_col]:.2f} | Vulnerability={row[vulnerability_col]:.2f} | Heat_Index={row[heat_col]:.2f}"
        )

    lines.extend([
        "",
        "Pattern notes:",
        f"- High-density wards are defined as population density >= {density_threshold:.0f}.",
        f"- Low-green-cover wards are defined as green cover <= {green_threshold:.2f}.",
    ])

    if not pattern_df.empty:
        pattern_summary = ", ".join(
            f"{ward} (Pop={int(pop)}, Green={green:.2f}%)"
            for ward, pop, green in zip(pattern_df[ward_col].head(8), pattern_df[population_col].head(8), pattern_df[green_col].head(8))
        )
        lines.append(f"- Wards showing high density + low green cover: {pattern_summary}.")
    else:
        lines.append("- No wards matched the high-density + low-green-cover pattern in this dataset slice.")

    lines.extend([
        "",
        "Quick interpretation:",
        "- Higher Heat_Index indicates more intense surface heat stress.",
        "- Higher Vulnerability_Index indicates wards where heat risk is amplified by people and urban form.",
        "- Elderly population is important because vulnerable residents are more likely to suffer during heatwaves.",
    ])

    return "\n".join(lines)


def _build_prompt(question: str, context: str) -> str:
    """Assemble the prompt text sent to an LLM using `context` and `question`."""
    return (
        "You are an urban heat risk advisor.\n"
        "Use the given city data context to answer the user's question clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{question}\n\n"
        "Instructions:\n"
        "- Answer in 2–4 sentences\n"
        "- Mention ward names explicitly\n"
        "- Give reason (temperature, density, green cover, elderly)\n"
        "- Be precise and actionable"
    )


def smart_answer(df: pd.DataFrame, question: str) -> Optional[str]:
    q = question.lower()

    ward_col = _find_column(df, ["Ward Name"])
    heat_col = _find_column(df, ["Heat_Index"])
    vulnerability_col = _find_column(df, ["Vulnerability_Index"])
    green_col = _find_column(df, ["Green Cover", "Green Cover %", "Green Cover (%)"])

    if "explain" in q:
        ward_name = _extract_ward_name(df, question)
        if ward_name:
            return _explain_ward_risk(df, ward_name)

    # COOLING REQUIRED
    if "cooling" in q:
        areas = df.sort_values(heat_col, ascending=False).head(3)
        return f"Urgent cooling measures are required in {', '.join(areas[ward_col].astype(str).values)}."

    # COOL AREAS
    if "cool" in q or "low heat" in q:
        cool_wards = df.sort_values(heat_col, ascending=True).head(3)
        return f"The coolest areas are {', '.join(cool_wards[ward_col].astype(str).values)} with the lowest heat index."

    # HIGH RISK / DANGEROUS
    if "danger" in q or "risk" in q:
        risky = df.sort_values(vulnerability_col, ascending=False).head(3)
        return f"The most vulnerable areas are {', '.join(risky[ward_col].astype(str).values)} due to high heat and population density."

    # TREE PLANTING
    if "tree" in q or "green" in q:
        areas = df[df[green_col] < 20].sort_values(heat_col, ascending=False).head(3)
        if not areas.empty:
            return f"Tree plantation is most needed in {', '.join(areas[ward_col].astype(str).values)} due to low green cover and high heat."

    return None


def _rule_based_answer(question: str, context: str) -> str:
    ward_lines = [line.strip() for line in context.splitlines() if line.strip().startswith("-")]
    top_wards = []
    for line in ward_lines:
        match = re.match(r"-\s+(.*?)\s+\|", line)
        if match:
            top_wards.append(match.group(1))
    top_wards = top_wards[:3]

    question_lower = question.lower()
    if any(keyword in question_lower for keyword in ["elderly", "old"]):
        return (
            f"The most dangerous wards for elderly residents are {', '.join(top_wards[:3])}. "
            "These wards appear in the highest-risk group, so they should get priority for cooling shelters, hydration points, and shade support."
        )
    if any(keyword in question_lower for keyword in ["tree", "plant"]):
        return (
            f"Tree planting should start in {', '.join(top_wards[:3])}. "
            "These wards combine stronger heat stress with higher vulnerability, so increasing green cover there will have the biggest cooling impact."
        )
    if any(keyword in question_lower for keyword in ["cool", "urgent", "danger"]):
        return (
            f"Urgent cooling measures are needed in {', '.join(top_wards[:3])}. "
            "They should be prioritized because the context shows the highest heat and vulnerability scores in those wards."
        )
    return (
        f"The most important wards to watch are {', '.join(top_wards[:3])}. "
        "They have the strongest combination of heat stress, density, and vulnerability, so these are the best candidates for immediate intervention."
    )


def _ask_openai(question: str, context: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = _build_prompt(question, context)
        try:
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
        except Exception:
            chat_response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an urban heat risk advisor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return chat_response.choices[0].message.content.strip()
    except Exception:
        return None


def _ask_gemini(question: str, context: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        prompt = _build_prompt(question, context)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


def ask_ai(question: str, context: str, df: pd.DataFrame) -> str:
    """
    Ask an LLM to answer a heat-risk question using compact city context.
    Falls back to a rule-based response if API access fails.
    """
    # STEP 1: Try smart rule-based answer
    rule_answer = smart_answer(df, question)
    if rule_answer:
        return rule_answer

    # STEP 2: LLM fallback (if needed)
    try:
        answer = _ask_openai(question, context)
        if answer:
            return answer

        answer = _ask_gemini(question, context)
        if answer:
            return answer
    except Exception:
        return "Unable to generate response"

    return _rule_based_answer(question, context)


def answer_questions(df: pd.DataFrame, questions: list[str]) -> list[str]:
    context = build_context(df)
    return [ask_ai(question, context, df) for question in questions]


def get_top_alerts(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Return top k wards with highest Heat_Index with advisory information.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with heat and advisory data.
    - k (int): Number of top wards to return (default: 5).
    
    Returns:
    - pd.DataFrame: DataFrame containing Ward Name, Heat_Index, Vulnerability_Index,
        and Heat_Advisory for the top k wards sorted by Heat_Index (descending).
    
    Raises:
    - KeyError: If required columns are missing from the DataFrame.
    """
    required_cols = ["Ward Name", "Heat_Index", "Vulnerability_Index", "Heat_Advisory"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Select and sort by Heat_Index in descending order, then take top k
    top_alerts = (
        df[required_cols]
        .sort_values("Heat_Index", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
    
    return top_alerts


def generate_green_cover_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate ward-level, actionable green cover recommendations.

    The function evaluates multiple, possibly overlapping conditions and
    combines specific, actionable recommendations into a single
    `Green_Cover_Recommendation` column.

    Expected input columns (case-insensitive, flexible names are accepted):
    - Heat_Index
    - Green Cover (or Green_Cover, Green Cover (%))
    - Population Density (or Population_Density, Population Density (people/km²))
    - Building Density (or Building_Density, Building Density (0-1))

    Returns a copy of the DataFrame with `Green_Cover_Recommendation` added.
    """
    # Resolve column names present in the DataFrame
    heat_col = _find_column(df, ["Heat_Index"])
    green_col = _find_column(df, ["Green Cover", "Green Cover (%)", "Green_Cover", "Green_Cover (%)"]) 
    pop_col = _find_column(df, ["Population Density", "Population_Density", "Population Density (people/km²)"])
    build_col = _find_column(df, ["Building Density", "Building_Density", "Building Density (0-1)"])

    # Vectorized rule evaluation -> recommendation strings (empty string when not applicable)
    r1 = np.where((df[heat_col] > 75) & (df[green_col] < 20),
                  "Urgent tree plantation required (increase canopy by 15–25%)",
                  "")
    r2 = np.where(df[build_col] > 0.7,
                  "Implement green roofs and vertical gardens",
                  "")
    r3 = np.where(df[pop_col] > 10000,
                  "Develop micro parks and shaded public spaces",
                  "")
    r4 = np.where(df[heat_col] > 85,
                  "Declare heat-risk zone and prioritize cooling infrastructure",
                  "")
    r5 = np.where(df[green_col] > 40,
                  "Maintain and protect existing green cover",
                  "")

    # Combine recommendations cleanly per-row
    rec_df = pd.DataFrame({
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "r5": r5,
    }, index=df.index)

    combined = rec_df.apply(lambda row: "; ".join([s for s in row if s]), axis=1)

    out = df.copy()
    out["Green_Cover_Recommendation"] = combined
    # If no rule matched, provide a neutral, actionable default
    out["Green_Cover_Recommendation"] = out["Green_Cover_Recommendation"].replace("", "No specific action recommended")

    return out


if __name__ == "__main__":
    sample_path = "data/ward_data.csv"
    df = pd.read_csv(sample_path)
    context = build_context(df)
    example_questions = [
        "Which areas are most dangerous for elderly people?",
        "Which wards need urgent cooling measures?",
        "Where should trees be planted first?",
    ]
    print(context)
    print("\n--- AI Answers ---\n")
    for question in example_questions:
        print(f"Q: {question}")
        print(f"A: {ask_ai(question, context, df)}")
        print()
