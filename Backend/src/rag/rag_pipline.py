"""
RAG pipeline for Alloyance
--------------------------
Responsibilities:
 - Retrieve reference data for Life Cycle Assessment (LCA)
 - Generate Executive Summary and Circularity Analysis via Gemini or fallback text
 - Serve as a clean interface for report_tech.py

Supports:
 - Gemini API (via google.generativeai)
 - Static fallback summaries if no API key or model response

Usage:
 - Ensure GEMINI_API_KEY is set in environment variables
"""

import os
import logging
from typing import Dict

# Optional: dotenv for local testing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# ------------------------------------------------------
# OpenRouter Model Setup (Gemma 3 27B)
# ------------------------------------------------------
# ------------------------------------------------------
# OpenRouter Model Setup (Llama 3.3 70B Instruct)
# ------------------------------------------------------
from openai import OpenAI
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

PRIMARY_MODEL = "google/gemini-2.0-flash-exp:free"
FALLBACK_MODEL = "mistralai/mistral-nemo:free"

def _call_openrouter(prompt: str, label: str) -> str:
    """Call OpenRouter using Llama 3.3 70B as primary and Mistral Nemo as fallback."""
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No OPENROUTER_API_KEY found; using fallback for %s", label)
        return ""

    def _try_model(model_name):
        logger.info("Generating %s using %s...", label, model_name)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sustainability and circular-economy expert. "
                        "Respond in clear, professional report language without markdown or emojis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=900,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Alloyance-RAG",
            },
        )
        return completion.choices[0].message.content.strip()

    # Try primary, then fallback
    try:
        return _try_model(PRIMARY_MODEL)
    except Exception as e:
        logger.error("Primary model failed (%s): %s", PRIMARY_MODEL, e)
        time.sleep(2)
        try:
            return _try_model(FALLBACK_MODEL)
        except Exception as e2:
            logger.exception("Fallback model also failed: %s", e2)
            return ""



# ------------------------------------------------------
# Prompt Templates (clean, no emojis or Unicode symbols)
# ------------------------------------------------------

EXECUTIVE_SUMMARY_PROMPT = """
You are an expert sustainability analyst preparing the Executive Summary section of a Life Cycle Assessment (LCA) report.

Context:
Material: {material}
Circularity Score: {circ_score}
Recycled Content: {recycled_content}
Reuse Potential: {reuse_potential}
Recovery Rate: {recovery_rate}

Task:
1. Summarize the overall environmental and circularity performance of the material.
2. Include the following four parts:
   - Introduction
   - Key Metrics
   - Assessment
   - Recommendations (3 to 4 actionable suggestions)
3. Write in a professional, concise, and factual style suitable for executive-level reports.
4. Keep it under 250 words.
5. Use plain English text only. Do not include emojis, arrows, bullets, or special symbols.
"""

CIRCULARITY_ANALYSIS_PROMPT = """
You are a circular economy expert preparing the Circularity Analysis section of an LCA report.

Context:
Material: {material}
Circularity Score: {circ_score}
Recycled Content: {recycled_content}
Reuse Potential: {reuse_potential}
Recovery Rate: {recovery_rate}

Task:
1. Write three structured sections:
   - Material Flow
   - Circular Economy Indicators
   - Opportunities for Improvement
2. Discuss how recycled inputs, reuse, and recovery affect resource efficiency.
3. Keep the writing clear, factual, and action-oriented (approximately 300 to 400 words).
4. Use plain text only. Avoid emojis, decorative characters, or special symbols.
"""

EMISSION_INTERPRETATION_PROMPT = """
You are an environmental analyst. Based on the following data, provide an insightful emission interpretation.

Material: {material}
Process Stage: {process_stage}
CO2: {emission_co2} kg
SOx: {emission_sox} kg
NOx: {emission_nox} kg
Particulate Matter: {emission_pm} kg
Water AMD: {emission_amd} kg
Water Heavy Metals: {emission_hm} kg
Water BOD: {emission_bod} kg
Total GHG Emissions: {ghg_emissions} kg CO2-eq

Task:
Write two concise paragraphs explaining the significance of these emissions and potential mitigation opportunities.
Use clear, factual, professional language without emojis or symbols.
"""

ENERGY_EFFICIENCY_PROMPT = """
Analyze the energy efficiency of the following manufacturing configuration:

Material: {material}
Stage: {process_stage}
Technology: {technology}
Energy Input: {energy_input_qty} MJ

Task:
Discuss efficiency, potential energy savings, and circular implications.
Keep the tone technical and professional, and avoid using emojis or non-ASCII characters.
"""

BENCHMARK_ANALYSIS_PROMPT = """
Compare the following circularity metrics to typical benchmarks for similar materials.

Material: {material}
Circularity Score: {circ_score}%
Recycled Content: {recycled_content}%
Reuse Potential: {reuse_potential}%
Recovery Rate: {recovery_rate}%

Task:
Summarize how this material performs relative to standard benchmarks and identify improvement areas.
Maintain a neutral, factual tone. Do not include emojis, arrows, or other special characters.
"""

ACTION_RECOMMENDATIONS_PROMPT = """
You are a circular economy consultant. Based on the data below, provide actionable recommendations.

Material: {material}
Stage: {process_stage}
Technology: {technology}
Circularity Score: {circ_score}%
Recycled Content: {recycled_content}%
Reuse Potential: {reuse_potential}%
Recovery Rate: {recovery_rate}%
GHG Emissions: {ghg_emissions} kg CO2-eq
Energy Input: {energy_input_qty} MJ

Task:
List three to four prioritized strategies for improvement that are specific, measurable, and actionable.
Write in a professional report style and use only plain ASCII text (no emojis, arrows, or symbols).
"""


# ------------------------------------------------------
# Fallback Static Templates
# ------------------------------------------------------
def _fallback_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate):
    """Static text if Gemini is unavailable."""
    return (
        f"This Life Cycle Assessment evaluates the environmental and circularity performance of {material}. "
        f"The analysis indicates a Circularity Score of {circ_score}%, supported by {recycled_content}% recycled content, "
        f"a reuse potential of {reuse_potential}%, and a recovery rate of {recovery_rate}%. "
        f"\n\nCircularity Assessment:\nThe material demonstrates moderate circular potential, "
        f"with strong reuse and recycled input levels but room for improvement in end-of-life recovery. "
        f"\n\nRecommendations:\n"
        f"1. Increase post-use collection and recovery efficiency.\n"
        f"2. Integrate more secondary materials in production.\n"
        f"3. Implement design-for-reuse and modular strategies."
    )


def _fallback_circularity(material, circ_score, recycled_content, reuse_potential, recovery_rate):
    """Static text if Gemini is unavailable."""
    return (
        f"Material Flow:\nApproximately {recycled_content}% of {material} comes from recycled inputs, "
        f"reducing reliance on virgin extraction. The reuse potential of {reuse_potential}% helps extend "
        f"product lifecycles, while {recovery_rate}% of materials are currently recovered at end-of-life.\n\n"
        f"Circular Economy Indicators:\nThe Circularity Score of {circ_score}% indicates a balanced performance "
        f"across recycling, reuse, and recovery dimensions, though system inefficiencies still limit overall retention.\n\n"
        f"Opportunities for Improvement:\n"
        f"- Increase use of recycled feedstock and expand take-back systems.\n"
        f"- Improve recovery processes through better sorting and reprocessing.\n"
        f"- Promote product design strategies that facilitate disassembly and reuse."
    )


def _fallback_emission_interpretation(material, process_stage, *args):
    return (f"The {process_stage.lower()} stage for {material.lower()} shows moderate emissions. "
            f"CO₂ is the dominant contributor, followed by SOx and NOx, which may originate from energy or fuel combustion. "
            f"Water emissions such as heavy metals and BOD indicate minor wastewater impact.")

def _fallback_energy_efficiency(material, process_stage, technology, energy_input_qty):
    return (f"The {technology.lower()} technology used in the {process_stage.lower()} stage "
            f"consumes approximately {energy_input_qty:.1f} MJ of energy. "
            "Energy efficiency improvements, such as heat recovery or renewable electricity sourcing, could reduce the footprint.")

def _fallback_benchmark_analysis(material, circ_score, recycled_content, reuse_potential, recovery_rate):
    return (f"With a circularity score of {circ_score:.1f}%, {material.lower()} performs moderately compared to industry averages. "
            f"Recycled content ({recycled_content:.1f}%) and recovery rate ({recovery_rate:.1f}%) suggest partial circular adoption.")

def _fallback_action_recommendations(material, process_stage, technology, circ_score,
                                     recycled_content, reuse_potential, recovery_rate,
                                     ghg_emissions, energy_input_qty):
    return (f"To enhance sustainability of {material.lower()} in the {process_stage.lower()} stage, consider:\n"
            f"- Upgrading {technology.lower()} processes to lower GHG emissions ({ghg_emissions:.1f} kg CO₂-eq)\n"
            f"- Boosting recycled content (currently {recycled_content:.1f}%) to cut material intensity\n"
            f"- Improving reuse potential and recovery beyond {reuse_potential:.1f}% and {recovery_rate:.1f}%\n"
            f"- Targeting energy efficiency gains from the current {energy_input_qty:.1f} MJ per functional unit.")



# ------------------------------------------------------
# Public API
# ------------------------------------------------------
def generate_summary(material: str, circ_score: float, recycled_content: float, reuse_potential: float, recovery_rate: float) -> str:
    """Generate Executive Summary for PDF report."""
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        material=material,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate,
    )
    result = _call_openrouter(prompt, "Executive Summary")
    return result or _fallback_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate)


def generate_circularity_analysis(material: str, circ_score: float, recycled_content: float, reuse_potential: float, recovery_rate: float) -> str:
    """Generate Circularity Analysis for PDF report."""
    prompt = CIRCULARITY_ANALYSIS_PROMPT.format(
        material=material,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate,
    )
    result = _call_openrouter(prompt, "Circularity Analysis")
    return result or _fallback_circularity(material, circ_score, recycled_content, reuse_potential, recovery_rate)


def generate_emission_interpretation(
    material: str,
    process_stage: str,
    emission_co2: float,
    emission_sox: float,
    emission_nox: float,
    emission_pm: float,
    emission_amd: float,
    emission_hm: float,
    emission_bod: float,
    ghg_emissions: float
) -> str:
    """Generate emission interpretation section for the PDF report."""
    prompt = EMISSION_INTERPRETATION_PROMPT.format(
        material=material,
        process_stage=process_stage,
        emission_co2=emission_co2,
        emission_sox=emission_sox,
        emission_nox=emission_nox,
        emission_pm=emission_pm,
        emission_amd=emission_amd,
        emission_hm=emission_hm,
        emission_bod=emission_bod,
        ghg_emissions=ghg_emissions
    )
    result = _call_openrouter(prompt, "Emission Interpretation")
    return result or _fallback_emission_interpretation(
        material, process_stage,
        emission_co2, emission_sox, emission_nox,
        emission_pm, emission_amd, emission_hm,
        emission_bod, ghg_emissions
    )


def generate_energy_efficiency_analysis(
    material: str,
    process_stage: str,
    technology: str,
    energy_input_qty: float
) -> str:
    """Generate energy efficiency analysis for the report."""
    prompt = ENERGY_EFFICIENCY_PROMPT.format(
        material=material,
        process_stage=process_stage,
        technology=technology,
        energy_input_qty=energy_input_qty
    )
    result = _call_openrouter(prompt, "Energy Efficiency Analysis")
    return result or _fallback_energy_efficiency(material, process_stage, technology, energy_input_qty)


def generate_benchmark_analysis(
    material: str,
    circ_score: float,
    recycled_content: float,
    reuse_potential: float,
    recovery_rate: float
) -> str:
    """Generate benchmark comparison analysis for the circularity metrics."""
    prompt = BENCHMARK_ANALYSIS_PROMPT.format(
        material=material,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate
    )
    result = _call_openrouter(prompt, "Benchmark Analysis")
    return result or _fallback_benchmark_analysis(material, circ_score, recycled_content, reuse_potential, recovery_rate)


def generate_action_recommendations(
    material: str,
    process_stage: str,
    technology: str,
    circ_score: float,
    recycled_content: float,
    reuse_potential: float,
    recovery_rate: float,
    ghg_emissions: float,
    energy_input_qty: float
) -> str:
    """Generate AI-driven recommendations for circularity and sustainability improvement."""
    prompt = ACTION_RECOMMENDATIONS_PROMPT.format(
        material=material,
        process_stage=process_stage,
        technology=technology,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate,
        ghg_emissions=ghg_emissions,
        energy_input_qty=energy_input_qty
    )
    result = _call_openrouter(prompt, "Action Recommendations")
    return result or _fallback_action_recommendations(
        material, process_stage, technology, circ_score,
        recycled_content, reuse_potential, recovery_rate,
        ghg_emissions, energy_input_qty
    )



# ------------------------------------------------------
# CLI test (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    mat = "Copper Ore"
    circ, rec, reuse, recov = 52.8, 51.7, 60.2, 46.5
    print("\n=== EXECUTIVE SUMMARY ===\n")
    print(generate_summary(mat, circ, rec, reuse, recov))
    print("\n=== CIRCULARITY ANALYSIS ===\n")
    print(generate_circularity_analysis(mat, circ, rec, reuse, recov))
