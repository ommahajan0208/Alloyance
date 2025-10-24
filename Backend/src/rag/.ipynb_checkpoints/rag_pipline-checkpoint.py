import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate

# ✅ Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 📊 Data Structures
@dataclass
class UserInputData:
    energy_source: str
    transport_mode: str
    transport_distance_km: float
    recycled_content_percent: float
    location: str
    functional_unit: str
    raw_material_type: str
    processing_method: str

@dataclass
class PredictedValues:
    gwp_kg_co2_eq: float
    material_circularity_indicator: float
    water_consumption_m3: float
    end_of_life_recycling_rate_percent: float
    energy_per_material_mj: float
    total_air_emissions_kg: float
    total_water_emissions_kg: float
    circularity_score: float
    potential_gwp_reduction_renewable_percent: Optional[float] = None
    potential_mci_improvement_recycling_percent: Optional[float] = None

@dataclass
class BenchmarkData:
    industry_average_gwp: float
    best_in_class_mci: float
    sector_average_water_m3: float

@dataclass
class DynamicContext:
    product_name: str
    process_route: str
    user_inputs: UserInputData
    ai_predictions: PredictedValues
    benchmarks: BenchmarkData
    
    def to_context_string(self) -> str:
        return f"""
SUSTAINABILITY ANALYSIS CONTEXT
==============================
Product: {self.product_name}
Process Route: {self.process_route}

USER-PROVIDED DATA:
- Energy Source: {self.user_inputs.energy_source}
- Transport: {self.user_inputs.transport_mode} ({self.user_inputs.transport_distance_km} km)
- Recycled Content Input: {self.user_inputs.recycled_content_percent}%
- Location: {self.user_inputs.location}
- Functional Unit: {self.user_inputs.functional_unit}
- Raw Material: {self.user_inputs.raw_material_type}
- Processing Method: {self.user_inputs.processing_method}

AI-PREDICTED PERFORMANCE METRICS:
- Global Warming Potential: {self.ai_predictions.gwp_kg_co2_eq} kg CO2-eq
- Material Circularity Indicator: {self.ai_predictions.material_circularity_indicator}
- Water Consumption: {self.ai_predictions.water_consumption_m3} m³
- End-of-Life Recycling Rate: {self.ai_predictions.end_of_life_recycling_rate_percent}%
- Energy Intensity: {self.ai_predictions.energy_per_material_mj} MJ per unit
- Total Air Emissions: {self.ai_predictions.total_air_emissions_kg} kg
- Total Water Emissions: {self.ai_predictions.total_water_emissions_kg} kg
- Overall Circularity Score: {self.ai_predictions.circularity_score}

IMPROVEMENT POTENTIALS:
- GWP Reduction (Renewable Energy): {self.ai_predictions.potential_gwp_reduction_renewable_percent or 'Not calculated'}%
- MCI Improvement (Increased Recycling): {self.ai_predictions.potential_mci_improvement_recycling_percent or 'Not calculated'}%

INDUSTRY BENCHMARKS:
- Industry Average GWP: {self.benchmarks.industry_average_gwp} kg CO2-eq
- Best-in-Class MCI: {self.benchmarks.best_in_class_mci}
- Sector Average Water Use: {self.benchmarks.sector_average_water_m3} m³
"""

# 🔧 PROMPTS (unchanged, same as your original)
ANALYST_PROMPT = """
You are a senior Life Cycle Assessment (LCA) analyst preparing a sustainability report.

Context:
{context}

Task:
1. Extract only the quantitative and qualitative environmental data found in the context. 
2. Where data is missing, make reasonable assumptions based on industry best practices and label them clearly as "Assumptions."
3. Identify and rank the **top three environmental hotspots** (e.g., GHG emissions, water use, waste generation) across the product life cycle.
4. For each hotspot, explain its relevance, potential impact, and contribution to overall sustainability performance.
5. Write in a clear, narrative style suitable for inclusion in the "Hotspots" section of a professional sustainability report.
6. Be concise but insightful — aim for 2–3 well-developed sentences per hotspot.
"""
CIRCULARITY_PROMPT = """
You are a circular economy strategist creating actionable recommendations.

Context:
{context}

Identified Hotspots:
{hotspots}

Task:
For each hotspot, propose exactly **one practical and high-impact circularity improvement strategy**.
For each strategy:
- Explain why it addresses the hotspot effectively.
- Where possible, reference a relevant real-world case study or industry benchmark to strengthen credibility.
- Keep recommendations specific and actionable (not generic statements).
Write in a professional, persuasive tone suitable for a sustainability report section titled "Circularity Recommendations."
"""

COMPLIANCE_PROMPT = """
You are a sustainability compliance officer preparing a board-level report.

Context:
{context}

Analysis & Recommendations:
{analysis_and_strategies}

Task:
1. Identify the most relevant environmental regulations (e.g., EU CBAM, REACH, water discharge regulations, ESG disclosure requirements).
2. Describe the key regulatory risks and explain why each risk matters from a financial, operational, or reputational perspective.
3. Highlight compliance opportunities (ways to reduce exposure, gain competitive advantage, or unlock incentives).
4. Provide 3–5 strategic recommendations to stay ahead of regulations and minimize business risk.
Write in a structured, concise, and executive-friendly format for a "Compliance & Risk Assessment" report section.
"""
DATA_ASSESSOR_PROMPT = """
You are a data quality analyst reviewing environmental data for an LCA.

Context:
{context}

Task:
1. Identify all missing, incomplete, or unclear data points required for a full and credible LCA.
2. Present findings in two sections:
   - "Missing Data": List the key gaps (e.g., energy by process step, specific emissions, transport details).
   - "Recommended Next Steps": Suggest what data to collect next, prioritizing by impact on decision-making and hotspot identification.
Write in plain, actionable language for a sustainability manager who is not an LCA expert.
Keep the response clear and easy to follow.
"""

SCENARIO_PROMPT = """
You are an LCA data strategist modeling decision scenarios.

Context:
{context}

Task:
1. Identify data gaps that prevent building robust "what-if" scenarios (e.g., renewable energy adoption, recycling rates, transport mode shifts).
2. Suggest which scenarios would be most impactful to model (e.g., 100% renewable energy supply, closed-loop water systems, higher recycling rate).
3. Explain how filling these gaps would improve scenario analysis and decision-making.
Write clearly for a sustainability manager looking to plan improvement projects.
"""
SUMMARY_PROMPT = """
You are an expert sustainability communicator preparing a CEO-ready executive summary.

Hotspots:
{hotspots}

Strategies:
{strategies}

Compliance & Risks:
{compliance}

Scenario Outcomes:
{scenarios}

Task:
1. Write an executive summary that distills the findings into business-relevant insights.
2. Include:
   - **Three key insights** (concise, high-impact takeaways)
   - **Most critical recommended actions** (prioritized, actionable)
   - **Urgency level** (low, medium, or high) with a short justification
   - **Business value** (cost savings, compliance advantage, market access, reputational benefit)
3. Use a persuasive and confident tone that motivates leadership to act.
4. Keep it clear, executive-friendly, and focused on strategic decision-making.
"""

def call_gemini(model, prompt, label=None):
    """Helper to send prompt to Gemini and return clean text."""
    if label:
        print(f"\n🔎 Asking Gemini: {label}...\n")
    response = model.generate_content(prompt)
    return response.text

def run_pipeline(json_file_path: str) -> Dict[str, str]:
    """Run the multi-agent workflow and return all outputs as a dictionary."""

    # 1️⃣ Load JSON input and build context
    try:
        with open(json_file_path, 'r') as f:
            json_input = json.load(f)

        user_inputs = UserInputData(**json_input['user_inputs'])
        ai_predictions = PredictedValues(**json_input['ai_predictions'])
        benchmarks = BenchmarkData(**json_input['benchmarks'])

        dynamic_data = DynamicContext(
            product_name=json_input['product_name'],
            process_route=json_input['process_route'],
            user_inputs=user_inputs,
            ai_predictions=ai_predictions,
            benchmarks=benchmarks
        )
        context_text = dynamic_data.to_context_string()

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ JSON file not found: {json_file_path}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"❌ Invalid data structure in JSON input file: {e}")

    # 2️⃣ Configure Gemini model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("⚠️ GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # 3️⃣ Run multi-agent prompts and collect outputs
    outputs = {}

    data_assessor_prompt = ChatPromptTemplate.from_template(DATA_ASSESSOR_PROMPT).format(context=context_text)
    outputs["data_gaps"] = call_gemini(model, data_assessor_prompt, label="Data Quality Assessment")

    analyst_prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT).format(context=context_text)
    outputs["hotspots"] = call_gemini(model, analyst_prompt, label="LCA Hotspot Analysis")

    circularity_prompt = ChatPromptTemplate.from_template(CIRCULARITY_PROMPT).format(context=context_text, hotspots=outputs["hotspots"])
    outputs["strategies"] = call_gemini(model, circularity_prompt, label="Circularity Recommendations")

    scenario_prompt = ChatPromptTemplate.from_template(SCENARIO_PROMPT).format(context=context_text, strategies=outputs["strategies"])
    outputs["scenarios"] = call_gemini(model, scenario_prompt, label="What-If Scenario Modeling")

    combined_analysis = f"Hotspots:\n{outputs['hotspots']}\n\nStrategies:\n{outputs['strategies']}\n\nScenarios:\n{outputs['scenarios']}"
    compliance_prompt = ChatPromptTemplate.from_template(COMPLIANCE_PROMPT).format(context=context_text, analysis_and_strategies=combined_analysis)
    outputs["compliance"] = call_gemini(model, compliance_prompt, label="Compliance & Risk Assessment")

    summary_prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT).format(
        hotspots=outputs["hotspots"],
        strategies=outputs["strategies"],
        compliance=outputs["compliance"],
        scenarios=outputs["scenarios"]
    )
    outputs["executive_summary"] = call_gemini(model, summary_prompt, label="Executive Summary")

    return outputs
def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline from CLI")
    parser.add_argument("json_file", help="Path to input JSON file")
    args = parser.parse_args()

    results = run_pipeline(args.json_file)

    print("\n✅ MULTI-AGENT PIPELINE COMPLETE ✅\n")
    for k, v in results.items():
        print(f"\n--- {k.upper()} ---\n")
        print(v)

if __name__ == "__main__":
    main()
