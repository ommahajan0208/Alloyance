import sys
import os

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,Image , KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ------------------------------------------------------
# Import RAG-based text generation functions
# ------------------------------------------------------
try:
    from src.rag.rag_pipline import generate_summary, generate_circularity_analysis, generate_emission_interpretation, generate_energy_efficiency_analysis, generate_benchmark_analysis, generate_action_recommendations
except ModuleNotFoundError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAG_DIR = os.path.join(BASE_DIR, "rag")
    sys.path.append(RAG_DIR)
    from rag_pipline import generate_summary, generate_circularity_analysis, generate_emission_interpretation, generate_energy_efficiency_analysis, generate_benchmark_analysis, generate_action_recommendations


# ------------------------------------------------------
# PDF STYLE HELPERS
# ------------------------------------------------------
def create_styles():
    """Create and return a set of reusable PDF styles."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.HexColor('#1f4e79'),
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2e5c8a')
    ))
    return styles


# ------------------------------------------------------
# PDF SECTION HELPERS
# ------------------------------------------------------
def add_title_page(story, styles):
    """Create and append the title page to the report."""
    story.append(Paragraph("Sustainability & Circularity Analysis Report", styles['CustomTitle']))
    story.append(Spacer(1, 50))

    report_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Prepared By:', 'AI-Powered Multi-Agent System'],
        ['Methodology:', 'Retrieval-Augmented Generation + Multi-Agent Workflow']
    ]
    report_table = Table(report_data, colWidths=[2 * inch, 3 * inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(report_table)
    story.append(PageBreak())


def add_section(story, styles, title, content):
    """Add a section with title and text."""
    story.append(Paragraph(title, styles['SectionHeader']))
    story.append(Paragraph(content if content else "No data available for this section.", styles['Normal']))
    story.append(Spacer(1, 12))


# ------------------------------------------------------
# MAIN REPORT GENERATOR (DICT → PDF)
# ------------------------------------------------------
def generate_report_from_dict(data: dict, output_file=None): #V1 - version 1 
    """Generate a sustainability report PDF directly from a Python dictionary."""
    if not isinstance(data, dict):
        raise ValueError("❌ Input must be a dictionary.")

    # ✅ Define output directory relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → ml_Alloyance/
    output_dir = os.path.join(base_dir, "generatedpdf")

    # ✅ Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Default filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sustainability_report_{timestamp}.pdf")

    styles = create_styles()
    story = []

    # Title Page
    add_title_page(story, styles)

    # Extract values
    material = data.get("Raw Material Type", "Unknown Material")
    circ_score = data.get("Circularity_Score", 0)
    recycled_content = data.get("Recycled Content (%)", 0)
    reuse_potential = data.get("Reuse Potential (%)", 0)
    recovery_rate = data.get("Recovery Rate (%)", 0)

    # RAG-based generation
    exec_summary = generate_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate)
    circ_analysis = generate_circularity_analysis(material, circ_score, recycled_content, reuse_potential, recovery_rate)

    # Add sections
    add_section(story, styles, "Executive Summary", exec_summary)
    add_section(story, styles, "Circularity Analysis", circ_analysis)
    add_section(story, styles, "Data Quality Assessment", data.get("Data Gaps"))
    add_section(story, styles, "Hotspot Analysis", data.get("Hotspot Analysis"))
    add_section(story, styles, "Circularity Strategies", data.get("Circular Strategies"))
    add_section(story, styles, "Scenario Modeling", data.get("Scenario Modeling"))
    add_section(story, styles, "Compliance & Risk Assessment", data.get("Compliance"))

    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Appendices", styles['SectionHeader']))
    story.append(Paragraph(
        "This report was automatically generated using an AI-powered multi-agent RAG pipeline. "
        "Results should be validated by domain experts before being used for critical decision-making.",
        styles['Normal']
    ))

    # Build final PDF
    doc = SimpleDocTemplate(
        output_file, pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )
    doc.build(story)

    return output_file














# Version 2 


# -------------------------
# ENHANCEMENT: Enhanced PDF layout (3 pages)
# -------------------------

def _safe_get(data, key, default="N/A"):
    v = data.get(key)
    return v if v is not None and v != "" else default

def _small_kv_table(kv_pairs, col_widths=(2.2*inch, 3.8*inch)):
    """Return a small styled table for key-value pairs."""
    t = Table(kv_pairs, colWidths=col_widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#d9e2ef')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e6eef8')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return t

def generate_report_from_dict_enhanced(data: dict, output_file=None):
    """
    Enhanced 3-page report generator.

    - Does NOT change your existing modules or imports.
    - Uses generate_summary and generate_circularity_analysis from your RAG pipeline.
    - Reads images from images/alloyance.jpg and images/ghg_emissions_histogram.png (relative to project root).
    - If Prediction_Accuracy dict is present in `data` it will be displayed; otherwise shows placeholders.
    """
    if not isinstance(data, dict):
        raise ValueError("❌ Input must be a dictionary.")

    # ---------------------------
    # Extract and initialize variables
    # ---------------------------
    material = data.get("Raw Material Type", "Unknown Material")
    circ_score = data.get("Circularity_Score", 0)
    technology = data.get("Technology", "Unknown")
    process_stage = data.get("Process Stage", "Unknown")
    energy_input_qty = data.get("Energy Input Quantity (MJ)", 0)

    # Emission-related values
    emission_co2 = data.get("Emissions to Air CO2 (kg)", 0)
    emission_sox = data.get("Emissions to Air SOx (kg)", 0)
    emission_nox = data.get("Emissions to Air NOx (kg)", 0)
    emission_pm = data.get("Emissions to Air Particulate Matter (kg)", 0)
    emission_amd = data.get("Emissions to Water Acid Mine Drainage (kg)", 0)
    emission_hm = data.get("Emissions to Water Heavy Metals (kg)", 0)
    emission_bod = data.get("Emissions to Water BOD (kg)", 0)
    ghg_emissions = data.get("Greenhouse Gas Emissions (kg CO2-eq)", 0)

    recycled_content = data.get("Recycled Content (%)", 0)
    reuse_potential = data.get("Reuse Potential (%)", 0)
    recovery_rate = data.get("Recovery Rate (%)", 0)
    

    # Keep same base_dir/output_dir logic as original generate_report_from_dict to preserve structure
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "generatedpdf")
    os.makedirs(output_dir, exist_ok=True)

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sustainability_report_enhanced_{timestamp}.pdf")

    styles = create_styles()
    # small custom styles
    styles.add(ParagraphStyle(name='CenterSubTitle', alignment=TA_CENTER, fontSize=12, spaceAfter=12))
    styles.add(ParagraphStyle(name='SmallNote', alignment=TA_LEFT, fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='StatCaption', alignment=TA_CENTER, fontSize=10, spaceAfter=8))
    styles.add(ParagraphStyle(name='MonoSmall', alignment=TA_LEFT, fontSize=8, fontName='Helvetica'))

    story = []

    # Use RAG to generate executive summary (keeps your pipeline usage)
    try:
        exec_summary = generate_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate)
    except Exception as e:
        exec_summary = ("(Error producing executive summary from RAG pipeline.)\n"
                        f"Pipeline error: {e}")
    
    # A short AI-generated circularity assessment summary (fallback to rag if available)
    try:
        circ_assessment = generate_circularity_analysis(material, circ_score, recycled_content, reuse_potential, recovery_rate)
    except Exception as e:
        circ_assessment = ("(Error producing circularity analysis from RAG pipeline.)\n"
                           f"Pipeline error: {e}")
    try:
        emission_analysis = generate_emission_interpretation(
            material,
            process_stage,
            emission_co2,
            emission_sox,
            emission_nox,
            emission_pm,
            emission_amd,
            emission_hm,
            emission_bod,
            ghg_emissions
        )
    except Exception as e:
        emission_analysis = f"(Error producing emission interpretation: {e})"

    try:
        energy_comment = generate_energy_efficiency_analysis(
            material,
            process_stage,
            technology,
            energy_input_qty
        )
    except Exception as e:
        energy_comment = f"(Error generating energy efficiency analysis: {e})"

    try:
        benchmark_text = generate_benchmark_analysis(
            material,
            circ_score,
            recycled_content,
            reuse_potential,
            recovery_rate
        )
    except Exception as e:
        benchmark_text = f"(Error generating benchmark analysis: {e})"

    try:
        recommendations_text = generate_action_recommendations(
            material,
            process_stage,
            technology,
            circ_score,
            recycled_content,
            reuse_potential,
            recovery_rate,
            ghg_emissions,
            energy_input_qty
        )
    except Exception as e:
        recommendations_text = f"(Error generating recommendations: {e})"



    # ---------- Page 1 ----------
    # Try to load logo image from images/alloyance.jpg (relative to project root)
    logo_path = os.path.join(base_dir, "images", "alloyance.jpg")
    logo_img = None
    if os.path.exists(logo_path):
        try:
            logo_img = Image(logo_path, width=1.2*inch, height=1.2*inch)
            logo_img.hAlign = 'LEFT'
        except Exception:
            logo_img = None

    # Header: logo left + centered title
    header_table_data = []
    title_para = Paragraph("Life Cycle Assessment - Report<br/><i>Circularity Assessment</i>", styles['CustomTitle'])
    left_cell = logo_img if logo_img is not None else Paragraph(" ", styles['Normal'])
    header_table_data.append([left_cell, title_para])
    header_table = Table(header_table_data, colWidths=[1.5*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (0,0), 6),
        ('RIGHTPADDING', (1,0), (1,0), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # Dynamic content lines
    material = _safe_get(data, "Raw Material Type", "Unknown Material")
    process_stage = _safe_get(data, "Process Stage", "Unknown")
    technology = _safe_get(data, "Technology", "Unknown")
    dynamic_lines = [
        Paragraph(f"<b>Material:</b> {material}", styles['Normal']),
        Paragraph(f"<b>Process Stage:</b> {process_stage}", styles['Normal']),
        Paragraph(f"<b>Technology:</b> {technology}", styles['Normal'])
    ]
    story.extend(dynamic_lines)
    story.append(Spacer(1, 12))

    # Key info table: Report generation time, Location, Functional Unit, Time Period
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    kvt = [
        ["Report Generated:", gen_time],
        ["Location:", _safe_get(data, "Location")],
        ["Functional Unit:", _safe_get(data, "Functional Unit")],
        ["Time Period:", _safe_get(data, "Time Period")]
    ]
    story.append(_small_kv_table(kvt))
    story.append(Spacer(1, 12))

    # Short descriptive note
    note = ("This report is generated using AI/ML models for LCA estimation. "
            "Results should be validated with actual measurement where possible.")
    story.append(Paragraph(note, styles['SmallNote']))
    story.append(Spacer(1, 12))

    # Small table with selected user input parameters (you can expand keys as needed)
    user_params = [
        ["Raw Material Quantity", _safe_get(data, "Raw Material Quantity (kg or unit)")],
        ["Energy Input", f"{_safe_get(data, 'Energy Input Quantity (MJ)')} {_safe_get(data,'Energy Input Type','')}".strip()],
        ["Processing Method", _safe_get(data, "Processing Method")],
        ["Transport", f"{_safe_get(data,'Transport Mode')} / {_safe_get(data,'Transport Distance (km)')} km"]
    ]
    story.append(Paragraph("Input Parameters", styles['SectionHeader']))
    story.append(_small_kv_table(user_params))

    # Adding Energy efficiency Analysis
    story.append(Paragraph("Energy Efficiency Analysis", styles['SectionHeader']))
    story.append(Paragraph(energy_comment, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(PageBreak())

    # ---------- Page 2: Executive Summary + Histogram ----------
    # Executive Summary generated by your RAG pipeline
    circ_score = data.get("Circularity_Score", 0)
    recycled_content = data.get("Recycled Content (%)", 0)
    reuse_potential = data.get("Reuse Potential (%)", 0)
    recovery_rate = data.get("Recovery Rate (%)", 0)

    
    # Insert executive summary and a compact key-metrics box (mirrors your example)
    story.append(Paragraph("Executive Summary", styles['SectionHeader']))
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 8))

    # Key metrics box (use numeric formatting)
    def _fmt_percent(x):
        try:
            return f"{float(x):.1f}%"
        except Exception:
            return _safe_get({}, "", "N/A")

    metrics_table = [
        ["Overall Circularity Score:", _fmt_percent(circ_score)],
        ["Recycled Content:", _fmt_percent(recycled_content)],
        ["Reuse Potential:", _fmt_percent(reuse_potential)],
        ["Recovery Rate:", _fmt_percent(recovery_rate)]
    ]
    story.append(_small_kv_table(metrics_table))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Circularity Assessment", styles['SectionHeader']))
    # If RAG output is long, keep it as a single paragraph block
    story.append(Paragraph(circ_assessment, styles['Normal']))
    story.append(Spacer(1, 12))

    # Statistical Distribution of Emissions Data (centered)
    story.append(Paragraph("Statistical Distribution of Emissions Data", styles['CenterSubTitle']))
    hist_img_path = os.path.join(base_dir, "images", "ghg_emissions_histogram.png")
    if os.path.exists(hist_img_path):
        try:
            hist_img = Image(hist_img_path, width=5.5*inch, height=3.5*inch)
            hist_img.hAlign = 'CENTER'
            story.append(hist_img)
        except Exception:
            story.append(Paragraph("(Could not load emissions histogram image.)", styles['Normal']))
    else:
        story.append(Paragraph("(Emissions histogram image not found at images/ghg_emissions_histogram.png)", styles['Normal']))

    # Caption under histogram (use the text user provided; keep it literal)
    caption_text = ("The histogram shows GHG emissions are mostly below 5000 kg CO₂-eq, with fewer high-emission observations. "
                    "The mean (5269.52) is higher than the median (4002.43), indicating a right-skewed distribution due to high-emission outliers.")
    story.append(Paragraph(caption_text, styles['StatCaption']))
    story.append(PageBreak())

    # Adding emission interpretation


    story.append(Paragraph("Environmental Impact Interpretation", styles['SectionHeader']))
    story.append(Paragraph(emission_analysis, styles['Normal']))
    story.append(Spacer(1, 12))


    # ---------- Page 3: Prediction Accuracy + Circularity Analysis ----------
    story.append(Paragraph("Our LCA Prediction Accuracy", styles['SectionHeader']))

    # If caller provided a dict of accuracies, display them; else show placeholders collected from data["Prediction_Accuracy"]
    accuracy_dict = data.get("Prediction_Accuracy", None)
    if isinstance(accuracy_dict, dict) and len(accuracy_dict) > 0:
        acc_rows = [["Target", "R² (score)"]]
        for k, v in accuracy_dict.items():
            acc_rows.append([k, f"{v:.3f}" if isinstance(v, (int, float)) else str(v)])
        acc_table = Table(acc_rows, colWidths=[4.2*inch, 2.0*inch])
        acc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#d0d7e6')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        story.append(acc_table)
    else:
        # Produce a table of numeric fields with "Not provided" placeholder
        # Collect numeric-looking keys from data for display
        numeric_keys = [k for k, v in data.items() if isinstance(v, (int, float)) and k not in ("Circularity_Score",)]
        if not numeric_keys:
            story.append(Paragraph("No prediction accuracy data provided. Add a dict under key 'Prediction_Accuracy' with target:R2 pairs.", styles['SmallNote']))
        else:
            acc_rows = [["Target", "R² (score)"]]
            for k in numeric_keys:
                acc_rows.append([k, "Not provided"])
            story.append(Table(acc_rows, colWidths=[4.2*inch, 2.0*inch], style=TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#d0d7e6')),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
            ])))

    story.append(Spacer(1, 12))
    # Circularity Analysis (use RAG output to satisfy "must be generated by our rag architecture")
    story.append(Paragraph("Circularity Analysis", styles['SectionHeader']))
    story.append(Paragraph(circ_assessment, styles['Normal']))
    story.append(Spacer(1, 6))

    # Benchmark analysis

    story.append(Paragraph("Benchmark Comparison", styles['SectionHeader']))
    story.append(Paragraph(benchmark_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Add a short structured "Material Flow" style block with the numbers you requested, using actual numeric values when available
    try:
        # Try to format percentages from provided numbers
        recycled_pct = f"{float(recycled_content):.1f}%" if isinstance(recycled_content, (int, float)) else _safe_get({}, "", "N/A")
        reuse_pct = f"{float(reuse_potential):.1f}%" if isinstance(reuse_potential, (int, float)) else _safe_get({}, "", "N/A")
        recovery_pct = f"{float(recovery_rate):.1f}%" if isinstance(recovery_rate, (int, float)) else _safe_get({}, "", "N/A")
    except Exception:
        recycled_pct = reuse_pct = recovery_pct = "N/A"

    material_flow_lines = [
        Paragraph(f"<b>Material Flow</b>", styles['Normal']),
        Paragraph(f"Recycled Inputs: {recycled_pct} of {material} comes from recycled sources → less virgin mining needed.", styles['Normal']),
        Paragraph(f"Reuse Potential: {reuse_pct} of products/components can be reused → longer product life.", styles['Normal']),
        Paragraph(f"Recovery Rate: {recovery_pct} of materials recovered at end-of-life → but more than half still lost.", styles['Normal'])
    ]
    for p in material_flow_lines:
        story.append(p)
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))
    # Circular indicators and improvement suggestions
    circ_indicators = [
        ["Material Retention:", _safe_get(data, "Material Retention", "53.4%")],
        ["Circularity Index:", f"{circ_score:.1f}%"],
        ["Pathways:", "Circular model (reuse + recycle) outperforms linear model"]
    ]
    story.append(_small_kv_table(circ_indicators))
    story.append(Spacer(1, 8))

    # AI recommendation

    story.append(Paragraph("AI Recommendations", styles['SectionHeader']))
    story.append(Paragraph(recommendations_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Page 4 / Final disclaimer-appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix", styles['SectionHeader']))
    story.append(Paragraph(
        "This enhanced report was auto-generated using your RAG-based multi-agent pipeline. "
        "Please validate metrics and predictions with domain experts and measured data when possible.",
        styles['Normal']
    ))

    # Build final PDF
    doc = SimpleDocTemplate(output_file, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    doc.build(story)
    return output_file

# Optional convenience wrapper for backwards compatibility in interactive runs:
# def generate_report_from_dict_enhanced_main(data: dict):
#     """Convenience: call this from __main__ or other scripts to produce enhanced PDF."""
#     return generate_report_from_dict_enhanced(data)














# ------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------
if __name__ == "__main__":
    test_data = {
    "Process Stage": "Manufacturing",
    "Technology": "Emerging",
    "Time Period": "2020-2025",
    "Location": "Asia",
    "Functional Unit": "1 kg Aluminium Sheet",
    "Raw Material Type": "Aluminium Scrap",
    "Raw Material Quantity (kg or unit)": 100.0,
    "Energy Input Type": "Electricity",
    "Energy Input Quantity (MJ)": 250.0,
    "Processing Method": "Advanced",
    "Transport Mode": "Truck",
    "Transport Distance (km)": 300.0,
    "Fuel Type": "Diesel",
    "Metal Quality Grade": "High",
    "Material Scarcity Level": "Medium",
    "Material Cost (USD)": 500.0,
    "Processing Cost (USD)": 200.0,
    "Emissions to Air CO2 (kg)": 3081.47509765625,
    "Emissions to Air SOx (kg)": 23.762849807739258,
    "Emissions to Air NOx (kg)": 19.012903213500977,
    "Emissions to Air Particulate Matter (kg)": 11.879419326782227,
    "Emissions to Water Acid Mine Drainage (kg)": 5.273314476013184,
    "Emissions to Water Heavy Metals (kg)": 3.1632373332977295,
    "Emissions to Water BOD (kg)": 2.1091058254241943,
    "Greenhouse Gas Emissions (kg CO2-eq)": 5035.37841796875,
    "Scope 1 Emissions (kg CO2-eq)": 2504.817626953125,
    "Scope 2 Emissions (kg CO2-eq)": 1503.769775390625,
    "Scope 3 Emissions (kg CO2-eq)": 1044.5374755859375,
    "End-of-Life Treatment": "Recycling",
    "Environmental Impact Score": 56.69657897949219,
    "Metal Recyclability Factor": 0.5509214401245117,
    "Energy_per_Material": 11.394341468811035,
    "Total_Air_Emissions": 237.7609100341797,
    "Total_Water_Emissions": 10.54836368560791,
    "Transport_Intensity": 8.591459274291992,
    "GHG_per_Material": 5.105621337890625,
    "Time_Period_Numeric": 2017.448974609375,
    "Total_Cost": 780.7816162109375,
    "Circularity_Score": 44.7951545715332,
    "Circular_Economy_Index": 0.46311068534851074,
    "Recycled Content (%)": 70.25933837890625,
    "Resource Efficiency (%)": 69.86246490478516,
    "Extended Product Life (years)": 20.351980209350586,
    "Recovery Rate (%)": 87.98226165771484,
    "Reuse Potential (%)": 27.105358123779297
    }

    try:
        # path = generate_report_from_dict(test_data)
        path = generate_report_from_dict_enhanced(test_data)
        print(f" Report generated successfully: {path}")
    except Exception as e:
        print(f" Error generating report: {e}")
