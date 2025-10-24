from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Optional
import random
import datetime
import os
import sys
import uuid

# Imports
from src.autofill import autofill_missing_values # src/autofill.py
from src.predict import make_prediction # src/predict.py
from src.report.report_tech import generate_report_from_dict # src/report/report_tech.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "generatedpdf")
os.makedirs(REPORTS_DIR, exist_ok=True)

class UserLogin(BaseModel):
    email: str
    password: str

class InputData(BaseModel):
    # --- Core process details ---
    process_stage: Annotated[
        Literal['Transport', 'Use', 'Manufacturing', 'End-of-Life', 'Raw Material Extraction'],
        Field(..., alias='Process Stage', description="Stage of the industrial process")
    ]
    technology: Annotated[
        Literal['Conventional', 'Emerging', 'Advanced'],
        Field(..., alias='Technology', description="Technology or method used")
    ]
    time_period: Annotated[
        Literal['2020-2025', '2015-2019', '2010-2014'],
        Field(..., alias='Time Period', description="Time period or year of operation")
    ]
    # time_period: Annotated[
    #     Optional[Literal['2020-2025', '2015-2019', '2010-2014']],
    #     Field(None, alias='Time Period', description="Time period or year of operation")
    # ]
    location: Annotated[
        Literal['South America', 'Asia', 'North America', 'Europe'],
        Field(..., alias='Location', description="Geographical location")
    ]
    functional_unit: Annotated[
        Literal['1 kg Copper Wire', '1 m2 Aluminium Panel', '1 kg Aluminium Sheet'],
        Field(..., alias='Functional Unit', description="Reference unit for assessment")
    ]

    # --- Material & energy inputs ---
    raw_material_type: Annotated[
        Literal['Aluminium Scrap', 'Aluminium Ore', 'Copper Scrap', 'Copper Ore'],
        Field(..., alias='Raw Material Type', description="Type of raw material used")
    ]
    raw_material_quantity: Annotated[
        float,
        Field(..., alias='Raw Material Quantity (kg or unit)', gt=0, description="Quantity of raw material (kg or unit)")
    ]
    energy_input_type: Annotated[
        Literal['Electricity', 'Coal', 'Natural Gas'],
        Field(..., alias='Energy Input Type', description="Type of energy input (e.g., electricity, gas)")
    ]
    energy_input_quantity: Annotated[
        float,
        Field(..., alias='Energy Input Quantity (MJ)', gt=0, description="Energy consumed (MJ)")
    ]

    # --- Processing & transport ---
    processing_method: Annotated[
        Literal['Conventional', 'Emerging', 'Advanced'],
        Field(..., alias='Processing Method', description="Processing or manufacturing method")
    ]
    transport_mode: Annotated[
        Literal['Rail', 'Truck', 'Ship'],
        Field(..., alias='Transport Mode', description="Mode of transport used")
    ]
    transport_distance: Annotated[
        float,
        Field(..., alias='Transport Distance (km)', ge=0, description="Transport distance (km)")
    ]
    fuel_type: Annotated[
        Literal['Electric', 'Diesel', 'Heavy Fuel Oil'],
        Field(..., alias='Fuel Type', description="Type of fuel used in processing or transport")
    ]

    # --- Material & cost properties ---
    metal_quality_grade: Annotated[
        Literal['Low', 'Medium', 'High'],
        Field(..., alias='Metal Quality Grade', description="Quality or grade of metal used")
    ]
    material_scarcity_level: Annotated[
        Literal['Low', 'Medium', 'High'],
        Field(..., alias='Material Scarcity Level', description="Scarcity of material")
    ]
    material_cost_usd: Annotated[
        float,
        Field(..., alias='Material Cost (USD)', ge=0, description="Cost of material in USD")
    ]
    processing_cost_usd: Annotated[
        float,
        Field(..., alias='Processing Cost (USD)', ge=0, description="Processing cost in USD")
    ]

    # --- Emissions ---
    emissions_air_co2: Annotated[Optional[float], Field(None, alias='Emissions to Air CO2 (kg)', ge=0)]
    emissions_air_sox: Annotated[Optional[float], Field(None, alias='Emissions to Air SOx (kg)', ge=0)]
    emissions_air_nox: Annotated[Optional[float], Field(None, alias='Emissions to Air NOx (kg)', ge=0)]
    emissions_air_pm: Annotated[Optional[float], Field(None, alias='Emissions to Air Particulate Matter (kg)', ge=0)]
    emissions_water_amd: Annotated[Optional[float], Field(None, alias='Emissions to Water Acid Mine Drainage (kg)', ge=0)]
    emissions_water_hm: Annotated[Optional[float], Field(None, alias='Emissions to Water Heavy Metals (kg)', ge=0)]
    emissions_water_bod: Annotated[Optional[float], Field(None, alias='Emissions to Water BOD (kg)', ge=0)]
    ghg_emissions: Annotated[Optional[float], Field(None, alias='Greenhouse Gas Emissions (kg CO2-eq)', ge=0)]

    # --- Scope emissions ---
    scope1_emissions: Annotated[Optional[float], Field(None, alias='Scope 1 Emissions (kg CO2-eq)', ge=0)]
    scope2_emissions: Annotated[Optional[float], Field(None, alias='Scope 2 Emissions (kg CO2-eq)', ge=0)]
    scope3_emissions: Annotated[Optional[float], Field(None, alias='Scope 3 Emissions (kg CO2-eq)', ge=0)]

    # --- End-of-life & environmental metrics ---
    end_of_life_treatment: Annotated[
        Literal['Incineration', 'Recycling', 'Reuse', 'Landfill'],
        Field(..., alias='End-of-Life Treatment', description="Treatment method at end of product life")
    ]
    environmental_impact_score: Annotated[
        Optional[float],
        Field(None, alias='Environmental Impact Score', ge=0, description="Overall environmental impact score")
    ]
    metal_recyclability_factor: Annotated[
        Optional[float],
        Field(None, alias='Metal Recyclability Factor', ge=0, le=1, description="Metal recyclability factor (0–1)")
    ]


app = FastAPI(title="Circular Metals API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ALL API ROUTES FIRST =====
#just for reference -these apis are not production ready and can be atatcked easily 
#before deploing need to check their security


@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/login")
async def login_user(user_login: UserLogin):
    if user_login.email.endswith("@gmail.com") and user_login.password:
        return {
            "success": True,
            "token": "real-backend-token-" + str(datetime.datetime.utcnow().timestamp()),
            "user": {"email": user_login.email},
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/assessment")
async def run_assessment(request: Request):
    payload = await request.json()
    recycled = float(payload.get("recycledContent", 50) or 0)
    
    base_circularity = 40 + (recycled * 0.45)
    if payload.get("energyUse") == "renewable": 
        base_circularity += 10
    if payload.get("transport") in ["local", "rail"]: 
        base_circularity += 5
    if payload.get("endOfLife") in ["recycle", "reuse"]: 
        base_circularity += 10

    environmental_score = 35 + (recycled * 0.5)
    if payload.get("energyUse") == "fossil": 
        environmental_score -= 15
    if payload.get("endOfLife") == "landfill": 
        environmental_score -= 20
    
    return {
        "circularityScore": max(0, min(100, round(base_circularity))),
        "environmentalScore": max(0, min(100, round(environmental_score))),
        "recommendations": random.sample([
            "Increase recycled content to reduce virgin input dependency.",
            "Optimize transport routes and prefer rail/sea logistics.",
            "Adopt closed-loop water systems in processing.",
            "Redesign product for easier disassembly and reuse.",
        ], 3),
        "missingParams": 0,
        "confidence": 95
    }

@app.get("/api/environmental-impact")
async def get_environmental_impact():
    return [
        {"name": "CO2 Emissions", "conventional": 850, "circular": 320},
        {"name": "Energy Use", "conventional": 1200, "circular": 680},
        {"name": "Water Use", "conventional": 400, "circular": 180},
        {"name": "Waste Gen.", "conventional": 200, "circular": 45},
    ]

@app.get("/api/circularity-indicators")
async def get_circularity_indicators():
    return [
        {"name": "Recycled Content", "value": 65, "target": 80},
        {"name": "Resource Efficiency", "value": 72, "target": 85},
        {"name": "Product Life Ext.", "value": 58, "target": 75},
        {"name": "Reuse Potential", "value": 43, "target": 60},
    ]

@app.get("/api/flow-data")
async def get_flow_data():
    return [
        {"stage": "Extraction", "material": 100, "recycled": 0},
        {"stage": "Processing", "material": 95, "recycled": 60},
        {"stage": "Manufacturing", "material": 90, "recycled": 85},
        {"stage": "Use", "material": 88, "recycled": 83},
        {"stage": "End-of-Life", "material": 25, "recycled": 75},
    ]

@app.get("/api/pie-data")
async def get_pie_data():
    return [
        {"name": "Recycled", "value": 45, "color": "#10b981"},
        {"name": "Virgin", "value": 35, "color": "#6366f1"},
        {"name": "Recovered", "value": 20, "color": "#f59e0b"},
    ]



@app.post("/api/reports/export")
async def export_report(input_data: InputData):
    """
    1. Autofill missing values (autofill.py)
    2. Generate predictions (predict.py)
    3. Create PDF report (report.py)
    4. Return PDF to user
    """
    try:
        # --- STEP 1: AUTOFILL ---
        autofilled_dict = autofill_missing_values(input_data.model_dump(by_alias=True))

        # --- STEP 2: PREDICT ---
        # prediction_dict should include both autofilled data + model outputs
        prediction_dict = make_prediction(autofilled_dict)

        # Step 3: Generate report
        report_filename = f"report_{uuid.uuid4().hex}.pdf"
        report_path = os.path.join(REPORTS_DIR, report_filename)

        generate_report_from_dict(prediction_dict, output_file=report_path)

        # Step 4: Return PDF file
        return FileResponse(
            path=report_path,
            media_type="application/pdf",
            filename=os.path.basename(report_path)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


# ===== STATIC FILES LAST (IMPORTANT!) =====
#these part serves the react app -make sure run buid first 
# or else white screen problem can appeaar
#--Ignore--


BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(_file_)), "my-app", "build")

if os.path.exists(BUILD_DIR):
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)