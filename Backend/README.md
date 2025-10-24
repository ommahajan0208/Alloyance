# Alloyance - AI-Driven Life Cycle Assessment Tool

Machine learning backend for LCAlloyance project — includes model training, evaluation, and deployment workflows.

## Overview

**Alloyance** is an intuitive, AI-powered software platform that enables metallurgists, engineers, and decision-makers to perform automated Life Cycle Assessments (LCA) for metals such as aluminium, copper, and critical minerals, with a special emphasis on **circularity** and **sustainability**.

Traditional linear models in metallurgy and mining often result in resource depletion and waste. Alloyance transforms this approach by assessing not only emissions and resource use but also the potential for reuse, recycling, and closed-loop systems, enabling a shift toward a **circular economy**.

## Problem Statement

### Background

As industries increasingly emphasize sustainability, **Life Cycle Assessment (LCA)** is emerging not just as a tool for measuring environmental impact, but as a key strategy for advancing circularity. Metals like aluminium and copper, vital to sectors from energy to infrastructure, undergo multiple stages — from extraction and processing to use and end-of-life. 

Traditional linear models often result in resource depletion and waste. Modern LCA frameworks now assess not only emissions and resource use, but also the potential for reuse, recycling, and closed-loop systems. By informing decisions on product design, manufacturing, and end-of-life recovery, LCA enables a shift toward a **circular economy** where materials are kept in use longer and waste is minimized.

### Challenge

There is a need to design an **intuitive, AI-powered software platform** that enables metallurgists, engineers, and decision-makers to perform automated LCAs for metals such as aluminium, copper, or critical minerals, with a special emphasis on **circularity**.

## Solution

### Core Capabilities

The platform provides the following capabilities:

- **Input Flexibility**: Allow users to input or select process and production details including raw vs. recycled routes, energy use, transport, and end-of-life options
- **AI-Powered Prediction**: Use AI/ML models to estimate missing parameters and predict both environmental and circularity indicators
- **Visualization**: Visualize circular flow opportunities alongside environmental impacts across the full value chain — from raw material extraction through reuse or recycling
- **Pathway Comparison**: Enable easy comparison of conventional and circular processing pathways
- **Actionable Insights**: Generate actionable reports and recommendations for reducing impacts and enhancing circularity, even when users have limited data or specialized expertise

### Circularity Indicators

- Recycled content percentage
- Resource efficiency scores
- Extended product life potential
- Recovery rates
- Reuse potential

## Repository Structure

```
ml_Alloyance/
├── .ipynb_checkpoints/          # Jupyter notebook checkpoints
├── data/                        # Dataset storage and generation
│   ├── rag_data/               # Research papers and reference materials
│   │   └── Sanjuan-Delmsetal.2022.pdf
│   ├── DummyDatasetGeneration.py
│   └── lca_dataset.csv         # LCA training and testing data
├── images/                      # Generated visualization plots
│   ├── co2_vs_nox_scatter.png
│   ├── energy_vs_ghg.png
│   ├── ghg_emissions_histogram.png
│   └── ghg_emissions_trend.png
├── model/                       # Trained ML models and encoders
│   ├── model_*.pkl             # XGBoost circularity prediction models
│   ├── label_encoders.pkl      # Categorical feature encoders
│   ├── rf_imputer.pkl          # Random Forest imputation model
│   └── xgb_imputer.pkl         # XGBoost imputation model
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── DataExploration.ipynb   # Initial data analysis
│   └── main.ipynb              # Main workflow notebook
├── src/                         # Core application source code
│   ├── rag/                    # RAG pipeline implementation
│   │   ├── chroma/            # ChromaDB vector storage
│   │   ├── create_database.py # Database initialization
│   │   ├── rag_pipline.py     # RAG pipeline logic
│   │   └── .env               # Environment configuration
│   ├── api.py                  # Backend API endpoints
│   ├── predict.py              # Prediction logic
│   ├── autofill.py             # Automated data completion
│   └── report_tech.py          # Report generation
├── .gitattributes              # Git LFS configuration for pkl files
└── app.py                       # Main application entry point
```

## Technical Architecture

### Machine Learning Pipeline

#### 1. Data Processing
- **Comprehensive Dataset**: LCA dataset covering complete lifecycle stages (Raw Material Extraction, Manufacturing, Transport, Use, End-of-Life) with 40+ features including process parameters, emissions, and circularity metrics
- **Multi-dimensional Features**: 
  - Process characteristics (stage, technology type, time period, location)
  - Material specifications (type, quantity, quality grade, scarcity level)
  - Energy and transport data (input types, quantities, distances, fuel types)
  - Environmental emissions (CO₂, SOx, NOx, particulate matter, water pollutants)
  - Cost factors (material cost, processing cost)
  - Circularity indicators (recycled content, resource efficiency, recovery rates)
- **Feature Engineering**: Label encoding for categorical variables (process stage, technology, location, etc.) and scaling of numerical features
- **Advanced Imputation**: MICE (Multiple Imputation by Chained Equations) technique using Random Forest and XGBoost algorithms for handling missing data

#### 2. Prediction Models
Five specialized XGBoost models trained for circularity metrics:
- **Recycled Content Model**: Predicts proportion of recycled materials in production
- **Resource Efficiency Model**: Estimates optimization of material and energy use
- **Extended Product Life Model**: Forecasts potential for product longevity
- **Recovery Rate Model**: Calculates end-of-life material recovery potential
- **Reuse Potential Model**: Assesses opportunities for direct reuse

#### 3. RAG Pipeline
- **Vector Database**: ChromaDB for efficient similarity search
- **Document Processing**: Extraction and chunking of research papers
- **Context-Aware Responses**: Retrieval-Augmented Generation for evidence-based recommendations

### Environmental Metrics Tracked

**Emissions to Air:**
- CO₂ Emissions (kg)
- SOx Emissions (kg)
- NOx Emissions (kg)
- Particulate Matter (kg)

**Emissions to Water:**
- Acid Mine Drainage (kg)
- Heavy Metals (kg)
- BOD - Biological Oxygen Demand (kg)

**Greenhouse Gas Accounting:**
- Total GHG Emissions (kg CO₂-eq)
- Scope 1 Emissions (direct emissions from owned/controlled sources)
- Scope 2 Emissions (indirect emissions from purchased energy)
- Scope 3 Emissions (value chain emissions)

**Resource Metrics:**
- Energy Consumption (MJ)
- Raw Material Quantity (kg)
- Water Usage (implicit in water emissions)
- Transport Distance (km)

**Economic Metrics:**
- Material Cost (USD)
- Processing Cost (USD)
- Environmental Impact Score (0-100)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git LFS (for large model files)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LCAlloyance/ml_Alloyance.git
cd ml_Alloyance
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cd src/rag
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Initialize the RAG database (if needed):
```bash
python src/rag/create_database.py
```

### Running the Application

```bash
python app.py
```

## Usage Workflow

### Step 1: Input Production Details
Specify metal production process parameters:
- **Process Stage**: Raw Material Extraction, Manufacturing, Transport, Use, or End-of-Life
- **Metal Type**: Aluminium (ore/scrap), Copper (ore/scrap), or other critical minerals
- **Production Route**: Primary extraction vs. recycled materials
- **Technology Type**: Conventional, Emerging, or Advanced
- **Location**: Geographic region (Asia, Europe, North America, South America)
- **Functional Unit**: 1 kg Copper Wire, 1 kg Aluminium Sheet, 1 m² Aluminium Panel
- **Energy Sources**: Coal, Natural Gas, Electricity, or hybrid
- **Energy Consumption**: Quantity in MJ
- **Transport Details**: Mode (truck, rail, ship), distance (km), fuel type
- **Material Quality**: Low, Medium, or High grade
- **Material Scarcity**: Low, Medium, or High
- **End-of-Life Treatment**: Recycling, Reuse, Incineration, or Landfill
- **Time Period**: 2010-2014, 2015-2019, or 2020-2025

### Step 2: Automated Parameter Completion
The AI models automatically estimate missing parameters using:
- Historical data patterns
- MICE Technique

### Step 3: Circularity and Environmental Analysis
View comprehensive metrics including:
- Environmental impact across lifecycle stages
- Circularity performance indicators
- Resource flow visualization
- Hotspot identification

### Step 4: Pathway Comparison
Compare different scenarios:
- Conventional linear production
- Circular economy alternatives
- Hybrid approaches
- Optimization opportunities

### Step 5: Report Generation
Generate detailed reports with:
- Executive summary
- Detailed analysis by lifecycle stage
- Circularity recommendations
- Environmental impact reduction strategies
- Evidence-based best practices from literature

## Technology Stack

**Machine Learning & Data Science**
- XGBoost: Gradient boosting for prediction models
- Scikit-learn: Model training and preprocessing
- Pandas: Data manipulation and analysis
- NumPy: Numerical computing

**Visualization**
- Matplotlib: Static plotting
- Seaborn: Statistical visualization

**RAG & Knowledge Management**
- ChromaDB: Vector database for embeddings
- LangChain: RAG pipeline orchestration

**Development Tools**
- Jupyter: Interactive notebooks for exploration
- Python: Core programming language

## Model Performance

The trained models provide:
- Automated parameter estimation with high accuracy
- Missing data imputation based on contextual patterns
- Circularity metric predictions aligned with industry standards
- Environmental impact assessments based on validated LCA methodologies

## Impact

With this tool, the metals sector will be empowered to:
- Make practical, data-driven choices for sustainability
- Advance circular, resource-efficient production systems
- Reduce environmental footprint across the value chain
- Identify opportunities for waste minimization and material recovery
- Support evidence-based decision-making with limited expertise

## Contact

**Organization**: LCAlloyance  
**Repository**: [https://github.com/LCAlloyance/ml_Alloyance](https://github.com/LCAlloyance/ml_Alloyance)  
**Maintainer**: @ommahajan0208

---

*Empowering the metals sector to make practical, data-driven choices that foster environmental sustainability while advancing circular, resource-efficient systems.*
