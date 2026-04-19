import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Volkswagen parts if you want a clean reset
Part.objects.filter(make="VOLKSWAGEN").delete()

vw_models = [
    "POLO",
    "GOLF",
    "PASSAT",
    "TIGUAN",
    "TOUAREG",
    "T-ROC",
    "TOURAN",
    "SCIROCCO",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 65),
        ("Brake Discs", 130),
    ],
    "electrical": [
        ("Battery", 120),
        ("Alternator", 200),
    ],
    "engine": [
        ("Spark Plugs", 50),
        ("Ignition Coil", 75),
    ],
    "cooling": [
        ("Water Pump", 95),
        ("Radiator", 170),
    ],
    "transmission": [
        ("Clutch Kit", 250),
        ("Transmission Fluid Service", 110),
    ],
    "suspension": [
        ("Shock Absorber", 130),
        ("Control Arm", 150),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "POLO": -15,
    "GOLF": 0,
    "PASSAT": 10,
    "TIGUAN": 15,
    "TOUAREG": 35,
    "T-ROC": 10,
    "TOURAN": 5,
    "SCIROCCO": 10,
}

# Year-based price adjustments
year_adjustments = {
    2005: -10,
    2010: -5,
    2015: 0,
    2020: 10,
    2025: 20,
}

count = 0

for model in vw_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="VOLKSWAGEN",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Volkswagen part records.")