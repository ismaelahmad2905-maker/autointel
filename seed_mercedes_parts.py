import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Mercedes parts if you want a clean reset
Part.objects.filter(make="MERCEDES").delete()

mercedes_models = [
    "A-CLASS",
    "B-CLASS",
    "C-CLASS",
    "E-CLASS",
    "CLA",
    "GLA",
    "GLC",
    "GLE",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 78),
        ("Brake Discs", 155),
    ],
    "electrical": [
        ("Battery", 135),
        ("Alternator", 225),
    ],
    "engine": [
        ("Spark Plugs", 55),
        ("Ignition Coil", 88),
    ],
    "cooling": [
        ("Water Pump", 102),
        ("Radiator", 180),
    ],
    "transmission": [
        ("Clutch Kit", 285),
        ("Transmission Fluid Service", 125),
    ],
    "suspension": [
        ("Shock Absorber", 145),
        ("Control Arm", 170),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "A-CLASS": -10,
    "B-CLASS": -5,
    "C-CLASS": 0,
    "E-CLASS": 15,
    "CLA": 5,
    "GLA": 10,
    "GLC": 20,
    "GLE": 30,
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

for model in mercedes_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="MERCEDES",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Mercedes part records.")