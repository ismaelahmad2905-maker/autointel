import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Ford parts if you want a clean reset
Part.objects.filter(make="FORD").delete()

ford_models = [
    "FIESTA",
    "FOCUS",
    "MONDEO",
    "KUGA",
    "ECOSPORT",
    "PUMA",
    "S-MAX",
    "GALAXY",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 55),
        ("Brake Discs", 110),
    ],
    "electrical": [
        ("Battery", 105),
        ("Alternator", 180),
    ],
    "engine": [
        ("Spark Plugs", 40),
        ("Ignition Coil", 65),
    ],
    "cooling": [
        ("Water Pump", 80),
        ("Radiator", 145),
    ],
    "transmission": [
        ("Clutch Kit", 220),
        ("Transmission Fluid Service", 95),
    ],
    "suspension": [
        ("Shock Absorber", 115),
        ("Control Arm", 135),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "FIESTA": -15,
    "FOCUS": 0,
    "MONDEO": 10,
    "KUGA": 15,
    "ECOSPORT": 5,
    "PUMA": 10,
    "S-MAX": 20,
    "GALAXY": 25,
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

for model in ford_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="FORD",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Ford part records.")