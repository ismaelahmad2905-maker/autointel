import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Lexus parts if you want a clean reset
Part.objects.filter(make="LEXUS").delete()

lexus_models = [
    "CT",
    "IS",
    "ES",
    "GS",
    "LS",
    "NX",
    "RX",
    "UX",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 70),
        ("Brake Discs", 145),
    ],
    "electrical": [
        ("Battery", 125),
        ("Alternator", 210),
    ],
    "engine": [
        ("Spark Plugs", 50),
        ("Ignition Coil", 80),
    ],
    "cooling": [
        ("Water Pump", 95),
        ("Radiator", 170),
    ],
    "transmission": [
        ("Clutch Kit", 260),
        ("Transmission Fluid Service", 115),
    ],
    "suspension": [
        ("Shock Absorber", 135),
        ("Control Arm", 160),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "CT": -10,
    "IS": 0,
    "ES": 10,
    "GS": 15,
    "LS": 30,
    "NX": 15,
    "RX": 20,
    "UX": 10,
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

for model in lexus_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="LEXUS",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Lexus part records.")