import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Audi parts if you want a clean reset
Part.objects.filter(make="AUDI").delete()

audi_models = [
    "A1",
    "A3",
    "A4",
    "A5",
    "A6",
    "Q2",
    "Q3",
    "Q5",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 72),
        ("Brake Discs", 145),
    ],
    "electrical": [
        ("Battery", 130),
        ("Alternator", 215),
    ],
    "engine": [
        ("Spark Plugs", 52),
        ("Ignition Coil", 82),
    ],
    "cooling": [
        ("Water Pump", 98),
        ("Radiator", 175),
    ],
    "transmission": [
        ("Clutch Kit", 270),
        ("Transmission Fluid Service", 120),
    ],
    "suspension": [
        ("Shock Absorber", 140),
        ("Control Arm", 165),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "A1": -10,
    "A3": 0,
    "A4": 10,
    "A5": 15,
    "A6": 20,
    "Q2": 5,
    "Q3": 15,
    "Q5": 25,
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

for model in audi_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="AUDI",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Audi part records.")