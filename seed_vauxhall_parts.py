import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Vauxhall parts if you want a clean reset
Part.objects.filter(make="VAUXHALL").delete()

vauxhall_models = [
    "CORSA",
    "ASTRA",
    "INSIGNIA",
    "MOKKA",
    "CROSSLAND",
    "GRANDLAND",
    "ZAFIRA",
    "VIVARO",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 50),
        ("Brake Discs", 100),
    ],
    "electrical": [
        ("Battery", 95),
        ("Alternator", 170),
    ],
    "engine": [
        ("Spark Plugs", 38),
        ("Ignition Coil", 60),
    ],
    "cooling": [
        ("Water Pump", 75),
        ("Radiator", 135),
    ],
    "transmission": [
        ("Clutch Kit", 210),
        ("Transmission Fluid Service", 90),
    ],
    "suspension": [
        ("Shock Absorber", 105),
        ("Control Arm", 125),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "CORSA": -10,
    "ASTRA": 0,
    "INSIGNIA": 10,
    "MOKKA": 10,
    "CROSSLAND": 5,
    "GRANDLAND": 15,
    "ZAFIRA": 10,
    "VIVARO": 20,
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

for model in vauxhall_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="VAUXHALL",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Vauxhall part records.")