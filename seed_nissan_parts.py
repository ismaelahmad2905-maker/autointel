import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old Nissan parts if you want a clean reset
Part.objects.filter(make="NISSAN").delete()

nissan_models = [
    "MICRA",
    "JUKE",
    "QASHQAI",
    "X-TRAIL",
    "NOTE",
    "LEAF",
    "NAVARA",
    "370Z",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 58),
        ("Brake Discs", 115),
    ],
    "electrical": [
        ("Battery", 110),
        ("Alternator", 185),
    ],
    "engine": [
        ("Spark Plugs", 42),
        ("Ignition Coil", 68),
    ],
    "cooling": [
        ("Water Pump", 82),
        ("Radiator", 150),
    ],
    "transmission": [
        ("Clutch Kit", 225),
        ("Transmission Fluid Service", 100),
    ],
    "suspension": [
        ("Shock Absorber", 118),
        ("Control Arm", 138),
    ],
}

# Model-based price adjustments
model_adjustments = {
    "MICRA": -15,
    "JUKE": 0,
    "QASHQAI": 10,
    "X-TRAIL": 20,
    "NOTE": -5,
    "LEAF": 10,
    "NAVARA": 25,
    "370Z": 30,
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

for model in nissan_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="NISSAN",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} Nissan part records.")