import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Part

# Clear old BMW parts if you want a clean reset
Part.objects.filter(make="BMW").delete()

bmw_models = [
    "1 SERIES",
    "2 SERIES",
    "3 SERIES",
    "4 SERIES",
    "5 SERIES",
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
]

years = [2005, 2010, 2015, 2020, 2025]

base_parts = {
    "braking": [
        ("Front Brake Pads", 85),
        ("Brake Discs", 170),
    ],
    "electrical": [
        ("Battery", 150),
        ("Alternator", 240),
    ],
    "engine": [
        ("Spark Plugs", 60),
        ("Ignition Coil", 90),
    ],
    "cooling": [
        ("Water Pump", 110),
        ("Radiator", 190),
    ],
    "transmission": [
        ("Clutch Kit", 320),
        ("Transmission Fluid Service", 140),
    ],
    "suspension": [
        ("Shock Absorber", 160),
        ("Control Arm", 180),
    ],
}

# Small model-based price adjustments
model_adjustments = {
    "1 SERIES": -10,
    "2 SERIES": -5,
    "3 SERIES": 0,
    "4 SERIES": 10,
    "5 SERIES": 20,
    "X1": 10,
    "X2": 15,
    "X3": 20,
    "X4": 25,
    "X5": 40,
}

# Small year-based price adjustments
year_adjustments = {
    2005: -10,
    2010: -5,
    2015: 0,
    2020: 10,
    2025: 20,
}

count = 0

for model in bmw_models:
    for year in years:
        model_adj = model_adjustments[model]
        year_adj = year_adjustments[year]

        for category, parts in base_parts.items():
            for part_name, base_price in parts:
                final_price = base_price + model_adj + year_adj

                Part.objects.create(
                    category=category,
                    make="BMW",
                    model=model,
                    year=year,
                    part_name=part_name,
                    avg_cost_gbp=final_price,
                )
                count += 1

print(f"Inserted {count} BMW part records.")