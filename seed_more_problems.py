import os
import django
import random

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Problem

# Optional: clear old data if you want a clean dataset
# Problem.objects.all().delete()

verbs = [
    "is", "feels", "sounds", "seems", "keeps", "starts", "becomes"
]

time_phrases = [
    "", "when driving", "when stopping", "at low speed", "at high speed",
    "in traffic", "after a few minutes", "when accelerating", "when braking"
]

# Category-specific keywords
categories = {
    "braking": [
        "brakes squeaking", "brakes grinding", "brake pedal soft",
        "car pulling when braking", "brakes vibrating", "long stopping distance",
        "brakes making noise", "brake warning light", "burning smell from brakes"
    ],
    "cooling": [
        "engine overheating", "coolant leaking", "temperature too high",
        "steam from engine", "coolant dropping", "engine running hot",
        "radiator leaking", "cooling fan not working", "heater not working"
    ],
    "electrical": [
        "battery dying", "car not starting", "lights flickering",
        "dashboard lights on", "electrical faults", "battery not charging",
        "radio cutting out", "power loss", "alternator issue"
    ],
    "engine": [
        "engine misfiring", "rough idle", "loss of power",
        "engine stalling", "engine knocking", "poor acceleration",
        "engine shaking", "engine hesitation", "check engine light"
    ],
    "transmission": [
        "gears slipping", "hard to change gears", "clutch feels loose",
        "gearbox jerking", "burning smell from gearbox", "gear delay",
        "transmission slipping", "gear grinding", "gear engagement issues"
    ],
    "suspension": [
        "knocking over bumps", "car unstable", "steering vibrating",
        "clunking noise", "uneven tyre wear", "car pulling",
        "suspension loose", "bouncing too much", "rough ride"
    ]
}

extra_descriptions = [
    "", "badly", "a lot", "randomly", "sometimes", "constantly",
    "more than usual", "recently", "all the time"
]

generated = []

# Generate MANY variations
for category, phrases in categories.items():
    for base in phrases:
        for _ in range(25):  # increase tnumber for more data
            verb = random.choice(verbs)
            time = random.choice(time_phrases)
            extra = random.choice(extra_descriptions)

            sentence = f"{base} {verb} {extra} {time}".strip()
            sentence = " ".join(sentence.split())  # clean spacing

            generated.append((sentence.lower(), category))


# Insert into DB
count = 0
for text, category in generated:
    Problem.objects.create(problem_text=text, category=category)
    count += 1

print(f"Inserted {count} generated problem records.")
print(f"Total Problem rows now: {Problem.objects.count()}")