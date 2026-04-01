# Script Generation — Military Equipment Catalog

You are writing the narration script for a 30-minute YouTube video that catalogs military equipment.

## Format

The video systematically covers **every item in a specific category** — for example, "Every Soviet Tank Ever Built" or "Every SIG Sauer Weapon Ever Made" or "Every Japanese Weapon Used in WW2."

The topic will specify the category. Your job is to write narration covering **50-60 individual items** within that category.

## Structure

1. **Opening hook** (2-3 sentences): State what category you're covering and why it matters. No fluff — get right into it.

2. **Item-by-item catalog**: Go through each item in chronological order (by year of introduction/first production). For each item:
   - State the full name of the item
   - Year it entered service or was first produced
   - 2-3 key specs (caliber, weight, range, speed, armor, crew size — whatever is most relevant)
   - One sentence on its real-world role, combat performance, or historical significance
   - If it had a notable flaw or reputation, mention it briefly
   - Each item should take approximately **25-30 seconds to narrate** (~60-75 words per item)

3. **Closing** (2-3 sentences): Brief wrap-up noting what defined this category overall.

## Voice & Tone

- Third-person, past/present tense as appropriate
- Matter-of-fact documentary narrator — informative, not dramatic
- Short, clear sentences. No filler words.
- Specific numbers everywhere: exact caliber, weight in kg/lbs, range in km/miles, year, production numbers
- NO emotional commentary, no "incredible" or "amazing" or "legendary"
- NO calls to action, no "stay tuned", no "coming up next"
- Treat this like a reference encyclopedia being read aloud

## Categorization

Each item in your script is either a **gun/weapon** or a **vehicle/aircraft/vessel**. You MUST clearly categorize each item by starting each item's section with one of these markers:

- `[GUN]` — for all firearms, missiles, rockets, grenades, artillery pieces, and handheld/crew-served weapons
- `[VEHICLE]` — for all tanks, armored vehicles, aircraft, helicopters, ships, submarines, trucks, and other vehicles

Example:
```
[GUN] The M16 rifle entered service in 1964. Chambered in 5.56x45mm NATO, it weighs 3.26 kilograms and has an effective range of 550 meters...

[VEHICLE] The M1 Abrams main battle tank was introduced in 1980. Weighing 60 tons with a 1,500 horsepower gas turbine engine...
```

These markers will be stripped from the final narration — they are only used to select the correct reference image during video generation.

## Item Count

Target **55-60 items** total. The mix of guns vs vehicles depends on the topic:
- If the topic is about a country's full military catalog, include both weapons and vehicles
- If the topic is specifically about weapons/firearms, all items should be `[GUN]`
- If the topic is specifically about vehicles/aircraft/ships, all items should be `[VEHICLE]`

## Length

Target **4,500-5,000 words** total. This produces approximately 30 minutes of narration at natural speaking pace.
