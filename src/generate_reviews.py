"""Generate a synthetic app-store review dataset for a FICTIONAL smart-home app.

The product ("NestMate") is entirely made up for this project -- it is not a
real company or product. Reviews are synthetic, generated from randomized
phrase templates with a fixed random seed so the dataset is reproducible.

Each review carries ground-truth labels:
    - is_defect: 1 if the review reports a software defect, else 0
    - category: one of the defect categories below, or a non-defect type

Run:  python src/generate_reviews.py   (from the project root)
Writes: data/reviews.csv
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
N_REVIEWS = 1300
PRODUCT = "NestMate"  # fictional smart-home app -- not a real company

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = [
    "thermostat", "video doorbell", "security camera", "smart lock",
    "living room bulb", "kitchen plug", "bedroom speaker", "garage sensor",
    "smoke detector", "smart plug",
]

VERSIONS = ["3.2.0", "3.2.1", "3.3.0", "3.3.2", "3.4.0", "3.4.1"]

# ---------------- defect template pools (category -> templates) ----------------
# {dev} / {ver} get filled in; slots pick extra flavour words.

CRASH = [
    "App keeps crashing every time I open the {dev} controls on version {ver}.",
    "{dev} screen just force-closes the app. Completely unusable since {ver}.",
    "Crash after crash. The app dies the moment I tap my {dev}.",
    "Opening the {dev} tab makes the app freeze then crash. Please fix this.",
    "The latest update crashes on launch whenever my {dev} is connected.",
    "App crashes when I try to view the {dev} live feed. Was fine before {ver}.",
    "Every time I adjust the {dev} settings the app force quits. So frustrating.",
    "Instant crash when switching between the {dev} and the dashboard.",
    "The app won't stay open for more than a few seconds if the {dev} is online.",
    "Crashes constantly since the {ver} update. My {dev} is basically bricked in the app.",
    "Tapping notifications for the {dev} crashes the app every single time.",
    "App force closes during {dev} setup on the very last step. Unacceptable.",
    "Background crash loop drained my patience -- {dev} alerts kill the app.",
    "Rotated my phone while on the {dev} page and the app crashed instantly.",
    "Crash report sent three times this week, all from the {dev} screen.",
]

BATTERY = [
    "This app eats my phone battery alive -- lost 40% overnight with the {dev} idle.",
    "Since {ver}, the app drains battery even in the background. {dev} polling is out of control.",
    "Battery usage doubled after the update. The app keeps the {dev} connection awake nonstop.",
    "My phone gets hot and the battery tanks whenever the {dev} is armed.",
    "Background battery drain is ridiculous. Turned off {dev} notifications and it still drains.",
    "The app used 25% of my battery in 3 hours just syncing the {dev}.",
    "Woke up to a dead phone. The app was top of battery stats because of the {dev}.",
    "Battery saver mode does nothing -- the {dev} stream keeps running in the background.",
    "Massive battery drain on {ver}. Had to uninstall to get through the day.",
    "Phone battery dropped 15% during a 20-minute {dev} live view. Fix the power usage.",
    "The widget alone drains my battery keeping the {dev} status updated.",
    "Constant location polling for the {dev} geofence is killing my battery.",
]

CONNECTIVITY = [
    "My {dev} shows offline in the app constantly, but my Wi-Fi is fine.",
    "App can't find the {dev} half the time. Re-pairing works for a day then it drops again.",
    "Connection keeps dropping -- the {dev} goes 'unreachable' every few hours.",
    "Since {ver}, the app loses connection to the {dev} and never reconnects on its own.",
    "The {dev} pairs fine in Bluetooth settings but the app says device not found.",
    "'Device offline' error all day for my {dev} even though it's two feet from the router.",
    "Sync fails repeatedly with the {dev}. Spinning loader, then 'connection timeout'.",
    "App disconnects from the {dev} whenever my phone switches between Wi-Fi and data.",
    "Firmware update for the {dev} failed halfway and now the app won't connect at all.",
    "The {dev} works in the old app but this version {ver} can't maintain a connection.",
    "Intermittent connectivity: {dev} responds maybe one time out of three.",
    "Had to factory reset the {dev} three times this month because the app loses it.",
]

UI_BUG = [
    "The {dev} temperature shows in Fahrenheit even though I set Celsius in settings.",
    "Dark mode is broken on the {dev} page -- white text on a white background.",
    "Buttons on the {dev} schedule screen overlap so I can't tap save.",
    "The {dev} history graph renders blank after midnight. Graphs worked in the old version.",
    "Toggle switches on the {dev} page don't reflect the real state. Confusing.",
    "Text is cut off on the {dev} details screen on my small phone.",
    "The back button from the {dev} settings dumps me to the login screen instead of home.",
    "Slider for the {dev} jumps around and won't land on the value I pick.",
    "Icons for the {dev} are missing -- just empty boxes on {ver}.",
    "The {dev} status card shows yesterday's data until I force-refresh the app.",
    "Notification badges never clear for the {dev}, even after I read everything.",
]

PERF = [
    "App takes forever to load the {dev} dashboard -- 20+ seconds of spinner.",
    "Everything is sluggish since {ver}. Opening the {dev} feed lags badly.",
    "Scrolling the {dev} event history stutters and freezes the whole app.",
    "Live view from the {dev} lags 10 seconds behind real time. Useless for a doorbell.",
    "The app is painfully slow to arm/disarm the {dev}. Takes three tries.",
    "Search in the {dev} history times out on anything older than a week.",
    "App startup is so slow now that the {dev} alerts arrive before the app opens.",
    "Switching rooms with the {dev} takes ages -- the UI just hangs.",
    "Video from the {dev} buffers constantly even on fast Wi-Fi.",
    "Every tap has a delay on {ver}. The {dev} controls feel like they're on dial-up.",
]

AUTH = [
    "Can't log in since {ver} -- 'invalid credentials' even though my password is right.",
    "The app logs me out every day and I have to sign back in to see my {dev}.",
    "Two-factor code never arrives, so I'm locked out of controlling my {dev}.",
    "Login screen loops forever: enter password, spinner, back to login.",
    "Face ID stopped working with the app on {ver}. Have to type the password each time.",
    "Password reset email never comes. Can't access my {dev} at all now.",
    "App says my session expired in the middle of {dev} setup and lost my progress.",
    "Can't stay signed in on two phones -- the second login kicks out the first.",
    "Biometric login crashes back to the password field on {ver}.",
    "'Account not found' error on login, but the website works fine with the same email.",
]

DEFECT_TEMPLATES = {
    "crash": CRASH,
    "battery_drain": BATTERY,
    "connectivity": CONNECTIVITY,
    "ui_bug": UI_BUG,
    "performance": PERF,
    "login_auth": AUTH,
}

# ---------------- non-defect template pools ----------------

PRAISE = [
    "Love this app! My {dev} setup took five minutes and everything just works.",
    "The new {ver} update made the {dev} controls so much snappier. Great work.",
    "Best smart home app I've tried. The {dev} automations are rock solid.",
    "Setup for the {dev} was painless and the interface is beautiful.",
    "Five stars. The {dev} routines run perfectly every morning.",
    "Really impressed with how reliable the {dev} connection is.",
    "The app keeps getting better. My {dev} has never been easier to manage.",
    "Clean design, fast, and the {dev} widgets are super handy.",
    "Switched from a competitor and the {dev} experience here is miles ahead.",
    "Everything about the {dev} integration feels polished. Highly recommend.",
    "The {dev} schedules are flexible and the app never misses a beat.",
    "Simple, fast, reliable. Exactly what I want for my {dev}.",
]

FEATURE_REQUEST = [
    "Please add a dark mode schedule for the {dev} page. Would love auto-switching at sunset.",
    "It would be great if the app supported widgets for the {dev} on the lock screen.",
    "Feature request: let me export {dev} history to CSV for my energy audit.",
    "Can you add guest access for the {dev}? I want my family to control it without my login.",
    "Would love Apple Watch support to check the {dev} quickly.",
    "Please add multi-home support -- I have a {dev} at two houses.",
    "It'd be nice to get a weekly summary email of my {dev} activity.",
    "Request: IFTTT-style custom triggers for the {dev}. The presets are too limited.",
    "Please add a vacation mode that randomizes the {dev} schedule.",
    "Would love voice control shortcuts for the {dev} beyond the basics.",
]

QUESTION = [
    "How do I share {dev} access with my partner? Can't find the option in {ver}.",
    "Does the {dev} work without Wi-Fi? Wondering about offline mode.",
    "Question: can I rename the {dev} after setup? Don't see an edit option.",
    "Is there a way to get push notifications only for the {dev} and not the other devices?",
    "How do I reset the {dev} energy stats? They seem off this month.",
    "Does anyone know if the app supports the older {dev} model from 2021?",
    "Where do I find the firmware version of my {dev} in the app?",
    "Can the {dev} trigger routines based on sunrise/sunset times?",
]

NON_DEFECT = {"praise": PRAISE, "feature_request": FEATURE_REQUEST, "question": QUESTION}

# Harder examples: unambiguous labels but lexically tricky.
# Softened defects read positive yet still report a real defect (label: defect).
# Each template is paired with its TRUE category so labels stay clean.
SOFTENED_DEFECT = [
    ("crash", "I love this app overall, but it crashes every time I open the {dev} controls."),
    ("connectivity", "Great app otherwise -- however my {dev} keeps showing offline since {ver}."),
    ("battery_drain", "Five stars for the features, but the battery drain from the {dev} sync is brutal."),
    ("login_auth", "Really like the design, but logging in has been broken for me since {ver}."),
    ("performance", "Best smart home app I've used, if only the {dev} page didn't lag so badly."),
    ("performance", "Love the automations, but the {dev} live view buffers constantly."),
    ("crash", "The app is great when it works -- sadly the {dev} screen crashes on launch."),
    ("ui_bug", "Huge fan of the product, but the UI on the {dev} page is glitchy and unreadable."),
]
# Negated praise: uses defect vocabulary in negated form (label: non-defect).
NEGATED_PRAISE = [
    "Zero crashes so far and no battery drain. The {dev} just works.",
    "No login issues, no connectivity drops -- the {dev} has been rock solid.",
    "Haven't seen a single crash since {ver}. The {dev} controls are fast and reliable.",
    "No lag, no freezes, no battery drain. Really happy with the {dev} integration.",
    "Never had the app crash on me once. The {dev} setup was smooth.",
    "No offline errors at all -- the {dev} stays connected all day.",
]
# Genuinely ambiguous reviews: strong counter-sentiment, but the label follows
# the dominant clause. Realistic label noise -- these are the ones humans argue about.
AMBIGUOUS_DEFECT = [
    ("crash", "The app crashes on my {dev} screen. Support was great though and sent a workaround."),
    ("connectivity", "Love the new dashboard design, but my {dev} won't stay connected since {ver}."),
    ("battery_drain", "Five stars for features, one star for the battery drain -- the {dev} sync never sleeps."),
    ("performance", "The app looks beautiful but everything lags; opening the {dev} takes forever."),
    ("ui_bug", "Really happy with the {dev} itself, but the app UI shows the wrong status half the time."),
    ("login_auth", "Great app when you're in, but I get logged out daily and can't reach my {dev}."),
    ("crash", "It crashed twice this week on the {dev} page. Still better than the competitor's app."),
    ("connectivity", "Setup was a breeze and I love the widgets, but the {dev} drops offline every night."),
]
AMBIGUOUS_NONDEFECT = [
    ("praise", "Had one crash right after the {ver} update, but a reinstall fixed everything. Loving it now."),
    ("praise", "The login screen glitched once, but support helped in minutes. Great app overall."),
    ("praise", "Battery use was high for a day after the update, then settled. The {dev} works flawlessly."),
    ("praise", "App froze once during {dev} setup; restarting my phone fixed it. Smooth ever since."),
    ("praise", "Saw a weird UI glitch on the {dev} page yesterday, gone after the restart. Still five stars."),
]
HARD_DEFECT_RATE = 0.10   # share of defect reviews drawn from SOFTENED_DEFECT
HARD_PRAISE_RATE = 0.25   # share of praise reviews drawn from NEGATED_PRAISE
AMBIGUOUS_RATE = 0.06     # share of each class drawn from the AMBIGUOUS_* pools

# star rating distribution per review type: (weights for 1..5 stars)
DEFECT_STARS = [0.45, 0.30, 0.15, 0.07, 0.03]
PRAISE_STARS = [0.02, 0.03, 0.10, 0.30, 0.55]
FEATURE_STARS = [0.03, 0.05, 0.25, 0.35, 0.32]
QUESTION_STARS = [0.05, 0.08, 0.30, 0.32, 0.25]

CATEGORY_MIX = {
    "crash": 0.22, "battery_drain": 0.14, "connectivity": 0.24,
    "ui_bug": 0.14, "performance": 0.14, "login_auth": 0.12,
}
NONDEFECT_MIX = {"praise": 0.55, "feature_request": 0.25, "question": 0.20}
DEFECT_SHARE = 0.62

START_DATE = datetime(2026, 1, 5)
END_DATE = datetime(2026, 9, 20)

FILLERS = [
    "", "", "", " Honestly pretty annoyed.",
    " Hope this gets fixed soon.", " Otherwise I like the app.",
    " Support was no help.", " This started after the update.",
    " Reinstalling didn't help.", " Anyone else seeing this?",
    " Really disappointing.", " Considering switching apps.",
]


def _weighted_choice(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def _random_date(rng):
    delta = END_DATE - START_DATE
    return (START_DATE + timedelta(days=rng.randint(0, delta.days))).strftime("%Y-%m-%d")


def generate(n_reviews: int = N_REVIEWS, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_reviews):
        is_defect = rng.random() < DEFECT_SHARE
        if is_defect:
            r = rng.random()
            if r < AMBIGUOUS_RATE:
                category, text = rng.choice(AMBIGUOUS_DEFECT)
            elif r < AMBIGUOUS_RATE + HARD_DEFECT_RATE:
                category, text = rng.choice(SOFTENED_DEFECT)
            else:
                category = _weighted_choice(rng, list(CATEGORY_MIX), list(CATEGORY_MIX.values()))
                text = rng.choice(DEFECT_TEMPLATES[category])
            stars = _weighted_choice(rng, [1, 2, 3, 4, 5], DEFECT_STARS)
        else:
            kind = _weighted_choice(rng, list(NONDEFECT_MIX), list(NONDEFECT_MIX.values()))
            category = kind  # keep the non-defect subtype as the category label
            if kind == "praise" and rng.random() < AMBIGUOUS_RATE:
                _, text = rng.choice(AMBIGUOUS_NONDEFECT)
            elif kind == "praise" and rng.random() < HARD_PRAISE_RATE:
                text = rng.choice(NEGATED_PRAISE)
            else:
                text = rng.choice(NON_DEFECT[kind])
            weights = {"praise": PRAISE_STARS, "feature_request": FEATURE_STARS, "question": QUESTION_STARS}[kind]
            stars = _weighted_choice(rng, [1, 2, 3, 4, 5], weights)

        text = text.format(dev=rng.choice(DEVICES), ver=rng.choice(VERSIONS))
        # light variety: occasional filler clause + rare typo-style noise
        text += rng.choice(FILLERS)
        text = text.strip()

        rows.append({
            "review_id": f"R{i:05d}",
            "text": text,
            "star_rating": stars,
            "date": _random_date(rng),
            "app_version": rng.choice(VERSIONS),
            "is_defect": int(is_defect),
            "category": category,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    out = DATA_DIR / "reviews.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} reviews to {out}")
    print(df["is_defect"].value_counts().to_dict())
    print(df["category"].value_counts().to_dict())


if __name__ == "__main__":
    main()
