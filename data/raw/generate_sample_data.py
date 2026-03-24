"""
generate_sample_data.py
-----------------------
Generates a synthetic demo dataset of ~500 Kenyan political discourse records
for the sentiment analysis prototype.

IMPORTANT: This is entirely synthetic/fabricated data created for demonstration
purposes only. It does NOT represent real tweets, real persons, or real events.
It is clearly labelled as a prototype dataset throughout the application.

Author : Nichodemus Werre Amollo
Created: 2025
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_RECORDS = 500
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2022, 8, 31)   # covers the 2022 Kenyan general election period

# ── Topic pools ───────────────────────────────────────────────────────────────
TOPICS = ["Elections", "Economy", "Security", "Healthcare", "Governance"]

TOPIC_WEIGHTS = [0.35, 0.25, 0.15, 0.12, 0.13]   # elections dominant in 2022

REGIONS = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Thika", "Machakos", "Nyeri", "Garissa", "Kakamega"
]

# ── Text templates by topic and sentiment polarity ────────────────────────────
# Each list contains template strings. {leader} and {issue} are filled randomly.

LEADERS = ["the government", "IEBC", "the president", "the opposition",
           "parliament", "county officials", "the judiciary", "the treasury"]

TEXT_TEMPLATES = {
    "Elections": {
        "positive": [
            "The electoral process is transparent and credible. Kenyans deserve fair elections. #KenyaDecides2022",
            "IEBC has done a commendable job in voter registration. Democracy is winning! #Elections2022",
            "Peaceful voting is a sign of maturity. Proud to be Kenyan today. #Uchaguzi2022",
            "Results coming in smoothly. The system is working. Trust the process. #KenyaVotes",
            "Voter turnout is impressive across the country. This is democracy at work.",
        ],
        "negative": [
            "IEBC cannot be trusted to deliver credible results. The system is rigged! #RiggedElections",
            "Voter intimidation reported in several constituencies. This is unacceptable.",
            "The electoral commission has failed Kenyans again. We demand accountability.",
            "Results being delayed without explanation. What is IEBC hiding? #ElectionFraud",
            "Violence at polling stations is shameful. Our democracy is under threat.",
        ],
        "neutral": [
            "Voting underway in all 47 counties. Results expected by evening. #KenyaElections2022",
            "IEBC confirms voter turnout at 63% nationally as of midday.",
            "Presidential results will be announced within 7 days per constitutional requirement.",
            "Observers from the AU and EU deployed to monitor the electoral process.",
            "Tallying of results continues at constituency level. Updates pending.",
        ],
    },
    "Economy": {
        "positive": [
            "Kenya's GDP growth projected at 5.5% this year. Strong fundamentals. #KenyaEconomy",
            "New manufacturing jobs created in Nairobi corridor. Positive momentum for workers.",
            "Exports of tea and horticulture hit record highs. Agriculture sector thriving.",
            "NSE gains 3% this week on investor confidence. Markets looking healthy.",
            "Youth enterprise fund disbursements have helped thousands of small businesses.",
        ],
        "negative": [
            "Cost of unga and basic commodities unbearable. Ordinary Kenyans are suffering.",
            "Unemployment among youth remains at crisis levels. Nothing is being done.",
            "Fuel prices at an all-time high. The government must act now. #FuelCrisis",
            "Kenya's debt burden will cripple future generations. Reckless borrowing must stop.",
            "Inflation eroding purchasing power. The economy is failing ordinary people.",
        ],
        "neutral": [
            "Kenya Revenue Authority reports KSh 1.2 trillion in tax collection for FY2021/22.",
            "Central Bank holds base rate at 7.5%. MPC cites stable inflation outlook.",
            "IMF approves $2.3B extended credit facility for Kenya. Conditions attached.",
            "Treasury releases mid-year budget review. Spending adjustments noted across ministries.",
            "Kenya Power announces tariff review pending Energy Regulatory Commission approval.",
        ],
    },
    "Security": {
        "positive": [
            "Police deployment ahead of elections praised for professionalism. Calm maintained.",
            "KDF and NPS cooperation securing borders effectively. Kenya is safe.",
            "Community policing initiatives reducing crime in Nairobi estates. Good progress.",
        ],
        "negative": [
            "Al-Shabaab attacks in the North East continue. Government response inadequate.",
            "Banditry in Laikipia threatening lives and livelihoods. Enough is enough.",
            "Police brutality during protests is deeply troubling. Officers must be held accountable.",
            "Crime wave in Mombasa Old Town worrying residents. Where is the state?",
        ],
        "neutral": [
            "GSU units deployed to election hotspots in advance of voting day.",
            "Interior Ministry issues security advisory ahead of elections. Citizens urged to remain calm.",
            "Multi-agency security teams conducting patrols in volatile regions.",
        ],
    },
    "Healthcare": {
        "positive": [
            "NHIF coverage expansion reaches more households in rural Kenya. A step forward.",
            "New Level 5 hospital commissioned in Kisumu. Healthcare access improving.",
            "Kenya's COVID-19 vaccination coverage exceeds 60%. Strong public health response.",
        ],
        "negative": [
            "Doctors still on strike after 3 weeks. Patients dying in hospital corridors.",
            "NHIF reforms failing to protect the poorest Kenyans. Reform the system now.",
            "Drug shortage at government hospitals is alarming. Patients buying meds privately.",
            "Maternal mortality in rural Kenya remains unacceptably high. Neglect must end.",
        ],
        "neutral": [
            "Ministry of Health launches malaria prevention campaign in lake region counties.",
            "Kenya Medical Supplies Authority to distribute free ARVs to 54 public hospitals.",
            "National Cancer Institute of Kenya opens second facility in Mombasa.",
        ],
    },
    "Governance": {
        "positive": [
            "Devolution has brought services closer to citizens. Counties are delivering.",
            "Transparency in public procurement improving under new regulations. Keep it up.",
            "Open data portal launched by State House praised by civil society. Good governance.",
        ],
        "negative": [
            "Corruption in government tenders is endemic. Nothing changes regardless of who wins.",
            "Public funds being looted while hospitals lack medicine. Accountability needed now.",
            "Parliament rubber-stamping executive decisions. Where is our oversight? #Bunge",
            "Nepotism in public service appointments is entrenched. Merit must prevail.",
        ],
        "neutral": [
            "Cabinet approves National Development Plan 2022–2027. Implementation timelines unclear.",
            "Ethics and Anti-Corruption Commission tables annual report to parliament.",
            "County governors meet to discuss revenue sharing formula ahead of budget cycle.",
        ],
    },
}


def random_timestamp(start: datetime, end: datetime) -> str:
    """Return a random timestamp string between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    ts = start + timedelta(seconds=random_seconds)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def pick_polarity() -> str:
    """Randomly assign a sentiment polarity with realistic distribution."""
    # ~35% positive, ~30% neutral, ~35% negative — reflects contested political climate
    return random.choices(
        ["positive", "neutral", "negative"],
        weights=[0.35, 0.30, 0.35]
    )[0]


def generate_record(record_id: int) -> dict:
    """Generate a single synthetic record."""
    topic    = random.choices(TOPICS, weights=TOPIC_WEIGHTS)[0]
    polarity = pick_polarity()
    region   = random.choice(REGIONS)
    texts    = TEXT_TEMPLATES[topic][polarity]
    text     = random.choice(texts)
    ts       = random_timestamp(START_DATE, END_DATE)

    return {
        "record_id"        : f"KE-{record_id:04d}",
        "text"             : text,
        "timestamp"        : ts,
        "topic"            : topic,
        "region"           : region,
        "true_polarity"    : polarity,    # ground-truth label for evaluation only
        "data_source"      : "SYNTHETIC_DEMO",
    }


def main() -> None:
    """Generate and save the synthetic dataset."""
    print(f"Generating {N_RECORDS} synthetic records...")
    records = [generate_record(i + 1) for i in range(N_RECORDS)]
    df = pd.DataFrame(records)

    # Sort by timestamp for time-series coherence
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    output_path = "data/raw/kenya_political_sample.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    print("\nTopic distribution:")
    print(df["topic"].value_counts())
    print("\nPolarity distribution (ground truth):")
    print(df["true_polarity"].value_counts())


if __name__ == "__main__":
    main()
