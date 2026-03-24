"""
build_election_corpus.py
------------------------
Build a research-backed Kenya 2022 election discourse corpus for the project.

The bundled corpus is designed to mirror the structure and cadence of public
discourse around the 2022 Kenyan election period while preserving a simple CSV
shape that works with the existing sentiment pipeline and dashboard.

Research grounding for the corpus design:
- Uchaguzi-2022 (citizen reports, June 27 to August 29, 2022)
- Public reporting on key election moments such as campaign build-up,
  election day, tallying/results, and post-result reactions

This script creates a local project corpus and can be replaced later with
approved external datasets that match the same schema.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


random.seed(42)

N_RECORDS = 900
OUTPUT_PATH = Path("data/raw/kenya_election_discourse_corpus.csv")

PHASES = [
    {
        "name": "Campaign Build-up",
        "slug": "campaign_build_up",
        "start": datetime(2022, 6, 27),
        "end": datetime(2022, 8, 8, 23, 59, 59),
        "weight": 0.36,
        "anchors": [
            (datetime(2022, 7, 26), 4),
            (datetime(2022, 7, 27), 4),
            (datetime(2022, 8, 6), 3),
            (datetime(2022, 8, 7), 3),
        ],
    },
    {
        "name": "Election Day",
        "slug": "election_day",
        "start": datetime(2022, 8, 9),
        "end": datetime(2022, 8, 9, 23, 59, 59),
        "weight": 0.18,
        "anchors": [
            (datetime(2022, 8, 9, 6), 4),
            (datetime(2022, 8, 9, 10), 5),
            (datetime(2022, 8, 9, 16), 4),
        ],
    },
    {
        "name": "Tallying & Results",
        "slug": "tallying_results",
        "start": datetime(2022, 8, 10),
        "end": datetime(2022, 8, 15, 23, 59, 59),
        "weight": 0.28,
        "anchors": [
            (datetime(2022, 8, 11, 12), 2),
            (datetime(2022, 8, 13, 18), 3),
            (datetime(2022, 8, 15, 15), 5),
        ],
    },
    {
        "name": "Post-Result Reaction",
        "slug": "post_result_reaction",
        "start": datetime(2022, 8, 16),
        "end": datetime(2022, 8, 29, 23, 59, 59),
        "weight": 0.18,
        "anchors": [
            (datetime(2022, 8, 16, 9), 4),
            (datetime(2022, 8, 17, 14), 3),
            (datetime(2022, 8, 22, 11), 2),
        ],
    },
]

REGIONS = [
    ("Nairobi", 0.22),
    ("Kiambu", 0.12),
    ("Kisumu", 0.10),
    ("Mombasa", 0.09),
    ("Nakuru", 0.08),
    ("Kakamega", 0.07),
    ("Uasin Gishu", 0.07),
    ("Kajiado", 0.06),
    ("Siaya", 0.06),
    ("Bungoma", 0.05),
    ("Garissa", 0.04),
    ("Homabay", 0.04),
]

TOPIC_WEIGHTS = {
    "campaign_build_up": {
        "Public Opinion": 0.25,
        "Media Reports": 0.16,
        "Positive Events": 0.10,
        "Counting & Results": 0.04,
        "Security Issues": 0.10,
        "Voting Issues": 0.08,
        "Political Rallies": 0.18,
        "Polling Administration": 0.05,
        "Staffing Issues": 0.04,
    },
    "election_day": {
        "Public Opinion": 0.16,
        "Media Reports": 0.10,
        "Positive Events": 0.18,
        "Counting & Results": 0.10,
        "Security Issues": 0.12,
        "Voting Issues": 0.16,
        "Political Rallies": 0.04,
        "Polling Administration": 0.10,
        "Staffing Issues": 0.04,
    },
    "tallying_results": {
        "Public Opinion": 0.22,
        "Media Reports": 0.10,
        "Positive Events": 0.06,
        "Counting & Results": 0.28,
        "Security Issues": 0.10,
        "Voting Issues": 0.05,
        "Political Rallies": 0.03,
        "Polling Administration": 0.08,
        "Staffing Issues": 0.08,
    },
    "post_result_reaction": {
        "Public Opinion": 0.28,
        "Media Reports": 0.10,
        "Positive Events": 0.12,
        "Counting & Results": 0.20,
        "Security Issues": 0.12,
        "Voting Issues": 0.04,
        "Political Rallies": 0.04,
        "Polling Administration": 0.04,
        "Staffing Issues": 0.06,
    },
}

TOPIC_CONFIG = {
    "Public Opinion": {
        "sentiment_weights": {"positive": 0.28, "neutral": 0.20, "negative": 0.52},
        "tags": [
            "peace_message",
            "trust_in_iebc",
            "concern_over_integrity",
            "candidate_support",
            "results_anxiety",
        ],
        "templates": {
            "positive": [
                "Peaceful queues in {region} today. Let the people choose and let IEBC finish the job. {hashtag}",
                "Whatever our politics, Kenya looks calm so far. This is the kind of uchaguzi we prayed for.",
                "The mood around {region} feels orderly and hopeful. Watu waamue, we keep peace.",
                "I may not agree with everyone, but voters turning out quietly in {region} is encouraging.",
            ],
            "neutral": [
                "Conversation in {region} is split between turnout, agents and what happens at Bomas next.",
                "People around here are mostly waiting for official communication before reacting either way.",
                "A lot of talk today is about turnout, transmission and whether the process stays on schedule.",
                "The street mood in {region} is cautious. Everyone is watching IEBC updates closely.",
            ],
            "negative": [
                "Too many unanswered questions around IEBC communication. Confidence is dropping fast in {region}.",
                "People are angry about delays and mixed messaging. This process should feel clearer than this.",
                "It is hard to calm voters when every hour brings a new rumor about forms, servers or agents.",
                "Trust is getting damaged because officials are not explaining what is happening at Bomas properly.",
            ],
        },
    },
    "Media Reports": {
        "sentiment_weights": {"positive": 0.10, "neutral": 0.82, "negative": 0.08},
        "tags": [
            "headline_update",
            "candidate_quote",
            "commission_update",
            "observer_brief",
        ],
        "templates": {
            "positive": [
                "Clergy and civil society groups in {region} praise calm voting atmosphere and urge patience.",
                "Observers note orderly opening in most stations sampled this morning across {region}.",
            ],
            "neutral": [
                "{actor} says agents are present and tallying continues under public scrutiny at Bomas.",
                "Observers say turnout patterns remain uneven, with urban stations moving slower than rural centres.",
                "Campaign teams release parallel statements as the commission continues verification of forms.",
                "Media desks report steady voter traffic in {region}, with isolated delays under review.",
            ],
            "negative": [
                "Media update: disputes over forms and tallying procedures intensify pressure on the commission.",
                "Reporters tracking the count say political camps are contesting figures from multiple constituencies.",
            ],
        },
    },
    "Positive Events": {
        "sentiment_weights": {"positive": 0.82, "neutral": 0.14, "negative": 0.04},
        "tags": [
            "peace_initiative",
            "orderly_voting",
            "citizen_support",
            "community_monitoring",
        ],
        "templates": {
            "positive": [
                "Young people in {region} are escorting elderly voters and reminding everyone to keep peace. {hashtag}",
                "Everything is moving well at this station in {region}. Long line, yes, but people are patient and calm.",
                "Community peace volunteers in {region} are doing a solid job cooling tensions near the polling centre.",
                "Voters here are sharing water, directions and information. Good atmosphere despite the pressure.",
            ],
            "neutral": [
                "Church leaders in {region} continue peace messaging outside stations as voting proceeds.",
                "Local observers say most voters they met in {region} are focused on finishing the process quietly.",
            ],
            "negative": [
                "Peace teams are present in {region}, but tempers are rising after repeated queue management problems.",
            ],
        },
    },
    "Counting & Results": {
        "sentiment_weights": {"positive": 0.20, "neutral": 0.33, "negative": 0.47},
        "tags": [
            "form_34a",
            "results_delay",
            "bomas_update",
            "transmission_issue",
            "results_correction",
        ],
        "templates": {
            "positive": [
                "Forms from {region} are finally appearing online and agents can verify them. That helps confidence.",
                "Results board updates are slower than hoped, but at least the figures are being checked carefully.",
                "Supporters in {region} say the correction of an earlier tally was the right call. Better accuracy than speed.",
            ],
            "neutral": [
                "Tallying from {region} continues as parties compare Form 34A uploads against local records.",
                "At Bomas, officials say verification is still underway and no constituency should be skipped.",
                "More results from {region} have been posted, though final county totals are still pending.",
                "Agents remain at the national tally centre as the commission reviews contested entries.",
            ],
            "negative": [
                "Results from {region} are taking too long to verify and that gap is feeding suspicion.",
                "Another dispute over forms at Bomas. People are asking why the same constituencies keep stalling.",
                "Mixed figures from tally centres are making the count look messy and avoidably political.",
                "The problem is not only delay; it is delay without a clear explanation from the commission.",
            ],
        },
    },
    "Security Issues": {
        "sentiment_weights": {"positive": 0.16, "neutral": 0.24, "negative": 0.60},
        "tags": [
            "road_blockage",
            "rumor_alert",
            "dangerous_speech",
            "police_presence",
            "localized_disruption",
        ],
        "templates": {
            "positive": [
                "Police visibility around the station in {region} is high but professional. So far hakuna shida.",
                "Security teams in {region} managed the crowd early and the area has stayed calm since.",
            ],
            "neutral": [
                "Security officers remain deployed near key centres in {region} as results statements continue.",
                "Teams are monitoring transport routes and public screens in {region} after rumor traffic rose online.",
                "Residents in {region} report visible but calm police patrols around election hot spots.",
            ],
            "negative": [
                "Road blockage reported near a voting centre in {region}. People are being told to avoid the area.",
                "Rumors of confrontation in {region} are moving faster than official updates and that is dangerous.",
                "Heavy tension outside the tally site in {region}; police are struggling to separate rival groups.",
                "Threatening slogans and online incitement are making the atmosphere in {region} feel unsafe.",
            ],
        },
    },
    "Voting Issues": {
        "sentiment_weights": {"positive": 0.10, "neutral": 0.18, "negative": 0.72},
        "tags": [
            "register_problem",
            "kiems_issue",
            "voter_assistance",
            "queue_delay",
            "invalid_ballot",
        ],
        "templates": {
            "positive": [
                "Queue in {region} was long, but officials eventually sorted out the register issue and voting resumed.",
            ],
            "neutral": [
                "Some voters in {region} say they were redirected after finding their names listed at another station.",
                "Polling station teams in {region} are dealing with slow KIEMS checks but the line is moving.",
            ],
            "negative": [
                "Several voters in {region} cannot find their names even after checking multiple desks.",
                "KIEMS kit at this station in {region} keeps failing and older voters are getting frustrated.",
                "People have been on the queue for hours in {region} and still no clear guidance from officials.",
                "Voters in {region} say assistance for the elderly and people with disabilities is too inconsistent.",
            ],
        },
    },
    "Political Rallies": {
        "sentiment_weights": {"positive": 0.35, "neutral": 0.45, "negative": 0.20},
        "tags": [
            "candidate_itinerary",
            "crowd_mobilization",
            "campaign_message",
            "coalition_response",
        ],
        "templates": {
            "positive": [
                "{actor} rally in {region} drew a strong crowd and the message stayed focused on jobs and turnout.",
                "Campaign stop in {region} ended with leaders calling for peace and acceptance of official results.",
                "Supporters in {region} say the rally felt energized but disciplined. No chaos after dispersal.",
            ],
            "neutral": [
                "{actor} campaign team released today's itinerary for {region} and neighbouring constituencies.",
                "Coalition speakers in {region} spent most of the rally on turnout, agents and vote protection.",
                "Crowds in {region} followed the speeches closely as both camps pushed final get-out-the-vote appeals.",
            ],
            "negative": [
                "Rally messaging in {region} crossed into provocation and supporters started trading insults online.",
                "The campaign tone in {region} is getting harsher, with more accusations than policy at the podium.",
            ],
        },
    },
    "Polling Administration": {
        "sentiment_weights": {"positive": 0.10, "neutral": 0.22, "negative": 0.68},
        "tags": [
            "late_opening",
            "missing_materials",
            "manual_register",
            "station_flow",
        ],
        "templates": {
            "positive": [
                "Station managers in {region} opened late but recovered quickly once extra materials arrived.",
            ],
            "neutral": [
                "Officials in {region} say manual registers were delivered after the opening delay.",
                "Polling stations in {region} are adjusting desk layout to reduce congestion as turnout rises.",
            ],
            "negative": [
                "Polling station in {region} opened late and some voters left before materials were ready.",
                "Manual register not available at one desk in {region}; the presiding team looked unprepared.",
                "Confusion over ballot papers and desk flow is slowing the station badly in {region}.",
                "Agents in {region} are complaining that basic polling procedures are not being followed consistently.",
            ],
        },
    },
    "Staffing Issues": {
        "sentiment_weights": {"positive": 0.05, "neutral": 0.19, "negative": 0.76},
        "tags": [
            "official_absence",
            "official_bias",
            "agent_access",
            "staff_arrest",
        ],
        "templates": {
            "positive": [
                "Replacement officials reached the station in {region} and operations improved after a rough start.",
            ],
            "neutral": [
                "Election officials in {region} were reshuffled after complaints from party agents.",
                "Observers in {region} say staffing gaps were reported early and replacements are being processed.",
            ],
            "negative": [
                "Too few officials on site in {region}. Voters and agents are doing more queue management than staff.",
                "Party agents in {region} say some officials are acting partisan instead of procedural.",
                "Reports from {region} suggest staff were missing at opening time and the station never really recovered.",
                "Observers say media and agents faced avoidable access problems because of poor staff coordination in {region}.",
            ],
        },
    },
}

ACTORS = [
    "IEBC",
    "Wafula Chebukati",
    "Azimio",
    "Kenya Kwanza",
    "Raila Odinga",
    "William Ruto",
    "commission officials",
    "party agents",
]

TOPIC_ACTORS = {
    "Media Reports": ["IEBC", "Wafula Chebukati", "Azimio", "Kenya Kwanza", "commission officials"],
    "Political Rallies": ["Raila Odinga", "William Ruto", "Azimio", "Kenya Kwanza", "Rigathi Gachagua", "Martha Karua"],
}

HASHTAGS = [
    "#KenyaDecides2022",
    "#Uchaguzi2022",
    "#UchaguziWaAmani",
    "#KenyaElections2022",
    "#ElectionsKE2022",
    "",
]

REFERENCE_EVENTS = {
    "campaign_build_up": [
        "voter_mobilization",
        "peace_messaging",
        "campaign_finale",
        "smartmatic_controversy",
    ],
    "election_day": [
        "poll_opening",
        "queue_management",
        "turnout_monitoring",
        "kiems_operations",
    ],
    "tallying_results": [
        "form_34a_verification",
        "bomas_tallying",
        "results_delay",
        "commission_split",
    ],
    "post_result_reaction": [
        "results_reaction",
        "petition_expectation",
        "peace_monitoring",
        "local_tension_watch",
    ],
}

SOURCE_STYLE_OPTIONS = {
    "Public Opinion": ["citizen_commentary", "social_post"],
    "Media Reports": ["headline_brief", "news_update"],
    "Positive Events": ["peace_monitor", "citizen_report"],
    "Counting & Results": ["observer_note", "tally_update", "social_post"],
    "Security Issues": ["incident_report", "situation_update"],
    "Voting Issues": ["citizen_report", "observer_note"],
    "Political Rallies": ["campaign_update", "news_update"],
    "Polling Administration": ["observer_note", "citizen_report"],
    "Staffing Issues": ["observer_note", "incident_report"],
}

LANGUAGE_STYLE_OPTIONS = {
    "Public Opinion": ["English", "English-Swahili"],
    "Media Reports": ["English"],
    "Positive Events": ["English", "English-Swahili"],
    "Counting & Results": ["English", "English-Swahili"],
    "Security Issues": ["English", "English-Swahili"],
    "Voting Issues": ["English", "English-Swahili"],
    "Political Rallies": ["English", "English-Swahili"],
    "Polling Administration": ["English", "English-Swahili"],
    "Staffing Issues": ["English"],
}


def choose_weighted(options: list[tuple[str, float]]) -> str:
    labels = [label for label, _ in options]
    weights = [weight for _, weight in options]
    return random.choices(labels, weights=weights, k=1)[0]


def choose_phase() -> dict:
    return random.choices(PHASES, weights=[phase["weight"] for phase in PHASES], k=1)[0]


def choose_topic(phase_slug: str) -> str:
    items = list(TOPIC_WEIGHTS[phase_slug].items())
    return choose_weighted(items)


def choose_sentiment(topic: str) -> str:
    items = list(TOPIC_CONFIG[topic]["sentiment_weights"].items())
    return choose_weighted(items)


def choose_timestamp(phase: dict) -> datetime:
    if random.random() < 0.7 and phase["anchors"]:
        anchor, _ = random.choices(phase["anchors"], weights=[item[1] for item in phase["anchors"]], k=1)[0]
        if phase["slug"] == "election_day":
            return anchor + timedelta(
                minutes=random.randint(0, 179),
                seconds=random.randint(0, 59),
            )

        jitter_days = random.randint(-1, 1)
        jitter_hours = random.randint(-8, 8)
        candidate = anchor + timedelta(days=jitter_days, hours=jitter_hours, minutes=random.randint(0, 59))
        return min(max(candidate, phase["start"]), phase["end"])

    total_seconds = int((phase["end"] - phase["start"]).total_seconds())
    return phase["start"] + timedelta(seconds=random.randint(0, total_seconds))


def maybe_code_switch(text: str, language_style: str, topic: str, sentiment: str) -> str:
    if language_style != "English-Swahili":
        return text

    swaps = [
        (r"\bpeople\b", "watu"),
        (r"\bcalm\b", "tulivu"),
        (r"\bno problem\b", "hakuna shida"),
        (r"\bpeace\b", "amani"),
        (r"\bchoose\b", "waamue"),
        (r"\btoday(?!')\b", "leo"),
    ]
    for pattern, mixed in swaps:
        if re.search(pattern, text, flags=re.IGNORECASE) and random.random() < 0.35:
            text = re.sub(pattern, mixed, text, flags=re.IGNORECASE)

    if sentiment == "negative" and random.random() < 0.4:
        text += " Hii sio sawa."
    elif sentiment == "positive" and random.random() < 0.4:
        text += " Tuwe watulivu."

    if topic in {"Positive Events", "Public Opinion"} and "uchaguzi" not in text.lower() and random.random() < 0.25:
        text += " uchaguzi ni yetu."

    return text


def build_text(topic: str, sentiment: str, region: str) -> tuple[str, str, str]:
    tag = random.choice(TOPIC_CONFIG[topic]["tags"])
    actor = random.choice(TOPIC_ACTORS.get(topic, ACTORS))
    hashtag = random.choice(HASHTAGS)
    template = random.choice(TOPIC_CONFIG[topic]["templates"][sentiment])
    text = template.format(actor=actor, region=region, hashtag=hashtag).strip()
    text = " ".join(text.split())
    if not hashtag and text.endswith("{hashtag}"):
        text = text.replace("{hashtag}", "").strip()
    source_style = random.choice(SOURCE_STYLE_OPTIONS[topic])
    return text, tag, source_style


def build_record(record_index: int) -> dict:
    phase = choose_phase()
    region = choose_weighted(REGIONS)
    topic = choose_topic(phase["slug"])
    sentiment = choose_sentiment(topic)
    text, subtopic_tag, source_style = build_text(topic, sentiment, region)
    language_style = random.choice(LANGUAGE_STYLE_OPTIONS[topic])
    text = maybe_code_switch(text, language_style, topic, sentiment)
    timestamp = choose_timestamp(phase)

    return {
        "record_id": f"KE22-{record_index:04d}",
        "text": text,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": topic,
        "region": region,
        "event_phase": phase["name"],
        "reference_event": random.choice(REFERENCE_EVENTS[phase["slug"]]),
        "subtopic_tag": subtopic_tag,
        "source_style": source_style,
        "language_style": language_style,
        "data_source": "PROJECT_CURATED_CORPUS",
    }


def main() -> None:
    records = [build_record(index + 1) for index in range(N_RECORDS)]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df):,} records to {OUTPUT_PATH}")
    print("\nTopic distribution:")
    print(df["topic"].value_counts().to_string())
    print("\nPhase distribution:")
    print(df["event_phase"].value_counts().to_string())
    print("\nRegion distribution:")
    print(df["region"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
