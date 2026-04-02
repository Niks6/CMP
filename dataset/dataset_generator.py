"""
dataset_generator.py — Synthetic post dataset generator for the Content Moderation RL environment.

Generates realistic social media posts across four categories:
  - BENIGN        : Safe, everyday content
  - SPAM          : Unwanted promotional / scam content
  - HATE_SPEECH   : Discriminatory / hostile content
  - MISINFORMATION: False or misleading claims

Each generated Post has realistic metadata (toxicity_score, spam_score, etc.)
consistent with its category but with plausible noise to avoid a trivially
solvable environment.
"""

from __future__ import annotations

import random
from typing import Optional

from env.observation_model import Post, PostCategory


# ---------------------------------------------------------------------------
# Template text banks per category
# ---------------------------------------------------------------------------

_BENIGN_TEXTS = [
    "Just had the most amazing sunset hike! 🌄",
    "Anyone else think the new season of that show is incredible?",
    "Made homemade pasta for the first time — turned out great!",
    "Looking for book recommendations — into sci-fi and history.",
    "Happy birthday to my best friend! You deserve all the love. 🎉",
    "The coffee shop on 5th Ave has the best espresso in the city.",
    "Finally finished my thesis. 5 years of hard work done!",
    "Morning run complete. Nothing like starting the day right.",
    "Tried a new Thai restaurant last night — absolutely delicious!",
    "My dog learned a new trick today. Proud dog parent moment! 🐕",
    "Spring is finally here — flowers are blooming everywhere.",
    "Just finished a 1000-piece puzzle. Took three days but worth it!",
    "Pro tip: always keep a spare charger at the office.",
    "Anyone going to the local farmers market this weekend?",
    "Finished reading 'The Three-Body Problem' — mind-blowing!",
    "The city library renovations look amazing. Can't wait to visit.",
    "Yoga at 6am might sound rough but it really sets the tone.",
    "My sourdough starter is finally active after two weeks!",
    "Volunteered at the animal shelter today — highly recommend.",
    "Does anyone know a good mechanic in the downtown area?",
    "I cannot believe how fast the kids are growing up.",
    "Binge-watched the whole documentary series this weekend.",
    "The community garden is looking absolutely gorgeous this year.",
    "Just got my flu shot. Remember to protect yourself and others!",
    "Working from home has its perks — lunch in the backyard.",
    "Finished my first 5K today. Slow but steady wins the race!",
    "The new movie at the cinema is worth every penny.",
    "Cycling to work saves time AND is great for the environment.",
    "Planning a road trip to the mountains next month. Any tips?",
    "Our local high school team won the state championship!",
]

_SPAM_TEXTS = [
    "BUY CRYPTO NOW!! 1000% returns GUARANTEED! DM for details 🚀🚀",
    "🔥 LIMITED TIME OFFER — Free iPhone 15! Click here: bit.ly/freephone99",
    "Make $5000 a week from home. No experience needed. WhatsApp me.",
    "URGENT: Your account is suspended. Verify at secure-bank-login.xyz",
    "Congratulations! You've been selected for a $1,000 Amazon gift card!",
    "Lose 30 lbs in 30 days with this ONE WEIRD TRICK doctors hate!",
    "Best deals on Ray-Ban sunglasses — 90% OFF today only!",
    "I made $10,000 last month with this simple forex signal bot. Join now!",
    "FREE followers! Get 10,000 Instagram followers in 24 hours.",
    "Click this link to claim your prize: prize-winner-2024.net",
    "EARN MONEY ONLINE — work 2 hours/day, $3000/week guaranteed!",
    "Your Netflix account will be cancelled. Update your payment: netflix-verify.ru",
    "Sell your old gold for CASH TODAY. Best rates in your area.",
    "Exclusive MLM opportunity — be your own boss! Message me NOW.",
    "Download this app and earn $50 instantly. No hidden fees!",
    "FLASH SALE: Designer handbags 95% off. Shop before midnight!",
    "Meet singles in your area! Sign up FREE: date-tonight.xyz",
    "Claim your inheritance from deceased relative in Nigeria. Contact us.",
    "Get verified on ALL social media platforms for just $10.",
    "BREAKING: Invest in this altcoin before it goes to the moon! 🚀",
    "Win a brand new Tesla! Just follow and retweet. Drawing tonight!",
    "Your computer has a virus! Call 1-800-TECH-FIX immediately!",
    "Order from our pharmacy — no prescription needed, ships worldwide.",
    "Double your Bitcoin in 24 hours with our AI trading algorithm!",
    "You have an unclaimed tax refund of $2,547. Click to claim now!",
]

_HATE_SPEECH_TEXTS = [
    "People like you should not be allowed in this country.",
    "All [group] are criminals. They should be deported.",
    "I can't stand how these people are taking over everything.",
    "Those [slur] ruined the neighborhood. Get them out.",
    "Women should stay in the kitchen and stop trying to compete.",
    "No [religious group] deserves rights in a civilized country.",
    "These [ethnicity] are subhuman and deserve what they get.",
    "Anyone who voted for [party] is a traitor and should be jailed.",
    "I hate these immigrants coming here and stealing our jobs.",
    "Disabled people are a burden on society and shouldn't breed.",
    "Trans people are a mental illness and should be institutionalized.",
    "Go back to your own country if you don't like it here.",
    "[Slur] don't belong in positions of power. Ever.",
    "These refugees are invaders. Shoot them at the border.",
    "Old people are useless — they should pay higher taxes.",
    "I hope all [group] get what's coming to them.",
    "These [racial group] are genetically inferior — science proves it.",
    "Anyone with that religion shouldn't be allowed to vote.",
    "Fat people choose to be lazy. They deserve no sympathy.",
    "These [nationality] have ruined this country with their culture.",
    "Mildly annoys me when different groups try to act equal.",
    "I disagree with immigration policies — it's changing our culture.",
    "Sometimes I feel like certain communities don't integrate well.",
    "I think some traditional values are worth protecting.",
    "It's frustrating when people misrepresent historical facts.",
]

_MISINFO_TEXTS = [
    "Drinking bleach mixed with water kills COVID-19 — spread the word!",
    "5G towers are secretly emitting radiation to control human thoughts.",
    "Vaccines contain microchips that track your location for the government.",
    "The Earth is flat and NASA has been covering it up for decades.",
    "Scientists ADMIT: sun exposure cures cancer better than chemotherapy.",
    "Eating raw garlic daily prevents heart disease with 100% effectiveness.",
    "The moon landing in 1969 was filmed in a Hollywood studio.",
    "Drinking hydrogen peroxide can cure all bacterial infections at home.",
    "BREAKING: Scientists confirm that autism is caused by Wi-Fi signals.",
    "Climate change is a hoax invented by China to destroy Western economies.",
    "Bill Gates created COVID to sell vaccines and install tracking chips.",
    "You can recharge your phone battery by placing it in the microwave.",
    "Studies show that homeopathy cures Alzheimer's disease permanently.",
    "The government adds fluoride to water to make people docile and obedient.",
    "Essential oils can replace all antibiotics for bacterial infections.",
    "JFK Jr. is still alive and will run for president in the next election.",
    "The COVID vaccine alters your DNA permanently and makes you infertile.",
    "Cancer can be cured by simply switching to an alkaline diet.",
    "Chemtrails from planes are chemical weapons used for population control.",
    "Scientific study proves coffee causes more cancer than cigarettes.",
    "Doctors refuse to tell you this herbal remedy cures diabetes in a week.",
    "New research confirms: the human body can survive without sleep entirely.",
    "Official report: masturbation causes blindness — suppressed by media.",
    "Tap water in major cities contains estrogen that feminizes male children.",
    "Ancient knowledge: inhaling onion fumes prevents viral infections.",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int = 500,
    seed: Optional[int] = 42,
    category_weights: Optional[dict[str, float]] = None,
) -> list[Post]:
    """
    Generate a synthetic dataset of social media posts.

    Args:
        n:                Number of posts to generate.
        seed:             Random seed for reproducibility.
        category_weights: Optional dict mapping category names to sampling weights.
                          Defaults to {benign: 0.40, spam: 0.25, hate_speech: 0.20,
                          misinformation: 0.15}.

    Returns:
        List of Post objects with realistic metadata.
    """
    rng = random.Random(seed)

    if category_weights is None:
        category_weights = {
            PostCategory.BENIGN: 0.40,
            PostCategory.SPAM: 0.25,
            PostCategory.HATE_SPEECH: 0.20,
            PostCategory.MISINFORMATION: 0.15,
        }

    categories = list(category_weights.keys())
    weights = list(category_weights.values())

    posts: list[Post] = []
    for post_id in range(1, n + 1):
        category: PostCategory = rng.choices(categories, weights=weights, k=1)[0]
        post = _generate_post(post_id, category, rng)
        posts.append(post)

    return posts


def generate_task_dataset(
    category: PostCategory,
    n: int = 200,
    benign_ratio: float = 0.4,
    seed: Optional[int] = 42,
) -> list[Post]:
    """
    Generate a task-specific dataset focused on one harmful category vs. benign.

    Args:
        category:     The harmful category to include (SPAM, HATE_SPEECH, or MISINFORMATION).
        n:            Total number of posts.
        benign_ratio: Fraction of posts that should be benign.
        seed:         Random seed.

    Returns:
        List of Post objects.
    """
    rng = random.Random(seed)
    posts: list[Post] = []

    n_benign = int(n * benign_ratio)
    n_harmful = n - n_benign

    for post_id in range(1, n_benign + 1):
        posts.append(_generate_post(post_id, PostCategory.BENIGN, rng))

    for post_id in range(n_benign + 1, n + 1):
        posts.append(_generate_post(post_id, category, rng))

    rng.shuffle(posts)
    return posts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_post(post_id: int, category: PostCategory, rng: random.Random) -> Post:
    """Generate a single Post with category-consistent metadata and realistic noise."""

    if category == PostCategory.BENIGN:
        text = rng.choice(_BENIGN_TEXTS)
        toxicity_score = _noisy(rng, base=0.05, noise=0.12, lo=0.0, hi=0.45)
        spam_score = _noisy(rng, base=0.05, noise=0.08, lo=0.0, hi=0.25)
        misinfo_score = _noisy(rng, base=0.03, noise=0.07, lo=0.0, hi=0.20)
        report_count = rng.randint(0, 3)
        user_reputation = _noisy(rng, base=0.75, noise=0.15, lo=0.4, hi=1.0)
        image_flag = rng.random() < 0.05

    elif category == PostCategory.SPAM:
        text = rng.choice(_SPAM_TEXTS)
        toxicity_score = _noisy(rng, base=0.35, noise=0.25, lo=0.05, hi=0.75)
        spam_score = _noisy(rng, base=0.85, noise=0.10, lo=0.55, hi=1.0)
        misinfo_score = _noisy(rng, base=0.20, noise=0.15, lo=0.0, hi=0.55)
        report_count = rng.randint(5, 50)
        user_reputation = _noisy(rng, base=0.20, noise=0.15, lo=0.0, hi=0.50)
        image_flag = rng.random() < 0.40

    elif category == PostCategory.HATE_SPEECH:
        text = rng.choice(_HATE_SPEECH_TEXTS)
        toxicity_score = _noisy(rng, base=0.75, noise=0.20, lo=0.40, hi=1.0)
        spam_score = _noisy(rng, base=0.10, noise=0.10, lo=0.0, hi=0.35)
        misinfo_score = _noisy(rng, base=0.15, noise=0.12, lo=0.0, hi=0.40)
        report_count = rng.randint(10, 100)
        user_reputation = _noisy(rng, base=0.25, noise=0.20, lo=0.0, hi=0.60)
        image_flag = rng.random() < 0.25

    else:  # MISINFORMATION
        text = rng.choice(_MISINFO_TEXTS)
        toxicity_score = _noisy(rng, base=0.40, noise=0.25, lo=0.10, hi=0.80)
        spam_score = _noisy(rng, base=0.30, noise=0.20, lo=0.05, hi=0.65)
        misinfo_score = _noisy(rng, base=0.88, noise=0.08, lo=0.60, hi=1.0)
        report_count = rng.randint(8, 80)
        user_reputation = _noisy(rng, base=0.30, noise=0.20, lo=0.0, hi=0.65)
        image_flag = rng.random() < 0.30

    return Post(
        post_id=post_id,
        post_text=text,
        toxicity_score=round(toxicity_score, 4),
        report_count=report_count,
        user_reputation=round(user_reputation, 4),
        image_flag=image_flag,
        category=category,
        spam_score=round(spam_score, 4),
        misinfo_score=round(misinfo_score, 4),
    )


def _noisy(
    rng: random.Random, base: float, noise: float, lo: float = 0.0, hi: float = 1.0
) -> float:
    """Return `base` perturbed by Gaussian noise, clamped to [lo, hi]."""
    value = base + rng.gauss(0, noise)
    return max(lo, min(hi, value))
