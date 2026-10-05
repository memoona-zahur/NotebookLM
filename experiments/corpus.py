"""A deliberately broad evaluation corpus.

Two axes of variety, because a corpus that varies only in file type but not in
subject will happily agree with a system tuned to one document:

* **Format**: prose and tabular, prose and structured, prose and code.
* **Subject**: brewing, pharmacology, contract law, athletics, catalysis,
  medieval history, glaciology, baroque music, horticulture, civil aviation,
  agronomy and curriculum design. Nothing in the application code knows any of
  these exist.

Every document gets both polite natural-language questions and terse
keyword-style ones, because terse queries are where dense retrieval is weakest,
and its own trap questions that share vocabulary with the document but not its
answers. Those "entity-overlap traps" are the ones that separate a working
retriever from a plausible-looking one.
"""

import io
import json
import tempfile
from pathlib import Path

PROSE_PAGES = {
    "brew-guide.pdf": [
        """Home Brewing Guide: Wort Production

Mashing converts the grain starches into fermentable sugars. The mash tun is
held at a single temperature for roughly an hour, and the temperature chosen
determines how much of the fermentable wort will be produced. A single infusion
mash at sixty-six degrees Celsius favours a balanced body.

After the wort is drained, it is boiled for at least sixty minutes. The boil
unwortifies the wort and drives off volatile hop character precursors. Whirlpool
and knockout steps concentrate the hop additions late in the boil.

Fermentation is carried out by Saccharomyces cerevisiae at a controlled
temperature. Ale strains prefer warmer fermentation around eighteen to twenty
degrees, while lager strains are pitched cooler and held for longer."""
        ,
        """Packaging and Carbonation

Bottling or kegging transfers the conditioned beer away from yeast. Residual
yeast is dropped with fining agents such as gelatin or potassium metabisulphite,
which also binds oxygen that would otherwise oxidise the finished beer.

Carbonation may come from priming sugar, which the yeast converts to carbon
dioxide, or from forced carbonation with carbon dioxide gas. A beer bottled with
four grams of priming sugar per litre will reach roughly two and a half volumes
of carbonation.

Cold crashing to near zero degrees for a day or two before packaging forces the
yeast to flocculate, which reduces the yeast load carried into the package and
cuts the risk of over-carbonation."""
        ,
        """Water Chemistry and Grain Bill

Malt extract and mineral content are adjusted with gypsum and calcium chloride
to bring the mash liquor into a desirable range. Sulphate in the finished beer
tightens the mouthfeel and bittering perception, while chloride softens it.

A balanced grain bill for an all-grain pale ale pairs a two row base malt with
a small proportion of crystal malt for body and a larger share of hops for
bitterness. The mash temperature must be recalculated whenever the grist
changes, because a grist rich in crystal malt will gel and can foul the mash tun.

Fermentability is a ratio, not an absolute. A wort that started at 1.048 and
finished at 1.010 is about 79 percent apparent attenuation, which is normal for
an ale, and 1.002 would suggest the mash or the attenuation curve is off."""
        ,
    ],
    "glaciology.pdf": [
        """Alpine Glacier Dynamics

A valley glacier flows downhill under its own weight. The driving stress is a
function of the surface slope, the ice thickness and the gravitational
acceleration, and it is resisted by basal friction and internal deformation.

Ice deforms by dislocation creep, in which crystals deform plastically over
centuries, and by recrystallisation, which is faster and dominates at depth.
The resulting velocity profile is close to linear with depth near the bed and
steepens upward.

A glacier advances when accumulation in the accumulation zone exceeds ablation
below the equilibrium line. A surge is a short episode of dramatic acceleration
caused by water reaching the bed, which decouples the ice from its substrate."""
        ,
        """Moraine Deposition and Landforms

Material transported at the glacier surface is supraglacial debris, while debris
carried at or within the base is englacial. When a glacier retreats, it leaves
lateral moraines marking former ice margins and a terminal moraine at the snout.

Till is the unsorted, unstratified sediment deposited directly from the ice.
Its matrix-supported, poorly sorted texture is diagnostic of direct deposition,
in contrast to outwash, which is rounded, sorted and bedded because meltwater
reworked it.

Drumlins are streamlined hills of till whose long axis records the direction of
ice flow. They are common in areas of former continental ice sheets and are
useful for reconstructing palaeo-ice flow even where the ice itself is long gone."""
        ,
    ],
}

TEXT_DOCS = {
    "pharm-notes.txt": (
        "Pharmacology Study Notes: Antimicrobials\n"
        "\n"
        "Beta-lactam antibiotics require an intact beta-lactam ring. Penicillins are\n"
        "destroyed by penicillinase, so isoxazolyl penicillins such as oxacillin are used\n"
        "against Staphylococcus aureus. There is no cross reactivity between the\n"
        "penicillins and the cephalosporins, so a penicillin allergy does not imply a\n"
        "cephalosporin allergy.\n"
        "\n"
        "Aminoglycosides are concentration dependent killers. They require a high peak\n"
        "to achieve killing, and they are ineffective against anaerobes because uptake\n"
        "depends on oxygen dependent transport. They are ototoxic and nephrotoxic, so\n"
        "monitoring of both drug level and renal function is required.\n"
        "\n"
        "Vancomycin must be infused slowly over at least one hour to avoid red man\n"
        "syndrome, which is a histamine mediated reaction rather than true allergy.\n"
        "Trough levels should be checked for renally impaired patients.\n"
        "\n"
        "Fluoroquinolones inhibit DNA gyrase and topoisomerase IV. They are associated\n"
        "with tendinitis, peripheral neuropathy and aortic aneurysm, so they are now\n"
        "reserved for infections with no safer alternative.\n"
    ),
    "catalysis.txt": (
        "Heterogeneous Catalysis Notes\n"
        "\n"
        "The rate of a heterogeneous reaction depends on the fraction of surface atoms\n"
        "that are active sites. Turnover frequency, defined as molecules converted per\n"
        "active site per second, separates intrinsic activity from surface area.\n"
        "\n"
        "Sabatier's principle states that a catalyst is best when it binds the reactant\n"
        "and the product neither too strongly nor too weakly. Binding too weakly leaves\n"
        "the activation barrier intact; binding too strongly poisons the site and stops\n"
        "the cycle.\n"
        "\n"
        "The Langmuir Hinshelwood mechanism assumes both reactants adsorb before\n"
        "reacting, and is distinguished from the Eley Rideal mechanism, in which a gas\n"
        "phase species collides directly with an adsorbed species.\n"
        "\n"
        "Deactivation of a catalyst occurs by coking, sintering, poisoning by sulfur or\n"
        "lead, or by hydrothermal degradation. Sintering is driven by the migration of\n"
        "small particles that merge into larger ones, reducing surface area irreversibly.\n"
    ),
    "music.txt": (
        "Basso Continuo Practice\n"
        "\n"
        "The basso continuo was the bass line plus a set of figures that directed the\n"
        "harpsichordist's chordal filling. Figures were shorthand rather than full\n"
        "harmony, and their meaning was conventional within a region rather than fixed.\n"
        "\n"
        "A cadential six four was written as a bass note with a six and a four above it,\n"
        "and functioned as a dominant embellishment rather than a stable chord, delaying\n"
        "the cadence that follows it.\n"
        "\n"
        "The rule of the octave was a practical guide for harmonising a scale descending\n"
        "or ascending, assigning consonant intervals on strong beats and dissonant\n"
        "intervals on weak ones.\n"
        "\n"
        "Basso continuo practice implies a flexible disposition of tempo and\n"
        "ornamentation, so a modern performance should treat the figures as a floor for\n"
        "the harpsichordist rather than a ceiling on what is permitted.\n"
    ),
}

MD_DOCS = {
    "horticulture.md": """# Allotment Growing Guide

## Soil Preparation

Allotment soil is rarely in good condition on first arrival. Break it to a
depth of about thirty centimetres and incorporate well rotted organic matter.
Clay soils need generous compost every season; sandy soils need it less often but
lose nutrients faster.

## Brassicas

Plant brassicas in a firm bed to encourage roots that anchor the plant. Space
cabbages at forty-five centimetres and protect them from cabbage white with
fine mesh insect proofing from the moment of planting, because a single cabbage
white lays eggs that produce a generation of caterpillars.

Rotate brassicas on a four year cycle to starve the soil borne diseases that
build up under a continuous crop.

## Blight

Potato blight appears in warm wet weather. Remove and destroy affected foliage
rather than composting it, and never save seed potatoes from blighted plants.
Water in the morning so that leaves dry before nightfall.
""",
    "football.md": """# Squad Analysis: Pressing Patterns

## The Base Build

The side build from the back in a 2-3-5 shape. Both centre backs split wide to
give the pivot a free lane, and the pivot receives facing forward so the first
pass is a progressive one rather than sideways.

## Counter Pressing

Counter pressing starts within two seconds of losing the ball. The nearest
player delays the restart rather than diving in, and the second man screens the
most obvious forward pass. Counter pressing fails when the first defender
presses without cover, which leaves the pivot free to play long.

## Rest Defending

When counter pressing is broken the side drops into a back five, conceding
territory in exchange for a compact block. The block narrows to nineteen metres
in behind the ball and thirty five metres in front of it.

## Set Pieces

Defending corners uses a near post screen and a zonal marking system on the
six yard line. The two worst aerial defenders in the squad are assigned to the
zone that is crossed least often.
""",
}

RST_DOCS = {
    "mining.rst": """Operations Manual: Deep Shaft Ventilation
====================================

Purpose
-------

This manual defines the minimum ventilation standard for underground workings
below two thousand metres depth.

Primary ventilation
-------------------

A primary fan station must provide a mean throughflow of 3.0 metres per second
per thousand tonnes of daily production. Two independent fan stations are
required for depths beyond 1500 metres, arranged so that a single power failure
does not stop ventilation.

Auxiliary ventilation
---------------------

Auxiliary forcing is used for development headings. A flexible duct of at least
450 millimetres diameter is required beyond 300 metres from the main circuit,
and duct length beyond that distance is deducted from the calculated flow.

Atmosphere monitoring
---------------------

Methane is monitored continuously at every working face. The statutory action
level is 1.0 per cent, at which point electrical power is removed and all
personnel withdraw. Two per cent is the immediate withdrawal level with no
power restoration.
""",
}

HTML_DOCS = {
    "contract.html": """<!doctype html>
<html><head><title>Master Services Agreement</title></head><body>
<h1>Master Services Agreement</h1>
<p>This agreement is made between the Client and the Supplier and takes effect
on the date of last signature.</p>
<h2>1. Definitions</h2>
<p>"Deliverable" means any output identified as a deliverable in a statement of
work. "Acceptance" means the Client confirming in writing that a Deliverable
meets the acceptance criteria.</p>
<h2>2. Term and Termination</h2>
<p>This agreement continues for an initial term of thirty six months. Either party
may terminate for convenience on ninety days written notice. Either party may
terminate immediately for material breach that remains uncured after thirty
days notice.</p>
<h2>3. Limitation of Liability</h2>
<p>The total aggregate liability of either party is capped at the fees paid in
the twelve months preceding the claim, except for liability for death or
personal injury caused by negligence, and for breach of confidentiality, which
is uncapped.</p>
<h2>4. Payment Terms</h2>
<p>Invoices are payable within forty five days of receipt. The Client may
withhold payment for Deliverables under dispute while the parties negotiate.</p>
</body></html>""",
}

CSV_ROWS = """item,region,category,units_sold,unit_price,revenue,return_rate_pct
Espresso Blend,EMEA,Beverage,1240,12.50,15500.00,2.1
Espresso Blend,APAC,Beverage,1402,12.50,17525.00,5.8
Espresso Blend,AMER,Beverage,1610,12.50,20125.00,1.9
Filter Papers,EMEA,Accessory,3200,3.20,10240.00,0.8
Filter Papers,APAC,Accessory,2870,3.20,9184.00,2.4
Filter Papers,AMER,Accessory,3480,3.20,11136.00,0.6
Burr Grinder,EMEA,Equipment,64,289.00,18496.00,3.9
Burr Grinder,APAC,Equipment,41,289.00,11849.00,7.6
Burr Grinder,AMER,Equipment,88,289.00,25432.00,2.2
Cupping Spoon,EMEA,Accessory,910,7.40,6734.00,1.2
Cupping Spoon,APAC,Accessory,760,7.40,5624.00,3.3
Cupping Spoon,AMER,Accessory,1020,7.40,7548.00,0.9
Scale 2kg,EMEA,Equipment,112,145.00,16240.00,2.6
Scale 2kg,APAC,Equipment,95,145.00,13775.00,5.1
Scale 2kg,AMER,Equipment,134,145.00,19430.00,1.8
"""

TSV_ROWS = """sensor_id\tzone\treading\tunit\ttimestamp
TH-001\tstorefront\t21.4\tC\t2024-05-01T08:00:00
TH-002\tstorefront\t21.6\tC\t2024-05-01T08:00:00
TH-003\twarehouse\t17.9\tC\t2024-05-01T08:00:00
TH-004\twarehouse\t18.1\tC\t2024-05-01T08:00:00
TH-005\tcoldroom\t3.2\tC\t2024-05-01T08:00:00
TH-006\tcoldroom\t3.4\tC\t2024-05-01T08:00:00
TH-007\tcoldroom\t-19.5\tC\t2024-05-01T08:00:00
TH-008\tdelivery_bay\t14.0\tC\t2024-05-01T08:00:00
TH-009\tdelivery_bay\t14.3\tC\t2024-05-01T08:00:00
TH-010\tstorefront\t22.1\tC\t2024-05-01T12:00:00
TH-011\twarehouse\t19.4\tC\t2024-05-01T12:00:00
TH-012\tcoldroom\t-19.1\tC\t2024-05-01T12:00:00
"""

JSON_DOC = {
    "model": "wide-residual-cascade",
    "version": 7,
    "training": {
        "epochs": 140,
        "batch_size": 512,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "early_stopping_patience": 12,
    },
    "data": {
        "train_rows": 4820000,
        "validation_rows": 320000,
        "test_rows": 310000,
        "positive_rate": 0.042,
    },
    "features": {
        "categorical": ["merchant_category", "country", "device_type"],
        "numerical": ["amount", "hour_of_day", "days_since_signup"],
        "text": ["description"],
    },
    "thresholds": {"review": 0.35, "decline": 0.82},
    "monitoring": {"drift_alert_auc_drop": 0.03, "retrain_frequency_days": 14},
}

YAML_DOC = """service: ledger-api
version: 3.2.0
runtime:
  replicas: 6
  memory_limit_mb: 2048
  cpu_limit: 1500
  graceful_shutdown_seconds: 45
database:
  engine: postgres
  host: ledger-db.internal
  port: 5432
  pool_size: 40
  statement_timeout_ms: 8000
cache:
  backend: redis
  ttl_seconds: 600
  max_entries: 250000
rate_limit:
  requests_per_minute: 3000
  burst: 500
observability:
  log_level: info
  trace_sampling_rate: 0.05
features:
  - double_entry_enforcement
  - idempotency_keys
  - webhook_retries
"""

TOML_DOC = """[package]
name = "harvest-planner"
version = "0.9.3"
edition = "2021"

[dependencies]
serde = "1.0"
chrono = "0.4"
polars = "0.20"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
"""

INI_DOC = """[database]
host = warehouse.internal
port = 5432
pool_size = 25
connect_timeout = 10

[scheduler]
interval_seconds = 300
max_parallel_jobs = 4

[alerts]
channel = ops-primary
escalate_after_minutes = 15
"""

LOG_TEXT = """2024-11-03 06:00:01 INFO  scheduler: harvest plan run starting
2024-11-03 06:00:01 INFO  scheduler: loaded 14 fields from parcels_v3.parquet
2024-11-03 06:00:04 WARN  io: field soil_ph missing for 38 of 12480 rows
2024-11-03 06:00:04 INFO  io: imputed soil_ph with field median 6.4
2024-11-03 06:00:09 INFO  gis: reprojected 12480 geometries to EPSG:4326
2024-11-03 06:00:12 ERROR gis: 6 geometries failed self-intersection repair
2024-11-03 06:00:12 ERROR gis: rows 8811-8816 excluded from yield model
2024-11-03 06:00:18 INFO  model: training on 11864 rows, 22 features
2024-11-03 06:00:41 INFO  model: rmse 412.7 kg/ha, r2 0.71
2024-11-03 06:00:41 WARN  model: drift check flagged precipitation feature
2024-11-03 06:00:42 INFO  alert: posted summary to ops-primary
2024-11-03 06:00:42 INFO  scheduler: harvest plan run finished in 41s
"""

PY_DOC = '''"""Crop yield model training and evaluation."""

import math
from dataclasses import dataclass, field

RMSE_TARGET_KG_HA = 380.0
SPATIAL_FOLDS = 5


@dataclass
class FoldResult:
    index: int
    rmse: float
    r2: float
    excluded_rows: list = field(default_factory=list)


def spatial_folds(geometries, count: int = SPATIAL_FOLDS):
    """Split by spatial block so neighbouring plots cannot straddle a split."""
    ordered = sorted(range(len(geometries)), key=lambda i: geometries[i].centroid.x)
    size = math.ceil(len(ordered) / count)
    return [ordered[i * size : (i + 1) * size] for i in range(count)]


def impute_median(values, fallback):
    """Replace missing entries with the median of the observed values."""
    observed = sorted(v for v in values if v is not None)
    if not observed:
        return list(fallback)
    middle = len(observed) // 2
    median = (
        observed[middle]
        if len(observed) % 2
        else (observed[middle - 1] + observed[middle]) / 2
    )
    return [median if v is None else v for v in values]


def evaluate(model, train_index, test_index, target):
    """Score the model on one held out spatial fold."""
    predictions = model.predict(train_index, test_index)
    errors = [p - t for p, t in zip(predictions, target[test_index])]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    r2 = 1 - sum(errors) ** 2 / sum((t - sum(target[test_index]) / len(errors)) ** 2 for t in target[test_index])
    return FoldResult(index=test_index[0], rmse=rmse, r2=r2)
'''

JS_DOC = """// Cart totals with progressive discount tiers.
const TIERS = [
  { minSpend: 500, rate: 0.12 },
  { minSpend: 250, rate: 0.07 },
  { minSpend: 0, rate: 0 },
];

export function discountRate(subtotal) {
  const tier = TIERS.find((t) => subtotal >= t.minSpend);
  return tier ? tier.rate : 0;
}

export function applyVat(subtotal, vatRate = 0.2) {
  return round2(subtotal * (1 + vatRate));
}

export function round2(value) {
  return Math.round(value * 100) / 100;
}

export function cartTotal(lines, vatRate) {
  const subtotal = lines.reduce((sum, line) => sum + line.price * line.qty, 0);
  const discounted = subtotal * (1 - discountRate(subtotal));
  return { subtotal: round2(subtotal), total: applyVat(discounted, vatRate) };
}
"""

JAVA_DOC = """package com.example.shipping;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

public class CustomsDeclaration {

    private final String declarationNumber;
    private final LocalDate exportDate;
    private final List<LineItem> lineItems;

    public CustomsDeclaration(String declarationNumber, LocalDate exportDate, List<LineItem> lineItems) {
        this.declarationNumber = declarationNumber;
        this.exportDate = exportDate;
        this.lineItems = lineItems;
    }

    /** Value for duty: invoice value plus freight and insurance, excluding tax. */
    public BigDecimal customsValue(BigDecimal freight, BigDecimal insurance) {
        BigDecimal invoice = lineItems.stream()
            .map(LineItem::declaredValue)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        return invoice.add(freight).add(insurance);
    }

    public boolean requiresLicence(String countryOfDestination) {
        return lineItems.stream().anyMatch(item -> item.isControlled() && item.origin() != countryOfDestination);
    }

    public int lineCount() {
        return lineItems.size();
    }
}
"""

SQL_DOC = """-- Warehouse picking performance by zone.
SELECT
    z.zone_name,
    COUNT(p.pick_id) AS picks,
    ROUND(AVG(EXTRACT(EPOCH FROM (p.completed_at - p.started_at))), 1) AS avg_seconds,
    SUM(CASE WHEN p.completed_at IS NULL THEN 1 ELSE 0 END) AS abandoned
FROM picks p
JOIN zones z ON z.zone_id = p.zone_id
WHERE p.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY z.zone_name
HAVING COUNT(p.pick_id) > 50
ORDER BY avg_seconds DESC;
"""

QUESTIONS = {
    "brew-guide.pdf": {
        "relevant": [
            "What mash temperature favours a balanced body?",
            "How long must wort be boiled?",
            "What volume does priming sugar produce?",
            "mash tun temperature",
            "cold crashing",
        ],
        "trap": [
            "How do I replace a bicycle chain?",
            "What is the capital of Peru?",
        ],
    },
    "glaciology.pdf": {
        "relevant": [
            "What causes a glacier surge?",
            "How is till different from outwash?",
            "What do drumlins record?",
            "englacial debris",
        ],
        "trap": ["How do I change a tyre?", "Who wrote Hamlet?"],
    },
    "pharm-notes.txt": {
        "relevant": [
            "Why are aminoglycosides ineffective against anaerobes?",
            "What is red man syndrome?",
            "Which penicillins work against staph aureus?",
            "fluoroquinolone tendonitis",
        ],
        "trap": ["How do I bake bread?", "What is the capital of Peru?"],
    },
    "catalysis.txt": {
        "relevant": [
            "What is turnover frequency?",
            "State Sabatier's principle.",
            "How do Langmuir-Hinshelwood and Eley-Rideal differ?",
            "catalyst sintering",
        ],
        "trap": ["How do I knit a scarf?", "Who invented the telephone?"],
    },
    "music.txt": {
        "relevant": [
            "What was the basso continuo?",
            "What is a cadential six four?",
            "What is the rule of the octave?",
        ],
        "trap": ["How do I change a car tyre?", "What is the population of Tokyo?"],
    },
    "horticulture.md": {
        "relevant": [
            "How do I deal with cabbage white?",
            "What causes potato blight?",
            "Why rotate brassicas?",
            "blight compost",
        ],
        "trap": ["How do I fix a bicycle chain?", "Who won the World Cup?"],
    },
    "football.md": {
        "relevant": [
            "What shape does the side build from the back?",
            "When does counter pressing start?",
            "How do they defend corners?",
            "rest defending block size",
        ],
        "trap": ["How do I make pasta?", "What is the capital of Japan?"],
    },
    "mining.rst": {
        "relevant": [
            "What is the statutory methane action level?",
            "What duct length is required beyond 300 metres?",
            "How many fan stations are needed past 1500 metres?",
        ],
        "trap": ["How do I knit a scarf?", "Give me a pancake recipe"],
    },
    "contract.html": {
        "relevant": [
            "How long is the initial term?",
            "What is the liability cap?",
            "What are the payment terms?",
            "termination for convenience notice period",
        ],
        "trap": ["How do I bake sourdough?", "What is the tallest mountain?"],
    },
    "coffee_sales.csv": {
        "relevant": [
            "What was the revenue for Burr Grinder in AMER?",
            "Which region has the highest return rate?",
            "How many espresso blends were sold in APAC?",
            "Filter Papers return_rate_pct",
        ],
        "trap": ["What is the capital of France?", "How do I tie a bowline?"],
    },
    "temperatures.tsv": {
        "relevant": [
            "What is the coldroom temperature?",
            "Which zone is warmest?",
            "What did TH-012 read at midday?",
            "sensor TH-005",
        ],
        "trap": ["What is the boiling point of water?", "Who wrote Don Quixote?"],
    },
    "model.json": {
        "relevant": [
            "What is the learning rate?",
            "How many training rows are there?",
            "What is the decline threshold?",
            "early stopping patience",
        ],
        "trap": ["How do I make compost?", "What is the capital of Norway?"],
    },
    "app.yaml": {
        "relevant": [
            "What is the database pool size?",
            "How long is the graceful shutdown window?",
            "What is the statement timeout?",
            "trace_sampling_rate",
        ],
        "trap": ["How do I change a tyre?", "Who invented the printing press?"],
    },
    "pyproject.toml": {
        "relevant": [
            "What is the package version?",
            "Which compression is enabled in release builds?",
            "What dependencies are declared?",
        ],
        "trap": ["How do I bake bread?", "What is the population of Rome?"],
    },
    "warehouse.ini": {
        "relevant": [
            "What is the database pool size?",
            "How often does the scheduler run?",
            "When does it escalate alerts?",
        ],
        "trap": ["How do I knit a scarf?", "What is the capital of Peru?"],
    },
    "harvest.log": {
        "relevant": [
            "Why were some rows excluded from the yield model?",
            "What happened to the missing soil_ph values?",
            "What was the model rmse?",
            "GIS self-intersection failure",
        ],
        "trap": ["How do I make pasta?", "Who won the World Cup?"],
    },
    "train.py": {
        "relevant": [
            "How are spatial folds chosen?",
            "What does impute_median do with missing values?",
            "What is the rmse target?",
            "SPATIAL_FOLDS",
        ],
        "trap": ["How do I fix a bike?", "What is the capital of Peru?"],
    },
    "cart.js": {
        "relevant": [
            "What discount applies above 500?",
            "What is the default vat rate?",
            "How is the cart total computed?",
            "discountRate tiers",
        ],
        "trap": ["How do I bake bread?", "What is the capital of Peru?"],
    },
    "Customs.java": {
        "relevant": [
            "What does customsValue include?",
            "When is a licence required?",
            "How is the declaration number used?",
        ],
        "trap": ["How do I change a tyre?", "Who invented the telephone?"],
    },
    "picks.sql": {
        "relevant": [
            "What does this query order by?",
            "Which zones are included in the results?",
            "How is the average pick duration computed?",
        ],
        "trap": ["How do I knit a scarf?", "Give me a pancake recipe"],
    },
}

REAL_WORLD = {
    "uploaded PDF": {
        "file": None,
        "relevant": [
            "what does the Data Characteristics Handbook describe",
            "why are Data Release Notes produced each quarter",
            "what historical events degraded the data",
            "Barycentric Julian Date",
            "acronyms expanded",
        ],
        "trap": [
            "boiling point of water",
            "capital of France",
            "best pizza recipe",
            "three kepler laws of planetary motion",
        ],
    }
}

# Traps that deliberately share an entity with the document but ask about
# something the document does not contain. These are the cases a keyword or
# embedding model cannot catch, because the shared vocabulary is genuine.
ENTITY_TRAPS = {
    "brew-guide.pdf": ("home brewing", ["how does brewing differ from baking?"]),
    "pharm-notes.txt": ("antibiotics", ["which antibiotics treat tuberculosis?"]),
    "contract.html": ("agreement", ["how do I break a lease early?"]),
    "glaciology.pdf": ("glacier", ["how long does a glacier take to reach the sea?"]),
    "music.txt": ("basso continuo", ["who composed the Brandenburg Concertos?"]),
    "football.md": ("pressing", ["what is the offside rule?"]),
    "train.py": ("spatial folds", ["how do I install a python package?"]),
    "model.json": ("threshold", ["what is the legal blood alcohol limit?"]),
    "app.yaml": ("database pool", ["how do I optimise a postgres query?"]),
    "coffee_sales.csv": ("revenue", ["what is the inflation rate in Brazil?"]),
}


def _pdf(pages):
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 545, 760), text, fontsize=9)
    return doc


def _docx(paragraphs, table):
    from docx import Document

    word = Document()
    for style, text in paragraphs:
        word.add_paragraph(text, style=style)
    if table:
        grid = word.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    return word


def _needs_write(payload: bytes, path: Path) -> bool:
    """Whether `payload` differs in content from what is already at `path`.

    Compared on parsed text rather than bytes: the writers embed a creation
    timestamp, so a byte comparison reports a change on every run and rewrites a
    file whose content is identical, leaving the checkout permanently dirty.
    """
    if not path.exists():
        return True
    if path.read_bytes() == payload:
        return False
    with tempfile.TemporaryDirectory() as folder:
        candidate = Path(folder) / path.name
        candidate.write_bytes(payload)
        try:
            from app.parsers import parse as _parse

            fresh = " ".join(b.text for b in _parse(candidate))
            current = " ".join(b.text for b in _parse(path))
        except Exception:
            # If either will not parse, fall back to bytes. An unnecessary rewrite
            # is harmless; skipping a genuinely changed file is not.
            return True
    return current != fresh


def build(target):
    """Write the whole corpus into `target` and return {filename: questions}.

    A PDF or DOCX carries a creation timestamp, so regenerating one that is
    already there produces a different file with identical content and leaves the
    checkout permanently dirty. The corpus is committed, so only write a document
    that is missing or whose bytes actually differ.
    """
    target.mkdir(parents=True, exist_ok=True)
    written = {}

    for name, pages in PROSE_PAGES.items():
        doc = _pdf(pages)
        if _needs_write(doc.tobytes(), target / name):
            doc.save(target / name)
        doc.close()
        written[name] = name

    for name, text in TEXT_DOCS.items():
        (target / name).write_text(text, encoding="utf-8")
        written[name] = name

    for name, text in MD_DOCS.items():
        (target / name).write_text(text, encoding="utf-8")
        written[name] = name

    for name, text in RST_DOCS.items():
        (target / name).write_text(text, encoding="utf-8")
        written[name] = name

    for name, text in HTML_DOCS.items():
        (target / name).write_text(text, encoding="utf-8")
        written[name] = name

    (target / "coffee_sales.csv").write_text(CSV_ROWS, encoding="utf-8")
    (target / "temperatures.tsv").write_text(TSV_ROWS, encoding="utf-8")
    (target / "model.json").write_text(json.dumps(JSON_DOC, indent=2), encoding="utf-8")
    (target / "app.yaml").write_text(YAML_DOC, encoding="utf-8")
    (target / "pyproject.toml").write_text(TOML_DOC, encoding="utf-8")
    (target / "warehouse.ini").write_text(INI_DOC, encoding="utf-8")
    (target / "harvest.log").write_text(LOG_TEXT, encoding="utf-8")
    (target / "train.py").write_text(PY_DOC, encoding="utf-8")
    (target / "cart.js").write_text(JS_DOC, encoding="utf-8")
    (target / "Customs.java").write_text(JAVA_DOC, encoding="utf-8")
    (target / "picks.sql").write_text(SQL_DOC, encoding="utf-8")

    word = _docx(
        [
            ("Heading 1", "Agronomy Trial Protocol"),
            ("Normal", "Field trials are laid out as randomised complete blocks with four replicates per treatment, and the plot size is fixed so that the harvest machinery can service every plot without turning at the headland."),
            ("Heading 2", "Trial Design"),
            ("Normal", "The protocol owner is the trial manager, who signs off the randomisation seed before drilling so the layout can be reproduced exactly."),
            ("Heading 2", "Data Capture"),
            ("Normal", "Yield is recorded at a moisture content of fourteen percent, and the target recovery time for processing a trial is ten working days after harvest."),
        ],
        [
            ["treatment", "nitrogen_kg_ha", "owner", "days_to_harvest"],
            ["control", "0", "trial manager", "104"],
            ["low N", "60", "trial manager", "110"],
            ["high N", "140", "field lead", "118"],
            ["foliar", "0", "field lead", "101"],
        ],
    )
    buffer = io.BytesIO()
    word.save(buffer)
    if _needs_write(buffer.getvalue(), target / "agronomy.docx"):
        (target / "agronomy.docx").write_bytes(buffer.getvalue())

    written["agronomy.docx"] = "agronomy.docx"

    return written
