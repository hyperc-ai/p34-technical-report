import tqdm

DIFFICULTY_LEVEL = 6.5
GENERATE = True
LOAD = True
SAVE = False

BAD_FEATURE_ENABLED = False

DATA_SIZE = "medium"


import json
import math
import random

def sample_poisson(lam: float) -> int:
    """
    Pure-Python Poisson sampler using Knuth's algorithm.
    Good for lam up to ~100 (your case).
    """
    if lam <= 0:
        return 0

    L = math.exp(-lam)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= random.random()

    return k - 1
                        
def blend_weight(sv: float,
                 center: float = 10.0,
                 width: float = 0.4) -> float:
    """
    Smoothly blend from Poisson (low sv) to Gaussian (high sv)
    using a logistic function on sv (in log10 space).

    Returns w in [0, 1]:
      w ≈ 0 → mostly Poisson
      w ≈ 1 → mostly Gaussian
    """
    if sv <= 0:
        return 0.0

    x = math.log10(sv)
    x0 = math.log10(center)
    z = (x - x0) / width
    return 1.0 / (1.0 + math.exp(-z))  # logistic


def sample_sales_mixed(sv: float,
                       rel_sigma: float = 0.2) -> int:
    """
    Mixed model:
      - Poisson for low sv
      - Gaussian with relative stddev for high sv
      - Smooth probabilistic blend between them
    """
    if sv <= 0:
        return 0

    # probability of using Gaussian
    w_gauss = blend_weight(sv)

    # draw from one distribution or the other
    if random.random() < w_gauss:
        # Gaussian branch
        sigma = sv * rel_sigma
        val = random.gauss(mu=sv, sigma=sigma)
        val = max(0.0, val)
        return int(round(val))
    else:
        # Poisson branch
        return sample_poisson(lam=sv)


def pure_sampling_key(cluster_id, lcogs, qty_ordered):
    return f"{int(cluster_id)}_{int(lcogs)}_{int(qty_ordered)}"

import math
import random
import numpy as np

def sample_skewed_sales(
    sv: float,
    rel_sigma: float = 0.2,
    skew_mode: str = "none",  # "none", "left", "right", "extreme_left", "extreme_right"
    skew_strength: float = 2.0
) -> int:
    """
    Mixed model with controllable skewness:
      - Poisson for low sv
      - Gaussian with relative stddev for high sv
      - Smooth probabilistic blend between them
      - Additional skewness control via mode transformation
    
    Args:
        sv: sales velocity (expected value)
        rel_sigma: relative standard deviation for Gaussian mode
        skew_mode: type of skewness to apply
            - "none": symmetric (original behavior)
            - "left": skew left (heavy tail on low side)
            - "right": skew right (heavy tail on high side)
            - "extreme_left": very heavy left tail
            - "extreme_right": very heavy right tail
        skew_strength: strength of skewness transformation (higher = more skewed)
    
    Returns:
        int: sampled sales value
    """
    if sv <= 0:
        return 0

    # Probability of using Gaussian
    w_gauss = blend_weight(sv)

    if random.random() < w_gauss:
        # Gaussian branch with skewness
        sigma = sv * rel_sigma
        
        if skew_mode == "none":
            # Standard symmetric Gaussian
            val = random.gauss(mu=sv, sigma=sigma)
        
        elif skew_mode == "left":
            # Left skew: sample from log-normal transformed
            # Generate from standard normal, then apply asymmetric transformation
            z = random.gauss(0, 1)
            # Transform to create left skew (long tail on left)
            if z < 0:
                z = z * skew_strength
            val = sv + z * sigma
        
        elif skew_mode == "right":
            # Right skew: opposite transformation
            z = random.gauss(0, 1)
            if z > 0:
                z = z * skew_strength
            val = sv + z * sigma
        
        elif skew_mode == "extreme_left":
            # Very heavy left tail using log-normal
            # Sample from log-normal, then flip and shift
            log_sv = max(0.1, sv)
            log_sigma = rel_sigma
            val = np.random.lognormal(np.log(log_sv), log_sigma)
            val = 2 * sv - val  # Flip around mean
        
        elif skew_mode == "extreme_right":
            # Very heavy right tail using log-normal directly
            log_sv = max(0.1, sv)
            log_sigma = rel_sigma * skew_strength
            val = np.random.lognormal(np.log(log_sv), log_sigma)
        
        else:
            raise ValueError(f"Unknown skew_mode: {skew_mode}")
        
        val = max(0.0, val)
        return int(round(val))
    
    else:
        # Poisson branch (inherently has slight right skew)
        base_sample = sample_poisson(lam=sv)
        
        if skew_mode in ["left", "extreme_left"]:
            # For Poisson, simulate left skew by capping high values
            cap = int(sv * (2.0 - skew_strength * 0.3))
            return min(base_sample, cap)
        
        elif skew_mode in ["right", "extreme_right"]:
            # For Poisson, amplify right tail
            if base_sample > sv:
                excess = base_sample - sv
                return int(sv + excess * skew_strength)
        
        return base_sample


def sample_bimodal_sales(
    sv: float,
    rel_sigma: float = 0.2,
    mode_separation: float = 2.0,
    mode_mix: float = 0.5
) -> int:
    """
    Creates a bimodal distribution (two peaks).
    
    Args:
        sv: center sales velocity
        rel_sigma: relative standard deviation for each mode
        mode_separation: how far apart the modes are (in std devs)
        mode_mix: probability of sampling from first mode vs second (0.5 = equal)
    
    Returns:
        int: sampled sales value
    """
    if sv <= 0:
        return 0
    
    sigma = sv * rel_sigma
    separation = mode_separation * sigma
    
    if random.random() < mode_mix:
        # First mode (lower)
        mu = sv - separation / 2
    else:
        # Second mode (higher)
        mu = sv + separation / 2
    
    val = random.gauss(mu=mu, sigma=sigma)
    val = max(0.0, val)
    return int(round(val))


def sample_fat_tailed_sales(
    sv: float,
    rel_sigma: float = 0.2,
    tail_heaviness: float = 3.0
) -> int:
    """
    Creates a fat-tailed distribution using Student's t-distribution.
    Lower degrees of freedom = heavier tails.
    
    Args:
        sv: expected sales velocity
        rel_sigma: relative standard deviation
        tail_heaviness: inverse of degrees of freedom (higher = heavier tails)
            Recommended range: 1.0 (very heavy) to 10.0 (light)
    
    Returns:
        int: sampled sales value
    """
    if sv <= 0:
        return 0
    
    # Use Student's t with low degrees of freedom for heavy tails
    df = max(1.0, 30.0 / tail_heaviness)  # degrees of freedom
    
    # Sample from t-distribution and scale
    sigma = sv * rel_sigma
    t_sample = np.random.standard_t(df)
    val = sv + t_sample * sigma
    
    val = max(0.0, val)
    return int(round(val))


# Example usage in your data generation:
def generate_skewed_portfolio():
    """
    Example showing how to use the skewed distributions
    """
    # Portfolio with different skew types
    SCENARIOS = [
        {"sv": 5.0, "skew_mode": "left", "label": "Low SV, Left Skew"},
        {"sv": 20.0, "skew_mode": "right", "label": "High SV, Right Skew"},
        {"sv": 10.0, "skew_mode": "extreme_left", "label": "Mid SV, Extreme Left"},
        {"sv": 15.0, "skew_mode": "extreme_right", "label": "Mid SV, Extreme Right"},
    ]
    
    samples = []
    for scenario in SCENARIOS:
        scenario_samples = [
            sample_skewed_sales(
                sv=scenario["sv"],
                skew_mode=scenario["skew_mode"],
                skew_strength=2.5
            )
            for _ in range(1000)
        ]
        samples.append({
            "label": scenario["label"],
            "samples": scenario_samples,
            "mean": np.mean(scenario_samples),
            "std": np.std(scenario_samples),
            "skewness": float(np.mean([(x - np.mean(scenario_samples))**3 for x in scenario_samples]) / (np.std(scenario_samples)**3))
        })
    
    return samples

def sample_casino_sales(
    sv: float,
    rel_sigma: float = 0.3,
    casino_chance: float = 0.001,
    casino_sales: int = 20
):
    """
    Sample sales for casino-type products, which may have unique distribution characteristics.
    
    Args:
        sv: expected sales velocity
        rel_sigma: relative standard deviation
        casino_chance: probability of a casino-like event occurring
    Returns:
        int: sampled sales value
    """

    if random.random() < casino_chance:
        # Simulate a rare high-sales event
        return casino_sales

    return sample_fat_tailed_sales(
        sv=sv,
        rel_sigma=rel_sigma,
        tail_heaviness=5.0  # Moderate tail heaviness for casino products
    )

if DIFFICULTY_LEVEL < 4:
    sample_casino_sales = sample_sales_mixed
if DIFFICULTY_LEVEL < 3:
    sample_sales_mixed = sample_skewed_sales
if DIFFICULTY_LEVEL < 2:
    sample_fat_tailed_sales = sample_skewed_sales

# Visualize the distributions
def plot_skewed_distributions():
    """
    Visualize different skewness modes
    """
    import matplotlib.pyplot as plt
    
    samples_data = generate_skewed_portfolio()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, data in enumerate(samples_data):
        axes[idx].hist(data["samples"], bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(data["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {data["mean"]:.1f}')
        axes[idx].set_title(f'{data["label"]}\nSkewness: {data["skewness"]:.2f}')
        axes[idx].set_xlabel('Sales')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import math

def sine_multiplier(time_tick: int,
                    period: int = 20_000,
                    min_val: float = 0.3,
                    max_val: float = 1.2) -> float:
    """
    Returns a sine-wave-based multiplier in [min_val, max_val]
    with the given period, based on time_tick.
    """
    # Angular frequency
    omega = 2.0 * math.pi / period

    # Raw sine in [-1, 1]
    s = math.sin(omega * time_tick)

    # Normalize to [0, 1]
    s_norm = 0.5 * (s + 1.0)

    # Scale to [min_val, max_val]
    return min_val + s_norm * (max_val - min_val)


import random
import numpy as np

def generate_exponential_clusters(n_clusters, base_sv_range=(0.01, 20.0), sigma_range=(0.1, 5.0)):
    """
    Generate cluster parameters following an exponential pattern.
    
    Args:
        n_clusters: Number of clusters to generate
        base_sv_range: (min, max) for base SV values before randomization
        sigma_range: (min, max) for sigma values before randomization
    
    Returns:
        tuple: (SV_FIXED, SIGMAS, SAMPLERS)
    """
    # Generate exponentially spaced base values
    sv_min, sv_max = base_sv_range
    sigma_min, sigma_max = sigma_range
    
    # Create exponentially spaced values (log-scale)
    sv_base = np.logspace(np.log10(sv_min), np.log10(sv_max), n_clusters)
    
    # Generate sigmas with correlation to SV (higher SV tends to have higher sigma)
    # but with some randomization
    sigma_base = np.linspace(sigma_min, sigma_max, n_clusters)
    
    # Apply random variation (0.5x to 2.0x)
    SV_FIXED = [sv * random.uniform(0.5, 2.0) for sv in sv_base]
    SIGMAS = [sigma * random.uniform(0.5, 2.0) for sigma in sigma_base]
    
    # Available sampler types with their characteristics
    sampler_pool = [
        sample_sales_mixed,      # Standard mixed Poisson/Gaussian
        sample_skewed_sales,     # Left/right skewed
        sample_bimodal_sales,    # Two-mode distribution
        sample_fat_tailed_sales, # Heavy tails
        sample_casino_sales,     # Rare high-sales events
    ]
    
    # Assign samplers with weighted preferences based on SV magnitude
    SAMPLERS = []
    for i, sv in enumerate(SV_FIXED):
        if sv < 0.1:
            # Very low SV: prefer Poisson-like (mixed or skewed)
            weights = [0.5, 0.3, 0.1, 0.05, 0.05]
        elif sv < 1.0:
            # Low SV: mixed behavior
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        elif sv < 5.0:
            # Medium SV: more complex distributions
            weights = [0.2, 0.2, 0.25, 0.2, 0.15]
        else:
            # High SV: all distributions equally likely
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        sampler = random.choices(sampler_pool, weights=weights, k=1)[0]
        SAMPLERS.append(sampler)
    
    return SV_FIXED, SIGMAS, SAMPLERS


MAX_CLUSTERS = 50  # Or any arbitrary number

from collections import defaultdict

QUANTILE = 0.055  # 0.025
MARKET_BIAS_DIFFICULTY = 18  # dollars minimum LCOGS available on market

telemetry = []
sales_per_period = []  # for visual check only
START_PRICE = 28
price = START_PRICE  # selling price
if DIFFICULTY_LEVEL < 2:
    price = price * 1.3
holding_cost_per_unit = 2  # cost to hold one unit in inventory per week

# TODO: need to replicate this! The purpose of cluster is to create a different mode / non-nice function in this cluster
# then, cluster is being fitted with "good" convergence
# next, to test the "quantile" we test the entire cluster of this SKU - "all SKUs of this type will give us X return"
# need more "good behaving" clusters? Like with just normal distribution, or some easy features within (easy to learn "nice" functions)
# TODO: add heteroskedasticity within cluster? More QTY --> different distribution tail/skewness (or, +rel_sigma + -sv_shift may be OK)

pure_sampling_source = defaultdict(list)
time_pure_sampling_source = defaultdict(lambda: defaultdict(list))

do_remove_index = random.random() < 0.4  # for testing only
# do_remove_index = True  # for testing only

DISASTROUS_CLUSTER = 2  # cluster to add random disasters
DISASTER_COST = -100  # large negative profit event
ORDERS_PER_CLUSTER = 5  # was 5 --> very bad result
SKU=0

SKU_DUP_SIZE = 20  # to create more "realistic" data with multiple SKUs per order
LCOGS_SEEDS = 48*2
QTY_ORDERED_RANGE = 50  # was 50

if DATA_SIZE == "small":
    LCOGS_SEEDS = 2
    QTY_ORDERED_RANGE = 10
elif DATA_SIZE == "medium":
    LCOGS_SEEDS = 5
    QTY_ORDERED_RANGE = 30
elif DATA_SIZE == "large":
    LCOGS_SEEDS = 20
    QTY_ORDERED_RANGE = 50

import pandas as pd
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModernMarketSynthConfig:
    max_ticks: int = 100
    future_eval_fraction: float = 0.20
    cluster_count: int = 30
    cluster_sv_range: tuple[float, float] = (0.01, 3.0)
    cluster_sigma_range: tuple[float, float] = (0.1, 3.0)
    lcogs_seeds: int = 5
    qty_ordered_range: int = 40
    weekly_demand_n_weeks: int = 8

    market_observed_period: float = 100.0
    market_observed_phase: float = 0.10
    volume_period: float = 25.0
    volume_phase: float = 4.40
    market_latent_period: float = 100.0
    market_latent_phase: float = 0.90
    policy_period: float = 31.0
    policy_phase: float = 3.00
    policy_latent_period: float = 23.0
    policy_latent_phase: float = 1.30
    policy_sv_latent_period: float = 29.0
    policy_sv_latent_phase: float = 2.10

    lcogs_floor_base: float = 18.0
    lcogs_floor_amplitude: float = 8.50
    base_price: float = 34.0
    lcogs_draw_min: float = 2.0
    lcogs_draw_max: float = 38.0
    lcogs_draw_skew: float = 1.0
    base_orders_per_tick: float = 1000 / (30 * 5)
    volume_amplitude: float = 0.65
    holding_cost_per_unit: float = 0.5
    leftover_writeoff_fraction: float = 1.0
    latent_offer_edge_noise_scale: float = 0.18
    latent_offer_fragility_noise_scale: float = 0.10
    downside_regime_base_prob: float = 0.01
    downside_regime_max_prob: float = 0.28
    downside_regime_best_demand_mult: float = 0.90
    downside_regime_worst_demand_mult: float = 0.10
    downside_regime_base_extra_cost_frac: float = 0.00
    downside_regime_max_extra_cost_frac: float = 0.55

    market_latent_amplitude: float = 0.16
    market_latent_noise_amplitude: float = 0.10
    market_profit_trend_strength: float = 1.75
    market_trend_segments: tuple[dict[str, float | int], ...] | None = None
    # Selection is now an ex-ante policy:
    # accept if noisy utility(margin, sv) clears a latent strictness threshold.
    # By default only the threshold process moves over time; weight drift is off.
    policy_threshold_base: float = 1.90
    market_policy_coupling: float = 0.00
    policy_threshold_amplitude: float = 0.30
    policy_threshold_noise_amplitude: float = 0.08
    policy_latent_amplitude: float = 0.65
    policy_latent_noise_amplitude: float = 0.10
    policy_sv_latent_amplitude: float = 0.45
    policy_sv_latent_noise_amplitude: float = 0.10
    policy_margin_interaction: float = 0.0
    policy_sv_interaction: float = 0.0
    # Legacy target-rate knobs are kept only so older configs still load.
    # The simplified policy does not force same-day acceptance to match them.
    target_acceptance_rate_min: float = 0.02
    target_acceptance_rate_max: float = 0.10
    target_acceptance_rate_scale: float = 1.35
    target_acceptance_rate_market_coupling: float = 0.90
    policy_score_noise_scale: float = 0.75
    policy_exploration_accept_rate: float = 0.01
    business_qty_mode: str = "sv_coverage"
    business_qty_coverage_low_sv: float = 2.00
    business_qty_coverage_mid_sv: float = 1.60
    business_qty_coverage_high_sv: float = 1.25
    business_qty_lognormal_sigma: float = 0.0
    business_qty_beta_alpha: float = 4.5
    business_qty_beta_beta: float = 2.5
    business_qty_horizon_weeks: float = 2.0
    business_qty_rel_sigma_low_sv: float = 0.60
    business_qty_rel_sigma_mid_sv: float = 0.35
    business_qty_rel_sigma_high_sv: float = 0.20
    business_qty_inventory_risk_aversion: float = 1.00
    business_operational_cost_per_unit: float = 2.5
    accepted_safe_unsafe_labels_from_business_qty: bool = False
    low_sv_threshold: float = 0.5
    mid_sv_threshold: float = 3.0
    rel_sigma_min_cap: float = 0.05
    rel_sigma_low_cap: float = 1.00
    rel_sigma_mid_cap: float = 0.55
    rel_sigma_high_cap: float = 0.25
    weekly_demand_low_rho: float = 0.35
    weekly_demand_mid_rho: float = 0.60
    weekly_demand_high_rho: float = 0.82
    weekly_demand_low_latent_rel_sigma: float = 0.30
    weekly_demand_mid_latent_rel_sigma: float = 0.18
    weekly_demand_high_latent_rel_sigma: float = 0.08

    disable_cluster_drop_for_testing: bool = True


DEFAULT_SYNTH_CONFIG = ModernMarketSynthConfig()

NON_MODEL_COLUMNS = {
    "profit",
    "business_profit_true",
    "raw_profit_true",
    "business_qty",
    "sales_true",
    "qty_left_true",
    "business_sales_true",
    "business_qty_left_true",
    "business_demand_censored_true",
    "label_observed",
    "inventory_cost_basis",
    "latent_offer_edge",
    "latent_offer_fragility",
    "downside_regime_prob_true",
    "downside_regime_demand_mult_true",
    "downside_regime_extra_cost_frac_true",
    "downside_regime_triggered_true",
    "downside_regime_extra_cost_true",
    "date",
    "sku_cluster",
    "key",
    "accepted",
    "accepted_by_policy",
    "accepted_via_exploration",
    "policy_gap_to_threshold",
}


def _effective_rel_sigma(
    sv_effective: float,
    raw_rel_sigma: float,
    config: ModernMarketSynthConfig,
) -> float:
    raw_rel_sigma = float(raw_rel_sigma)
    if sv_effective < float(config.low_sv_threshold):
        rel_sigma_cap = float(config.rel_sigma_low_cap)
    elif sv_effective < float(config.mid_sv_threshold):
        rel_sigma_cap = float(config.rel_sigma_mid_cap)
    else:
        rel_sigma_cap = float(config.rel_sigma_high_cap)
    return float(max(float(config.rel_sigma_min_cap), min(raw_rel_sigma, rel_sigma_cap)))


def _business_qty_rel_sigma_for_sv(sv_observed: float, config: ModernMarketSynthConfig) -> float:
    if sv_observed < float(config.low_sv_threshold):
        business_qty_rel_sigma = float(config.business_qty_rel_sigma_low_sv)
    elif sv_observed < float(config.mid_sv_threshold):
        business_qty_rel_sigma = float(config.business_qty_rel_sigma_mid_sv)
    else:
        business_qty_rel_sigma = float(config.business_qty_rel_sigma_high_sv)
    assert business_qty_rel_sigma >= 0.0, "business_qty_rel_sigma must be non-negative"
    return float(business_qty_rel_sigma)


def _business_qty_for_sv(sv_observed: float, max_qty: int, config: ModernMarketSynthConfig) -> int:
    assert max_qty >= 1, "max_qty must be >= 1"
    assert sv_observed >= 0.0, "sv_observed must be non-negative"
    business_qty_mode = str(config.business_qty_mode)
    if business_qty_mode == "sv_coverage":
        if sv_observed < float(config.low_sv_threshold):
            coverage_weeks = float(config.business_qty_coverage_low_sv)
        elif sv_observed < float(config.mid_sv_threshold):
            coverage_weeks = float(config.business_qty_coverage_mid_sv)
        else:
            coverage_weeks = float(config.business_qty_coverage_high_sv)
        business_qty_center = max(1.0, float(sv_observed) * coverage_weeks)
        business_qty_lognormal_sigma = float(config.business_qty_lognormal_sigma)
        assert business_qty_lognormal_sigma >= 0.0, "business_qty_lognormal_sigma must be non-negative"
        if business_qty_lognormal_sigma > 0.0:
            lognormal_multiplier = random.lognormvariate(
                -0.5 * (business_qty_lognormal_sigma ** 2),
                business_qty_lognormal_sigma,
            )
            business_qty_center *= float(lognormal_multiplier)
        business_qty = int(round(business_qty_center))
    elif business_qty_mode == "beta_dist":
        business_qty_beta_alpha = float(config.business_qty_beta_alpha)
        business_qty_beta_beta = float(config.business_qty_beta_beta)
        assert business_qty_beta_alpha > 0.0, "business_qty_beta_alpha must be positive"
        assert business_qty_beta_beta > 0.0, "business_qty_beta_beta must be positive"
        business_qty_share = random.betavariate(business_qty_beta_alpha, business_qty_beta_beta)
        business_qty = int(1 + round(float(business_qty_share) * float(max_qty - 1)))
    elif business_qty_mode == "conservative_sv":
        business_qty_horizon_weeks = float(config.business_qty_horizon_weeks)
        business_qty_inventory_risk_aversion = float(config.business_qty_inventory_risk_aversion)
        assert business_qty_horizon_weeks > 0.0, "business_qty_horizon_weeks must be positive"
        assert business_qty_inventory_risk_aversion >= 0.0, "business_qty_inventory_risk_aversion must be non-negative"
        demand_mean = float(sv_observed) * business_qty_horizon_weeks
        demand_rel_sigma = _business_qty_rel_sigma_for_sv(
            sv_observed=float(sv_observed),
            config=config,
        )
        demand_sigma = float(demand_mean * demand_rel_sigma)
        business_qty = int(round(demand_mean - (business_qty_inventory_risk_aversion * demand_sigma)))
    else:
        raise ValueError(f"Unknown business_qty_mode: {business_qty_mode}")
    return int(min(max(1, business_qty), max_qty))


def _simulate_profit_for_weekly_demand_path(
    weekly_demand_path: list[float],
    price: float,
    lcogs: float,
    qty_ordered: float,
    config: ModernMarketSynthConfig,
) -> tuple[float, float, float]:
    profit = 0.0
    sales = 0.0
    qty_left = float(qty_ordered)
    for demand_this_week in weekly_demand_path:
        profit -= qty_left * float(config.holding_cost_per_unit)
        sales_this_week = min(float(demand_this_week), qty_left)
        sales += sales_this_week
        profit += sales_this_week * (float(price) - float(lcogs))
        qty_left -= sales_this_week
    assert qty_left >= -1e-9, "qty_left must stay non-negative"
    qty_left = max(0.0, qty_left)
    profit -= qty_left * float(lcogs) * float(config.leftover_writeoff_fraction)
    return float(profit), float(qty_left), float(sales)


def _inventory_cost_basis(lcogs: float, qty_ordered: float) -> float:
    inventory_cost_basis = float(qty_ordered) * float(lcogs)
    assert inventory_cost_basis > 0.0, "inventory_cost_basis must be positive"
    return float(inventory_cost_basis)


def _observed_label_mask(full_df: pd.DataFrame, config: ModernMarketSynthConfig) -> pd.Series:
    accepted_mask = full_df["accepted"].astype(bool)
    if not bool(config.accepted_safe_unsafe_labels_from_business_qty):
        return accepted_mask

    required_cols = {"qty", "business_qty", "business_demand_censored_true"}
    missing_cols = required_cols.difference(full_df.columns)
    if missing_cols:
        raise ValueError(
            f"accepted_safe_unsafe_labels_from_business_qty requires columns: {sorted(missing_cols)}"
        )

    qty = pd.to_numeric(full_df["qty"], errors="coerce")
    business_qty = pd.to_numeric(full_df["business_qty"], errors="coerce")
    assert qty.notna().all(), "qty must be finite for safe/unsafe label routing"
    assert business_qty.notna().all(), "business_qty must be finite for safe/unsafe label routing"
    censored_mask = full_df["business_demand_censored_true"].astype(bool)
    safe_mask = (~censored_mask) | (qty <= (business_qty + 1e-9))
    return accepted_mask & safe_mask


def _sample_lcogs(
    lcogs_draw_min: float,
    lcogs_draw_max: float,
    config: ModernMarketSynthConfig,
) -> float:
    lcogs_draw_min = float(lcogs_draw_min)
    lcogs_draw_max = float(lcogs_draw_max)
    lcogs_draw_skew = float(config.lcogs_draw_skew)
    assert lcogs_draw_max > lcogs_draw_min, "lcogs_draw_max must exceed lcogs_draw_min"
    assert lcogs_draw_skew >= 1.0, "lcogs_draw_skew must be >= 1.0"
    if lcogs_draw_skew == 1.0:
        return float(random.uniform(lcogs_draw_min, lcogs_draw_max))
    draw_u = 1.0 - (random.random() ** lcogs_draw_skew)
    return float(lcogs_draw_min + ((lcogs_draw_max - lcogs_draw_min) * draw_u))


def _latent_offer_edge(
    price: float,
    lcogs: float,
    sv_observed: float,
    raw_rel_sigma: float,
    config: ModernMarketSynthConfig,
    edge_noise: float | None = None,
) -> float:
    margin_ratio = (float(price) - float(lcogs)) / max(float(price), 1e-9)
    sv_low = max(float(config.cluster_sv_range[0]), 1e-9)
    sv_high = max(float(config.cluster_sv_range[1]), sv_low)
    sigma_low = float(config.cluster_sigma_range[0])
    sigma_high = max(float(config.cluster_sigma_range[1]), sigma_low + 1e-9)
    z_log_sv = (
        math.log1p(max(float(sv_observed), 0.0))
        - ((math.log1p(sv_low) + math.log1p(sv_high)) / 2.0)
    ) / max((math.log1p(sv_high) - math.log1p(sv_low)) / 4.0, 1e-9)
    z_sigma = (
        float(raw_rel_sigma)
        - ((sigma_low + sigma_high) / 2.0)
    ) / max((sigma_high - sigma_low) / 4.0, 1e-9)
    if edge_noise is None:
        edge_noise = random.gauss(0.0, float(config.latent_offer_edge_noise_scale))
    return float((2.4 * margin_ratio) + (0.55 * z_log_sv) - (0.35 * z_sigma) + float(edge_noise))


def _latent_offer_fragility(
    offer_edge: float,
    config: ModernMarketSynthConfig,
    fragility_noise: float | None = None,
) -> float:
    if fragility_noise is None:
        fragility_noise = random.gauss(0.0, float(config.latent_offer_fragility_noise_scale))
    fragility_logit = float(-offer_edge + float(fragility_noise))
    fragility_logit = max(-40.0, min(40.0, fragility_logit))
    return float(1.0 / (1.0 + math.exp(-fragility_logit)))


def _sample_offer_downside_regime(
    offer_fragility: float,
    config: ModernMarketSynthConfig,
) -> dict:
    base_prob = float(config.downside_regime_base_prob)
    max_prob = float(config.downside_regime_max_prob)
    best_demand_mult = float(config.downside_regime_best_demand_mult)
    worst_demand_mult = float(config.downside_regime_worst_demand_mult)
    base_extra_cost_frac = float(config.downside_regime_base_extra_cost_frac)
    max_extra_cost_frac = float(config.downside_regime_max_extra_cost_frac)
    assert 0.0 <= base_prob <= max_prob <= 1.0, "downside regime probabilities must satisfy 0 <= base <= max <= 1"
    assert 0.0 < worst_demand_mult <= best_demand_mult <= 1.0, "downside regime demand multipliers must satisfy 0 < worst <= best <= 1"
    assert 0.0 <= base_extra_cost_frac <= max_extra_cost_frac, "downside regime extra cost fractions must satisfy 0 <= base <= max"
    fragility = float(min(max(float(offer_fragility), 0.0), 1.0))
    downside_regime_prob = float(base_prob + fragility * (max_prob - base_prob))
    if random.random() >= downside_regime_prob:
        return {
            "downside_regime_prob": downside_regime_prob,
            "downside_regime_triggered": False,
            "downside_regime_demand_mult": 1.0,
            "downside_regime_extra_cost_frac": 0.0,
        }
    fragility_severity = random.uniform(0.35, 1.0)
    severity = float(fragility * fragility_severity)
    downside_regime_demand_mult = float(best_demand_mult - severity * (best_demand_mult - worst_demand_mult))
    downside_regime_extra_cost_frac = float(base_extra_cost_frac + severity * (max_extra_cost_frac - base_extra_cost_frac))
    return {
        "downside_regime_prob": downside_regime_prob,
        "downside_regime_triggered": True,
        "downside_regime_demand_mult": downside_regime_demand_mult,
        "downside_regime_extra_cost_frac": downside_regime_extra_cost_frac,
    }


def _apply_offer_downside_to_weekly_demand_path(
    weekly_demand_path: list[float],
    downside_regime: dict,
) -> list[float]:
    demand_mult = float(downside_regime["downside_regime_demand_mult"])
    assert demand_mult > 0.0, "downside_regime_demand_mult must be positive"
    return [float(demand) * demand_mult for demand in weekly_demand_path]


def _apply_offer_downside_extra_cost(
    profit_before_downside: float,
    inventory_cost_basis: float,
    downside_regime: dict,
) -> tuple[float, float]:
    assert inventory_cost_basis > 0.0, "inventory_cost_basis must be positive"
    downside_regime_extra_cost_frac = float(downside_regime["downside_regime_extra_cost_frac"])
    assert downside_regime_extra_cost_frac >= 0.0, "downside_regime_extra_cost_frac must be non-negative"
    downside_regime_extra_cost = float(inventory_cost_basis * downside_regime_extra_cost_frac)
    return float(profit_before_downside - downside_regime_extra_cost), downside_regime_extra_cost


def _profit_from_grounding_inputs(
    weekly_demand_path: list[float],
    price: float,
    lcogs: float,
    qty_ordered: float,
    extra_cost_frac: float,
    config: ModernMarketSynthConfig,
) -> tuple[float, float, float, float]:
    inventory_cost_basis = _inventory_cost_basis(lcogs=lcogs, qty_ordered=qty_ordered)
    profit, qty_left, sales = _simulate_profit_for_weekly_demand_path(
        weekly_demand_path=weekly_demand_path,
        price=price,
        lcogs=lcogs,
        qty_ordered=qty_ordered,
        config=config,
    )
    extra_cost = float(inventory_cost_basis * float(extra_cost_frac))
    return float(profit - extra_cost), float(qty_left), float(sales), float(inventory_cost_basis)


def _effective_sampler_for_sv(sampler, sv_effective: float):
    if sv_effective >= 3.0:
        if sampler in {sample_fat_tailed_sales, sample_bimodal_sales, sample_casino_sales}:
            if random.random() < 0.85:
                return sample_sales_mixed
        if sampler == sample_skewed_sales and random.random() < 0.50:
            return sample_sales_mixed
    elif sv_effective >= 1.0:
        if sampler == sample_casino_sales and random.random() < 0.85:
            return sample_sales_mixed
        if sampler in {sample_fat_tailed_sales, sample_bimodal_sales} and random.random() < 0.60:
            return sample_sales_mixed
    return sampler


def _generate_weekly_demand_path(
    sv_effective: float,
    raw_rel_sigma: float,
    sampler,
    config: ModernMarketSynthConfig,
    n_weeks: int = 20,
) -> list[float]:
    assert n_weeks > 0, "n_weeks must be > 0"
    if sv_effective <= 0.0:
        return [0.0] * n_weeks

    effective_rel_sigma = _effective_rel_sigma(sv_effective, raw_rel_sigma, config=config)
    effective_sampler = _effective_sampler_for_sv(sampler, sv_effective)

    if sv_effective < float(config.low_sv_threshold):
        rho = float(config.weekly_demand_low_rho)
        latent_rel_sigma = float(config.weekly_demand_low_latent_rel_sigma)
    elif sv_effective < float(config.mid_sv_threshold):
        rho = float(config.weekly_demand_mid_rho)
        latent_rel_sigma = float(config.weekly_demand_mid_latent_rel_sigma)
    else:
        rho = float(config.weekly_demand_high_rho)
        latent_rel_sigma = float(config.weekly_demand_high_latent_rel_sigma)

    latent_mean = float(sv_effective)
    path = []
    for _ in range(n_weeks):
        innovation = random.gauss(0.0, sv_effective * latent_rel_sigma)
        latent_mean = max(
            0.0,
            (rho * latent_mean) + ((1.0 - rho) * sv_effective) + innovation,
        )
        path.append(float(effective_sampler(latent_mean, rel_sigma=effective_rel_sigma)))

    return path


def _cycle_value(time_tick: int, period: float, phase: float) -> float:
    assert period > 0.0, f"period must be > 0, got {period}"
    return math.sin((2.0 * math.pi * float(time_tick) / float(period)) + float(phase))


def _compute_test_dates(unique_dates, future_eval_fraction: float):
    unique_dates = pd.Series(unique_dates).dropna().sort_values().unique()
    n_test_dates = max(1, int(np.ceil(len(unique_dates) * future_eval_fraction)))
    test_dates = set(unique_dates[-n_test_dates:])
    assert len(test_dates) > 0, "test_dates must not be empty"
    return test_dates


def _generate_smooth_noise(n_rows: int, rng: np.random.Generator, rho: float = 0.88) -> np.ndarray:
    assert n_rows > 0, "n_rows must be > 0"
    noise = np.zeros(n_rows, dtype=float)
    innovation_scale = math.sqrt(max(1e-9, 1.0 - rho ** 2))
    for idx in range(1, n_rows):
        noise[idx] = rho * noise[idx - 1] + innovation_scale * float(rng.normal())
    noise_std = float(noise.std(ddof=0))
    if not np.isfinite(noise_std) or noise_std <= 1e-12:
        return np.zeros(n_rows, dtype=float)
    return (noise - float(noise.mean())) / noise_std


def _build_market_trend_path(config: ModernMarketSynthConfig) -> np.ndarray:
    if not config.market_trend_segments:
        return -float(config.market_profit_trend_strength) * np.linspace(0.0, 1.0, config.max_ticks)
    trend = np.zeros(config.max_ticks, dtype=float)
    level = 0.0
    next_start = 0
    for segment in config.market_trend_segments:
        start = int(segment["start"])
        end = int(segment["end"])
        strength = float(segment["strength"])
        assert 0 <= start <= end < config.max_ticks, f"market_trend_segments has invalid bounds: {segment}"
        assert np.isfinite(strength), f"market_trend_segments strength must be finite: {segment}"
        assert start == next_start, f"market_trend_segments must cover all dates with no gaps/overlaps: expected next start {next_start}, got {start}"
        n_ticks = end - start + 1
        if n_ticks == 1:
            trend[start] = level - strength
        else:
            trend[start : end + 1] = level - strength * np.linspace(0.0, 1.0, n_ticks)
        level = float(trend[end])
        next_start = end + 1
    assert next_start == config.max_ticks, f"market_trend_segments must cover final date {config.max_ticks - 1}"
    return trend


def _build_date_process_df(config: ModernMarketSynthConfig, random_state: int) -> pd.DataFrame:
    assert config.max_ticks > 0, "max_ticks must be > 0"
    assert np.isfinite(config.market_profit_trend_strength), "market_profit_trend_strength must be finite"
    market_trend_path = _build_market_trend_path(config)
    rng = np.random.default_rng(random_state + 1729)
    market_latent_noise = _generate_smooth_noise(config.max_ticks, rng=rng, rho=0.90)
    policy_latent_noise = _generate_smooth_noise(config.max_ticks, rng=rng, rho=0.86)
    policy_sv_latent_noise = _generate_smooth_noise(config.max_ticks, rng=rng, rho=0.84)
    policy_threshold_noise = _generate_smooth_noise(config.max_ticks, rng=rng, rho=0.84)

    records = []
    for time_tick in range(config.max_ticks):
        market_observed_t = _cycle_value(
            time_tick=time_tick,
            period=config.market_observed_period,
            phase=config.market_observed_phase,
        )
        volume_cycle_t = _cycle_value(
            time_tick=time_tick,
            period=config.volume_period,
            phase=config.volume_phase,
        )
        market_trend_t = float(market_trend_path[time_tick])
        market_latent_t = config.market_latent_amplitude * _cycle_value(
            time_tick=time_tick,
            period=config.market_latent_period,
            phase=config.market_latent_phase,
        ) + config.market_latent_noise_amplitude * float(market_latent_noise[time_tick]) + market_trend_t
        policy_cycle_t = _cycle_value(
            time_tick=time_tick,
            period=config.policy_period,
            phase=config.policy_phase,
        )
        policy_latent_t = config.policy_latent_amplitude * _cycle_value(
            time_tick=time_tick,
            period=config.policy_latent_period,
            phase=config.policy_latent_phase,
        ) + config.policy_latent_noise_amplitude * float(policy_latent_noise[time_tick])
        policy_sv_latent_t = config.policy_sv_latent_amplitude * _cycle_value(
            time_tick=time_tick,
            period=config.policy_sv_latent_period,
            phase=config.policy_sv_latent_phase,
        ) + config.policy_sv_latent_noise_amplitude * float(policy_sv_latent_noise[time_tick])
        lcogs_floor_t = config.lcogs_floor_base + config.lcogs_floor_amplitude * market_observed_t
        offer_volume_t = int(round(config.base_orders_per_tick * (1.0 + config.volume_amplitude * volume_cycle_t)))
        policy_threshold_t = (
            config.policy_threshold_base
            + config.policy_threshold_amplitude * policy_cycle_t
            + config.policy_threshold_noise_amplitude * float(policy_threshold_noise[time_tick])
        )
        # `policy_threshold_t` is the hidden day-level strictness knob `a_t`.
        # Market difficulty should still drive most acceptance variation through
        # changing offer composition, while this latent threshold adds only
        # moderate business-side drift by default.
        policy_margin_weight_t = 1.0 + config.policy_margin_interaction * policy_latent_t
        policy_sv_weight_t = 1.0 + config.policy_sv_interaction * policy_sv_latent_t
        records.append(
            {
                "date": float(time_tick),
                "market_observed_t": float(market_observed_t),
                "volume_cycle_t": float(volume_cycle_t),
                "market_trend_t": float(market_trend_t),
                "market_latent_t": float(market_latent_t),
                "policy_cycle_t": float(policy_cycle_t),
                "policy_threshold_t": float(policy_threshold_t),
                "policy_latent_t": float(policy_latent_t),
                "policy_sv_latent_t": float(policy_sv_latent_t),
                "policy_margin_weight_t": float(policy_margin_weight_t),
                "policy_sv_weight_t": float(policy_sv_weight_t),
                "policy_preference_t": float(policy_margin_weight_t - policy_sv_weight_t),
                "lcogs_floor_t": float(lcogs_floor_t),
                "offer_volume_t": int(max(1, offer_volume_t)),
                "effective_demand_multiplier_t": float(math.exp(market_latent_t)),
            }
        )

    date_process_df = pd.DataFrame(records)
    assert len(date_process_df) == config.max_ticks, "date_process_df length mismatch"
    assert np.isfinite(date_process_df.select_dtypes(include=[np.number]).to_numpy()).all(), "date_process_df contains non-finite values"
    assert (date_process_df["offer_volume_t"] >= 1).all(), "offer_volume_t must be >= 1"
    assert (date_process_df["policy_margin_weight_t"] > 0.05).all(), "policy_margin_weight_t must stay positive"
    assert (date_process_df["policy_sv_weight_t"] > 0.05).all(), "policy_sv_weight_t must stay positive"
    assert np.allclose(
        date_process_df["market_trend_t"].to_numpy(),
        market_trend_path,
    ), "market_trend_t must match the configured linear trend path"
    return date_process_df


def _safe_zscore(values: pd.Series, mean_val: float, std_val: float) -> pd.Series:
    std_val = float(std_val)
    if not np.isfinite(std_val) or std_val <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(mean_val)) / std_val


def _build_policy_tables(
    full_df: pd.DataFrame,
    date_process_df: pd.DataFrame,
    config: ModernMarketSynthConfig,
):
    # See documentation.txt for the intended post-redesign business-selection
    # semantics: ex-ante observable acceptance signals, latent strictness drift
    # via policy_threshold_t by default, and ex-post realized business profit.
    key_policy_df = full_df[["key", "date", "price", "lcogs", "sv"]].drop_duplicates(subset=["key"]).copy()
    assert key_policy_df["key"].is_unique, "key_policy_df must have unique keys"

    unique_dates = pd.Series(full_df["date"]).dropna().sort_values().unique()
    test_dates = _compute_test_dates(unique_dates, config.future_eval_fraction)
    historical_dates = set(unique_dates) - test_dates
    assert historical_dates, "historical_dates must not be empty"

    key_policy_df["margin"] = key_policy_df["price"] - key_policy_df["lcogs"]
    key_policy_df["log_sv_observed"] = np.log1p(key_policy_df["sv"])
    key_policy_df = key_policy_df.merge(
        date_process_df[
            [
                "date",
                "market_observed_t",
                "market_latent_t",
                "policy_threshold_t",
                "policy_latent_t",
                "policy_sv_latent_t",
                "policy_margin_weight_t",
                "policy_sv_weight_t",
                "policy_preference_t",
                "offer_volume_t",
                "lcogs_floor_t",
                "effective_demand_multiplier_t",
            ]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    key_policy_historical = key_policy_df[key_policy_df["date"].isin(historical_dates)].copy()
    assert len(key_policy_historical) > 0, "No historical rows available for policy standardization"

    margin_mean = float(key_policy_historical["margin"].mean())
    margin_std = float(key_policy_historical["margin"].std(ddof=0))
    log_sv_mean = float(key_policy_historical["log_sv_observed"].mean())
    log_sv_std = float(key_policy_historical["log_sv_observed"].std(ddof=0))

    key_policy_df["z_margin"] = _safe_zscore(key_policy_df["margin"], margin_mean, margin_std)
    key_policy_df["z_log_sv_observed"] = _safe_zscore(
        key_policy_df["log_sv_observed"],
        log_sv_mean,
        log_sv_std,
    )
    # Acceptance is based only on ex-ante observable offer signals.
    # Realized business profit is computed later from the chosen operational qty
    # and is never used inside the acceptance rule itself.
    key_policy_df["policy_score_obs"] = key_policy_df["z_margin"] + key_policy_df["z_log_sv_observed"]
    key_policy_df["policy_score_true_raw"] = (
        key_policy_df["policy_margin_weight_t"] * key_policy_df["z_margin"]
        + key_policy_df["policy_sv_weight_t"] * key_policy_df["z_log_sv_observed"]
    )
    key_policy_df["policy_score_true"] = key_policy_df["policy_score_true_raw"]
    policy_score_noise_scale = float(config.policy_score_noise_scale)
    assert policy_score_noise_scale >= 0.0, "policy_score_noise_scale must be non-negative"
    key_policy_df["policy_score_noise_true"] = 0.0
    if policy_score_noise_scale > 0.0:
        key_policy_df["policy_score_noise_true"] = pd.Series(
            np.random.logistic(loc=0.0, scale=policy_score_noise_scale, size=len(key_policy_df)),
            index=key_policy_df.index,
            dtype=float,
        )
        key_policy_df["policy_score_true"] = (
            key_policy_df["policy_score_true"] + key_policy_df["policy_score_noise_true"]
        )
        key_policy_df["policy_accept_proba_true"] = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        (key_policy_df["policy_score_true_raw"] - key_policy_df["policy_threshold_t"])
                        / policy_score_noise_scale,
                        -40.0,
                        40.0,
                    )
                )
            )
        )
    else:
        key_policy_df["policy_accept_proba_true"] = (
            key_policy_df["policy_score_true_raw"] >= key_policy_df["policy_threshold_t"]
        ).astype(float)
    exploration_accept_rate = float(config.policy_exploration_accept_rate)
    assert 0.0 <= exploration_accept_rate <= 1.0, "policy_exploration_accept_rate must be between 0 and 1"
    key_policy_df["policy_gap_to_threshold"] = (
        key_policy_df["policy_threshold_t"] - key_policy_df["policy_score_true"]
    )
    key_policy_df["accepted_by_policy"] = (
        key_policy_df["policy_score_true"] >= key_policy_df["policy_threshold_t"]
    )
    key_policy_df["accepted_via_exploration"] = False
    if exploration_accept_rate > 0.0:
        exploratory_accept_count = 0
        exploratory_target_count = 0
        for date, date_index in key_policy_df.groupby("date").groups.items():
            date_index = pd.Index(date_index)
            accepted_by_policy_count_for_date = int(key_policy_df.loc[date_index, "accepted_by_policy"].sum())
            exploratory_target_float = exploration_accept_rate * accepted_by_policy_count_for_date
            exploratory_target_floor = int(np.floor(exploratory_target_float))
            exploratory_target_count_for_date = exploratory_target_floor + int(
                np.random.random() < (exploratory_target_float - exploratory_target_floor)
            )
            exploratory_target_count += exploratory_target_count_for_date
            rejected_index = date_index[~key_policy_df.loc[date_index, "accepted_by_policy"].to_numpy()]
            exploratory_accept_count_for_date = min(exploratory_target_count_for_date, len(rejected_index))
            if exploratory_accept_count_for_date == 0:
                continue
            exploratory_index = np.random.choice(
                rejected_index.to_numpy(),
                size=exploratory_accept_count_for_date,
                replace=False,
            )
            key_policy_df.loc[exploratory_index, "accepted_via_exploration"] = True
            exploratory_accept_count += exploratory_accept_count_for_date
        print(
            "Policy exploration accepts: "
            f"{exploratory_accept_count:,} / {int(key_policy_df['accepted_by_policy'].sum()):,} accepted-by-policy "
            f"({exploratory_accept_count / max(int(key_policy_df['accepted_by_policy'].sum()), 1):.2%}) "
            f"vs target {exploration_accept_rate:.2%} "
            f"[requested picks={exploratory_target_count:,}]"
        )
    assert not (
        key_policy_df["accepted_by_policy"] & key_policy_df["accepted_via_exploration"]
    ).any(), "accepted_via_exploration must only flip below-threshold deals"
    max_business_qty = int(config.qty_ordered_range) - 1
    # Operational qty is chosen from an ex-ante heuristic, not from realized
    # curve-optimal profit. This keeps one business row per key without leaking
    # future outcomes into the selection policy.
    key_policy_df["business_qty"] = key_policy_df["sv"].map(
        lambda sv_observed: _business_qty_for_sv(
            sv_observed=float(sv_observed),
            max_qty=max_business_qty,
            config=config,
        )
    )
    business_profit_lookup = (
        full_df[["key", "qty", "profit", "sales_true", "qty_left_true"]]
        .rename(
            columns={
                "qty": "business_qty",
                "profit": "business_profit_curve_true",
                "sales_true": "business_sales_true",
                "qty_left_true": "business_qty_left_true",
            }
        )
        .copy()
    )
    key_policy_df = key_policy_df.merge(
        business_profit_lookup,
        on=["key", "business_qty"],
        how="left",
        validate="one_to_one",
    )
    operational_cost_per_unit = float(config.business_operational_cost_per_unit)
    assert operational_cost_per_unit >= 0.0, "business_operational_cost_per_unit must be non-negative"
    key_policy_df["business_profit_true"] = (
        key_policy_df["business_profit_curve_true"]
        - (operational_cost_per_unit * key_policy_df["business_qty"])
    )
    assert key_policy_df["business_profit_true"].notna().all(), "Every key must have a business_profit_true value"
    key_policy_df["business_demand_censored_true"] = np.isclose(
        key_policy_df["business_sales_true"],
        key_policy_df["business_qty"],
        atol=1e-9,
        rtol=0.0,
    )
    business_qty_left_zero = np.isclose(
        key_policy_df["business_qty_left_true"],
        0.0,
        atol=1e-9,
        rtol=0.0,
    )
    assert (
        key_policy_df["business_demand_censored_true"].to_numpy() == business_qty_left_zero
    ).all(), "business stockout state must match business_qty_left_true"
    key_policy_df["accepted"] = (
        key_policy_df["accepted_by_policy"] | key_policy_df["accepted_via_exploration"]
    )

    full_df = full_df.merge(
        key_policy_df[
            [
                "key",
                "accepted",
                "accepted_by_policy",
                "accepted_via_exploration",
                "policy_gap_to_threshold",
                "business_qty",
                "business_sales_true",
                "business_qty_left_true",
                "business_demand_censored_true",
                "business_profit_true",
            ]
        ],
        on="key",
        how="left",
        validate="many_to_one",
    )
    assert full_df["accepted"].notna().all(), "Every row must receive an acceptance label"

    policy_standardization_stats = {
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "log_sv_mean": log_sv_mean,
        "log_sv_std": log_sv_std,
        "historical_date_count": int(len(historical_dates)),
        "future_eval_date_count": int(len(test_dates)),
    }
    return full_df, key_policy_df, policy_standardization_stats


def generate_market_df(random_state=42, config: ModernMarketSynthConfig = DEFAULT_SYNTH_CONFIG):
    random.seed(random_state)
    np.random.seed(random_state)

    telemetry = []
    offer_rows = []
    grounding_demand_rows = []
    grounding_offer_cost_rows = []
    sales_per_period = []
    pure_sampling_source = defaultdict(list)
    time_pure_sampling_source = defaultdict(lambda: defaultdict(list))
    date_process_df = _build_date_process_df(config=config, random_state=random_state)
    date_process_by_tick = {
        float(row["date"]): row
        for row in date_process_df.to_dict(orient="records")
    }

    max_clusters = int(config.cluster_count)
    SV_FIXED, SIGMAS, SAMPLERS = generate_exponential_clusters(
        n_clusters=max_clusters,
        base_sv_range=tuple(config.cluster_sv_range),
        sigma_range=tuple(config.cluster_sigma_range),
    )

    do_remove_index = False
    if not config.disable_cluster_drop_for_testing:
        do_remove_index = random.random() < 0.4

    assert len(SAMPLERS) == max_clusters
    assert len(SIGMAS) == max_clusters
    assert len(SV_FIXED) == max_clusters

    if do_remove_index:
        remove_index = random.randint(0, max_clusters - 1)
        print(f"Removing cluster index {remove_index} for testing...")
        SV_FIXED.pop(remove_index)
        SIGMAS.pop(remove_index)
        SAMPLERS.pop(remove_index)
        max_clusters -= 1

    offer_key = 0

    if GENERATE:
        for sku_cluster in tqdm.tqdm(range(max_clusters)):
            sv_avg = SV_FIXED[sku_cluster]
            sigma = SIGMAS[sku_cluster]
            sampler = SAMPLERS[sku_cluster]
            price = float(config.base_price)
            for time_tick in range(config.max_ticks):
                date_state = date_process_by_tick[float(time_tick)]
                lcogs_floor_t = float(date_state["lcogs_floor_t"])
                offer_volume_t = int(date_state["offer_volume_t"])
                market_latent_t = float(date_state["market_latent_t"])
                effective_demand_multiplier_t = float(date_state["effective_demand_multiplier_t"])

                for order in range(offer_volume_t):
                    lcogs_draw_min = max(float(config.lcogs_draw_min), lcogs_floor_t)
                    lcogs_draw_max = float(config.lcogs_draw_max)
                    assert lcogs_draw_max > lcogs_draw_min, "lcogs_draw_max must exceed effective lcogs floor"
                    for LCOGS in [
                        _sample_lcogs(
                            lcogs_draw_min=lcogs_draw_min,
                            lcogs_draw_max=lcogs_draw_max,
                            config=config,
                        )
                        for _ in range(int(config.lcogs_seeds))
                    ]:
                        if LCOGS < lcogs_floor_t:
                            continue

                        offer_key += 1
                        sv_observed = float(sv_avg)
                        sv_effective = float(sv_observed * effective_demand_multiplier_t)
                        assert sv_effective >= 0.0, "sv_effective must be non-negative"
                        weekly_demand_path = _generate_weekly_demand_path(
                            sv_effective=sv_effective,
                            raw_rel_sigma=sigma,
                            sampler=sampler,
                            config=config,
                            n_weeks=int(config.weekly_demand_n_weeks),
                        )
                        assert all(demand >= 0.0 for demand in weekly_demand_path), "weekly_demand_path must be non-negative"
                        latent_offer_edge = _latent_offer_edge(
                            price=price,
                            lcogs=LCOGS,
                            sv_observed=sv_observed,
                            raw_rel_sigma=sigma,
                            config=config,
                        )
                        latent_offer_fragility = _latent_offer_fragility(
                            offer_edge=latent_offer_edge,
                            config=config,
                        )
                        downside_regime = _sample_offer_downside_regime(
                            offer_fragility=latent_offer_fragility,
                            config=config,
                        )
                        weekly_demand_path_for_offer = _apply_offer_downside_to_weekly_demand_path(
                            weekly_demand_path=weekly_demand_path,
                            downside_regime=downside_regime,
                        )
                        assert all(demand >= 0.0 for demand in weekly_demand_path_for_offer), "weekly_demand_path_for_offer must be non-negative"
                        offer_rows.append({
                            "key": float(offer_key),
                            "date": float(time_tick),
                            "sku_cluster": float(sku_cluster),
                            "lcogs": float(LCOGS),
                            "price": float(price),
                            "sv": float(sv_observed),
                            "isv": float(1.0 / max(sv_observed, 1e-9)),
                        })
                        grounding_offer_cost_rows.append({
                            "key": float(offer_key),
                            "extra_cost_frac": float(downside_regime["downside_regime_extra_cost_frac"]),
                        })
                        grounding_demand_rows.extend(
                            {
                                "key": float(offer_key),
                                "week_idx": int(week_idx),
                                "demand_units": float(demand_units),
                            }
                            for week_idx, demand_units in enumerate(weekly_demand_path_for_offer)
                        )

                        for qty_ordered in range(1, int(config.qty_ordered_range)):
                            inventory_cost_basis = _inventory_cost_basis(
                                lcogs=LCOGS,
                                qty_ordered=qty_ordered,
                            )
                            profit, qty_left, sales = _simulate_profit_for_weekly_demand_path(
                                weekly_demand_path=weekly_demand_path_for_offer,
                                price=price,
                                lcogs=LCOGS,
                                qty_ordered=qty_ordered,
                                config=config,
                            )
                            profit, downside_regime_extra_cost = _apply_offer_downside_extra_cost(
                                profit_before_downside=profit,
                                inventory_cost_basis=inventory_cost_basis,
                                downside_regime=downside_regime,
                            )
                            assert abs((float(qty_ordered) - float(qty_left)) - float(sales)) <= 1e-9, (
                                "sales_true and qty_left_true must reconcile to qty"
                            )

                            if sku_cluster == DISASTROUS_CLUSTER and DIFFICULTY_LEVEL >= 5:
                                if random.random() < 0.002:
                                    profit += DISASTER_COST

                            telemetry.append({
                                'lcogs': float(LCOGS),
                                'qty': float(qty_ordered),
                                'price': float(price),
                                'sv': float(sv_observed),
                                'isv': float(1.0 / max(sv_observed, 1e-9)),
                                'broken_feature_increasing_noise': random.uniform(10.0, 11.0) * float(sku_cluster) if (DIFFICULTY_LEVEL >= 8) or (BAD_FEATURE_ENABLED) else random.uniform(0.1, 10),
                                'profit': float(profit),
                                'sales_true': float(sales),
                                'qty_left_true': float(qty_left),
                                'key': float(offer_key),
                                'sku_cluster': float(sku_cluster),
                                'date': float(time_tick),
                                'inventory_cost_basis': float(inventory_cost_basis),
                                'latent_offer_edge': float(latent_offer_edge),
                                'latent_offer_fragility': float(latent_offer_fragility),
                                'downside_regime_prob_true': float(downside_regime["downside_regime_prob"]),
                                'downside_regime_demand_mult_true': float(downside_regime["downside_regime_demand_mult"]),
                                'downside_regime_extra_cost_frac_true': float(downside_regime["downside_regime_extra_cost_frac"]),
                                'downside_regime_triggered_true': bool(downside_regime["downside_regime_triggered"]),
                                'downside_regime_extra_cost_true': float(downside_regime_extra_cost),
                            })
                            pure_sampling_source[pure_sampling_key(sku_cluster, LCOGS, qty_ordered)].append(profit)
                            time_pure_sampling_source[time_tick][pure_sampling_key(sku_cluster, LCOGS, qty_ordered)].append(profit)
                            sales_per_period.append(sales)

        print(f"Generated {len(telemetry)} telemetry points.")

    pure_sampling_sources_means = {k: np.mean(v) for k, v in pure_sampling_source.items()}
    df = pd.DataFrame(telemetry)
    assert len(df) > 0
    offers_df = pd.DataFrame(offer_rows)
    grounding_demand_df = pd.DataFrame(grounding_demand_rows)
    grounding_offer_costs_df = pd.DataFrame(grounding_offer_cost_rows)
    df, key_policy_df, policy_standardization_stats = _build_policy_tables(
        full_df=df,
        date_process_df=date_process_df,
        config=config,
    )
    operational_df = df[df["qty"] == df["business_qty"]].copy()
    operational_df = operational_df.sort_values(["date", "key"]).reset_index(drop=True)
    assert operational_df["key"].is_unique, "operational_df must contain exactly one row per key"
    assert operational_df["business_profit_true"].notna().all(), "operational_df must have business_profit_true"
    operational_df["raw_profit_true"] = operational_df["profit"]
    operational_df["profit"] = operational_df["business_profit_true"]
    assert (
        operational_df["raw_profit_true"]
        - (float(config.business_operational_cost_per_unit) * operational_df["business_qty"])
        == operational_df["business_profit_true"]
    ).all(), "operational business profit must equal raw profit less operational cost"

    return {
        "df": operational_df,
        "offers_df": offers_df,
        "menu_df": df,
        "operational_df": operational_df,
        "grounding_demand_df": grounding_demand_df,
        "grounding_offer_costs_df": grounding_offer_costs_df,
        "date_process_df": date_process_df,
        "key_policy_df": key_policy_df,
        "policy_standardization_stats": policy_standardization_stats,
        "max_clusters": max_clusters,
        "SV_FIXED": SV_FIXED,
        "SIGMAS": SIGMAS,
        "SAMPLERS": SAMPLERS,
        "sales_per_period": sales_per_period,
        "pure_sampling_source": pure_sampling_source,
        "time_pure_sampling_source": time_pure_sampling_source,
        "pure_sampling_sources_means": pure_sampling_sources_means,
        "config": config,
    }


def split_market_df(df, random_state=42, config: ModernMarketSynthConfig = DEFAULT_SYNTH_CONFIG):
    sort_columns = [col for col in ["date", "key", "qty"] if col in df.columns]
    assert sort_columns, "split_market_df requires at least one sort column"
    df = df.sort_values(by=sort_columns).reset_index(drop=True)
    if "accepted" not in df.columns:
        raise ValueError("split_market_df requires accepted column before label routing")
    df["label_observed"] = _observed_label_mask(df, config=config)
    unique_dates = pd.Series(df["date"]).dropna().sort_values().unique()
    test_dates = _compute_test_dates(unique_dates, config.future_eval_fraction)
    test_mask = df["date"].isin(test_dates)

    historical_df = df.loc[~test_mask].copy()
    future_eval_df = df.loc[test_mask].copy()

    print(f"Min date in X_test: {future_eval_df['date'].min()}")
    print(f"Max date in X_train: {historical_df['date'].max()}")

    if BAD_FEATURE_ENABLED:
        print("Adding broken counter feature...")
        total_counter_rows = len(historical_df) + len(future_eval_df)
        counter_vals = np.arange(total_counter_rows)
        historical_df["index_counter"] = counter_vals[:len(historical_df)]
        future_eval_df["index_counter"] = counter_vals[len(historical_df):]

    biased_labeled_history = historical_df[historical_df["label_observed"]].copy()
    unlabeled_history = historical_df[~historical_df["label_observed"]].copy()
    assert len(biased_labeled_history) + len(unlabeled_history) == len(historical_df)
    if bool(config.accepted_safe_unsafe_labels_from_business_qty):
        historical_accepted_rows = int(historical_df["accepted"].sum())
        historical_safe_rows = int(biased_labeled_history["accepted"].sum())
        historical_unsafe_rows = int((unlabeled_history["accepted"]).sum())
        assert historical_accepted_rows == (historical_safe_rows + historical_unsafe_rows)
        print(
            "Historical safe/unsafe label routing: "
            f"accepted={historical_accepted_rows:,}, "
            f"safe_labeled={historical_safe_rows:,}, "
            f"unsafe_unlabeled={historical_unsafe_rows:,}"
        )

    return {
        "df": df,
        "historical_df": historical_df,
        "future_eval_df": future_eval_df,
        "biased_labeled_history": biased_labeled_history,
        "unlabeled_history": unlabeled_history,
    }


def build_fit_business_grounded_kwargs(split_data, random_state=42):
    biased_labeled_history = split_data["biased_labeled_history"]
    unlabeled_history = split_data["unlabeled_history"]

    X_grounded_biased = biased_labeled_history.drop(columns=[col for col in NON_MODEL_COLUMNS if col in biased_labeled_history.columns])
    assert not any(col in X_grounded_biased.columns for col in ["sku_cluster", "key"]), "X_grounded_biased contains bad columns"
    y_grounded_biased = biased_labeled_history["profit"]
    x_grounded_biased_dates = biased_labeled_history[["date"]]
    x_grounded_biased_keys = biased_labeled_history[["key"]]
    X_grounded_deployment_unlabeled = unlabeled_history.drop(columns=[col for col in NON_MODEL_COLUMNS if col in unlabeled_history.columns])
    assert not any(col in X_grounded_deployment_unlabeled.columns for col in ["profit", "date", "sku_cluster", "key"]), "X_grounded_deployment_unlabeled contains bad columns"
    x_grounded_deployment_unlabeled_dates = unlabeled_history[["date"]]
    x_grounded_deployment_unlabeled_keys = unlabeled_history[["key"]]
    safety_checks = True

    return {
        "X_grounded_biased": X_grounded_biased,
        "y_grounded_biased": y_grounded_biased,
        "x_grounded_biased_dates": x_grounded_biased_dates,
        "x_grounded_biased_keys": x_grounded_biased_keys,
        "X_grounded_deployment_unlabeled": X_grounded_deployment_unlabeled,
        "x_grounded_deployment_unlabeled_dates": x_grounded_deployment_unlabeled_dates,
        "x_grounded_deployment_unlabeled_keys": x_grounded_deployment_unlabeled_keys,
        "safety_checks": safety_checks,
        "random_state": random_state,
    }


def build_predict_kwargs(split_data, random_state=42):
    future_eval_df = split_data["future_eval_df"]

    X_fully_grounded_menu = future_eval_df.drop(columns=[col for col in NON_MODEL_COLUMNS if col in future_eval_df.columns])
    assert not any(col in X_fully_grounded_menu.columns for col in ["profit", "date", "sku_cluster", "key"]), "X_fully_grounded_menu contains bad columns"
    x_fully_grounded_menu_dates = future_eval_df[["date"]]
    x_fully_grounded_menu_keys = future_eval_df[["key"]]

    return {
        "X_fully_grounded_menu": X_fully_grounded_menu,
        "x_fully_grounded_menu_dates": x_fully_grounded_menu_dates,
        "x_fully_grounded_menu_keys": x_fully_grounded_menu_keys,
        "random_state": random_state,
    }


def _model_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in NON_MODEL_COLUMNS if col in df.columns]).copy()


def _saved_meta(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    meta = df[["key", "date", "qty", "accepted", label_col, "business_qty"]].rename(
        columns={"business_qty": "business_chosen_qty"}
    ).copy()
    meta["is_business_chosen_qty"] = meta["qty"] == meta["business_chosen_qty"]
    return meta


def _write_saved_frames(output_dir: Path, frames: dict[str, pd.DataFrame]):
    for rel_path, frame in frames.items():
        path = output_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _saved_shapes(frames: dict[str, pd.DataFrame]) -> dict[str, list[int]]:
    return {path.removesuffix(".parquet"): list(frame.shape) for path, frame in frames.items()}


def _inspection_part(df: pd.DataFrame, label_col: str, segment: str) -> pd.DataFrame:
    part = df.reset_index(drop=True).copy()
    if "business_chosen_qty" not in part.columns:
        assert "business_qty" in part.columns, "inspection frame missing business_qty"
        part = part.rename(columns={"business_qty": "business_chosen_qty"})
    part["segment"] = segment
    part["is_business_chosen_qty"] = part["qty"] == part["business_chosen_qty"]
    required_cols = [
        "date",
        "key",
        "qty",
        "profit",
        "accepted",
        label_col,
        "business_chosen_qty",
        "is_business_chosen_qty",
        "segment",
    ]
    missing_cols = sorted(set(required_cols) - set(part.columns))
    assert not missing_cols, f"inspection frame missing columns: {missing_cols}"
    return part


def _market_inspection_from_bundle(bundle, state: str, label_col="label_observed") -> dict[str, pd.DataFrame]:
    if state == "grounded":
        hist_observed_source = bundle["biased_labeled_history"]
        hist_unobserved_source = bundle["unlabeled_history"]
        future_eval_df = bundle["future_eval_df"]
    else:
        historical_df = bundle["operational_historical_df"]
        hist_observed_source = historical_df[historical_df[label_col]]
        hist_unobserved_source = historical_df[~historical_df[label_col]]
        future_eval_df = bundle["operational_future_eval_df"]
    hist_observed = _inspection_part(hist_observed_source, label_col, "observed")
    hist_unobserved = _inspection_part(hist_unobserved_source, label_col, "unobserved")
    eval_observed = _inspection_part(future_eval_df[future_eval_df[label_col]], label_col, "observed")
    eval_unobserved = _inspection_part(future_eval_df[~future_eval_df[label_col]], label_col, "unobserved")
    train_all = pd.concat([hist_observed, hist_unobserved], ignore_index=True)
    eval_all = pd.concat([eval_observed, eval_unobserved], ignore_index=True)
    key_policy_df = bundle["key_policy_df"].reset_index(drop=True).copy()
    date_process_df = bundle["date_process_df"].reset_index(drop=True).copy()
    required_policy_cols = ["accepted", "policy_score_true", "policy_threshold_t"]
    required_driver_cols = [
        "date",
        "market_observed_t",
        "market_trend_t",
        "market_latent_t",
        "effective_demand_multiplier_t",
        "policy_threshold_t",
        "policy_cycle_t",
    ]
    missing_policy_cols = sorted(set(required_policy_cols) - set(key_policy_df.columns))
    missing_driver_cols = sorted(set(required_driver_cols) - set(date_process_df.columns))
    assert not missing_policy_cols, f"inspection key_policy_df missing columns: {missing_policy_cols}"
    assert not missing_driver_cols, f"inspection date_process_df missing columns: {missing_driver_cols}"
    return {
        "hist_observed": hist_observed,
        "hist_unobserved": hist_unobserved,
        "train_all": train_all,
        "eval_all": eval_all,
        "inspect_all": pd.concat([train_all, eval_all], ignore_index=True),
        "key_policy_df": key_policy_df,
        "date_process_df": date_process_df,
    }


def _grounded_saved_frames(bundle, label_col="label_observed") -> dict[str, pd.DataFrame]:
    fit = bundle["fit_business_grounded_kwargs"]
    future_eval_df = bundle["future_eval_df"]
    future_labeled = future_eval_df[future_eval_df[label_col]].copy()
    future_unlabeled = future_eval_df[~future_eval_df[label_col]].copy()
    historical_unlabeled_meta = _saved_meta(bundle["unlabeled_history"], label_col)
    future_unlabeled_meta = _saved_meta(future_unlabeled, label_col)
    historical_unlabeled_meta["unlabeled_source"] = np.where(
        historical_unlabeled_meta["accepted"], "accepted_unsafe", "original_unlabeled"
    )
    future_unlabeled_meta["unlabeled_source"] = np.where(
        future_unlabeled_meta["accepted"], "accepted_unsafe", "original_unlabeled"
    )
    return {
        "X_labeled.parquet": fit["X_grounded_biased"],
        "y_labeled.parquet": fit["y_grounded_biased"].rename("profit").to_frame(),
        "x_labeled_dates.parquet": fit["x_grounded_biased_dates"],
        "x_labeled_keys.parquet": fit["x_grounded_biased_keys"],
        "X_unlabeled.parquet": fit["X_grounded_deployment_unlabeled"],
        "y_unlabeled.parquet": bundle["unlabeled_history"]["profit"].rename("profit").to_frame(),
        "x_unlabeled_dates.parquet": fit["x_grounded_deployment_unlabeled_dates"],
        "x_unlabeled_keys.parquet": fit["x_grounded_deployment_unlabeled_keys"],
        "metadata/labeled_meta.parquet": _saved_meta(bundle["biased_labeled_history"], label_col),
        "metadata/unlabeled_meta.parquet": historical_unlabeled_meta,
        "eval/X_labeled.parquet": _model_X(future_labeled),
        "eval/y_labeled.parquet": future_labeled["profit"].rename("profit").to_frame(),
        "eval/x_labeled_dates.parquet": future_labeled[["date"]],
        "eval/x_labeled_keys.parquet": future_labeled[["key"]],
        "metadata/eval_labeled_meta.parquet": _saved_meta(future_labeled, label_col),
        "eval/X_unlabeled.parquet": _model_X(future_unlabeled),
        "eval/y_unlabeled.parquet": future_unlabeled["profit"].rename("profit").to_frame(),
        "eval/x_unlabeled_dates.parquet": future_unlabeled[["date"]],
        "eval/x_unlabeled_keys.parquet": future_unlabeled[["key"]],
        "metadata/eval_unlabeled_meta.parquet": future_unlabeled_meta,
        "metadata/grounding_demand.parquet": bundle["generated"]["grounding_demand_df"],
        "metadata/grounding_offer_costs.parquet": bundle["generated"]["grounding_offer_costs_df"],
    }


def _business_saved_part(df: pd.DataFrame, label_col: str, labeled: bool):
    part = df[df[label_col]].copy() if labeled else df[~df[label_col]].copy()
    X = _model_X(part)
    if not labeled:
        X["qty"] = np.nan
    y_col = "raw_profit_true" if "raw_profit_true" in part.columns else "profit"
    y = part[y_col].rename("profit").to_frame() if labeled else pd.DataFrame({"profit": [np.nan] * len(part)})
    return X, y, part[["date"]], part[["key"]], _saved_meta(part, label_col)


def _historical_saved_frames(bundle, label_col="label_observed") -> dict[str, pd.DataFrame]:
    hist = bundle["operational_historical_df"]
    future = bundle["operational_future_eval_df"]
    hXl, hyl, hdl, hkl, hml = _business_saved_part(hist, label_col, labeled=True)
    hXu, hyu, hdu, hku, hmu = _business_saved_part(hist, label_col, labeled=False)
    eXl, eyl, edl, ekl, eml = _business_saved_part(future, label_col, labeled=True)
    eXu, eyu, edu, eku, emu = _business_saved_part(future, label_col, labeled=False)
    hmu["unlabeled_source"] = np.where(hmu["accepted"], "accepted_unsafe", "original_unlabeled")
    emu["unlabeled_source"] = np.where(emu["accepted"], "accepted_unsafe", "original_unlabeled")
    return {
        "X_labeled.parquet": hXl,
        "y_labeled.parquet": hyl,
        "x_labeled_dates.parquet": hdl,
        "x_labeled_keys.parquet": hkl,
        "X_unlabeled.parquet": hXu,
        "metadata/y_unlabeled.parquet": hyu,
        "x_unlabeled_dates.parquet": hdu,
        "x_unlabeled_keys.parquet": hku,
        "metadata/labeled_meta.parquet": hml,
        "metadata/unlabeled_meta.parquet": hmu,
        "eval/X_labeled.parquet": eXl,
        "eval/y_labeled.parquet": eyl,
        "eval/x_labeled_dates.parquet": edl,
        "eval/x_labeled_keys.parquet": ekl,
        "metadata/eval_labeled_meta.parquet": eml,
        "eval/X_unlabeled.parquet": eXu,
        "metadata/eval_y_unlabeled.parquet": eyu,
        "eval/x_unlabeled_dates.parquet": edu,
        "eval/x_unlabeled_keys.parquet": eku,
        "metadata/eval_unlabeled_meta.parquet": emu,
        "metadata/grounding_demand.parquet": bundle["generated"]["grounding_demand_df"],
        "metadata/grounding_offer_costs.parquet": bundle["generated"]["grounding_offer_costs_df"],
    }


def _market_dataset_from_bundle(
    bundle,
    state: str = "grounded",
    metadata_extra: dict | None = None,
):
    assert state in {"grounded", "historical"}, f"Unknown state: {state}"
    frames = (
        _grounded_saved_frames(bundle)
        if state == "grounded"
        else _historical_saved_frames(bundle)
    )
    metadata = {
        "state": state,
        "label_col": "label_observed",
        "config": asdict(bundle["config"]),
        "saved_shapes": _saved_shapes(frames),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return {"frames": frames, "inspection": _market_inspection_from_bundle(bundle, state), "metadata": metadata}


def get_market_dataset(
    random_state=42,
    config: ModernMarketSynthConfig = DEFAULT_SYNTH_CONFIG,
    state: str = "grounded",
    metadata_extra: dict | None = None,
):
    bundle = build_market_synth_bundle(random_state=random_state, config=config)
    return _market_dataset_from_bundle(
        bundle=bundle,
        state=state,
        metadata_extra=metadata_extra,
    )


def save_market_dataset(dataset: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset["metadata"])
    metadata["saved_shapes"] = _saved_shapes(dataset["frames"])
    _write_saved_frames(output_dir, dataset["frames"])
    (output_dir / "metadata" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": output_dir, "metadata": metadata}


def load_market_dataset(input_dir: str | Path):
    input_dir = Path(input_dir)
    frames = {
        str(path.relative_to(input_dir)): pd.read_parquet(path)
        for path in sorted(input_dir.rglob("*.parquet"))
    }
    metadata = json.loads((input_dir / "metadata" / "metadata.json").read_text(encoding="utf-8"))
    return {"frames": frames, "metadata": metadata, "input_dir": input_dir}


def save_market_dataset_bundle(
    bundle,
    output_dir: str | Path,
    state: str = "grounded",
    metadata_extra: dict | None = None,
):
    dataset = _market_dataset_from_bundle(bundle, state=state, metadata_extra=metadata_extra)
    save_result = save_market_dataset(dataset, output_dir)
    return {**dataset, **save_result}


def _dataset_part(dataset: dict, split: str, labeled: bool):
    prefix = "eval/" if split == "eval" else ""
    meta_prefix = "metadata/eval_" if split == "eval" else "metadata/"
    name = "labeled" if labeled else "unlabeled"
    frames = dataset["frames"]
    y_path = f"{prefix}y_{name}.parquet"
    if y_path not in frames:
        y_path = f"metadata/{'eval_' if split == 'eval' else ''}y_{name}.parquet"
    return (
        frames[f"{prefix}X_{name}.parquet"],
        frames[y_path],
        frames[f"{prefix}x_{name}_dates.parquet"],
        frames[f"{prefix}x_{name}_keys.parquet"],
        frames[f"{meta_prefix}{name}_meta.parquet"].rename(columns={"business_qty": "business_chosen_qty"}),
    )


def _ground_saved_part(X, dates, keys, meta, demand_by_key, extra_by_key, config):
    x_rows, y_rows, date_rows, key_rows, meta_rows = [], [], [], [], []
    qty_values = range(1, int(config.qty_ordered_range))
    for i, (_, base_x) in enumerate(X.iterrows()):
        key = float(keys.iloc[i]["key"])
        demand_path = demand_by_key[key]
        extra_cost_frac = extra_by_key[key]
        base_x = base_x.to_dict()
        base_meta = meta.iloc[i].to_dict()
        for qty_ordered in qty_values:
            x_row = dict(base_x)
            x_row["qty"] = float(qty_ordered)
            profit, _, _, _ = _profit_from_grounding_inputs(
                weekly_demand_path=demand_path,
                price=float(x_row["price"]),
                lcogs=float(x_row["lcogs"]),
                qty_ordered=float(qty_ordered),
                extra_cost_frac=float(extra_cost_frac),
                config=config,
            )
            meta_row = dict(base_meta)
            meta_row["qty"] = float(qty_ordered)
            meta_row["is_business_chosen_qty"] = float(qty_ordered) == float(meta_row["business_chosen_qty"])
            x_rows.append(x_row)
            y_rows.append({"profit": float(profit)})
            date_rows.append(dates.iloc[i].to_dict())
            key_rows.append(keys.iloc[i].to_dict())
            meta_rows.append(meta_row)
    return (
        pd.DataFrame(x_rows, columns=X.columns),
        pd.DataFrame(y_rows, columns=["profit"]),
        pd.DataFrame(date_rows, columns=dates.columns),
        pd.DataFrame(key_rows, columns=keys.columns),
        pd.DataFrame(meta_rows, columns=meta.columns),
    )


def _grounded_from_dataset(dataset: dict, config: ModernMarketSynthConfig | None = None):
    metadata = dataset["metadata"]
    if config is None:
        config = ModernMarketSynthConfig(**metadata["config"])
    demand_df = dataset["frames"]["metadata/grounding_demand.parquet"].sort_values(["key", "week_idx"])
    costs_df = dataset["frames"]["metadata/grounding_offer_costs.parquet"]
    demand_by_key = {float(k): g["demand_units"].astype(float).tolist() for k, g in demand_df.groupby("key")}
    extra_by_key = {float(row.key): float(row.extra_cost_frac) for row in costs_df.itertuples(index=False)}
    frames = {}
    for split in ["historical", "eval"]:
        for labeled in [True, False]:
            X, _, dates, keys, meta = _dataset_part(dataset, split=split, labeled=labeled)
            gX, gy, gd, gk, gm = _ground_saved_part(X, dates, keys, meta, demand_by_key, extra_by_key, config)
            prefix = "eval/" if split == "eval" else ""
            meta_prefix = "metadata/eval_" if split == "eval" else "metadata/"
            name = "labeled" if labeled else "unlabeled"
            frames[f"{prefix}X_{name}.parquet"] = gX
            frames[f"{prefix}y_{name}.parquet"] = gy
            frames[f"{prefix}x_{name}_dates.parquet"] = gd
            frames[f"{prefix}x_{name}_keys.parquet"] = gk
            frames[f"{meta_prefix}{name}_meta.parquet"] = gm
    frames["metadata/grounding_demand.parquet"] = dataset["frames"]["metadata/grounding_demand.parquet"]
    frames["metadata/grounding_offer_costs.parquet"] = dataset["frames"]["metadata/grounding_offer_costs.parquet"]
    grounded_metadata = {
        **metadata,
        "state": "grounded",
        "source_state": metadata.get("state"),
        "saved_shapes": _saved_shapes(frames),
    }
    return {"frames": frames, "metadata": grounded_metadata}


def _historical_from_grounded_dataset(dataset: dict):
    frames = {}
    for split in ["historical", "eval"]:
        for labeled in [True, False]:
            X, y, dates, keys, meta = _dataset_part(dataset, split=split, labeled=labeled)
            mask = meta["is_business_chosen_qty"].to_numpy(dtype=bool)
            prefix = "eval/" if split == "eval" else ""
            meta_prefix = "metadata/eval_" if split == "eval" else "metadata/"
            name = "labeled" if labeled else "unlabeled"
            hX = X.loc[mask].reset_index(drop=True).copy()
            if not labeled:
                hX["qty"] = np.nan
            frames[f"{prefix}X_{name}.parquet"] = hX
            frames[f"{prefix}x_{name}_dates.parquet"] = dates.loc[mask].reset_index(drop=True)
            frames[f"{prefix}x_{name}_keys.parquet"] = keys.loc[mask].reset_index(drop=True)
            frames[f"{meta_prefix}{name}_meta.parquet"] = meta.loc[mask].reset_index(drop=True)
            y_path = f"{prefix}y_{name}.parquet" if labeled else f"metadata/{'eval_' if split == 'eval' else ''}y_{name}.parquet"
            frames[y_path] = y.loc[mask].reset_index(drop=True)
    frames["metadata/grounding_demand.parquet"] = dataset["frames"]["metadata/grounding_demand.parquet"]
    frames["metadata/grounding_offer_costs.parquet"] = dataset["frames"]["metadata/grounding_offer_costs.parquet"]
    metadata = {
        **dataset["metadata"],
        "state": "historical",
        "source_state": dataset["metadata"].get("state"),
        "saved_shapes": _saved_shapes(frames),
    }
    return {"frames": frames, "metadata": metadata}


def convert_market_dataset_state(dataset: dict, to_state: str, config: ModernMarketSynthConfig | None = None):
    assert to_state in {"grounded", "historical"}, f"Unknown to_state: {to_state}"
    if dataset["metadata"].get("state") == to_state:
        return dataset
    if to_state == "grounded":
        return _grounded_from_dataset(dataset, config=config)
    return _historical_from_grounded_dataset(dataset)


def ground_market_dataset(dataset: dict, config: ModernMarketSynthConfig | None = None):
    return convert_market_dataset_state(dataset, to_state="grounded", config=config)


def ground_saved_market_dataset(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    config: ModernMarketSynthConfig | None = None,
):
    grounded_dataset = ground_market_dataset(load_market_dataset(input_dir), config=config)
    if output_dir is not None:
        save_result = save_market_dataset(grounded_dataset, output_dir)
        return {**grounded_dataset, **save_result}
    return grounded_dataset


def build_market_synth_bundle(random_state=42, config: ModernMarketSynthConfig = DEFAULT_SYNTH_CONFIG):
    generated = generate_market_df(random_state=random_state, config=config)
    split_data = split_market_df(generated["menu_df"], random_state=random_state, config=config)
    operational_split_data = split_market_df(generated["operational_df"], random_state=random_state, config=config)
    fit_business_grounded_kwargs = build_fit_business_grounded_kwargs(split_data, random_state=random_state)
    predict_kwargs = build_predict_kwargs(split_data, random_state=random_state)

    historical_df = split_data["historical_df"]
    future_eval_df = split_data["future_eval_df"]
    biased_labeled_history = split_data["biased_labeled_history"]
    unlabeled_history = split_data["unlabeled_history"]
    operational_historical_df = operational_split_data["historical_df"]
    operational_future_eval_df = operational_split_data["future_eval_df"]
    business_df = operational_split_data["df"]

    assert historical_df["date"].max() < future_eval_df["date"].min()
    assert len(historical_df) + len(future_eval_df) == len(split_data["df"])
    assert biased_labeled_history.index.is_unique
    assert unlabeled_history.index.is_unique
    assert biased_labeled_history.index.intersection(unlabeled_history.index).empty
    assert biased_labeled_history.index.intersection(future_eval_df.index).empty
    assert unlabeled_history.index.intersection(future_eval_df.index).empty
    assert len(biased_labeled_history) + len(unlabeled_history) == len(historical_df)
    assert operational_historical_df["date"].max() < operational_future_eval_df["date"].min()
    assert len(operational_historical_df) + len(operational_future_eval_df) == len(operational_split_data["df"])
    assert operational_historical_df["key"].is_unique
    assert operational_future_eval_df["key"].is_unique
    assert len(historical_df) >= len(operational_historical_df)
    assert len(future_eval_df) >= len(operational_future_eval_df)

    return {
        "generated": generated,
        "date_process_df": generated["date_process_df"],
        "key_policy_df": generated["key_policy_df"],
        "policy_standardization_stats": generated["policy_standardization_stats"],
        "config": generated["config"],
        "full_df": split_data["df"],
        "historical_df": historical_df,
        "future_eval_df": future_eval_df,
        "business_df": business_df,
        "biased_labeled_history": biased_labeled_history,
        "unlabeled_history": unlabeled_history,
        "operational_historical_df": operational_historical_df,
        "operational_future_eval_df": operational_future_eval_df,
        "fit_business_grounded_kwargs": fit_business_grounded_kwargs,
        "predict_kwargs": predict_kwargs,
    }


def _compute_hidden_shift_diagnostics(bundle, date_summary: pd.DataFrame, output_dir: Path):
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    historical_df = bundle["historical_df"]
    future_eval_df = bundle["future_eval_df"]
    fit_kwargs = bundle["fit_business_grounded_kwargs"]
    predict_kwargs = bundle["predict_kwargs"]
    feature_cols = list(fit_kwargs["X_grounded_biased"].columns)

    historical_fit_df = historical_df
    if len(historical_fit_df) > 150000:
        historical_fit_df = historical_fit_df.sample(n=150000, random_state=42)

    baseline_regressor = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        device="cpu",
        verbosity=0,
        n_jobs=4,
    )
    baseline_regressor.fit(
        historical_fit_df[feature_cols],
        historical_fit_df["profit"],
        verbose=False,
    )
    future_pred = baseline_regressor.predict(predict_kwargs["X_fully_grounded_menu"])
    future_scored = future_eval_df[["date", "profit"]].copy()
    future_scored["pred_profit_observed_only"] = future_pred
    future_scored["residual_profit"] = future_scored["profit"] - future_scored["pred_profit_observed_only"]
    hidden_shift_date_residuals = (
        future_scored.groupby("date", dropna=False)
        .agg(
            profit_mean=("profit", "mean"),
            pred_profit_observed_only_mean=("pred_profit_observed_only", "mean"),
            residual_profit_mean=("residual_profit", "mean"),
        )
        .reset_index()
    )
    hidden_shift_date_residuals = hidden_shift_date_residuals.merge(
        bundle["date_process_df"][["date", "market_latent_t"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    hidden_shift_corr = float(
        hidden_shift_date_residuals["residual_profit_mean"].corr(hidden_shift_date_residuals["market_latent_t"])
    )

    observable_date_cols = [
        col
        for col in ["offer_count", "mean_lcogs", "mean_sv_observed", "mean_margin"]
        if col in date_summary.columns
    ]
    if not observable_date_cols:
        raise RuntimeError("No observable date-level columns available for latent observability check.")
    latent_ridge = Ridge(alpha=1.0)
    latent_ridge.fit(date_summary[observable_date_cols], date_summary["market_latent_t"])
    market_latent_pred = latent_ridge.predict(date_summary[observable_date_cols])
    market_latent_observable_r2 = float(r2_score(date_summary["market_latent_t"], market_latent_pred))

    hidden_shift_date_residuals.to_csv(output_dir / "hidden_shift_date_residuals.csv", index=False)

    diagnostics = {
        "hidden_shift_residual_corr": hidden_shift_corr,
        "market_latent_observable_r2": market_latent_observable_r2,
        "feature_cols": feature_cols,
    }
    return diagnostics


def _mean_gap(left: pd.Series, right: pd.Series) -> float:
    pooled = float(np.sqrt(((left.std(ddof=0) ** 2) + (right.std(ddof=0) ** 2)) / 2.0))
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def _build_validation_summary(bundle, date_summary: pd.DataFrame, coverage_cube: pd.DataFrame, diagnostics: dict, deep_diagnostics: dict, output_dir: Path):
    historical_df = bundle["historical_df"].copy()
    labeled_hist = bundle["biased_labeled_history"].copy()
    unlabeled_hist = bundle["unlabeled_history"].copy()
    latent_predictability_summary = deep_diagnostics["latent_predictability_summary"]
    driver_contribution_summary = deep_diagnostics["driver_contribution_summary"]
    selection_counterfactual_summary = deep_diagnostics["selection_counterfactual_summary"].iloc[0]

    policy_axis_col = "policy_preference_t" if "policy_preference_t" in date_summary.columns else "policy_threshold_t"
    market_policy_centered = date_summary[policy_axis_col] - float(date_summary[policy_axis_col].mean())
    aligned_dates = int(((date_summary["market_observed_t"] * market_policy_centered) > 0).sum())
    opposed_dates = int(((date_summary["market_observed_t"] * market_policy_centered) < 0).sum())

    labeled_margin = labeled_hist["price"] - labeled_hist["lcogs"]
    unlabeled_margin = unlabeled_hist["price"] - unlabeled_hist["lcogs"]
    labeled_sv = labeled_hist["sv"]
    unlabeled_sv = unlabeled_hist["sv"]

    observed_feature_corrs = {}
    for feature_col in ["lcogs", "qty", "price", "sv", "isv"]:
        if feature_col in historical_df.columns:
            feature_std = float(historical_df[feature_col].std(ddof=0))
            if not np.isfinite(feature_std) or feature_std <= 1e-12:
                continue
            observed_feature_corrs[feature_col] = float(historical_df[feature_col].corr(historical_df["profit"]))
    max_abs_feature_profit_corr = float(max((abs(v) for v in observed_feature_corrs.values()), default=0.0))

    support_cells = int((coverage_cube["date_count"] > 0).sum())
    total_cells = int(len(coverage_cube))
    policy_predictability_target = "policy_preference_t" if "policy_preference_t" in set(latent_predictability_summary["target"]) else "policy_latent_t"
    best_market_latent_time_oof_r2 = float(
        latent_predictability_summary[latent_predictability_summary["target"] == "market_latent_t"]["time_oof_r2"].max()
    )
    best_policy_latent_time_oof_r2 = float(
        latent_predictability_summary[latent_predictability_summary["target"] == policy_predictability_target]["time_oof_r2"].max()
    )
    acceptance_q90 = float(date_summary["acceptance_rate"].quantile(0.90))
    acceptance_q10 = float(date_summary["acceptance_rate"].quantile(0.10))
    acceptance_min = float(date_summary["acceptance_rate"].min())
    acceptance_max = float(date_summary["acceptance_rate"].max())

    summary_metrics = {
        "offer_volume_cv": float(date_summary["offer_count"].std(ddof=0) / max(date_summary["offer_count"].mean(), 1e-9)),
        "observed_market_span": float(date_summary["market_observed_t"].max() - date_summary["market_observed_t"].min()),
        "policy_state_span": float(date_summary[policy_axis_col].max() - date_summary[policy_axis_col].min()),
        "aligned_dates": aligned_dates,
        "opposed_dates": opposed_dates,
        "margin_smd_labeled_vs_unlabeled": float(_mean_gap(labeled_margin, unlabeled_margin)),
        "sv_smd_labeled_vs_unlabeled": float(_mean_gap(labeled_sv, unlabeled_sv)),
        "acceptance_rate_q10": acceptance_q10,
        "acceptance_rate_q90": acceptance_q90,
        "acceptance_rate_spread_q90_q10": float(acceptance_q90 - acceptance_q10),
        "acceptance_rate_min": acceptance_min,
        "acceptance_rate_max": acceptance_max,
        "support_cells_present": support_cells,
        "support_cells_total": total_cells,
        "support_cell_fraction": float(support_cells / max(total_cells, 1)),
        "hidden_shift_residual_corr": float(diagnostics["hidden_shift_residual_corr"]),
        "market_latent_observable_r2": float(diagnostics["market_latent_observable_r2"]),
        "best_market_latent_time_oof_r2": best_market_latent_time_oof_r2,
        "best_policy_latent_time_oof_r2": best_policy_latent_time_oof_r2,
        "max_abs_feature_profit_corr": max_abs_feature_profit_corr,
        "selection_market_variance_share": float(selection_counterfactual_summary["market_variance_share"]),
        "selection_policy_variance_share": float(selection_counterfactual_summary["policy_variance_share"]),
        "selection_interaction_variance_share": float(selection_counterfactual_summary["interaction_variance_share"]),
        "selection_market_counterfactual_mean_range": float(selection_counterfactual_summary["market_counterfactual_mean_range"]),
        "selection_policy_counterfactual_mean_range": float(selection_counterfactual_summary["policy_counterfactual_mean_range"]),
    }

    checks = [
        ("offer_volume_varies", summary_metrics["offer_volume_cv"] > 0.20),
        ("observed_market_varies", summary_metrics["observed_market_span"] > 1.0),
        ("policy_varies", summary_metrics["policy_state_span"] > 0.15),
        ("market_policy_align_and_oppose", aligned_dates >= 10 and opposed_dates >= 10),
        ("labeled_unlabeled_differ", max(abs(summary_metrics["margin_smd_labeled_vs_unlabeled"]), abs(summary_metrics["sv_smd_labeled_vs_unlabeled"])) > 0.25),
        ("hidden_shift_changes_y", abs(summary_metrics["hidden_shift_residual_corr"]) > 0.20),
        ("broad_support", summary_metrics["support_cell_fraction"] > 0.40),
        ("acceptance_not_stuck", summary_metrics["acceptance_rate_spread_q90_q10"] > 0.15),
        ("no_single_feature_silently_explains_everything", summary_metrics["max_abs_feature_profit_corr"] < 0.95),
        ("acceptance_extremes_present", summary_metrics["acceptance_rate_min"] <= 0.02 and summary_metrics["acceptance_rate_max"] >= 0.98),
        ("latent_not_easily_predictable_from_observed_x", summary_metrics["best_market_latent_time_oof_r2"] < 0.35 and summary_metrics["best_policy_latent_time_oof_r2"] < 0.35),
        ("market_meaningfully_changes_selection_beyond_policy", summary_metrics["selection_market_variance_share"] > 0.10 and summary_metrics["selection_market_counterfactual_mean_range"] > 0.10),
    ]

    lines = []
    for metric_name, metric_val in summary_metrics.items():
        if isinstance(metric_val, float):
            lines.append(f"{metric_name}: {metric_val:,.4f}")
        else:
            lines.append(f"{metric_name}: {metric_val}")
    lines.append("")
    for check_name, passed in checks:
        lines.append(f"{check_name}: {'PASS' if passed else 'WARN'}")

    summary_path = output_dir / "validation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_metrics": summary_metrics,
        "checks": checks,
        "summary_path": summary_path,
    }


def run_market_synth_validation(
    random_state=42,
    config: ModernMarketSynthConfig = DEFAULT_SYNTH_CONFIG,
    output_dir: str | Path | None = None,
):
    from market_synth_2.telemetry._SCRIPTS.deep_diagnostics import export_deep_diagnostics
    from market_synth_2.telemetry._SCRIPTS.telemetry_export import export_market_telemetry

    bundle = build_market_synth_bundle(random_state=random_state, config=config)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "telemetry" / "latest"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_export = export_market_telemetry(
        full_df=bundle["full_df"],
        output_dir=output_dir,
        split_bundle=bundle,
        key_col="key",
        accepted_col="accepted",
        driver_df=bundle["date_process_df"],
        driver_date_col="date",
        driver_cols=[
            "market_observed_t",
            "market_latent_t",
            "policy_threshold_t",
            "policy_latent_t",
            "policy_sv_latent_t",
            "policy_margin_weight_t",
            "policy_sv_weight_t",
            "policy_preference_t",
        ],
    )

    bundle["date_process_df"].to_csv(output_dir / "date_process.csv", index=False)
    bundle["key_policy_df"].head(10000).to_csv(output_dir / "key_policy_debug_sample.csv", index=False)
    bundle["full_df"].sample(n=min(len(bundle["full_df"]), 10000), random_state=random_state).to_csv(
        output_dir / "full_market_debug_sample.csv",
        index=False,
    )

    diagnostics = _compute_hidden_shift_diagnostics(
        bundle=bundle,
        date_summary=telemetry_export["date_summary"],
        output_dir=output_dir,
    )
    deep_diagnostics = export_deep_diagnostics(
        bundle=bundle,
        telemetry_export=telemetry_export,
        output_dir=output_dir,
    )
    validation = _build_validation_summary(
        bundle=bundle,
        date_summary=telemetry_export["date_summary"],
        coverage_cube=telemetry_export["coverage_cube"],
        diagnostics=diagnostics,
        deep_diagnostics=deep_diagnostics,
        output_dir=output_dir,
    )

    return {
        "bundle": bundle,
        "telemetry_export": telemetry_export,
        "diagnostics": diagnostics,
        "deep_diagnostics": deep_diagnostics,
        "validation": validation,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    result = run_market_synth_validation()
    print(f"Telemetry written to: {result['output_dir']}")
    print(f"Validation summary: {result['validation']['summary_path']}")
