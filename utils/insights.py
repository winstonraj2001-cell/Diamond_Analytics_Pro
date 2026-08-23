# ============================================================
# DIAMOND ANALYTICS PRO
# BUSINESS INSIGHTS UTILITIES
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# BUSINESS METRICS
# ============================================================

def calculate_business_metrics(df):

    data = df.copy()

    total_diamonds = len(data)

    average_price = (
        data["price"].mean()
        if "price" in data.columns
        else 0
    )

    median_price = (
        data["price"].median()
        if "price" in data.columns
        else 0
    )

    average_carat = (
        data["carat"].mean()
        if "carat" in data.columns
        else 0
    )

    total_value = (
        data["price"].sum()
        if "price" in data.columns
        else 0
    )

    highest_price = (
        data["price"].max()
        if "price" in data.columns
        else 0
    )

    lowest_price = (
        data["price"].min()
        if "price" in data.columns
        else 0
    )

    if (
        "price" in data.columns
        and "carat" in data.columns
    ):

        valid_carat = data["carat"] > 0

        price_per_carat = (
            data.loc[
                valid_carat,
                "price"
            ]
            /
            data.loc[
                valid_carat,
                "carat"
            ]
        ).mean()

    else:

        price_per_carat = 0

    popular_cut = (
        data["cut"].mode().iloc[0]
        if "cut" in data.columns
        and not data["cut"].dropna().empty
        else "N/A"
    )

    popular_color = (
        data["color"].mode().iloc[0]
        if "color" in data.columns
        and not data["color"].dropna().empty
        else "N/A"
    )

    popular_clarity = (
        data["clarity"].mode().iloc[0]
        if "clarity" in data.columns
        and not data["clarity"].dropna().empty
        else "N/A"
    )

    return {
        "total_diamonds": int(total_diamonds),
        "average_price": float(average_price),
        "median_price": float(median_price),
        "average_carat": float(average_carat),
        "total_value": float(total_value),
        "highest_price": float(highest_price),
        "lowest_price": float(lowest_price),
        "average_price_per_carat": float(
            price_per_carat
        ),
        "popular_cut": popular_cut,
        "popular_color": popular_color,
        "popular_clarity": popular_clarity
    }


# ============================================================
# PRICE SEGMENTS
# ============================================================

def calculate_price_segments(df):

    data = df.copy()

    if "price" not in data.columns:
        return pd.DataFrame()

    data["price_segment"] = pd.cut(
        data["price"],
        bins=[
            -np.inf,
            1000,
            3000,
            6000,
            np.inf
        ],
        labels=[
            "Budget",
            "Value",
            "Premium",
            "Luxury"
        ]
    )

    return data


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def category_performance(df, column):

    if column not in df.columns:
        return pd.DataFrame()

    if "price" not in df.columns:

        return (
            df[column]
            .value_counts()
            .reset_index()
        )

    result = (
        df.groupby(column)
        .agg(
            diamond_count=(
                "price",
                "count"
            ),
            average_price=(
                "price",
                "mean"
            ),
            median_price=(
                "price",
                "median"
            ),
            total_value=(
                "price",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "average_price",
            ascending=False
        )
    )

    return result


# ============================================================
# CUT PRICE ANALYSIS
# ============================================================

def get_cut_price_analysis(df):

    if (
        "cut" not in df.columns
        or "price" not in df.columns
    ):
        return pd.DataFrame()

    result = (
        df.groupby("cut")
        .agg(
            diamond_count=(
                "price",
                "count"
            ),
            average_price=(
                "price",
                "mean"
            ),
            median_price=(
                "price",
                "median"
            ),
            total_value=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    total_count = result[
        "diamond_count"
    ].sum()

    if total_count > 0:

        result["market_share"] = (
            result["diamond_count"]
            /
            total_count
            *
            100
        )

    else:

        result["market_share"] = 0

    return result.sort_values(
        "average_price",
        ascending=False
    )


# ============================================================
# COLOR PRICE ANALYSIS
# ============================================================

def get_color_price_analysis(df):

    if (
        "color" not in df.columns
        or "price" not in df.columns
    ):
        return pd.DataFrame()

    result = (
        df.groupby("color")
        .agg(
            diamond_count=(
                "price",
                "count"
            ),
            average_price=(
                "price",
                "mean"
            ),
            median_price=(
                "price",
                "median"
            ),
            total_value=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    total_count = result[
        "diamond_count"
    ].sum()

    if total_count > 0:

        result["market_share"] = (
            result["diamond_count"]
            /
            total_count
            *
            100
        )

    else:

        result["market_share"] = 0

    return result.sort_values(
        "average_price",
        ascending=False
    )


# ============================================================
# CLARITY PRICE ANALYSIS
# ============================================================

def get_clarity_price_analysis(df):

    if (
        "clarity" not in df.columns
        or "price" not in df.columns
    ):
        return pd.DataFrame()

    result = (
        df.groupby("clarity")
        .agg(
            diamond_count=(
                "price",
                "count"
            ),
            average_price=(
                "price",
                "mean"
            ),
            median_price=(
                "price",
                "median"
            ),
            total_value=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    total_count = result[
        "diamond_count"
    ].sum()

    if total_count > 0:

        result["market_share"] = (
            result["diamond_count"]
            /
            total_count
            *
            100
        )

    else:

        result["market_share"] = 0

    return result.sort_values(
        "average_price",
        ascending=False
    )


# ============================================================
# PRICE PER CARAT ANALYSIS
# ============================================================

def get_price_per_carat_analysis(df):

    if (
        "price" not in df.columns
        or "carat" not in df.columns
    ):
        return pd.DataFrame()

    data = df.copy()

    data = data[
        data["carat"] > 0
    ].copy()

    data["price_per_carat"] = (
        data["price"]
        /
        data["carat"]
    )

    return data


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(df):

    data = df.copy()

    insights = []

    # --------------------------------------------------------
    # Cut
    # --------------------------------------------------------

    if "cut" in data.columns:

        counts = (
            data["cut"]
            .value_counts()
        )

        if not counts.empty:

            top_cut = counts.index[0]
            count = counts.iloc[0]

            insights.append(
                f"Most common cut is "
                f"{top_cut}, representing "
                f"{count:,} diamonds."
            )

    # --------------------------------------------------------
    # Highest average cut price
    # --------------------------------------------------------

    if (
        "cut" in data.columns
        and "price" in data.columns
    ):

        cut_prices = (
            data.groupby("cut")[
                "price"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not cut_prices.empty:

            best_cut = cut_prices.index[0]

            best_price = cut_prices.iloc[0]

            insights.append(
                f"{best_cut} has the "
                f"highest average price "
                f"at ${best_price:,.2f}."
            )

    # --------------------------------------------------------
    # Average carat
    # --------------------------------------------------------

    if "carat" in data.columns:

        average_carat = (
            data["carat"].mean()
        )

        insights.append(
            f"Average diamond size is "
            f"{average_carat:.2f} carats."
        )

    # --------------------------------------------------------
    # Price range
    # --------------------------------------------------------

    if "price" in data.columns:

        minimum = data["price"].min()
        maximum = data["price"].max()

        insights.append(
            f"Diamond prices range from "
            f"${minimum:,.0f} to "
            f"${maximum:,.0f}."
        )

    return insights


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

def generate_business_recommendations(df):

    data = df.copy()

    recommendations = []

    if (
        "cut" in data.columns
        and "price" in data.columns
    ):

        cut_analysis = (
            data.groupby("cut")
            .agg(
                count=("price", "count"),
                average_price=("price", "mean")
            )
            .sort_values(
                "average_price",
                ascending=False
            )
        )

        if not cut_analysis.empty:

            premium_cut = (
                cut_analysis.index[0]
            )

            recommendations.append(
                f"Consider focusing on "
                f"{premium_cut} diamonds "
                f"because this segment has "
                f"the highest average price."
            )

    if (
        "carat" in data.columns
        and "price" in data.columns
    ):

        correlation = (
            data["carat"]
            .corr(data["price"])
        )

        if pd.notna(correlation):

            if correlation > 0.5:

                recommendations.append(
                    "Carat has a strong positive "
                    "relationship with price. "
                    "Higher-carat inventory may "
                    "support premium positioning."
                )

            else:

                recommendations.append(
                    "Carat shows a moderate or "
                    "weak relationship with price, "
                    "so quality attributes should "
                    "also be considered."
                )

    if not recommendations:

        recommendations.append(
            "Use the market analysis charts "
            "to identify high-value diamond "
            "segments."
        )

    return recommendations


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def create_executive_summary(df):

    metrics = calculate_business_metrics(df)

    summary = {
        "Total Diamonds":
            metrics["total_diamonds"],

        "Average Price":
            metrics["average_price"],

        "Median Price":
            metrics["median_price"],

        "Average Carat":
            metrics["average_carat"],

        "Total Inventory Value":
            metrics["total_value"],

        "Highest Price":
            metrics["highest_price"],

        "Lowest Price":
            metrics["lowest_price"],

        "Popular Cut":
            metrics["popular_cut"],

        "Popular Color":
            metrics["popular_color"],

        "Popular Clarity":
            metrics["popular_clarity"]
    }

    return summary


# ============================================================
# FORMAT CURRENCY
# ============================================================

def format_currency(value):

    try:
        return f"₹{float(value):,.2f}"

    except (
        TypeError,
        ValueError
    ):
        return "₹0.00"