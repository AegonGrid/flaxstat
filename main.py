from math import ceil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw_apro_cpshr_flax.tsv"
OUTPUT_DIR = ROOT / "output"
NUTS_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_01M_2024_4326_LEVL_{level}.geojson"
)
METRICS = {"AR_THS_HA": "area_ha", "HPRD_HUMD_EU_THS_T": "prod_t"}
PLOT_COLORS = ["#d95f02", "#1b9e77", "#7570b3"]
COUNTRY_CODES = ["FR", "BE", "NL"]
N_REGIONS = 10
YEAR_OF_INTEREST = 2021


def load_flax_data(
    path: Path = RAW_PATH, country_codes: list[str] | None = None
) -> pd.DataFrame:
    """Load Eurostat's mixed comma/tabular export into a tidy NUTS2 table."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    key_columns = raw.iloc[:, 0].str.strip().str.split(",", expand=True)
    key_columns.columns = ["freq", "crops", "strucpro", "geo"]
    values = raw.iloc[:, 1:].copy()
    values.columns = [str(column).strip() for column in values.columns]
    tidy = pd.concat([key_columns, values], axis=1).melt(
        id_vars=["freq", "crops", "strucpro", "geo"],
        var_name="year",
        value_name="value",
    )
    tidy["geo"] = tidy["geo"].str.strip()
    tidy = tidy[tidy["geo"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{2}")].copy()
    tidy = tidy[tidy["geo"].str[:2].ne("EU")].copy()
    tidy["year"] = pd.to_numeric(tidy["year"], errors="coerce").astype("Int64")
    values = tidy["value"].str.strip().replace(":", pd.NA)
    tidy["value"] = pd.to_numeric(
        values.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    tidy = tidy.dropna(subset=["value"])
    tidy = tidy[tidy["strucpro"].isin((*METRICS, "YLD_HUMD_EU_T_HA"))]

    result = (
        tidy.assign(country_code=tidy["geo"].str[:2], region_code=tidy["geo"])
        .pivot_table(
            index=["country_code", "region_code", "year"],
            columns="strucpro",
            values="value",
            aggfunc="first",
        )
        .rename(columns=METRICS | {"YLD_HUMD_EU_T_HA": "yield_t_ha"})
        .reset_index()
    )
    for column in ("area_ha", "prod_t", "yield_t_ha"):
        if column not in result:
            result[column] = pd.NA
    result = result[result["area_ha"].notna() & result["area_ha"].ne(0)]
    derived_yield = result["prod_t"].div(result["area_ha"].replace(0, pd.NA))
    result["yield_t_ha"] = result["yield_t_ha"].fillna(derived_yield)
    result = result.dropna(subset=["area_ha", "prod_t", "yield_t_ha"], how="all")
    for column in ("area_ha", "prod_t", "yield_t_ha"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["yield_t_ha"] = result["yield_t_ha"].round(3)
    return (
        result[
            ["country_code", "region_code", "year", "area_ha", "prod_t", "yield_t_ha"]
        ]
        .query("country_code in @country_codes" if country_codes else "True")
        .sort_values(["country_code", "region_code", "year"])
    )


def save_timeseries(data: pd.DataFrame) -> None:
    country_totals = data.groupby(["country_code", "year"], as_index=False).agg(
        area_ha=("area_ha", lambda values: values.sum(min_count=1)),
        prod_t=("prod_t", lambda values: values.sum(min_count=1)),
    )
    country_totals["yield_t_ha"] = pd.to_numeric(
        country_totals["prod_t"].div(country_totals["area_ha"].replace(0, pd.NA)),
        errors="coerce",
    )
    top_countries = {
        metric: country_totals.groupby("country_code")[metric]
        .sum(min_count=1)
        .nlargest(3)
        .index
        for metric in ("yield_t_ha", "area_ha", "prod_t")
    }
    labels = {
        "yield_t_ha": "Yield (t/ha)",
        "area_ha": "Area (1000 ha)",
        "prod_t": "Production (1000 t)",
    }
    for metric, label in labels.items():
        figure, axis = plt.subplots(figsize=(10, 6))
        for country, country_data in data.groupby("country_code"):
            is_top = country in top_countries[metric]
            country_series = country_data.groupby("year", as_index=False).agg(
                area_ha=("area_ha", lambda values: values.sum(min_count=1)),
                prod_t=("prod_t", lambda values: values.sum(min_count=1)),
            )
            if metric == "yield_t_ha":
                country_series[metric] = (
                    country_series["prod_t"]
                    .div(country_series["area_ha"].replace(0, pd.NA))
                    .round(3)
                )
            country_series[metric] = pd.to_numeric(
                country_series[metric], errors="coerce"
            )
            axis.plot(
                country_series["year"],
                country_series[metric],
                color=(
                    PLOT_COLORS[list(top_countries[metric]).index(country)]
                    if is_top
                    else "#c7c7c7"
                ),
                linewidth=2 if is_top else 0.8,
                alpha=1 if is_top else 0.65,
                label=country if is_top else None,
            )
        axis.set(title=f"Flax {label.lower()} by country", xlabel="Year", ylabel=label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", color="#e5e5e5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(title="Top countries", frameon=False)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"timeseries_{metric}.png", dpi=160)
        plt.close(figure)

    top_regions = (
        data.groupby("region_code")["prod_t"].sum(min_count=1).nlargest(N_REGIONS).index
    )
    for metric, label in labels.items():
        figure, axis = plt.subplots(figsize=(10, 6))
        for region, region_data in data[data["region_code"].isin(top_regions)].groupby(
            "region_code"
        ):
            region_series = region_data.groupby("year", as_index=False).agg(
                area_ha=("area_ha", lambda values: values.sum(min_count=1)),
                prod_t=("prod_t", lambda values: values.sum(min_count=1)),
            )
            if metric == "yield_t_ha":
                region_series[metric] = (
                    region_series["prod_t"]
                    .div(region_series["area_ha"].replace(0, pd.NA))
                    .round(3)
                )
            region_series[metric] = pd.to_numeric(
                region_series[metric], errors="coerce"
            )
            axis.plot(region_series["year"], region_series[metric], label=region)
        axis.set(
            title=f"Flax {label.lower()} by top {N_REGIONS} NUTS2 regions",
            xlabel="Year",
            ylabel=label,
        )
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", color="#e5e5e5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(title="NUTS2 region", frameon=False)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"timeseries_nuts2_{metric}.png", dpi=160)
        plt.close(figure)


def load_boundaries(level: int) -> gpd.GeoDataFrame:
    cache_path = OUTPUT_DIR / f"nuts_level_{level}.geojson"
    if not cache_path.exists():
        gpd.read_file(NUTS_URL.format(level=level)).to_file(
            cache_path, driver="GeoJSON"
        )
    return gpd.read_file(cache_path)


def mainland_bounds(
    countries: gpd.GeoDataFrame, country_codes: list[str]
) -> tuple[float, float, float, float]:
    selected = countries[countries["CNTR_CODE"].isin(country_codes)].copy()
    parts = selected.explode(index_parts=False).reset_index(drop=True)
    equal_area = parts.to_crs("EPSG:3035")
    areas = equal_area.geometry.area
    largest_indices = areas.groupby(parts["CNTR_CODE"]).idxmax()
    largest_parts = parts.loc[largest_indices]
    return largest_parts.union_all().bounds


def save_maps(data: pd.DataFrame, year_of_interest: int) -> None:
    regions = load_boundaries(2)
    countries = load_boundaries(0)
    country_codes = data["country_code"].drop_duplicates().tolist()
    year_of_interest = year_of_interest
    mapped = regions.merge(
        data[data["year"] == year_of_interest],
        left_on="NUTS_ID",
        right_on="region_code",
    )
    yield_min = int(data["yield_t_ha"].min())
    yield_max = ceil(data["yield_t_ha"].max())
    yield_norm = Normalize(vmin=yield_min, vmax=yield_max)
    yield_ticks = [
        ceil(yield_norm.vmin + (yield_norm.vmax - yield_norm.vmin) * step / 4)
        for step in range(5)
    ]
    labels = {
        "yield_t_ha": "Yield (t/ha)",
        "area_ha": "Area (ha)",
        "prod_t": "Production (t)",
    }
    for metric, label in labels.items():
        figure, axis = plt.subplots(figsize=(11, 8))
        mapped.plot(
            column=metric,
            ax=axis,
            cmap="RdYlGn" if metric == "yield_t_ha" else "YlGnBu",
            norm=yield_norm if metric == "yield_t_ha" else None,
            vmin=yield_norm.vmin if metric == "yield_t_ha" else None,
            vmax=yield_norm.vmax if metric == "yield_t_ha" else None,
            legend=True,
            legend_kwds=({"ticks": yield_ticks} if metric == "yield_t_ha" else {}),
            missing_kwds={"color": "#eeeeee"},
        )
        selected_countries = countries[countries["CNTR_CODE"].isin(country_codes)]
        selected_countries.boundary.plot(ax=axis, color="#333333", linewidth=0.45)
        min_x, min_y, max_x, max_y = mainland_bounds(countries, country_codes)
        padding_x = max((max_x - min_x) * 0.08, 1)
        padding_y = max((max_y - min_y) * 0.08, 1)
        axis.set_xlim(min_x - padding_x, max_x + padding_x)
        axis.set_ylim(min_y - padding_y, max_y + padding_y)
        axis.set(title=f"Flax {label.lower()} by NUTS2 region, {year_of_interest}")
        axis.set_axis_off()
        figure.subplots_adjust(left=0.02, right=0.87, top=0.93, bottom=0.02)
        figure.savefig(OUTPUT_DIR / f"map_{metric}_{year_of_interest}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_flax_data(country_codes=COUNTRY_CODES)
    data.to_csv(OUTPUT_DIR / "clean_apro_cpshr_flax.csv", index=False)
    save_timeseries(data)
    save_maps(data, year_of_interest=YEAR_OF_INTEREST)
    print(f"Wrote cleaned data and nine plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
