"""Event Selection and STAC Catalog Writer for SST Events."""

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EventSelection:
    """Event selection from SST results for training and testing datasets."""

    def __init__(
        self,
        pq_file: str,
        arrival_rate: int = 10,
        window_ratio: float = 0.2,
        test_rp_range: list[int] | None = None,
        tol: float = 0.15,
    ) -> None:
        """Initialize class with input parameters and load storm data."""
        self.pq_file = pq_file
        self.arrival_rate = arrival_rate
        self.window_ratio = window_ratio
        self.tol = tol
        self.test_rp_range = test_rp_range or [5, 2000]

        self.df = pd.read_parquet(pq_file)
        self.event_max = self._calculate_return_periods()

    def _calculate_return_periods(self) -> pd.DataFrame:
        """Compute return periods for max precipitation and inflow."""
        event_max = (
            self.df.groupby("event_id")[["precip-cum", "inflow"]]
            .max()
            .reset_index()
            .sort_values("event_id")
            .reset_index(drop=True)
        )

        def get_return_period_function(series: pd.Series) -> interp1d:
            blocks = [
                series[i : i + self.arrival_rate].max()
                for i in range(0, len(series), self.arrival_rate)
            ]
            sorted_blocks = np.sort(blocks)[::-1]
            ranks = np.arange(1, len(sorted_blocks) + 1)
            unique_blocks, idx = np.unique(sorted_blocks, return_index=True)
            unique_ranks = ((len(sorted_blocks) + 1) / ranks)[idx]

            return interp1d(
                unique_blocks,
                unique_ranks,
                bounds_error=False,
                fill_value="extrapolate",
            )

        event_max["RP_precip-cum"] = get_return_period_function(
            event_max["precip-cum"]
        )(event_max["precip-cum"])
        event_max["RP_inflow"] = get_return_period_function(event_max["inflow"])(
            event_max["inflow"]
        )
        return event_max

    def _is_close(self, rp1: float, rp2: float) -> bool:
        """Check if two return periods are within the tolerance."""
        return abs(rp1 - rp2) / max(rp1, rp2) < self.tol

    def _select_aep_storms(self, target_rps: list[int]) -> pd.DataFrame:
        """Select storms matching target RPs using joint log-distance."""
        selected, sets, selected_ids = [], [], set()

        preselected = [
            ("Max", self.event_max.loc[self.event_max["RP_precip-cum"].idxmax()]),
            ("Max", self.event_max.loc[self.event_max["RP_inflow"].idxmax()]),
            (
                "Max",
                self.event_max.assign(
                    joint=(
                        (
                            (
                                self.event_max["precip-cum"]
                                - self.event_max["precip-cum"].min()
                            )
                            / (
                                self.event_max["precip-cum"].max()
                                - self.event_max["precip-cum"].min()
                            )
                        )
                        + (
                            (self.event_max["inflow"] - self.event_max["inflow"].min())
                            / (
                                self.event_max["inflow"].max()
                                - self.event_max["inflow"].min()
                            )
                        )
                    )
                ).loc[lambda df: df["joint"].idxmax()],
            ),
        ]

        for label, row in preselected:
            if row["event_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["event_id"])
                sets.append(label)

        for rp in target_rps:
            rmin, rmax = rp * (1 - self.window_ratio), rp * (1 + self.window_ratio)

            for rp_field, _ in [
                ("RP_precip-cum", "RP_inflow"),
                ("RP_inflow", "RP_precip-cum"),
            ]:
                window_df = self.event_max[
                    self.event_max[rp_field].between(rmin, rmax)
                ].copy()

                if not window_df.empty:
                    window_df = window_df.assign(
                        log_dist=np.sqrt(
                            (np.log10(window_df["RP_precip-cum"] / rp)) ** 2
                            + (np.log10(window_df["RP_inflow"] / rp)) ** 2
                        )
                    )
                    for _, s in window_df.sort_values("log_dist").iterrows():
                        if s["event_id"] not in selected_ids and not any(
                            self._is_close(s["RP_precip-cum"], x["RP_precip-cum"])
                            and self._is_close(s["RP_inflow"], x["RP_inflow"])
                            for x in selected
                        ):
                            selected.append(
                                s.drop(labels=["log_dist"], errors="ignore")
                            )
                            selected_ids.add(s["event_id"])
                            sets.append("AEP")
                            break

        df = pd.DataFrame(selected).copy()
        df["Set"] = sets
        df["Type"] = "Train"
        return df

    def _select_diverse_storms(
        self, selected_event_ids: list[int], num_to_select: int, n_components: int = 5
    ) -> pd.DataFrame:
        """Select diverse storms using PCA on time series."""
        df = self.df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(["event_id", "datetime"])
        df["hour"] = df.groupby("event_id").cumcount()

        p1 = df.pivot(index="event_id", columns="hour", values="precip-excess").fillna(
            0
        )
        p2 = df.pivot(index="event_id", columns="hour", values="inflow").fillna(0)

        pcs1 = PCA(n_components=n_components).fit_transform(p1)
        pcs2 = PCA(n_components=n_components).fit_transform(p2)
        pcs_comb = pd.DataFrame(np.concatenate([pcs1, pcs2], axis=1), index=p1.index)
        pcs_scaled = pd.DataFrame(
            StandardScaler().fit_transform(pcs_comb), index=p1.index
        )

        selected, candidates = set(selected_event_ids), set(pcs_scaled.index)
        selected_list, added = list(selected), []
        candidates -= selected

        for _ in range(num_to_select):
            sel_vecs = pcs_scaled.loc[selected_list].values
            cand_vecs = pcs_scaled.loc[list(candidates)].values
            dists = np.linalg.norm(cand_vecs[:, None, :] - sel_vecs[None, :, :], axis=2)
            best_id = list(candidates)[dists.min(axis=1).argmax()]
            added.append(best_id)
            selected_list.append(best_id)
            candidates.remove(best_id)

        df_div = self.event_max[self.event_max.event_id.isin(added)].copy()
        df_div["Set"] = "Diverse"
        df_div["Type"] = "Train"
        return df_div

    def _select_test_storms(
        self,
        test_rp_range: list[int],
        n_test_storms: int,
        excluded_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Select test storms from RP bins with fallback if bins are empty."""
        rp_min, rp_max = test_rp_range
        n_bins = n_test_storms // 2
        rng = np.random.default_rng(seed=42)

        eligible_df = self.event_max[
            (self.event_max["RP_precip-cum"].between(rp_min, rp_max))
            & (self.event_max["RP_inflow"].between(rp_min, rp_max))
        ].copy()

        if excluded_ids:
            eligible_df = eligible_df[~eligible_df["event_id"].isin(excluded_ids)]

        if eligible_df.empty:
            raise ValueError("No eligible storms found in the specified test RP range.")

        def sample_from_bins(rp_col: str) -> set[int]:
            bins = np.logspace(np.log10(rp_min), np.log10(rp_max), n_bins + 1)
            selected = set()
            for i in range(n_bins):
                rmin, rmax = bins[i], bins[i + 1]
                bin_df = eligible_df[
                    (eligible_df[rp_col] >= rmin) & (eligible_df[rp_col] <= rmax)
                ]
                if not bin_df.empty:
                    sample = bin_df.sample(1, random_state=rng.integers(0, 10000))
                    selected.add(sample.iloc[0]["event_id"])
            return selected

        test_ids = sample_from_bins("RP_precip-cum") | sample_from_bins("RP_inflow")

        if len(test_ids) < n_test_storms:
            remaining = eligible_df[~eligible_df["event_id"].isin(test_ids)].copy()
            remaining["mean_rp"] = (
                remaining["RP_precip-cum"] + remaining["RP_inflow"]
            ) / 2
            filler = remaining.sample(n=n_test_storms - len(test_ids), random_state=42)[
                "event_id"
            ].tolist()
            test_ids.update(filler)

        test_df = self.event_max[self.event_max.event_id.isin(test_ids)].copy()
        test_df["Set"] = "Test"
        test_df["Type"] = "Test"
        return test_df

    def run_selection(
        self, n_train_storms: int, n_test_storms: int, target_rps: list[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run event selection and return combined selected events and event maxima."""
        aep_df = self._select_aep_storms(target_rps)
        n_diverse = n_train_storms - len(aep_df)
        diverse_df = self._select_diverse_storms(aep_df.event_id.tolist(), n_diverse)

        train_df = pd.concat([aep_df, diverse_df], ignore_index=True)
        train_df["Type"] = "Train"

        test_df = self._select_test_storms(
            test_rp_range=self.test_rp_range,
            n_test_storms=n_test_storms,
            excluded_ids=train_df.event_id.tolist(),
        )

        selected = pd.concat([train_df, test_df], ignore_index=True)
        return selected, self.event_max


def write_events_stac(
    selected_df: pd.DataFrame, s3_prefix: str, out_path: str
) -> dict[str, Any]:
    """Write a STAC-style JSON catalog for selected hydrologic storm events."""
    assets = {}
    for _, row in selected_df.iterrows():
        event_id = int(row["event_id"])
        role = str(row["Type"])
        set_label = str(row["Set"]) if "Set" in row else "Unknown"

        assets[str(event_id)] = {
            "href": f"{s3_prefix}/{event_id}/hydrology/SST.dss",
            "type": "application/x.hec-dss",
            "roles": [role],
            "event_id": event_id,
            "properties": {
                "Set": set_label,
                "avg_precip_in": round(float(row["precip-cum"]), 2),
                "peak_inflow_cfs": round(float(row["inflow"]), 2),
                "recurrence_interval_precip": round(float(row["RP_precip-cum"]), 2),
                "recurrence_interval_inflow": round(float(row["RP_inflow"]), 2),
            },
        }

    stac_item: dict[str, Any] = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": "GPR_SST_Events",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": None,
            "train_event_count": int((selected_df["Type"] == "Train").sum()),
            "test_event_count": int((selected_df["Type"] == "Test").sum()),
        },
        "links": [],
        "assets": assets,
    }

    with open(out_path, "w") as f:
        json.dump(stac_item, f, indent=2)

    return stac_item


if __name__ == "__main__":
    ss = EventSelection(
        pq_file="./data_dir/precip_data/west-fork_s330_hms.pq",
        arrival_rate=10,
        window_ratio=0.2,
        test_rp_range=[2, 2000],
        tol=0.15,
    )

    storms, event_max = ss.run_selection(
        n_train_storms=35,
        n_test_storms=14,
        target_rps=[2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000],
    )

    write_events_stac(
        selected_df=storms,
        s3_prefix="s3://trinity-pilot/conformance/simulations/event-data",
        out_path="./data_dir/precip_data/events.stac.json",
    )
