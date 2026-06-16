"""Utilities for calculating mains statistics across NILMTK buildings."""

import logging

logger = logging.getLogger(__name__)


def _empty_stats(ac_type):
    return {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0,
        "data_points": 0,
        "ac_type": ac_type,
    }


def calculate_multi_building_mains_stats(
    dataset_path,
    building_ids,
    start_time,
    end_time,
    ac_type="active",
    sample_period=60,
    verbose=False,
):
    """Calculate mains statistics across multiple buildings.

    NILMTK is imported only when this function is called so importing this
    module stays cheap and does not access datasets.
    """
    import pandas as pd
    from nilmtk import DataSet

    ds = DataSet(dataset_path)
    try:
        ds.set_window(start=start_time, end=end_time)
        all_mains_data = []

        for building_id in building_ids:
            if verbose:
                logger.info("Processing Building %s...", building_id)
            try:
                mains = ds.buildings[building_id].elec.mains()
                power_data = mains.power_series_all_data(
                    ac_type=ac_type,
                    sample_period=sample_period,
                )

                if power_data is not None and not power_data.empty:
                    all_mains_data.append(power_data)
                elif verbose:
                    logger.info(
                        "No data found for Building %s in the specified timeframe.",
                        building_id,
                    )
            except KeyError:
                if verbose:
                    logger.info("Building %s not found in the dataset.", building_id)
            except Exception:
                if verbose:
                    logger.exception("Failed to process Building %s.", building_id)
                else:
                    logger.debug(
                        "Failed to process Building %s.",
                        building_id,
                        exc_info=True,
                    )

        if not all_mains_data:
            if verbose:
                logger.info("Could not retrieve data for any specified buildings.")
            return _empty_stats(ac_type)

        if verbose:
            logger.info("Combining data from all buildings.")
        clean_data = pd.concat(all_mains_data).dropna()

        return {
            "mean": clean_data.mean(),
            "std": clean_data.std(),
            "min": clean_data.min(),
            "max": clean_data.max(),
            "data_points": len(clean_data),
            "ac_type": ac_type,
        }
    finally:
        store = getattr(ds, "store", None)
        if store is not None:
            store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = calculate_multi_building_mains_stats(
        dataset_path="/home/ubuntu/downloads/refit.h5",
        building_ids=[2],
        start_time="2014-04-01",
        end_time="2014-04-30",
        ac_type="active",
        sample_period=60,
        verbose=True,
    )

    logger.info("--- Combined Mains Statistics ---")
    if stats["data_points"] > 0:
        logger.info("Combined Mains Mean: %.2fW", stats["mean"])
        logger.info("Combined Mains Std: %.2fW", stats["std"])
        logger.info("Data Range: %.2fW to %.2fW", stats["min"], stats["max"])
        logger.info("Total Data Points from all buildings: %s", stats["data_points"])
    else:
        logger.info("No data available to calculate statistics.")
