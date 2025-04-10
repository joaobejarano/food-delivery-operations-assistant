def get_demand_series_by_region(orders_df, region):
    orders_df["hour"] = orders_df["order_time"].dt.floor("H")
    grouped = orders_df.groupby(["hour", "region"]).size().reset_index(name="order_count")
    region_df = grouped[grouped["region"] == region][["hour", "order_count"]]
    region_df = region_df.rename(columns={"hour": "ds", "order_count": "y"})
    return region_df
