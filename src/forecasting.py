from prophet import Prophet
import pandas as pd

def get_demand_series_by_region(orders_df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Prepara uma série temporal com a contagem de pedidos por hora para uma região específica.

    Args:
        orders_df (pd.DataFrame): DataFrame com pedidos (inclui 'order_time' e 'region')
        region (str): Nome da região para filtrar

    Returns:
        pd.DataFrame: Colunas 'ds' (datetime) e 'y' (número de pedidos)
    """
    orders_df = orders_df.copy()
    orders_df["hour"] = orders_df["order_time"].dt.floor("h")
    grouped = orders_df.groupby(["hour", "region"]).size().reset_index(name="order_count")
    region_df = grouped[grouped["region"] == region][["hour", "order_count"]]
    region_df = region_df.rename(columns={"hour": "ds", "order_count": "y"})
    region_df = region_df.sort_values("ds").dropna()

    # ✅ Segurança extra
    region_df = region_df[region_df['y'] >= 0]
    region_df = region_df.drop_duplicates(subset='ds')

    return region_df

def train_prophet_model(df_region: pd.DataFrame, periods: int = 24):
    """
    Tenta treinar Prophet. Se falhar, retorna previsão com média móvel.
    """
    if df_region.empty:
        raise ValueError("Empty DataFrame passed to Prophet.")

    try:
        model = Prophet()
        model.fit(df_region)
        future = model.make_future_dataframe(periods=periods, freq='h')
        forecast = model.predict(future)
        return forecast, model
    except Exception as e:
        # Fallback usando média móvel simples
        fallback = df_region.copy()
        fallback['yhat'] = fallback['y'].rolling(3, min_periods=1).mean()
        last_timestamp = fallback['ds'].max()

        future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=periods, freq='h')
        fallback_forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': [fallback['yhat'].iloc[-1]] * periods
        })
        fallback_forecast['model'] = 'fallback_average'
        return fallback_forecast, None

