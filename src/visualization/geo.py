"""Map-based visualizations. Opt-in: requires `folium`."""


def plot_price_heatmap(df, lat_col="latitude", lon_col="longitude", price_col="price", center=(-22.9068, -43.1729), output_path="rio_price_heatmap.html"):
    """Render an interactive Folium heatmap of listing prices and save it as HTML."""
    import folium
    from folium.plugins import HeatMap

    map_df = df[[lat_col, lon_col, price_col]].dropna()
    heat_data = map_df.values.tolist()

    rio_map = folium.Map(location=list(center), zoom_start=12)
    HeatMap(heat_data, radius=8, blur=12, min_opacity=0.2).add_to(rio_map)
    rio_map.save(output_path)
    print(f"Map saved to {output_path}")
    return rio_map
