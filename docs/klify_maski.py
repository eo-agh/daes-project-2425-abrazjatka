#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pystac_client
import planetary_computer as pc
import rasterio
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
import matplotlib.pyplot as plt
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import transform_geom, transform_bounds, calculate_default_transform, reproject, Resampling
import matplotlib.colors as mcolors
import io
import base64
from tqdm import tqdm
from scipy.ndimage import median_filter
from shapely.geometry import shape
from rasterio.transform import from_origin, rowcol, from_bounds
import tempfile
import json
import math

# Połączenie z publicznym katalogiem STAC na Azure Planetary Computer
@st.cache_resource
def get_stac_client():
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    return pystac_client.Client.open(stac_url)

stac_client = get_stac_client()

# Funkcje pomocnicze
def clip_raster(dataset, aoi):
    aoi_transformed = transform_geom('EPSG:4326', dataset.crs, shape(aoi))
    aoi_geom = [json.loads(json.dumps(aoi_transformed))]
    clipped_array, clipped_transform = mask(dataset, aoi_geom, crop=True)
    return clipped_array[0], clipped_transform

def reproject_array(array, src_crs, dst_crs, src_transform):
    dst_transform, width, height = calculate_default_transform(
        src_crs, dst_crs, array.shape[1], array.shape[0], *rasterio.transform.array_bounds(array.shape[0], array.shape[1], src_transform)
    )
    dst_array = np.empty((height, width), dtype=np.float32)

    reproject(
        source=array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return dst_array, dst_transform

@st.cache_data(show_spinner=True)
def process_data_and_differences(cords, years):
    lat_center = cords[0]
    lon_center = cords[1]

    half_side_km = 10

    lat_deg_per_km = 1 / 111
    lon_deg_per_km = 1 / (111 * math.cos(math.radians(lat_center)))

    delta_lat = half_side_km * lat_deg_per_km
    delta_lon = half_side_km * lon_deg_per_km

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [lon_center - delta_lon, lat_center - delta_lat],
            [lon_center + delta_lon, lat_center - delta_lat],
            [lon_center + delta_lon, lat_center + delta_lat],
            [lon_center - delta_lon, lat_center + delta_lat],
            [lon_center - delta_lon, lat_center - delta_lat]
        ]]
    }

    aoi_shape = shape(aoi)
    resolution = 0.0001
    aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_shape.bounds

    vh_medians_by_year = {}
    vh_ds_crs = None
    red_transform_global = None

    # Sort years to ensure consistent comparison order
    years_sorted = sorted(years)

    for year in years_sorted:
        st.write(f"🔍 Przetwarzanie roku {year}")
        time_range = f"{year}-06-01/{year}-08-01"

        search = stac_client.search(
            collections=["sentinel-1-rtc"],
            intersects=aoi,
            datetime=time_range,
        )
        items = list(search.items())
        st.write(f"Znaleziono {len(items)} scen dla roku {year}")

        vh_stack = []

        progress_text = f"Pobieranie i przetwarzanie scen dla roku {year}. Proszę czekać."
        my_bar = st.progress(0, text=progress_text)

        for i, item in enumerate(items):
            try:
                href_vh = pc.sign(item.assets["vh"].href)

                with rasterio.open(href_vh) as vh_ds:
                    if vh_ds_crs is None:
                        vh_ds_crs = vh_ds.crs
                    vh, red_transform = clip_raster(vh_ds, aoi)
                    if red_transform_global is None:
                        red_transform_global = red_transform
                    vh = np.where(vh == -32768, np.nan, vh).astype(np.float32)

                if not vh_stack:
                    vh_stack.append(vh)
                elif vh.shape == vh_stack[0].shape:
                    vh_stack.append(vh)
                else:
                    st.warning(f"Pominięto scenę o innym wymiarze: {item.id}")

            except Exception as e:
                st.error(f"Błąd w scenie {item.id}: {e}")
                continue
            finally:
                my_bar.progress((i + 1) / len(items), text=progress_text)
        my_bar.empty()

        if vh_stack:
            vh_array = np.stack(vh_stack)
            vh_median = np.nanmedian(vh_array, axis=0)
            vh_medians_by_year[year] = vh_median
            st.success(f"✅ Obliczono medianę dla roku {year}")
        else:
            st.warning(f"⚠️ Brak poprawnych danych dla roku {year}")

    if vh_ds_crs is None or red_transform_global is None:
        st.error("Nie udało się przetworzyć żadnych danych. Sprawdź obszar zainteresowania i lata.")
        return None, None, None, None, None, None, None

    water_mask_filtered_by_year = {}
    for year in years_sorted:
        if year in vh_medians_by_year:
            water_mask = (vh_medians_by_year.get(year) < 0.007)
            water_mask_filtered = median_filter(water_mask.astype(np.uint8), size=3)
            water_mask_filtered_by_year[year] = water_mask_filtered
            st.write(f"Zrobiono maskę wody dla roku {year}")
        else:
            st.warning(f"Brak danych VH dla roku {year}, pomijam generowanie maski wody.")

    dataset_crs = vh_ds.crs
    height, width = vh.shape
    left, top = red_transform * (0, 0)
    right, bottom = red_transform * (width, height)
    bounds = transform_bounds(dataset_crs, 'EPSG:4326', left, bottom, right, top)

    water_mask_reprojected_by_year = {}
    for year in years_sorted:
        if year in water_mask_filtered_by_year:
            water_mask_reprojected, _ = reproject_array(water_mask_filtered_by_year.get(year), dataset_crs, 'EPSG:4326', red_transform)
            water_mask_reprojected_by_year[year] = water_mask_reprojected
            st.write(f"Zmieniono uklad wspulzednych dla roku: {year}")

    if water_mask_reprojected_by_year:
        last_year = sorted(water_mask_reprojected_by_year.keys())[-1]
        sample_mask = water_mask_reprojected_by_year[last_year]
        minx, miny, maxx, maxy = bounds
        height, width = sample_mask.shape
    else:
        st.error("Brak reprojekcji masek wodnych do określenia transformacji.")
        return None, None, None, None, None, None, None

    transform_water = from_bounds(minx, miny, maxx, maxy, width, height)
    row_start, col_start = rowcol(transform_water , aoi_minx, aoi_maxy)
    row_stop, col_stop = rowcol(transform_water , aoi_maxx, aoi_miny)

    row_start, row_stop = sorted([row_start, row_stop])
    col_start, col_stop = sorted([col_start, col_stop])

    water_mask_clipped_by_year = {}
    for year in years_sorted:
        if year in water_mask_reprojected_by_year:
            water_mask_clipped = water_mask_reprojected_by_year.get(year)
            row_start_clip = max(0, row_start)
            row_stop_clip = min(water_mask_clipped.shape[0], row_stop)
            col_start_clip = max(0, col_start)
            col_stop_clip = min(water_mask_clipped.shape[1], col_stop)

            # Ensure the clipped mask has content
            if row_stop_clip > row_start_clip and col_stop_clip > col_start_clip:
                water_mask_clipped = water_mask_clipped[row_start_clip:row_stop_clip, col_start_clip:col_stop_clip]
                water_mask_clipped_by_year[year] = water_mask_clipped
                st.write(f"Przyciąto maskę dla roku: {year}")
            else:
                st.warning(f"Przycięta maska dla roku {year} jest pusta lub ma niewłaściwe wymiary.")
                water_mask_clipped_by_year[year] = np.zeros((1,1)) # Placeholder to avoid errors
        else:
            st.warning(f"Brak reprojekcji maski wody dla roku {year}, pomijam przycinanie.")

    # Calculate differences between consecutive years
    water_mask_differences = {}
    if len(years_sorted) > 1:
        for i in range(len(years_sorted) - 1):
            year1 = years_sorted[i]
            year2 = years_sorted[i+1]

            if year1 in water_mask_clipped_by_year and year2 in water_mask_clipped_by_year:
                mask1 = water_mask_clipped_by_year[year1]
                mask2 = water_mask_clipped_by_year[year2]

                # Ensure masks have the same shape for comparison
                if mask1.shape != mask2.shape:
                    st.warning(f"Maski dla lat {year1} i {year2} mają różne wymiary. Nie można obliczyć różnicy.")
                    continue

                # Calculate difference:
                # 1: New water (was land in year1, is water in year2)
                # -1: Lost water (was water in year1, is land in year2)
                # 0: No change (land->land or water->water)
                difference_mask = np.zeros_like(mask1, dtype=np.int8)
                difference_mask[(mask1 == 0) & (mask2 == 1)] = 1  # New water
                difference_mask[(mask1 == 1) & (mask2 == 0)] = -1 # Lost water
                water_mask_differences[f"{year2}-{year1}"] = difference_mask
                st.success(f"✅ Obliczono różnicę między {year1} a {year2}")
            else:
                st.warning(f"Brak przyciętych masek dla lat {year1} lub {year2}. Pomijam obliczanie różnicy.")


    return water_mask_clipped_by_year, water_mask_differences, bounds, aoi_minx, aoi_miny, aoi_maxx, aoi_maxy

@st.cache_data
def generate_difference_image_overlays(water_mask_differences):
    image_colored_paths_by_diff = {}
    for diff_label, diff_mask in water_mask_differences.items():
        # Define custom colors for differences
        # Red: Lost water (-1)
        # Blue: New water (1)
        # Transparent: No change (0)
        # Ensure mask is converted to float for transparency blending later if needed, or handle directly
        colored_diff_array = np.zeros((*diff_mask.shape, 4), dtype=np.uint8) # RGBA

        # Lost water (-1) -> Red
        colored_diff_array[diff_mask == -1] = [255, 0, 0, 200] # Red with some transparency

        # New water (1) -> Blue
        colored_diff_array[diff_mask == 1] = [0, 0, 255, 200] # Blue with some transparency

        # No change (0) -> Transparent
        colored_diff_array[diff_mask == 0] = [0, 0, 0, 0] # Fully transparent

        image = Image.fromarray(colored_diff_array, mode="RGBA")

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_file.name, format='PNG')
        image_colored_paths_by_diff[diff_label] = temp_file.name
        st.write(f"Dodano warstwę dla różnicy: {diff_label}")
    return image_colored_paths_by_diff

# Streamlit UI
st.title("Detekcja Wody z Sentinel-1 RTC - Analiza Zmian")

st.sidebar.header("Ustawienia Obszaru Zainteresowania (AOI)")
default_lat = 53.87
default_lon = -0.04
latitude = st.sidebar.number_input("Szerokość geograficzna (Lat)", value=default_lat, format="%.2f")
longitude = st.sidebar.number_input("Długość geograficzna (Lon)", value=default_lon, format="%.2f")
cords_input = [latitude, longitude]

st.sidebar.header("Wybór Lat")
selected_years = st.sidebar.multiselect(
    "Wybierz lata do analizy zmian (min. 2 lata)",
    options=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    default=[2016, 2021, 2023]
)
if 2022 in selected_years:
    st.sidebar.warning("Rok 2022 może zawierać anomalie w danych.")

if st.sidebar.button("Generuj mapę zmian"):
    if len(selected_years) < 2:
        st.error("Proszę wybrać co najmniej dwa lata, aby obliczyć zmiany.")
    else:
        with st.spinner("Przetwarzanie danych satelitarnych i obliczanie zmian... To może potrwać kilka minut."):
            water_mask_clipped_by_year, water_mask_differences, bounds, aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = process_data_and_differences(cords_input, selected_years)

            if water_mask_differences: # Check if any differences were calculated
                image_colored_paths_by_diff = generate_difference_image_overlays(water_mask_differences)

                m = folium.Map(location=[(aoi_miny + aoi_maxy) / 2, (aoi_minx + aoi_maxx) / 2], zoom_start=12)

                for diff_label, img_path in image_colored_paths_by_diff.items():
                    image_overlay = ImageOverlay(
                        image=img_path,
                        bounds=[[aoi_miny, aoi_minx], [aoi_maxy, aoi_maxx]],
                        opacity=0.8, # Increase opacity for differences
                        name=f"Zmiany: {diff_label}"
                    )
                    image_overlay.add_to(m)

                folium.Polygon(
                    locations=[(aoi_miny, aoi_minx), (aoi_miny, aoi_maxx), (aoi_maxy, aoi_maxx), (aoi_maxy, aoi_minx), (aoi_miny, aoi_minx)],
                    color='blue',
                    weight=2,
                    fill=False,
                    popup='Obszar Zainteresowania (AOI)'
                ).add_to(m)

                folium.LayerControl().add_to(m)

                map_html = m._repr_html_()
                st.components.v1.html(map_html, height=700)
            else:
                st.error("Nie udało się wygenerować mapy zmian. Upewnij się, że wybrano co najmniej dwa lata i dane są dostępne dla wszystkich wybranych lat.")