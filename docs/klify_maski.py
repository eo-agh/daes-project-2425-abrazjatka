#!/usr/bin/env python
# coding: utf-8
import os
import pyproj

# Ustaw poprawną ścieżkę do bazy proj.db z pakietu pyproj
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()


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
from affine import Affine

import folium.plugins

# Połączenie z publicznym katalogiem STAC na Azure Planetary Computer
@st.cache_resource
def get_stac_client():
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    return pystac_client.Client.open(stac_url)

stac_client = get_stac_client()

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 400px;
    }
    .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .app-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-left: 1rem;
        }
         [data-testid="stSidebar"] {
        padding-top: 0rem;
    }
    </style>
    <div class="header-container">
        <div class="app-title">🌊 Detekcja zmian wód z Sentinel-1</div>
    </div>
    """,
    unsafe_allow_html=True
)


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
    transform_treshold = Affine(10.0, 0.0, 289640.00,
                   0.0, -10.0, 5983810.00)

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
                    vh = np.where(vh == -32768, np.nan, vh).astype(np.float32)

                    x1 = red_transform.c

                    x2 = transform_treshold.c

                result = str(int(x1))[0] == str(int(x2))[0] == '2'

                if not vh_stack and result:
                    vh_stack.append(vh)
                    red_transform_global = red_transform
                elif vh.shape == vh_stack[0].shape and result:
                    vh_stack.append(vh)
                    red_transform_global = red_transform
                else:
                    st.warning(f"Pominięto scenę o innym wymiarze lub transform: {item.id}")

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
            water_mask = (vh_medians_by_year.get(year) < 0.008)
            water_mask_filtered = median_filter(water_mask.astype(np.uint8), size=3)
            water_mask_filtered_by_year[year] = water_mask_filtered
            st.write(f"Zrobiono maskę wody dla roku {year}")
        else:
            st.warning(f"Brak danych VH dla roku {year}, pomijam generowanie maski wody.")

    dataset_crs = vh_ds.crs
    height, width = vh.shape
    left, top = red_transform_global * (0, 0)
    right, bottom = red_transform_global * (width, height)
    bounds = transform_bounds(dataset_crs, 'EPSG:4326', left, bottom, right, top)

    water_mask_reprojected_by_year = {}
    for year in years_sorted:
        if year in water_mask_filtered_by_year:
            water_mask_reprojected, _ = reproject_array(water_mask_filtered_by_year.get(year), dataset_crs, 'EPSG:4326', red_transform_global)
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
#st.title("Detekcja Wody z Sentinel-1 RTC - Analiza Zmian")

st.sidebar.header("Ustawienia Obszaru Zainteresowania (AOI)")
default_lat = 53.87
default_lon = -0.04

from streamlit_folium import st_folium
import folium.plugins

st.sidebar.markdown("### 🌍 Przykładowe lokalizacje")

locations = {
    "Luizjana (USA, delta Missisipi)": {"coords": (29.17, -89.31)},
    "Étretat (Francja, klify)": {"coords": (49.70, 0.19)},
}

# Gotowe lokalizacje
for name, info in locations.items():
    with st.sidebar.expander(name):
        lat, lon = info["coords"]
        m = folium.Map(location=[lat, lon], zoom_start=8)
        folium.Marker([lat, lon], tooltip=name).add_to(m)
        st.components.v1.html(m._repr_html_(), height=200)
        if st.button(f"📍 Ustaw lokalizację: {name}"):
            st.session_state["latitude"] = lat
            st.session_state["longitude"] = lon

# Nowy tryb: samodzielny wybór z mapy
with st.sidebar.expander("🖱️ Wybierz własny obszar z mapy"):
    draw_map = folium.Map(location=[45, 0], zoom_start=3)
    draw = folium.plugins.Draw(
        draw_options={"rectangle": True, "polygon": False, "circle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": False}
    )
    draw.add_to(draw_map)

    result = st_folium(draw_map, height=300, width=400)

    if result and "last_active_drawing" in result and result["last_active_drawing"]:
        geo = result["last_active_drawing"]["geometry"]
        coords = geo["coordinates"][0]
        lats = [c[1] for c in coords]
        lons = [c[0] for c in coords]

        # Środek zaznaczonego prostokąta
        lat_center = np.mean(lats)
        lon_center = np.mean(lons)

        st.session_state["latitude"] = lat_center
        st.session_state["longitude"] = lon_center

        st.success(f"✅ Wybrano obszar: ({lat_center:.4f}, {lon_center:.4f})")

#st.sidebar.header("🧭 Ustawienia Obszaru Zainteresowania (AOI)")

# Domyślnie puste (jeśli nie ustawiono wcześniej)
default_lat = st.session_state.get("latitude", "")
default_lon = st.session_state.get("longitude", "")

# Używamy text_input zamiast number_input, by umożliwić puste pola
lat_input = st.sidebar.text_input("Szerokość geograficzna (Lat)", value=default_lat)
lon_input = st.sidebar.text_input("Długość geograficzna (Lon)", value=default_lon)

# Walidacja i konwersja do float
try:
    latitude = float(lat_input)
    longitude = float(lon_input)
    cords_input = [latitude, longitude]
except ValueError:
    cords_input = None
    st.sidebar.warning("Proszę podać poprawne liczby dla szerokości i długości geograficznej.")


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

                import zipfile

                if image_colored_paths_by_diff:
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            for label, img_path in image_colored_paths_by_diff.items():
                                arcname = f"{label}.png"
                                zipf.write(img_path, arcname=arcname)

                    with open(tmp_zip.name, "rb") as f:
                        with st.sidebar:  # <- UMIESZCZAMY PRZYCISK W SIDEBARZE
                            st.markdown("---")
                            st.subheader("📦 Eksport zmian wodnych")
                            st.download_button(
                                label="📥 Pobierz wszystkie obrazy zmian (ZIP)",
                                data=f,
                                file_name="zmiany_wodne.zip",
                                mime="application/zip"
                            )

                else:
                    st.error(
                        "Nie udało się wygenerować mapy zmian. Upewnij się, że wybrano co najmniej dwa lata i dane są dostępne dla wszystkich wybranych lat.")
