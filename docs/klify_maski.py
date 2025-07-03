#!/usr/bin/env python
# coding: utf-8
import os
import pyproj

os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()


import streamlit as st
import pystac_client
import planetary_computer as pc
import rasterio
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import transform_geom, transform_bounds, calculate_default_transform, reproject, Resampling
from scipy.ndimage import median_filter
from shapely.geometry import shape
from rasterio.transform import rowcol, from_bounds
import tempfile
import json
import math
from affine import Affine
from pyproj import Transformer, Geod
import zipfile
from pyproj import Geod
from folium.plugins import Draw
from streamlit_folium import st_folium



st.set_page_config(page_title='Abrazjątka', page_icon='🌊', layout='wide')


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
        <div class="app-title">Detekcja zmian powierzchni lądowej z Sentinel-1</div>
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
        time_range = f"{year}-06-01/{year}-08-01"

        search = stac_client.search(
            collections=["sentinel-1-rtc"],
            intersects=aoi,
            datetime=time_range,
        )
        items = list(search.items())
        # st.write(f"Znaleziono {len(items)} scen dla roku {year}")

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
                # else:
                #     st.warning(f"Pominięto scenę o innym wymiarze lub transform: {item.id}")

            except Exception as e:
                # st.error(f"Błąd w scenie {item.id}: {e}")
                continue
            finally:
                my_bar.progress((i + 1) / len(items), text=progress_text)
        my_bar.empty()

        if vh_stack:
            vh_array = np.stack(vh_stack)
            vh_median = np.nanmedian(vh_array, axis=0)
            vh_medians_by_year[year] = vh_median
            # st.success(f"✅ Obliczono medianę dla roku {year}")
        # else:
            # st.warning(f"⚠️ Brak poprawnych danych dla roku {year}")

    if vh_ds_crs is None or red_transform_global is None:
        st.error("Nie udało się przetworzyć żadnych danych. Sprawdź obszar zainteresowania i lata.")
        return None, None, None, None, None, None, None

    water_mask_filtered_by_year = {}
    for year in years_sorted:
        if year in vh_medians_by_year:
            water_mask = (vh_medians_by_year.get(year) < 0.008)
            water_mask_filtered = median_filter(water_mask.astype(np.uint8), size=3)
            water_mask_filtered_by_year[year] = water_mask_filtered
            # st.write(f"Zrobiono maskę wody dla roku {year}")
        # else:
            # st.warning(f"Brak danych VH dla roku {year}, pomijam generowanie maski wody.")

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
            # st.write(f"Zmieniono uklad wspulzednych dla roku: {year}")

    if water_mask_reprojected_by_year:
        last_year = sorted(water_mask_reprojected_by_year.keys())[-1]
        sample_mask = water_mask_reprojected_by_year[last_year]
        minx, miny, maxx, maxy = bounds
        height, width = sample_mask.shape
    else:
        # st.error("Brak reprojekcji masek wodnych do określenia transformacji.")
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

            if row_stop_clip > row_start_clip and col_stop_clip > col_start_clip:
                water_mask_clipped = water_mask_clipped[row_start_clip:row_stop_clip, col_start_clip:col_stop_clip]
                water_mask_clipped_by_year[year] = water_mask_clipped
                # st.write(f"Przyciąto maskę dla roku: {year}")
            else:
                # st.warning(f"Przycięta maska dla roku {year} jest pusta lub ma niewłaściwe wymiary.")
                water_mask_clipped_by_year[year] = np.zeros((1,1)) 
        # else:
        #     st.warning(f"Brak reprojekcji maski wody dla roku {year}, pomijam przycinanie.")

    water_mask_differences = {}
    if len(years_sorted) > 1:
        for i in range(len(years_sorted) - 1):
            year1 = years_sorted[i]
            year2 = years_sorted[i+1]

            if year1 in water_mask_clipped_by_year and year2 in water_mask_clipped_by_year:
                mask1 = water_mask_clipped_by_year[year1]
                mask2 = water_mask_clipped_by_year[year2]

                if mask1.shape != mask2.shape:
                    # st.warning(f"Maski dla lat {year1} i {year2} mają różne wymiary. Nie można obliczyć różnicy.")
                    continue

                # Calculate difference:
                # 1: New water (was land in year1, is water in year2)
                # -1: Lost water (was water in year1, is land in year2)
                # 0: No change (land->land or water->water)
                difference_mask = np.zeros_like(mask1, dtype=np.int8)
                difference_mask[(mask1 == 0) & (mask2 == 1)] = 1  # New water
                difference_mask[(mask1 == 1) & (mask2 == 0)] = -1 # Lost water
                water_mask_differences[f"{year2}-{year1}"] = difference_mask
            #     st.success(f"✅ Obliczono różnicę między {year1} a {year2}")
            # else:
            #     st.warning(f"Brak przyciętych masek dla lat {year1} lub {year2}. Pomijam obliczanie różnicy.")


    return water_mask_clipped_by_year, water_mask_differences, bounds, aoi_minx, aoi_miny, aoi_maxx, aoi_maxy, red_transform_global


@st.cache_data
def generate_difference_image_overlays(water_mask_differences):
    image_colored_paths_by_diff = {}
    for diff_label, diff_mask in water_mask_differences.items():

        colored_diff_array = np.zeros((*diff_mask.shape, 4), dtype=np.uint8) 

        # Lost water (-1) -> Red
        colored_diff_array[diff_mask == -1] = [255, 0, 0, 200]

        # New water (1) -> Blue
        colored_diff_array[diff_mask == 1] = [0, 0, 255, 200] 

        # No change (0) -> Transparent
        colored_diff_array[diff_mask == 0] = [0, 0, 0, 0] 

        image = Image.fromarray(colored_diff_array, mode="RGBA")

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_file.name, format='PNG')
        image_colored_paths_by_diff[diff_label] = temp_file.name

    return image_colored_paths_by_diff


def compute_land_area_changes(diff_mask, transform, source_crs="EPSG:32616"):
    geod = Geod(ellps="WGS84")
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

    results = {}

    for value, label in [(-1, "lost_water"), (1, "new_water")]:
        rows, cols = np.where(diff_mask == value)
        total_area = 0.0

        for row, col in zip(rows, cols):
            x_left, y_top = transform * (col, row)
            x_right, y_bottom = transform * (col + 1, row + 1)

            lon_left, lat_top = transformer.transform(x_left, y_top)
            lon_right, lat_bottom = transformer.transform(x_right, y_bottom)

            lons = [lon_left, lon_right, lon_right, lon_left]
            lats = [lat_top, lat_top, lat_bottom, lat_bottom]

            if any(np.isnan(lons)) or any(np.isnan(lats)):
                raise ValueError("NaN in transformed coordinates!")

            area, _ = geod.polygon_area_perimeter(lons, lats)
            total_area += abs(area)

        count_pixels = len(rows)

        results[label] = {
            "area_m2": total_area,
            "area_ha": total_area / 10_000,
            "area_km2": total_area / 1_000_000,
            "num_pixels": count_pixels
        }

    return results



############################## STREAMLIT INTERFACE ##############################
# --- KONFIGURACJA I DANE ---

st.sidebar.header("Ustawienia Obszaru Zainteresowania (AOI)")

locations = {
    "Luizjana (USA, delta Missisipi)": {"coords": (29.17, -89.31)},
    "Étretat (Francja, klify)": {"coords": (49.70, 0.19)},
    "Jezioro Aralskie (Kazachstan / Uzbekistan)": {"coords": (59.436, 45.461)},
}

for name, info in locations.items():
    lat, lon = info["coords"]
    if st.sidebar.button(f"📍 {name}"):
        st.session_state["latitude"] = lat
        st.session_state["longitude"] = lon
        st.session_state['map_ready'] = False  #reset mapy zmian po zmianie lokalizacji

draw_aoi_enabled = st.sidebar.checkbox("Rysuj AOI na mapie")

default_lat = st.session_state.get("latitude", "")
default_lon = st.session_state.get("longitude", "")

lat_input = st.sidebar.text_input("Szerokość geograficzna (Lat)", value=default_lat)
lon_input = st.sidebar.text_input("Długość geograficzna (Lon)", value=default_lon)

try:
    latitude = float(lat_input)
    longitude = float(lon_input)
    cords_input = [latitude, longitude]
except ValueError:
    cords_input = None
    if lat_input or lon_input:
        st.sidebar.warning("Proszę podać poprawne liczby dla szerokości i długości geograficznej.")

if st.sidebar.button("Ustaw współrzędne AOI"):
    if cords_input is not None:
        st.session_state["latitude"] = latitude
        st.session_state["longitude"] = longitude
        st.session_state['map_ready'] = False  #reset mapy zmian po zmianie lokalizacji
    else:
        st.sidebar.error("Niepoprawne współrzędne. Proszę poprawić wpis.")


# --- WYBÓR LAT DO ANALIZY ---
st.sidebar.header("Wybór Lat")
selected_years = st.sidebar.multiselect(
    "Wybierz lata do analizy zmian (min. 2 lata)",
    options=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    default=[2016, 2021, 2023]
)

if 2022 in selected_years:
    st.sidebar.warning("Rok 2022 może zawierać anomalie w danych.")


# --- GENEROWANIE MAPY ZMIAN ---
if st.sidebar.button("Generuj mapę zmian"):
    if len(selected_years) < 2:
        st.error("Proszę wybrać co najmniej dwa lata, aby obliczyć zmiany.")
    elif cords_input is None:
        st.error("Proszę podać poprawne współrzędne AOI.")
    else:
        with st.spinner("Przetwarzanie danych satelitarnych i obliczanie zmian... To może potrwać kilka minut."):
            results = process_data_and_differences(cords_input, selected_years)

            if results:
                water_mask_clipped_by_year, water_mask_differences, bounds, aoi_minx, aoi_miny, aoi_maxx, aoi_maxy, red_transform_global = results

                image_colored_paths_by_diff = generate_difference_image_overlays(water_mask_differences)
                
                st.session_state['map_ready'] = True
                st.session_state['image_paths'] = image_colored_paths_by_diff
                st.session_state['aoi_bounds'] = (aoi_minx, aoi_miny, aoi_maxx, aoi_maxy)
                st.session_state['red_transform'] = red_transform_global
            else:
                st.error("Nie udało się wygenerować mapy zmian.")

#jeśli mapa została wygenerowana wczesniej to jest przechowywana w sesji i jest automatucznie wyświetlana
if st.session_state.get('map_ready'):
    aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = st.session_state['aoi_bounds']
    image_colored_paths_by_diff = st.session_state['image_paths']

    # land_area_changes = compute_land_area_changes(water_mask_differences, red_transform_global)
    # print(f"Zmiany powierzchni lądowej: {land_area_changes}")

    m = folium.Map(location=[(aoi_miny + aoi_maxy) / 2, (aoi_minx + aoi_maxx) / 2], zoom_start=12)

    for diff_label, img_path in image_colored_paths_by_diff.items():
        overlay = ImageOverlay(
            image=img_path,
            bounds=[[aoi_miny, aoi_minx], [aoi_maxy, aoi_maxx]],
            opacity=0.8,
            name=f"Zmiany: {diff_label}"
        )
        overlay.add_to(m)

    folium.Polygon(
        locations=[(aoi_miny, aoi_minx), (aoi_miny, aoi_maxx), (aoi_maxy, aoi_maxx), (aoi_maxy, aoi_minx), (aoi_miny, aoi_minx)],
        color='blue',
        weight=2,
        fill=False,
        popup='Obszar Zainteresowania (AOI)'
    ).add_to(m)

    folium.LayerControl().add_to(m)

    legend_html = """
        <div style="
            position: fixed; 
            bottom: 5px; left: 5px; width: 150px; height: 85px; 
            background-color: white;
            border:2px solid grey; 
            z-index:9999;
            font-size:14px;
            padding: 10px;
        ">
        <b>Legenda zmian</b><br>
        <i style="background: rgba(255,0,0,0.78); width: 15px; height: 15px; display: inline-block;"></i>
        Odsłonięty ląd<br>
        <i style="background: rgba(0,0,255,0.78); width: 15px; height: 15px; display: inline-block;"></i>
        Utracony ląd
        </div>
        """
    m.get_root().html.add_child(folium.Element(legend_html))

    st.components.v1.html(m._repr_html_(), height=700)

    if st.button("Wyczyść mapę zmian"):
        st.session_state['map_ready'] = False
        st.session_state.pop('image_paths', None)
        st.session_state.pop('aoi_bounds', None)


    #opcja eksportu obrazu w formie pliku zip
    if image_colored_paths_by_diff:
        import tempfile
        import zipfile
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                for label, img_path in image_colored_paths_by_diff.items():
                    arcname = f"{label}.png"
                    zipf.write(img_path, arcname=arcname)

        with open(tmp_zip.name, "rb") as f:
            with st.sidebar:
                st.markdown("---")
                st.subheader("Eksport zmian wodnych")
                st.download_button(
                    label="Pobierz wszystkie obrazy zmian (ZIP)",
                    data=f,
                    file_name="zmiany_wodne.zip",
                    mime="application/zip"
                )
if not st.session_state.get('map_ready', False):
    default_location = [45, 0]
    default_zoom = 2

    if "latitude" in st.session_state and "longitude" in st.session_state:
        center_lat = st.session_state["latitude"]
        center_lon = st.session_state["longitude"]
        zoom_level = 10
    else:
        center_lat, center_lon = default_location
        zoom_level = default_zoom

    m_base = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)

    for name, info in locations.items():
        lat, lon = info["coords"]
        folium.Marker(
            location=[lat, lon],
            tooltip=name,
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m_base)

    if cords_input:
        folium.CircleMarker(
            location=cords_input,
            radius=6,
            color="blue",
            fill=True,
            fill_color="blue",
            popup="Wybrane współrzędne"
        ).add_to(m_base)

    if draw_aoi_enabled:
        draw = Draw(
            draw_options={
                "rectangle": True,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
            edit_options={"edit": False},
        )
        draw.add_to(m_base)

    result = st_folium(m_base, height=700, width=1400)

    #obsługa rysowania AOI przez użytkownika
    if draw_aoi_enabled and result and "last_active_drawing" in result and result["last_active_drawing"]:
        geo = result["last_active_drawing"]["geometry"]
        coords = geo["coordinates"][0]
        lats = [c[1] for c in coords]
        lons = [c[0] for c in coords]

        lat_center = np.mean(lats)
        lon_center = np.mean(lons)

        st.session_state["latitude"] = lat_center
        st.session_state["longitude"] = lon_center

        st.success(f"Wybrano obszar AOI: ({lat_center:.4f}, {lon_center:.4f})")
