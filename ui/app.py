from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import folium
import requests
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

try:
    from streamlit_folium import st_folium
except ModuleNotFoundError:
    st.set_page_config(page_title="NYC Taxi Trip UI", page_icon=":taxi:", layout="wide")
    st.error(
        "Le module `streamlit-folium` est absent dans le virtualenv courant.\n\n"
        "Installe les dependances UI puis relance Streamlit:\n"
        "1. `./.venv/Scripts/python.exe -m ensurepip --upgrade`\n"
        "2. `./.venv/Scripts/python.exe -m pip install -r requirements.txt`\n"
        "3. `./.venv/Scripts/python.exe -m streamlit run ./ui/app.py`"
    )
    st.stop()

from api.config import SETTINGS

DEFAULT_CENTER = (40.758, -73.9855)
DEFAULT_ZOOM = 11
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
API_URL_DEFAULT = "http://127.0.0.1:8000"


def init_state() -> None:
    # Session state keeps the selected map points across Streamlit reruns.
    defaults = {
        "pickup_coords": None,
        "dropoff_coords": None,
        "active_map_target": "pickup",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def format_duration_hms(total_seconds: int) -> str:
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def combine_pickup_datetime(pickup_date: date, pickup_time: time) -> datetime:
    return datetime.combine(pickup_date, pickup_time)


@st.cache_data(show_spinner=False, ttl=300)
def search_nyc_addresses(query: str) -> list[dict]:
    # Address mode fetches geocoding suggestions, then stores the chosen result
    # as latitude/longitude so both input modes end up producing the same payload.
    if len(query.strip()) < 3:
        return []

    params = {
        "q": f"{query}, New York, NY, USA",
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
        "bounded": 1,
        "viewbox": "-74.25909,40.91758,-73.70018,40.4774",
    }
    headers = {"User-Agent": "nyc-taxi-streamlit-demo/1.0"}
    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def render_address_selector(label: str, state_key: str) -> None:
    # The text input is only a search helper: the coordinates actually sent to
    # the API are the selected suggestion stored in session_state[state_key].
    query = st.text_input(
        f"{label} - adresse a New York",
        key=f"{state_key}_query",
        placeholder="Ex: Times Square, Wall Street, JFK Airport",
    )
    if not query or len(query.strip()) < 3:
        st.caption("Saisis au moins 3 caracteres pour obtenir des suggestions.")
        return

    try:
        suggestions = search_nyc_addresses(query)
    except requests.RequestException:
        st.error("Recherche d'adresse indisponible pour le moment.")
        return

    if not suggestions:
        st.warning("Aucune adresse NYC trouvee pour cette recherche.")
        return

    options = {
        item["display_name"]: (float(item["lat"]), float(item["lon"]))
        for item in suggestions
    }
    selected_label = st.selectbox(
        f"{label} - suggestions",
        options=list(options.keys()),
        key=f"{state_key}_select",
    )
    if selected_label:
        st.session_state[state_key] = options[selected_label]
        lat, lon = options[selected_label]
        st.success(f"{label} selectionne: lat={lat:.6f}, lon={lon:.6f}")


def build_map() -> folium.Map:
    fmap = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, control_scale=True)

    folium.Rectangle(
        bounds=[
            [SETTINGS.latitude_range.min, SETTINGS.longitude_range.min],
            [SETTINGS.latitude_range.max, SETTINGS.longitude_range.max],
        ],
        color="#2563eb",
        fill=True,
        fill_opacity=0.05,
        weight=2,
        tooltip="Zone de validation NYC",
    ).add_to(fmap)

    for label, state_key, color in (
        ("Pickup", "pickup_coords", "green"),
        ("Dropoff", "dropoff_coords", "red"),
    ):
        coords = st.session_state.get(state_key)
        if coords:
            lat, lon = coords
            folium.Marker(
                location=[lat, lon],
                tooltip=f"{label}: {lat:.5f}, {lon:.5f}",
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(fmap)

    return fmap


def render_map_selector() -> None:
    # Map mode writes the last clicked point into session_state so the payload
    # builder can reuse the exact same pickup/dropoff structure as address mode.
    st.info(
        "Aide: choisis d'abord `Pickup` ou `Dropoff`, puis double-clique sur le point "
        "souhaite sur la carte pour le positionner."
    )

    st.radio(
        "Le prochain clic sur la carte definit",
        options=["pickup", "dropoff"],
        horizontal=True,
        key="active_map_target",
        format_func=lambda value: "Pickup" if value == "pickup" else "Dropoff",
    )

    map_data = st_folium(build_map(), height=520, width=None, key="trip_map")

    last_clicked = map_data.get("last_clicked") if map_data else None
    if last_clicked:
        target_key = f"{st.session_state.active_map_target}_coords"
        st.session_state[target_key] = (last_clicked["lat"], last_clicked["lng"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Effacer pickup", use_container_width=True):
            st.session_state["pickup_coords"] = None
    with col2:
        if st.button("Effacer dropoff", use_container_width=True):
            st.session_state["dropoff_coords"] = None


def show_selected_points() -> None:
    for label, state_key in (("Pickup", "pickup_coords"), ("Dropoff", "dropoff_coords")):
        coords = st.session_state.get(state_key)
        if coords:
            lat, lon = coords
            st.write(f"{label}: lat `{lat:.6f}` | lon `{lon:.6f}`")
        else:
            st.write(f"{label}: non defini")


def build_payload(
    vendor_id: int,
    pickup_date: date,
    pickup_time: time,
    passenger_count: int,
    store_and_fwd_flag: str,
) -> dict | None:
    # This is the single place where UI inputs are normalized into the JSON body
    # expected by POST /predict.
    pickup = st.session_state.get("pickup_coords")
    dropoff = st.session_state.get("dropoff_coords")
    if not pickup or not dropoff:
        return None

    pickup_datetime = combine_pickup_datetime(pickup_date, pickup_time)
    return {
        "vendor_id": vendor_id,
        "pickup_datetime": pickup_datetime.isoformat(),
        "passenger_count": passenger_count,
        "pickup_longitude": pickup[1],
        "pickup_latitude": pickup[0],
        "dropoff_longitude": dropoff[1],
        "dropoff_latitude": dropoff[0],
        "store_and_fwd_flag": store_and_fwd_flag,
    }


def extract_api_error_message(response: requests.Response) -> str:
    # Convert structured FastAPI errors into short messages that make sense in
    # the Streamlit interface instead of exposing raw JSON to the user.
    try:
        error_body = response.json()
    except ValueError:
        return "L'API a renvoye une erreur inattendue."

    if response.status_code == 422:
        lines = []
        message = error_body.get("message")
        if message:
            lines.append("La requete est invalide.")
            lines.append(message)

        for item in error_body.get("errors", []):
            field = item.get("field", "champ")
            field = field.replace("_", " ")
            detail = item.get("message", "valeur invalide")
            lines.append(f"- {field}: {detail}")

        longitude = error_body.get("bounding_box", {}).get("longitude")
        latitude = error_body.get("bounding_box", {}).get("latitude")
        if longitude and latitude:
            lines.append(
                "Zone autorisee: "
                f"longitude [{longitude[0]}, {longitude[1]}], "
                f"latitude [{latitude[0]}, {latitude[1]}]."
            )

        min_distance = error_body.get("min_trip_distance_meters")
        if min_distance is not None:
            lines.append(f"Distance minimale entre pickup et dropoff: {min_distance} m.")

        return "\n".join(lines)

    if response.status_code == 404:
        requested = error_body.get("requested_model_version")
        available = error_body.get("available_model_versions", [])
        if requested:
            message = [f"La version de modele `{requested}` est introuvable."]
            if available:
                message.append(f"Versions disponibles: {', '.join(available)}")
            return "\n".join(message)

    if "detail" in error_body:
        return str(error_body["detail"])

    return json.dumps(error_body, ensure_ascii=True, indent=2)


def call_prediction_api(api_url: str, payload: dict, model_version: str) -> dict:
    # The UI stays independent from inference internals: it only sends an HTTP
    # request to /predict and consumes the API response.
    params = {}
    if model_version.strip():
        params["model_version"] = model_version.strip()

    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json=payload,
        params=params,
        timeout=15,
    )
    if response.ok:
        return response.json()

    raise requests.HTTPError(extract_api_error_message(response), response=response)


def render_prediction_result(result: dict) -> None:
    # The API returns seconds; the UI handles presentation formatting locally.
    st.success("Prediction calculee.")
    st.metric("Duree du trajet prevue", format_duration_hms(result["trip_duration"]))
    st.caption(
        f"prediction_id={result['prediction_id']} | "
        f"model_version={result['model_version']} | "
        f"{result['trip_duration']} sec"
    )


def main() -> None:
    st.set_page_config(page_title="NYC Taxi Trip UI", page_icon=":taxi:", layout="wide")
    init_state()

    st.title("Interface Streamlit - Prediction de trajet taxi NYC")
    st.write(
        "Cette UI est separee du pipeline d'entrainement. Elle collecte les attributs "
        "d'une course, envoie une requete HTTP a `/predict`, puis affiche la duree predite."
    )

    with st.sidebar:
        st.subheader("Connexion API")
        api_url = st.text_input("URL API", value=API_URL_DEFAULT)
        model_version = st.text_input("Version du modele", value="")

        st.subheader("Attributs de prediction")
        vendor_id = st.number_input("Vendor ID", min_value=1, value=1, step=1)
        pickup_date = st.date_input("Date de pickup", value=datetime.now().date())
        pickup_time = st.time_input("Heure de pickup", value=datetime.now().time().replace(microsecond=0))
        passenger_count = st.slider("Nombre de passagers", min_value=0, max_value=6, value=1)
        store_and_fwd_flag = st.selectbox("Store and fwd flag", options=["N", "Y"], index=0)

    input_mode = st.radio(
        "Mode de saisie des coordonnees",
        options=["Carte", "Adresse"],
        horizontal=True,
    )

    if input_mode == "Carte":
        render_map_selector()
    else:
        col1, col2 = st.columns(2)
        with col1:
            render_address_selector("Pickup", "pickup_coords")
        with col2:
            render_address_selector("Dropoff", "dropoff_coords")

    st.subheader("Points selectionnes")
    show_selected_points()

    payload = build_payload(
        vendor_id=vendor_id,
        pickup_date=pickup_date,
        pickup_time=pickup_time,
        passenger_count=passenger_count,
        store_and_fwd_flag=store_and_fwd_flag,
    )

    st.subheader("Payload envoye a l'API")
    # Expose the request body to make the UI-to-API flow explicit during demos.
    st.json(payload if payload else {"detail": "Pickup et dropoff requis."})

    if st.button("Predire la duree", type="primary", use_container_width=True):
        if not payload:
            st.error("Il faut definir pickup et dropoff avant de lancer la prediction.")
            return

        try:
            result = call_prediction_api(api_url=api_url, payload=payload, model_version=model_version)
        except requests.HTTPError as exc:
            st.error(str(exc))
            return
        except requests.RequestException:
            st.error("API injoignable. Verifie que FastAPI tourne bien sur l'URL indiquee.")
            return

        render_prediction_result(result)


if __name__ == "__main__":
    main()
