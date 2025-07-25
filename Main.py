import os
import uuid
import base64
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from scipy.signal import butter, sosfilt

# Frequency Bands
bands = [
    (60, 150),     # Sub-bass
    (150, 400),    # Bass
    (400, 1000),   # Low-mid
    (1000, 3000),  # Mid
    (3000, 5000),  # Upper-mid
    (5000, 8000),  # Presence
    (8000, 12000), # Brilliance
]

# Butterworth bandpass filter
def butter_bandpass_sos(lowcut, highcut, fs, order=10):
    return butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', output='sos')

def apply_gain_band_sos(y, fs, lowcut, highcut, gain_db):
    sos = butter_bandpass_sos(lowcut, highcut, fs)
    filtered = sosfilt(sos, y)
    gain = 10 ** (gain_db / 20)
    return filtered * gain

def equalize_audio_from_array(y, sr, gains):
    processed = np.zeros_like(y)
    for (low, high), gain_db in zip(bands, gains):
        processed += apply_gain_band_sos(y, sr, low, high, gain_db)
    return processed / np.max(np.abs(processed))  # Normalize

def calculate_band_energies(y, sr):
    energies = []
    for low, high in bands:
        sos = butter_bandpass_sos(low, high, sr)
        band = sosfilt(sos, y)
        energy = np.sqrt(np.mean(band**2))
        energies.append(energy)
    return energies

# Dash App
app = Dash(__name__)
app.title = "🎚️ Audio Equalizer with Frequency Plot"

app.layout = html.Div([
    html.H2("🎵 Interactive Audio Equalizer with Visualization"),

    dcc.Upload(id='upload-audio', children=html.Button("Upload MP3 File"), multiple=False),

    html.Div(id='original-band-energy'),

    html.Div(id='sliders', children=[
        html.Div([
            html.Label(f"{low}-{high} Hz"),
            dcc.Slider(id=f"slider-{i}", min=-20, max=20, step=1, value=0,
                       marks={-20: '-20dB', 0: '0dB', 20: '20dB'})
        ], style={'margin': '10px'}) for i, (low, high) in enumerate(bands)
    ]),

    html.Button("Apply EQ", id="apply-button", style={'marginTop': '20px'}),

    html.Div([
        html.H4("🎧 Original vs Equalized Waveform"),
        dcc.Graph(id="waveform-plot")
    ]),

    html.Audio(id='player', controls=True, src='', style={'marginTop': '20px'})
])

@app.callback(
    Output("original-band-energy", "children"),
    Input("upload-audio", "contents")
)
def show_original_band_energy(upload_content):
    if not upload_content:
        return "Upload an MP3 file to view frequency energy."

    content_type, content_string = upload_content.split(',')
    decoded = base64.b64decode(content_string)
    uid = str(uuid.uuid4())
    input_path = f"input_{uid}.mp3"

    with open(input_path, "wb") as f:
        f.write(decoded)

    y, sr = librosa.load(input_path, sr=None, mono=True)
    os.remove(input_path)

    energies = calculate_band_energies(y, sr)
    energy_display = [html.P(f"{bands[i][0]}-{bands[i][1]} Hz: {energies[i]:.4f}") for i in range(len(bands))]
    return html.Div([
        html.H4("📊 Original Band Amplitudes (RMS):"),
        *energy_display
    ])

@app.callback(
    Output("player", "src"),
    Output("waveform-plot", "figure"),
    Input("apply-button", "n_clicks"),
    [State(f"slider-{i}", "value") for i in range(len(bands))],
    State("upload-audio", "contents")
)
def process_audio(n_clicks, *args):
    if not n_clicks or not args[-1]:
        return "", go.Figure()

    gains = args[:-1]
    upload_content = args[-1]

    content_type, content_string = upload_content.split(',')
    decoded = base64.b64decode(content_string)
    uid = str(uuid.uuid4())
    input_path = f"input_{uid}.mp3"
    output_path = f"output_{uid}.wav"

    with open(input_path, "wb") as f:
        f.write(decoded)

    y, sr = librosa.load(input_path, sr=None, mono=True)
    y_eq = equalize_audio_from_array(y, sr, gains)
    sf.write(output_path, y_eq, sr)

    # Create waveform plot
    duration = librosa.get_duration(y=y, sr=sr)
    time = np.linspace(0, duration, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=time, y=y_eq, mode='lines', name='Equalized'))
    fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")

    # Encode output audio
    with open(output_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    os.remove(input_path)
    os.remove(output_path)

    return f"data:audio/wav;base64,{audio_b64}", fig

if __name__ == '__main__':
    app.run_server(debug=True)
