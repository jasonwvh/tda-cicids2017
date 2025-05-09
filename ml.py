import pandas as pd
import numpy as np
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_and_prep_data():
    columns_to_keep = [
        ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp',
        ' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
        ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
        'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
        ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
    ]

    df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                     encoding='ISO-8859-1', usecols=columns_to_keep, low_memory=False)

    df.columns = df.columns.str.strip()

    for col in ['Flow Bytes/s', 'Flow Packets/s']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace([np.inf, -np.inf], -1, inplace=True)

    df.replace('Infinity', -1, inplace=True)
    df.fillna(-1, inplace=True)

    df["Source IP"] = df["Source IP"].astype(str).apply(
        lambda x: float(x.replace(".", "")) if x.replace('.', '').isdigit() else -1)
    df["Destination IP"] = df["Destination IP"].astype(str).apply(
        lambda x: float(x.replace(".", "")) if x.replace('.', '').isdigit() else -1)

    df = df.dropna()

    label_map = {"BENIGN": 0, "Web Attack – Brute Force": 1, "Web Attack – XSS": 1, "Web Attack – Sql Injection": 1}
    df["Label"] = df["Label"].map(label_map)

    if df["Label"].isnull().any():
        print("Warning: Some labels were not mapped:", df[df["Label"].isnull()]['Label'].unique())
        df.dropna(subset=['Label'], inplace=True)
        df['Label'] = df['Label'].astype(int)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df

df = load_and_prep_data()

time_series_counts = df.set_index('Timestamp').resample('60s').size()
# plt.figure(figsize=(15, 6))
# time_series_counts.plot()
# plt.title('Aggregated Time Series (Flow Counts per Minute)')
# plt.xlabel('Timestamp')
# plt.ylabel('Number of Flows')
# plt.grid(True)
# plt.show()

window_size = 30
window_stride = 10

te = SingleTakensEmbedding("fixed", 1, 3)
ph = VietorisRipsPersistence(homology_dimensions=[0, 1])
pl = PersistenceLandscape()

landscape_features = []
window_labels = []
window_end_times = []

print(f"Total time series points: {len(time_series_counts)}")
print(f"Processing with window size {window_size}, stride {window_stride}")

for i in range(0, len(time_series_counts) - window_size + 1, window_stride):
    window = time_series_counts.iloc[i: i + window_size]

    if window.empty or window.nunique() <= 1:
        continue

    window_array = window.values.reshape(-1, 1)

    plt.figure(figsize=(15, 6))
    window.plot()
    plt.title('Aggregated Time Series (Flow Counts per Minute)')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Flows')
    plt.grid(True)
    plt.show()

    try:
        embedding = te.fit_transform(window_array)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='b', s=10, alpha=0.5)
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t-τ)')
        ax.set_zlabel('x(t-2τ)')
        ax.set_title("Takens' Embedding (3D)")
        plt.show()

        embedding_reshaped = embedding[None, :, :]
        diagrams = ph.fit_transform(embedding_reshaped)
        fig = plt.figure(figsize=(10, 8))
        for i, dgm in enumerate(diagrams):
            if len(dgm) > 0:
                plt.scatter(dgm[:, 0], dgm[:, 1], label=f'H{i}', alpha=0.6)
                plt.plot([0, max(dgm[:, 1])], [0, max(dgm[:, 1])], 'k--', alpha=0.5)
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.title('Persistence Diagrams')
        plt.legend()
        plt.grid(True)
        plt.show()

        vectorized_landscape = pl.fit_transform(diagrams)
        fig = plt.figure(figsize=(10, 8))
        for i in range(vectorized_landscape.shape[1]):
            plt.plot(vectorized_landscape[0, i], label=f'Landscape {i}')
        plt.xlabel('Filtration value')
        plt.ylabel('Landscape value')
        plt.legend()
        plt.title('Persistence Landscapes')
        plt.show()

        landscape_features.append(vectorized_landscape.flatten())

        # determine label for the sliding window
        window_end_time = window.index[-1]
        window_end_times.append(window_end_time)
        window_start_time = window.index[0]

        # label is 1 if any attack occurred within the window
        original_df_window = df[(df['Timestamp'] >= window_start_time) & (df['Timestamp'] <= window_end_time)]
        if original_df_window.empty:
            window_label = 0
        else:
            window_label = 1 if original_df_window['Label'].sum() > 0 else 0
        window_labels.append(window_label)

        if (i // window_stride) % 50 == 0:
            print(
                f"Processed window {i // window_stride + 1} / {(len(time_series_counts) - window_size) // window_stride + 1}")

    except Exception as e:
        print(f"Error processing window starting at index {i}: {e}")

print(f"Successfully processed {len(landscape_features)} windows.")

X = np.array(landscape_features)
y = np.array(window_labels)

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")
print(f"Label distribution: {np.bincount(y)}")

if X.shape[0] > 0 and X.shape[0] == y.shape[0]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                   class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
else:
    print("Not enough data generated or data mismatch to proceed with ML.")