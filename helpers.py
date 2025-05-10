import pandas as pd

def load_and_prep_data():
    columns_to_keep = [
        ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp',
        ' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
        ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
        'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
        ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
    ]

    df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding='ISO-8859-1', usecols=columns_to_keep, low_memory=False)
    df.replace('Infinity', -1, inplace=True)
    df[" Source IP"] = df[" Source IP"].apply(lambda x: float(str(x).replace(".", "")))
    df[" Destination IP"] = df[" Destination IP"].apply(lambda x: float(str(x).replace(".", "")))
    df = df.dropna()

    df[" Label"] = df[" Label"].map({"BENIGN": 0, "Web Attack  Brute Force": 1, "Web Attack  XSS": 1, "Web Attack  Sql Injection": 1})
    return df
