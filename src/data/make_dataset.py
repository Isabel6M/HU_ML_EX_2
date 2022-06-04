import tensorflow as tf
from scipy.io import arff


#Tensor = torch.Tensor



def get_eeg(data_dir: Path= "../data/raw") -> Path:
    file = datadir / "eeg"
    if file.exists():
        logger.info(f"Found {file}, load from disk")
        data = arff.loadarff(file)
    else:
        logger.info(f"{file} does not exist, retrieving")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"  # noqa: E501
        datapath = tf.keras.utils.get_file(
        "eeg", origin=url, untar=False, cache_dir=data_dir
         )

        datapath = Path(datapath)
        logger.info(f"Data is downloaded to {datapath}.")
        data = arff.loadarff(datapath)
    return data

# def get_sunspots(datadir: Path) -> pd.DataFrame:
#     """loads sunspot data since 1749, selects year and monthly mean"""
#     file = datadir / "sunspots.csv"
#     if file.exists():
#         logger.info(f"Found {file}, load from disk")
#         data = pd.read_csv(file)
#     else:
#         logger.info(f"{file} does not exist, retrieving")
#         spots = requests.get("http://www.sidc.be/silso/INFO/snmtotcsv.php", stream=True)
#         spots_ = np.genfromtxt(spots.raw, delimiter=";")
#         data = pd.DataFrame(spots_[:, 2:4], columns=["year", "MonthlyMean"])  # type: ignore # noqa: E501
#         data.to_csv(file, index=False)
#     return data

# def get_eeg(data_dir: Path= "../../data/raw") -> Path:
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"  # noqa: E501
#     datapath = tf.keras.utils.get_file(
#         "eeg", origin=url, untar=False, cache_dir=data_dir
#     )

#     datapath = Path(datapath)
#     logger.info(f"Data is downloaded to {datapath}.")
#     data = arff.loadarff(datapath)
#     return data




