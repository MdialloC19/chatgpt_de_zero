from .dataset_loader import DatasetLoader


class Dataset:
    def __init__(self, dataset_ref="breandan/french-reddit-discussion", download_dir="data", max_length=None):
        """
        Initialize the Dataset class.
        :param dataset_ref: Reference to the dataset.
        :param download_dir: Directory to download the dataset.
        """
        self.raw_data = DatasetLoader(dataset_ref, download_dir).load()
        self.max_length = max_length
        self.data = None

    def preprocess(self):
        """
        Preprocess the dataset by removing unnecessary columns and rows.
        :return: Preprocessed dataset.
        """
        # Keep only the raw text column
        text_only = self.raw_data["utt"]

        # Remove empty rows or rows with only whitespace
        text_only = text_only[text_only.str.strip() != ""]
        # Remove rows that are too short
        text_only = text_only[text_only.str.len() > 5]

        # Convert text to lowercase
        text_only = text_only.str.lower()

        # Replace special spaces with regular space
        text_only = text_only.str.replace(r"[\u00A0\u202F\u2009]", " ", regex=True)

        # Remove emojis and unicode symbols
        text_only = text_only.str.replace(r"[\U00010000-\U0010ffff]", "", regex=True)

        # Remove URLs
        text_only = text_only.str.replace(r"http\S+|www\S+|https\S+", "", regex=True)

        # Keep only allowed characters (French letters, digits, basic punctuation)
        # This removes things like ė, χ, β, ロ, 漂, ⅔, etc.
        text_only = text_only.str.replace(r"[^a-zA-Z0-9\sàâäçéèêëîïôöùûüÿœæ.,!?;:()'\"-]", "", regex=True)

        # Save the cleaned text
        self.data = text_only.to_list()

        # Limit the length of the dataset if max_length is specified
        if self.max_length is not None:
            self.data = self.data[: self.max_length]
