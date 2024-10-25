from pathlib import Path

from torch.utils.data import Dataset

from utils import read_json_file


class KnownsDataset(Dataset):
    def __init__(self, data_path: str, jsonl=True, *args, **kwargs):
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"{data_path} does not exist.")

        self.data = read_json_file(data_path, jsonl=jsonl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item