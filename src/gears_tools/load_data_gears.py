from gears import PertData

def load_norman(split: str, seed: int, data_path: str, batch_size: int = 32) -> None:
    """Load and prepare the Norman dataset."""
    print("Loading 'norman' data.")
    norman_data = PertData(data_path=data_path)
    norman_data.load(data_name="norman")

    print("Preparing data split.")
    norman_data.prepare_split(split=split, seed=seed)
    norman_data.get_dataloader(batch_size=batch_size)
    return norman_data