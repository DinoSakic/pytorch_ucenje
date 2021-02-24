import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataIris(Dataset):
    def __init__(self):
        self.data = pd.read_csv('.\Iris.csv')  # citanje csv fajla sa pandas paketom

    def __getitem__(self, idx):  # dunder metod, koristeci komandu iloc vraca red csv fajla odredjenog indeksa
        return self.data.iloc[idx]

    def __len__(self):  # dunder metod, omogucava koristenje len() funkcije na objekat klase DataIris
        return len(self.data)


# naslijedjivanje, klasa DataIris cita csv podatke i omogucava metode len i indeks, a klasa TensorDataset
# sluzi da podatke dobijemo u odredjenom obliku.
class TensorDataset(DataIris):
    # naslijedjivanje funkcije iz klase DataIris: komanda super()
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return {
            # vraca podatke u 2 liste, u tensor i label su stavljeni header elementi koji nam trebaju; vraca Dict
            'tensor': torch.Tensor(
                [sample.SepalLengthCm,
                sample.SepalWidthCm,
                sample.PetalLengthCm,
                sample.PetalWidthCm]
            ),
            'label': sample.Species
        }

tensor = TensorDataset()
loader = DataLoader(tensor, batch_size=16, shuffle=True)
for batch in loader:
    print(batch)