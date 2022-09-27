import os
import pandas as pd
from glob import glob


class readDepthAvg:
    def __init__(
        self, 
        sem_folder: str, 
        avg_file: str
    ) -> None:
        self.sem_folder = sem_folder
        self.avg_file = avg_file

    @staticmethod
    def get_key(
        path: str
    ) -> str:
        path = os.path.normpath(path)
        names = path.split(os.sep)
        return f"{names[-3].lower()}_{names[-2]}"

    def __call__(
        self,
    ) -> pd.DataFrame:
        sem_files = glob(os.path.join(self.sem_folder, "*/*/*.png"))
        df_avg = pd.read_csv(self.avg_file)

        table = {}
        for index, key, depth_avg in zip(df_avg.index.values, df_avg["0"].values, df_avg["1"].values):
            table[key] = dict(id=index, depth_avg=depth_avg)

        df_out = pd.DataFrame()
        df_out["file"] = sem_files
        df_out["depth_avg"] = df_out["file"].apply(lambda x: table[self.get_key(x)]["depth_avg"])
        return df_out