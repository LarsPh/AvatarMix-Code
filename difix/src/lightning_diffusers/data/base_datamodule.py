import lightning as L
from typing import Any, Dict


class BaseDataModule(L.LightningDataModule):


    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def get_dataset_info(self) -> Dict[str, Any]:

        return {"base": "datamodule"}
