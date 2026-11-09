import abc
import pathlib
import pandas as pd
import onnxruntime as ort

class ADM_Base(abc.ABC):
    def load_model(
        self,
        onnx_file: pathlib.Path,
    ) -> None:
        """
        This function is used to load the model from given ONNX file.
        
        Args:
            onnx_file: Path to the ONNX file containing the model.
        """
        self.session = ort.InferenceSession(
            onnx_file,
            providers=["CPUExecutionProvider"],
        )

    @abc.abstractmethod
    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        """
        This function is used to detect activities from given DataFrame.

        The input timeseries DataFrame (index is a type of pd.DatetimeIndex) with following columns:
        - 'BVP': Blood Volume Pulse from photoplethysmograph
        - 'EDA': Electrodermal activity in microsiemens'
        - 'HR': Heart rate in BPM
        - 'IBI_dt': Inter Beat Interval delta t since last beat.
        - 'IBI_duration': Inter Beat Interval duration.
        - 'Temperature': Skin temperature in Celcius.
        
        Args:
            input: The input DataFrame.
        
        Return:
            A timeseries DataFrame with 'Activity' column. Valid activities are 'Stress', 'Activity', and 'Rest'
        """
        pass
