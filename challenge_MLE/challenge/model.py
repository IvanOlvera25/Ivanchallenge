import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
import xgboost as xgb
from sklearn.model_selection import train_test_split


class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        
    def _get_period_day(self, date: str) -> str:
        """
        Extract period of day from datetime string.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("04:59", '%H:%M').time()
        
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif (evening_min <= date_time <= evening_max) or (night_min <= date_time <= night_max):
            return 'noche'
        else:
            return 'mañana'
    
    def _is_high_season(self, fecha: str) -> int:
        """
        Check if date falls in high season.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
    
    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calculate difference in minutes between scheduled and actual time.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Create engineered features
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        data['min_diff'] = data.apply(self._get_min_diff, axis=1)
        
        # Create target variable if needed
        threshold_in_minutes = 15
        if target_column:
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        # One-hot encoding for categorical features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')], 
            axis=1
        )
        
        # Ensure all required features are present
        for feature in self._top_10_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Select only top 10 features
        features = features[self._top_10_features]
        
        if target_column:
            target = data[[target_column]]
            return features, target
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Calculate scale for class balancing
        n_y0 = len(target[target.delay == 0])
        n_y1 = len(target[target.delay == 1])
        scale = n_y0 / n_y1
        
        # Initialize and train XGBoost model with class balancing
        self._model = xgb.XGBClassifier(
            random_state=1, 
            learning_rate=0.01,
            scale_pos_weight=scale,
            n_estimators=100,
            max_depth=5
        )
        
        self._model.fit(features, target.values.ravel())

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
            
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        predictions = self._model.predict(features)
        return predictions.tolist()