import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class StudentScorePredictor:
    def __init__(self, data_path):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = pd.read_csv(data_path, sep=';')
        self.features = [
            'age', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
            'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
            'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 
            'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
        ]
        self.target = 'G3'
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        # Encoding categorical features
        for column in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        X = self.data[self.features]
        y = self.data[self.target]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, features):
        features_df = pd.DataFrame([features], columns=self.features)
        
        # Transform features
        for column in self.label_encoders:
            if column in features_df.columns:
                features_df[column] = self.label_encoders[column].transform(features_df[column])
        
        scaled_features = self.scaler.transform(features_df)
        return self.model.predict(scaled_features)[0]

    def run(self, input_features):
        self.preprocess()
        self.train()
        return self.predict(input_features)

