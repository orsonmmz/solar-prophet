import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class PowerPredictionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.targets[idx]
        }

class DataPreparer:
    def __init__(self, fotowoltaika_path, pogoda_path):
        self.fotowoltaika_path = fotowoltaika_path
        self.pogoda_path = pogoda_path

    def prepare_data(self):
        fotowoltaika_df, pogoda_df = self.load_data()
        features, targets = self.preprocess_data(fotowoltaika_df, pogoda_df)
        X_train, X_test, y_train, y_test = self.split_data(features, targets)
        return X_train, X_test, y_train, y_test
        
    def load_data(self):
        fotowoltaika_df = pd.read_csv(self.fotowoltaika_path)
        pogoda_df = pd.read_csv(self.pogoda_path)
        return fotowoltaika_df, pogoda_df
    
    def preprocess_data(self, fotowoltaika_df, pogoda_df):
        # Handle missing values
        fotowoltaika_df = fotowoltaika_df.dropna()
        pogoda_df = pogoda_df.dropna()

        # Ensure that the data is sorted by time
        fotowoltaika_df.sort_values('timestamp', inplace=True)
        pogoda_df.sort_values('time', inplace=True)

        # Merge the datasets by finding the closest match for the timestamps
        df_merged = pd.merge_asof(pogoda_df, fotowoltaika_df, left_on='time', right_on='timestamp', direction='nearest')

        
        # Feature selection
        features = df_merged[['temperature_2m', 'weathercode', 'cloudcover', 'shortwave_radiation', 'direct_radiation', 'diffuse_radiation']].values
        targets = df_merged['power'].values
        
        # Normalize the feature data
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features, targets
    
    def split_data(self, features, targets, test_size=0.2):
        return train_test_split(features, targets, test_size=test_size, random_state=42)

    def create_datasets(self, X_train, y_train, X_test, y_test):
        train_dataset = PowerPredictionDataset(X_train, y_train)
        test_dataset = PowerPredictionDataset(X_test, y_test)
        return train_dataset, test_dataset

def main():
    preparer = DataPreparer('data/fotowoltaika.csv', 'data/pogoda.csv')
    
    fotowoltaika_df, pogoda_df = preparer.load_data()
    features, targets = preparer.preprocess_data(fotowoltaika_df, pogoda_df)

    # save faetures and targets to combined csv file
    df = pd.DataFrame(features)
    df['target'] = targets
    df.to_csv('data/combined.csv', index=False, header=False)
    
    # Print out the shape of the features and targets
    print(f'Total number of examples: {features.shape[0]}')
    print(f'Number of features: {features.shape[1]}')
    
    # Calculate and print out the mean and standard deviation before scaling
    print(f'Features mean before scaling: {features.mean(axis=0)}')
    print(f'Features std before scaling: {features.std(axis=0)}')
    print(f'Targets mean: {targets.mean()}')
    print(f'Targets std: {targets.std()}')
    
    X_train, X_test, y_train, y_test = preparer.split_data(features, targets)
    
    # Print the number of examples in training and test sets
    print(f'Number of training examples: {X_train.shape[0]}')
    print(f'Number of test examples: {X_test.shape[0]}')
    
    # Create datasets
    train_dataset, test_dataset = preparer.create_datasets(X_train, y_train, X_test, y_test)
    
    # Example usage of PyTorch DataLoader to iterate through the dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Now you can use train_loader and test_loader in your model training and validation
    
    # Additional statistics after train-test split
    print(f'Training set: Mean of features = {X_train.mean(axis=0)}, Standard deviation = {X_train.std(axis=0)}')
    print(f'Training set: Mean of targets = {y_train.mean()}, Standard deviation = {y_train.std()}')
    print(f'Test set: Mean of features = {X_test.mean(axis=0)}, Standard deviation = {X_test.std(axis=0)}')
    print(f'Test set: Mean of targets = {y_test.mean()}, Standard deviation = {y_test.std()}')

if __name__ == '__main__':
    main()
