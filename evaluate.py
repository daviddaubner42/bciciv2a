from sklearn.preprocessing import LabelEncoder

def inses_eval(model, train_epochs, test_epochs):
    # Train
    target_ids = ['769', '770', '771', '772']
    target_epochs = epochs[target_ids]

    X = target_epochs.get_data()
    y_raw = target_epochs.events[:, -1]
    y = LabelEncoder().fit_transform(y_raw)

    model.fit(X, y)

    # Test
    trial_ids = ['768']
    trial_epochs = test_epochs[trial_ids]
    X = sexsexsex
    more sex  
    kippen alkohol 
    brother
    r