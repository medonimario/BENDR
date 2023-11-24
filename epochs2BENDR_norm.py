import mne
import torch
import numpy as np
import pandas as pd
from BENDR.dn3_ext import ConvEncoderBENDR

def import_data(folder_path, pair_nr):
    # Importing data for person 1 and person 2
    sample_data_folder = mne.datasets.sample.data_path()
    pair010_p1_coupled_file = (
        folder_path + f"pair{pair_nr}_p1_coupled.fif"
    )
    p1_coupled_epochs = mne.read_epochs(pair010_p1_coupled_file, preload=True)

    pair010_p1_uncoupled_file = (
        folder_path + f"pair{pair_nr}_p1_uncoupled.fif"
    )
    p1_uncoupled_epochs = mne.read_epochs(pair010_p1_uncoupled_file, preload=True)

    pair010_p2_coupled_file = (
        folder_path + f"pair{pair_nr}_p2_coupled.fif"
    )
    p2_coupled_epochs = mne.read_epochs(pair010_p2_coupled_file, preload=True)

    pair010_p2_uncoupled_file = (
        folder_path + f"pair{pair_nr}_p2_uncoupled.fif"
    )
    p2_uncoupled_epochs = mne.read_epochs(pair010_p2_uncoupled_file, preload=True)
    return p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs

def pick_19_ch(p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs):
    picks = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    p1_coupled_epochs_19 = p1_coupled_epochs.pick_channels(picks)
    p1_uncoupled_epochs_19 = p1_uncoupled_epochs.pick_channels(picks)
    p2_coupled_epochs_19 = p2_coupled_epochs.pick_channels(picks)
    p2_uncoupled_epochs_19 = p2_uncoupled_epochs.pick_channels(picks)
    return p1_uncoupled_epochs_19, p1_coupled_epochs_19, p2_uncoupled_epochs_19, p2_coupled_epochs_19

def normalize_data(epochs):
    """
    Normalize the data according to the described process in the paper.
    
    Args:
    - epochs (list): A list containing multiple MNE Epochs objects (each corresponding to a dataset)
    
    Returns:
    - normalized_epochs_list (list): A list containing the normalized MNE Epochs objects
    """
    
    # Combine all the data to find the global maximum and minimum
    all_data = np.concatenate([epoch.get_data() for epoch in epochs])
    global_max = np.max(all_data)
    global_min = np.min(all_data)
    
    normalized_epochs_list = []
    
    for epoch in epochs:
        data = epoch.get_data()
        new_data = np.zeros((data.shape[0], data.shape[1] + 1, data.shape[2]))  # +1 for the new channel
        
        # For each sequence (trial) in the dataset
        for i in range(data.shape[0]):
            max_val = np.max(data[i])
            min_val = np.min(data[i])
            
            # Scale and shift the data
            new_data[i, :-1] = 2 * (data[i] - min_val) / (max_val - min_val) - 1
            
            # Compute the constant value for the new channel
            const_value = (max_val - min_val) / (global_max - global_min)
            
            # Add a new channel with the constant value
            new_channel = np.full((1, data.shape[2]), const_value)
            new_data[i, -1] = new_channel
        
        # Create a new EpochsArray with updated data and channel names
        ch_names = epoch.info['ch_names'] + ['RA']
        info = mne.create_info(ch_names=ch_names, sfreq=epoch.info['sfreq'], ch_types='eeg')
        normalized_epoch = mne.EpochsArray(new_data, info, events=epoch.events, event_id=epoch.event_id)
        normalized_epochs_list.append(normalized_epoch)
    
    return normalized_epochs_list

def transform_epochs_to_bendr_embeddings(epochs, encoder_weights):
    # Get the EEG data from epochs and convert it to tensor format
    eeg_data = epochs.get_data().astype('float32')  # (n_epochs, channels, time)
    eeg_data_tensor = torch.tensor(eeg_data)
    print(eeg_data_tensor.shape)
    
    # Create the BENDR encoder model
    _, channels, _ = eeg_data_tensor.shape
    bendr_encoder = ConvEncoderBENDR(in_features=channels, encoder_h=512)
    
    # Load the pretrained encoder weights
    encoder_state_dict = torch.load(encoder_weights, map_location=torch.device('cpu'))
    bendr_encoder.load_state_dict(encoder_state_dict)
    #print('Pretrained encoder weights loaded')
    
    # Obtain embeddings for each epoch using the encoder
    bendr_embeddings = []
    for i in range(eeg_data_tensor.shape[0]):
        embedding = bendr_encoder(eeg_data_tensor[i][None, ...])  # Add batch dimension
        embedding = torch.mean(embedding, dim=-1) # Average over the 8 time points
        bendr_embeddings.append(embedding)
    
    # Stack the embeddings
    bendr_embeddings = torch.stack(bendr_embeddings)
    
    return bendr_embeddings

def bendr_to_df(bendr_p1_uncoupled, bendr_p1_coupled, bendr_p2_uncoupled, bendr_p2_coupled):
    # Reshape the embeddings to 2D and detach from computation graph before converting to numpy
    bendr_p1_uncoupled = bendr_p1_uncoupled.squeeze(1).detach().numpy()
    bendr_p1_coupled = bendr_p1_coupled.squeeze(1).detach().numpy()
    bendr_p2_uncoupled = bendr_p2_uncoupled.squeeze(1).detach().numpy()
    bendr_p2_coupled = bendr_p2_coupled.squeeze(1).detach().numpy()

    # Concatenate the embeddings for Person 1 and Person 2
    combined_uncoupled = np.hstack((bendr_p1_uncoupled, bendr_p2_uncoupled))
    combined_coupled = np.hstack((bendr_p1_coupled, bendr_p2_coupled))

    # Convert concatenated embeddings into dataframes
    df_uncoupled = pd.DataFrame(combined_uncoupled)
    df_uncoupled['label'] = 1  # Label for uncoupled

    df_coupled = pd.DataFrame(combined_coupled)
    df_coupled['label'] = 2  # Label for coupled

    # Concatenate the two dataframes
    df = pd.concat([df_uncoupled, df_coupled], axis=0).reset_index(drop=True)
    
    return df

folder_path = "C:/1_University/Thesis/Scripts/SC_epochs/"

pairs = ["003", "004", "005", "007", "008", "009", "010", "011","012", "013", "014", "016", "017", "018", "019", "020", "022", "023", "024", "025", "027"]
for pair_nr in pairs:
    print("--------------------")
    print("Pair " + pair_nr)

    # Import data
    p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs = import_data(folder_path, pair_nr)
    print("Pair " + pair_nr + "imported")

    # Pick 19 channels
    p1_uncoupled_epochs_19, p1_coupled_epochs_19, p2_uncoupled_epochs_19, p2_coupled_epochs_19 = pick_19_ch(p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs)
    print("Pair " + pair_nr + "19 channels picked")

    # Normalize data
    p1_uncoupled_epochs_19_norm, p1_coupled_epochs_19_norm, p2_uncoupled_epochs_19_norm, p2_coupled_epochs_19_norm = normalize_data([p1_uncoupled_epochs_19, p1_coupled_epochs_19, p2_uncoupled_epochs_19, p2_coupled_epochs_19])
    print("Pair " + pair_nr + "normalized")

    # Transform to BENDR representation
    encoder_path = 'data/checkpoints/encoder.pt'
    p1_uncoupled_bendr = transform_epochs_to_bendr_embeddings(p1_uncoupled_epochs_19_norm, encoder_path)
    print("Pair " + pair_nr + "p1 uncoupled transformed")
    p1_coupled_bendr = transform_epochs_to_bendr_embeddings(p1_coupled_epochs_19_norm, encoder_path)
    print("Pair " + pair_nr + "p1 coupled transformed")
    p2_uncoupled_bendr = transform_epochs_to_bendr_embeddings(p2_uncoupled_epochs_19_norm, encoder_path)
    print("Pair " + pair_nr + "p2 uncoupled transformed")
    p2_coupled_bendr = transform_epochs_to_bendr_embeddings(p2_coupled_epochs_19_norm, encoder_path)
    print("Pair " + pair_nr + "p2 coupled transformed")
    
    #Convert to dataframe
    df = bendr_to_df(p1_uncoupled_bendr, p1_coupled_bendr, p2_uncoupled_bendr, p2_coupled_bendr)
    print("Pair " + pair_nr + "df shape: " + str(df.shape))
    df.to_csv(f"C:/1_University/Thesis/Scripts/BENDR/BENDR_norm_av/pair{pair_nr}_df.csv", index=False)