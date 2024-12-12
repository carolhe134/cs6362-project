import optuna
from .train import *

def create_objective(settings_file, n_trials):
    def objective(trial):
        # Load preprocessed data
        data, data_train, data_valid, encoding_alphabet, settings = load_data(settings_file)

        # Name results file
        results_file_name = settings['data']['results_name'].replace('.', f'_{trial.number}of{n_trials}.', 1)
        settings['data']['results_name'] = results_file_name

        # Define search space for hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        latent_dimension = trial.suggest_int("latent_dimension", 10, 100, step=10)
        gru_neurons_num = trial.suggest_int("gru_neurons_num", 50, 300, step=50)
        gru_stack_size = trial.suggest_int("gru_stack_size", 1, 3)
        layer_1d = trial.suggest_int("layer_1d", 50, 200, step=50)
        layer_2d = trial.suggest_int("layer_2d", 50, 200, step=50)
        layer_3d = trial.suggest_int("layer_3d", 50, 200, step=50)
        KLD_alpha = trial.suggest_loguniform("KLD_alpha", 1e-6, 1e-2)
        lr_enc = trial.suggest_loguniform("lr_enc", 1e-5, 1e-3)
        lr_dec = trial.suggest_loguniform("lr_dec", 1e-5, 1e-3)

        # Update settings dynamically based on trial
        settings['data']['batch_size'] = batch_size
        settings['decoder']['latent_dimension'] = latent_dimension
        settings['decoder']['gru_neurons_num'] = gru_neurons_num
        settings['decoder']['gru_stack_size'] = gru_stack_size
        settings['encoder']['layer_1d'] = layer_1d
        settings['encoder']['layer_2d'] = layer_2d
        settings['encoder']['layer_3d'] = layer_3d
        settings['encoder']['latent_dimension'] = latent_dimension
        settings['training']['KLD_alpha'] = KLD_alpha
        settings['training']['lr_enc'] = lr_enc
        settings['training']['lr_dec'] = lr_dec

        # Initialize model
        vae_encoder, vae_decoder, data_params, training_params = initialize_model(settings, 
                                                                                  data, 
                                                                                  encoding_alphabet)
        
        # Train VAE model
        avg_loss = train_model(**training_params,
                               vae_encoder=vae_encoder,
                               vae_decoder=vae_decoder,
                               batch_size=data_params['batch_size'],
                               data_train=data_train,
                               data_valid=data_valid,
                               alphabet=encoding_alphabet,
                               sample_len=data.shape[1],
                               results_file=data_params['results_name'])

        return avg_loss

    return objective

# main function
def tune_hyperparameters(settings_file, n_trials):
    objective = create_objective(settings_file, n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")