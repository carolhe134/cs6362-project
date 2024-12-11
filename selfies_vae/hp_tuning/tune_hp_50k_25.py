from selfiesvae.tune_hp import tune_hyperparameters

def main():
    tune_hyperparameters("settings_50k_hp_tuning.yml", 25)

if __name__ == '__main__':
    main()