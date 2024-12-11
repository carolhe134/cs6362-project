from selfiesvae.train import train

def main():
    train(settings_file="settings_50k_optim.yml", 
          log_data=True, 
          log_dir='./training_logs_50k_optim', 
          model_dir='./saved_models_50k_optim')

if __name__ == '__main__':
    main()