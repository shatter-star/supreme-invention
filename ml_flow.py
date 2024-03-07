import mlflow
import os
import argparse
import matplotlib.pyplot as plt

from models.neural_style_transfer import NeuralStyleTransfer
from models.definitions.vgg19 import VggModel as Vgg19
from utils.config import load_config
from utils.logging import setup_logging

logger = setup_logging(__name__)

def setup_mlflow_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def log_artifacts(dump_path, config):
    output_image_path = os.path.join(dump_path, config['content_img_name'].split('.')[0] + '_' + config['style_img_name'].split('.')[0] + '.jpg')
    mlflow.log_artifact(output_image_path, artifact_path="output_images")

def log_metrics(loss_history):
    log_interval = 5
    for i, (total_loss, content_loss, style_loss, tv_loss) in enumerate(loss_history):
        if i % log_interval == 0:
            mlflow.log_metric(f'total_loss_{i}', total_loss)
            mlflow.log_metric(f'content_loss_{i}', content_loss)
            mlflow.log_metric(f'style_loss_{i}', style_loss)
            mlflow.log_metric(f'tv_loss_{i}', tv_loss)

def log_model(vgg_model):
    mlflow.pytorch.log_model(vgg_model, "model")
    mlflow.pytorch.log_model(vgg_model, "model", registered_model_name="VGG19Model")

def log_loss_curve(loss_history):
    total_loss_values = [loss[0] for loss in loss_history]
    style_loss_values = [loss[2] for loss in loss_history]
    content_loss_values = [loss[1] for loss in loss_history]
    tv_loss_values = [loss[3] for loss in loss_history]

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_values, label='Total Loss')
    plt.plot(style_loss_values, label='Style Loss')
    plt.plot(content_loss_values, label='Content Loss')
    plt.plot(tv_loss_values, label='TV Loss')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.yscale('log')
    plt.savefig('loss_curve.png')
    mlflow.log_artifact('loss_curve.png')

def run_neural_style_transfer(config):
    vgg_model = Vgg19()
    nst = NeuralStyleTransfer(config)
    loss_history = []

    with mlflow.start_run():
        for key, value in config.items():
            mlflow.log_param(key, value)

        dump_path, loss_history = nst.run()
        
        log_artifacts(dump_path, config)
        log_metrics(loss_history)
        log_model(vgg_model)
        log_loss_curve(loss_history)

def parse_arguments():
    config = load_config()
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    args = parser.parse_args()

    if args.config:
        config.update(load_config(args.config))

    return config

if __name__ == "__main__":
    config = parse_arguments()
    os.makedirs(config['output_img_dir'], exist_ok=True)

    setup_mlflow_experiment("Neural-Style-Transfer")
    run_neural_style_transfer(config)