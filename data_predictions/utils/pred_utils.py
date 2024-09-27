import json
import os
import torch
import torchmetrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.distributions import Normal
import numpy as np
import copy

# Local imports
from data_predictions.utils.pred_config import PATH_TO_SAVE_RESULTS, PATH_TO_SAVE_PLOTS
from models.lstm_base import LSTMBase
from models.transformer_base import TransformerBase
from models.autoformer import Autoformer
from models.informer import Informer
from models.prob_transformer import ProbabilisticTransformer

# Train/Validation/Test functions
def initialize_metrics(target_feature, device):
    """Initializes metrics and metric lists based on the target feature."""
    if target_feature == 'pct_price_change':
        # Initialize the metrics
        rmse_metric = torchmetrics.MeanSquaredError(squared=True).to(device)
        mse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(device)
        mape_metric = torchmetrics.MeanAbsolutePercentageError().to(device)
        r2_metric = torchmetrics.R2Score().to(device)

        # Define the metrics lists
        train_loss_list = []
        train_rmse_list = []
        train_mse_list = []
        train_r2_list = []
        train_mae_list = []
        train_mape_list = []
        val_loss_list = []
        val_rmse_list = []
        val_mse_list = []
        val_r2_list = []
        val_mae_list = []
        val_mape_list = []

        # Create a dictionary to store the metrics
        metrics = {
            'rmse': rmse_metric,
            'mse': mse_metric,
            'mae': mae_metric,
            'mape': mape_metric,
            'r2': r2_metric
        }

        # Create a dictionary to store the metric lists
        metrics_lists = {
            'train_loss': train_loss_list,
            'train_rmse': train_rmse_list,
            'train_mse': train_mse_list,
            'train_r2': train_r2_list,
            'train_mae': train_mae_list,
            'train_mape': train_mape_list,
            'val_loss': val_loss_list,
            'val_rmse': val_rmse_list,
            'val_mse': val_mse_list,
            'val_r2': val_r2_list,
            'val_mae': val_mae_list,
            'val_mape': val_mape_list
        }
    else:
        # Initialize the metrics
        accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=3).to(device)
        precision_metric = torchmetrics.Precision(task='multiclass', num_classes=3, average='macro').to(device)
        recall_metric = torchmetrics.Recall(task='multiclass', num_classes=3, average='macro').to(device)
        f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=3, average='macro').to(device)
        auroc_metric = torchmetrics.AUROC(task='multiclass', num_classes=3).to(device)

        # Define the metrics lists
        train_loss_list = []
        train_accuracy_list = []
        train_precision_list = []
        train_recall_list = []
        train_f1_list = []
        train_auroc_list = []
        val_loss_list = []
        val_accuracy_list = []
        val_precision_list = []
        val_recall_list = []
        val_f1_list = []
        val_auroc_list = []

        # Create a dictionary to store the metrics
        metrics = {
            'accuracy': accuracy_metric,
            'precision': precision_metric,
            'recall': recall_metric,
            'f1': f1_metric,
            'auroc': auroc_metric
        }

        # Create a dictionary to store the metric lists
        metrics_lists = {
            'train_loss': train_loss_list,
            'train_accuracy': train_accuracy_list,
            'train_precision': train_precision_list,
            'train_recall': train_recall_list,
            'train_f1': train_f1_list,
            'train_auroc': train_auroc_list,
            'val_loss': val_loss_list,
            'val_accuracy': val_accuracy_list,
            'val_precision': val_precision_list,
            'val_recall': val_recall_list,
            'val_f1': val_f1_list,
            'val_auroc': val_auroc_list
        }

    return metrics, metrics_lists

def calculate_metrics(loss, metrics, outputs, preds, target, target_feature):
    """Calculates and returns a dictionary of metrics."""
    if target_feature == 'pct_price_change':
        return {
            'loss': loss,
            'rmse': metrics['rmse'](outputs, target).item(),
            'mse': metrics['mse'](outputs, target).item(),
            'mae': metrics['mae'](outputs, target).item(),
            'mape': metrics['mape'](outputs, target).item(),
            'r2': metrics['r2'](outputs, target).item()
        }
    else:
        return {
            'loss': loss,
            'accuracy': metrics['accuracy'](outputs, target).item() * 100,
            'precision': metrics['precision'](outputs, target).item() * 100,
            'recall': metrics['recall'](outputs, target).item() * 100,
            'f1': metrics['f1'](outputs, target).item() * 100,
            'auroc': metrics['auroc'](preds.softmax(dim=1), target.long()).item() * 100
        }

def log_metrics(metrics, prefix):
    """Logs metrics to Weights & Biases."""
    wandb.log({f"{prefix} {k.upper()}": v for k, v in metrics.items()})

def print_metrics(metrics):
    """Prints metrics to the console."""
    for metric_name, metric_value in metrics.items():
        print(f", {metric_name.upper()}: {metric_value:.4f}", end='')
    print('')

def train_model(
    model,
    model_name,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    metrics,
    metrics_list,
    data_name,
    device,
    epochs,
    target_feature,
    use_wandb=False,
    use_llm_features=False
):
    """Trains a model and tracks metrics.

    Args:
        model: The model to train.
        model_name: The name of the model.
        train_dataloader: The training dataloader.
        val_dataloader: The validation dataloader.
        criterion: The loss function.
        optimizer: The optimizer.
        metrics: A dictionary of metrics.
        metrics_list: A dictionary to store metrics for each epoch.
        data_name: The name of the dataset.
        device: The device to use for training.
        epochs: The number of epochs to train for.
        target_feature: The target feature to predict.
        use_wandb: Whether to use Weights & Biases for logging.
        use_llm_features: Whether to use LLM features.

    Returns:
        A dictionary containing the training and validation metrics.
    """

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        train_loss = 0

        # Track all outputs and targets for epoch-level metrics
        all_train_outputs = []
        all_train_preds = []
        all_targets = []

        # Loop through the train dataloader
        for x_cont, *x_cat, y in train_dataloader:
            # Move data to device
            cont = x_cont.to(device)
            cat = {
                'fng_value_classification': x_cat[0].to(device),
                'fng_sentiment': x_cat[1].to(device),
                'cbbi_sentiment': x_cat[2].to(device)
            }
            if use_llm_features:
                cat.update({
                    'sentiment_class': x_cat[3].to(device),
                    'action_class': x_cat[4].to(device),
                    'action_score': x_cat[5].to(device)
                })

            target = y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if model_name != 'prob_transformer':
                outputs = model(cont, cat)
            else:
                mean, std = model(cont, cat)
                dist = Normal(mean, std)
                outputs = dist.sample()

            # Calculate loss
            if target_feature == 'trend':
                target = target.squeeze(1)
                loss = criterion(outputs, target)
            elif model_name == 'prob_transformer':
                loss = model.loss_fn(mean, std, target)
            else:
                loss = criterion(outputs, target)

            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # Update the loss
            train_loss += loss.item()

            # Append outputs and targets for epoch-level metrics
            all_train_preds.append(outputs)
            all_train_outputs.append(outputs if target_feature == 'pct_price_change' else outputs.argmax(dim=1))
            all_targets.append(target)

        # Calculate epoch-level metrics
        train_loss /= len(train_dataloader)
        train_preds = torch.cat(all_train_preds, dim=0)
        train_outputs = torch.cat(all_train_outputs, dim=0)
        target = torch.cat(all_targets, dim=0)
        with torch.no_grad():
            train_metrics = calculate_metrics(train_loss, metrics, train_outputs, train_preds, target, target_feature)

        # Log training metrics
        if use_wandb:
            log_metrics(train_metrics, 'Training')

        # Store training metrics
        for metric_name, metric_value in train_metrics.items():
            metrics_list[f'train_{metric_name}'].append(metric_value)

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0
        all_val_outputs = []
        all_targets = []
        all_val_preds = []

        with torch.no_grad():
            for x_cont, *x_cat, y in val_dataloader:
                # Prepare data
                cont = x_cont.to(device)
                cat = {
                    'fng_value_classification': x_cat[0].to(device),
                    'fng_sentiment': x_cat[1].to(device),
                    'cbbi_sentiment': x_cat[2].to(device)
                }
                if use_llm_features:
                    cat.update({
                        'sentiment_class': x_cat[3].to(device),
                        'action_class': x_cat[4].to(device),
                        'action_score': x_cat[5].to(device)
                    })
                target = y.to(device)

                # Forward pass
                if model_name != 'prob_transformer':
                    outputs = model(cont, cat)
                else:
                    mean, std = model(cont, cat)
                    dist = Normal(mean, std)
                    outputs = dist.sample()

                # Calculate loss
                if target_feature == 'trend':
                    target = target.squeeze(1)
                    loss = criterion(outputs, target)
                elif model_name == 'prob_transformer':
                    loss = model.loss_fn(mean, std, target)
                else:
                    loss = criterion(outputs, target)

                val_loss += loss.item()

                # Append outputs and targets for epoch-level metrics
                all_val_preds.append(outputs)
                all_val_outputs.append(outputs if target_feature == 'pct_price_change' else outputs.argmax(dim=1))
                all_targets.append(target)

            val_loss /= len(val_dataloader)
            val_preds = torch.cat(all_val_preds, dim=0)
            val_outputs = torch.cat(all_val_outputs, dim=0)
            target = torch.cat(all_targets, dim=0)
            val_metrics = calculate_metrics(val_loss, metrics, val_outputs, val_preds, target, target_feature)

        # Log validation metrics
        if use_wandb:
            log_metrics(val_metrics, 'Validation')

        # Store validation metrics
        for metric_name, metric_value in val_metrics.items():
            metrics_list[f'val_{metric_name}'].append(metric_value)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"- Train Loss: {train_loss:.4f}", end='')
        print_metrics(train_metrics)
        print(f"- Val Loss: {val_loss:.4f}", end='')
        print_metrics(val_metrics)

        # Save results and models
        current_results = {
            'epoch': epoch + 1,
            **{f'training_{k}': v for k, v in train_metrics.items()},
            **{f'validation_{k}': v for k, v in val_metrics.items()}
        }
        save_results(data_name, current_results)
        save_model(data_name, model, epoch)

    return metrics_list

def make_predictions(
    model,
    model_name,
    train_dataloader,
    test_dataloader,
    criterion,
    metrics,
    data_name,
    device,
    target_feature,
    use_wandb=False,
    use_llm_features=False
):
    """Makes predictions and calculates metrics.

    Args:
        model: The trained model.
        model_name: The name of the model.
        train_dataloader: The training dataloader.
        test_dataloader: The test dataloader.
        criterion: The loss function.
        metrics: A dictionary of metrics.
        data_name: The name of the dataset.
        device: The device to use for training.
        target_feature: The target feature to predict.
        use_wandb: Whether to use Weights & Biases for logging.
        use_llm_features: Whether to use LLM features.

    Returns:
        A tuple containing the training and test predictions.
    """

    model.eval()
    train_loss = 0
    test_loss = 0
    all_train_outputs = []
    all_train_targets = []
    all_test_outputs = []
    all_test_targets = []

    # Ensure metrics are on the same device
    for key, metric in metrics.items():
        metrics[key] = metric.to(device)

    with torch.no_grad():
        for dataloader, outputs_list, targets_list in [(train_dataloader, all_train_outputs, all_train_targets), (test_dataloader, all_test_outputs, all_test_targets)]:
            for x_cont, *x_cat, y in dataloader:
                cont = x_cont.to(device)
                cat = {
                    'fng_value_classification': x_cat[0].to(device),
                    'fng_sentiment': x_cat[1].to(device),
                    'cbbi_sentiment': x_cat[2].to(device)
                }
                if use_llm_features:
                    cat.update({
                        'sentiment_class': x_cat[3].to(device),
                        'action_class': x_cat[4].to(device),
                        'action_score': x_cat[5].to(device)
                    })
                target = y.to(device)

                if model_name != 'prob_transformer':
                    outputs = model(cont, cat)
                else:
                    mean, std = model(cont, cat)
                    dist = Normal(mean, std)
                    outputs = dist.sample()

                if target_feature == 'trend':
                    target = target.squeeze(1)
                    loss = criterion(outputs, target)
                elif model_name == 'prob_transformer':
                    loss = model.loss_fn(mean, std, target)
                else:
                    loss = criterion(outputs, target)

                if dataloader == train_dataloader:
                    train_loss += loss.item()
                else:
                    test_loss += loss.item()

                outputs_list.append(outputs.cpu().numpy() if target_feature == 'pct_price_change' else outputs.argmax(dim=1).cpu().numpy())
                targets_list.append(target.cpu().numpy())

    train_loss /= len(train_dataloader)
    test_loss /= len(test_dataloader)

    train_predictions = np.concatenate(all_train_outputs).flatten()
    train_targets = np.concatenate(all_train_targets).flatten()
    test_predictions = np.concatenate(all_test_outputs).flatten()
    test_targets = np.concatenate(all_test_targets).flatten()

    # Convert predictions and targets to tensors and move them to the device
    train_predictions_tensor = torch.tensor(train_predictions).to(device)
    train_targets_tensor = torch.tensor(train_targets).to(device)
    test_predictions_tensor = torch.tensor(test_predictions).to(device)
    test_targets_tensor = torch.tensor(test_targets).to(device)

    if target_feature == 'pct_price_change':
        # Regression task
        train_metrics = {
            'test_rmse': metrics['rmse'](train_predictions_tensor, train_targets_tensor).item(),
            'test_mse': metrics['mse'](train_predictions_tensor, train_targets_tensor).item(),
            'test_mae': metrics['mae'](train_predictions_tensor, train_targets_tensor).item(),
            'test_mape': metrics['mape'](train_predictions_tensor, train_targets_tensor).item(),
            'test_r2': metrics['r2'](train_predictions_tensor, train_targets_tensor).item()
        }
        test_metrics = {
            'test_rmse': metrics['rmse'](test_predictions_tensor, test_targets_tensor).item(),
            'test_mse': metrics['mse'](test_predictions_tensor, test_targets_tensor).item(),
            'test_mae': metrics['mae'](test_predictions_tensor, test_targets_tensor).item(),
            'test_mape': metrics['mape'](test_predictions_tensor, test_targets_tensor).item(),
            'test_r2': metrics['r2'](test_predictions_tensor, test_targets_tensor).item()
        }
    else:
        # Classification task
        train_metrics = {
            'test_accuracy': metrics['accuracy'](train_predictions_tensor, train_targets_tensor).item() * 100,
            'test_precision': metrics['precision'](train_predictions_tensor, train_targets_tensor).item() * 100,
            'test_recall': metrics['recall'](train_predictions_tensor, train_targets_tensor).item() * 100,
            'test_f1': metrics['f1'](train_predictions_tensor, train_targets_tensor).item() * 100,
        }
        # Handle AUROC only if predictions have multiple classes
        if train_predictions_tensor.dim() > 1:
            train_metrics['auroc'] = metrics['auroc'](train_predictions_tensor.softmax(dim=1), train_targets_tensor.long()).item() * 100

        test_metrics = {
            'test_accuracy': metrics['accuracy'](test_predictions_tensor, test_targets_tensor).item() * 100,
            'test_precision': metrics['precision'](test_predictions_tensor, test_targets_tensor).item() * 100,
            'test_recall': metrics['recall'](test_predictions_tensor, test_targets_tensor).item() * 100,
            'test_f1': metrics['f1'](test_predictions_tensor, test_targets_tensor).item() * 100,
        }
        # Handle AUROC for test set if it's multiclass
        if test_predictions_tensor.dim() > 1:
            test_metrics['auroc'] = metrics['auroc'](test_predictions_tensor.softmax(dim=1), test_targets_tensor.long()).item() * 100

    if use_wandb:
        wandb.log({"Final Train Loss": train_loss})
        wandb.log({"Final Test Loss": test_loss})
        for metric_name, metric_value in train_metrics.items():
            wandb.log({f"Final Train {metric_name.upper()}": metric_value})
        for metric_name, metric_value in test_metrics.items():
            wandb.log({f"Final Test {metric_name.upper()}": metric_value})

    print(f"Train Loss: {train_loss:.4f}", end='')
    print_metrics(train_metrics)
    print(f"Test Loss: {test_loss:.4f}", end='')
    print_metrics(test_metrics)

    test_results = {
        'test_loss': test_loss,
        **test_metrics
    }

    save_results(data_name, test_results, test=True)

    return train_predictions, test_predictions

# Utility functions
def plot_results(data_1, data_2, title, x_label, y_label, legend, config, save_plot=False):
    plot_title = f"{title} - {config['architecture']}" 

    plt.figure(figsize=(20, 5))
    plt.plot(data_1, label=legend[0])
    plt.plot(data_2, label=legend[1])
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.tight_layout()

    if save_plot:        
        # Lowercase the title and replace spaces with underscores and '(' and ')' with empty strings and '-' with '_'
        title = title.lower().replace(' ', '_')
        title = title.replace('(', '').replace(')', '')
        title = title.replace('-', '_')

        # Check if the directory exists
        if not os.path.exists(PATH_TO_SAVE_PLOTS):
            os.makedirs(PATH_TO_SAVE_PLOTS, exist_ok=True)
        file_name = f"{config['task']}/{config['architecture']}_{config['task']}_{title}.png"
        plt.savefig(f"{PATH_TO_SAVE_PLOTS}/{file_name}")
        print(f"Plot saved at {PATH_TO_SAVE_PLOTS}/{file_name}")

    plt.show()

def plot_confusion_matrix(data_1, data_2, title, x_label, y_label, config, save_plot=False):
    plot_title = f"{title} - {config['architecture']}" 

    # Calculate the confusion matrix
    cm = confusion_matrix(data_1, data_2)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Same', 'Up'], yticklabels=['Down', 'Same', 'Up'])
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_plot:
        # Lowercase the title and replace spaces with underscores and '(' and ')' with empty strings and '-' with '_'
        title = title.lower().replace(' ', '_')
        title = title.replace('(', '').replace(')', '')
        title = title.replace('-', '_')

        # Check if the directory exists
        if not os.path.exists(PATH_TO_SAVE_PLOTS):
            os.makedirs(PATH_TO_SAVE_PLOTS, exist_ok=True)
        file_name = f"{config['task']}/{config['architecture']}_{config['task']}_{title}_confusion_matrix.png"
        plt.savefig(f"{PATH_TO_SAVE_PLOTS}/{file_name}")
        print(f"Plot saved at {PATH_TO_SAVE_PLOTS}/{file_name}")

    plt.show()

def save_configurations(data_name, configurations):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'config.json'
    with open(results_file_path, 'w') as json_file:
        json.dump(configurations, json_file, indent=2)

def save_results(data_name, results, test=False):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/results/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'test_results.json' if test else path + 'tr_val_results.json'
    if os.path.exists(results_file_path):
        final_results = None
        with open(results_file_path, 'r') as json_file:
            final_results = json.load(json_file)
        final_results.append(results)
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)
    else:
        final_results = [results]
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)

def save_model(data_name, model, epoch=None, is_best=False, full=False):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/models/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if is_best:
        torch.save(model, f'{path}/best_model.pt')
    else:
        torch.save(model,
                   f'{path}/checkpoint_{epoch+1}.pt')

def select_model(model_name, parameters):
    if model_name == 'lstm_base':
        model = LSTMBase(parameters)
    elif model_name == 'transformer_base':
        model = TransformerBase(parameters)
    elif model_name == 'autoformer':
        model = Autoformer(parameters)
    elif model_name == 'informer':
        model = Informer(parameters)
    elif model_name == 'prob_transformer':
        model = ProbabilisticTransformer(parameters)
    else:
        raise ValueError(f"Model {model_name} not implemented")
    return model