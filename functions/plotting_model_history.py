
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, fold_histories=None, save_path=None):
    """
    Plot training and validation metrics history.
    
    Args:
        history: Dictionary containing training metrics for the best model.
        fold_histories: Optional list of histories from k-fold cross-validation.
        save_path: Optional path to save the plot. If None, the plot is not saved.
    
    Returns:
        The matplotlib.pyplot module.
    """
    if fold_histories:
        # We have k-fold histories to plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))  # Create figure and axes

        # Plot losses from all folds
        for i, fold_hist in enumerate(fold_histories):
            axes[0, 0].plot(fold_hist['train_loss'], linestyle='--', alpha=0.5, label=f'Fold {i+1} Train')
            axes[0, 0].plot(fold_hist['val_loss'], alpha=0.5, label=f'Fold {i+1} Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss (All Folds)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot mAP scores from all folds
        for i, fold_hist in enumerate(fold_histories):
            axes[0, 1].plot(fold_hist['val_mAP'], alpha=0.7, label=f'Fold {i+1}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP Score')
        axes[0, 1].set_title('Validation mAP Score (All Folds)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot average performance across folds
        avg_train_loss = np.mean([fold['train_loss'] for fold in fold_histories], axis=0)
        avg_val_loss = np.mean([fold['val_loss'] for fold in fold_histories], axis=0)
        std_train_loss = np.std([fold['train_loss'] for fold in fold_histories], axis=0)
        std_val_loss = np.std([fold['val_loss'] for fold in fold_histories], axis=0)
        
        epochs = range(1, len(avg_train_loss) + 1)
        axes[1, 0].plot(epochs, avg_train_loss, label='Avg Train Loss')
        axes[1, 0].plot(epochs, avg_val_loss, label='Avg Val Loss')
        axes[1, 0].fill_between(epochs, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, alpha=0.2)
        axes[1, 0].fill_between(epochs, avg_val_loss - std_val_loss, avg_val_loss + std_val_loss, alpha=0.2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Average Loss Across Folds')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot average mAP score across folds
        avg_map = np.mean([fold['val_mAP'] for fold in fold_histories], axis=0)
        std_map = np.std([fold['val_mAP'] for fold in fold_histories], axis=0)
        
        axes[1, 1].plot(epochs, avg_map, label='Avg mAP Score')
        axes[1, 1].fill_between(epochs, avg_map - std_map, avg_map + std_map, alpha=0.2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP Score')
        axes[1, 1].set_title('Average mAP Score Across Folds')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
    else:
        # Standard training history plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Create figure and axes
        
        # Plot training and validation loss
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot validation mAP score
        axes[1].plot(history['val_mAP'], label='Validation mAP')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP Score')
        axes[1].set_title('Validation mAP Score')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return plt # return the plt module so you can use plt.show() outside.