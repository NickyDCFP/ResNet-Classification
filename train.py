import os
import torch
import pandas as pd
import data
import models
import options


def train(opt):
    print(opt)
    train_dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    test_dataloader = data.get_dataloader(False, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt, train=True)
    history = pd.DataFrame(columns=["Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"])
    for epoch in range(opt.epochs):
        correct = 0
        total = 0
        test_loss = 0.0
        train_loss = 0.0
        print(f"Epoch {epoch + 1}")
        print("Training...")
        for batch in train_dataloader:
            inputs, labels = batch
            batch_correct, batch_total, batch_loss = model.optimize_params(inputs, labels)
            train_loss += batch_loss
            correct += batch_correct
            total += batch_total
        train_acc = correct / total
        print("Testing...")
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                correct, total, batch_loss = model.test(inputs, labels)
                test_loss += batch_loss
                correct += batch_correct
                total += batch_total
            test_acc = correct / total
        print(f'train loss: {train_loss}, train accuracy: {train_acc}, test loss: {test_loss}, test accuracy: {test_acc}')
        history.loc[len(history)] = [train_loss, test_loss, train_acc, test_acc]
        if epoch % opt.save_params_freq == 0:
            model.save_model(f'{opt.model_prefix}_{epoch // opt.save_params_freq}')
    history.to_csv(os.path.join(opt.checkpoint_dir, opt.csv_filename))
    model.save_model(f'{opt.model_prefix}_final')



if __name__ == '__main__':
    args = options.parse_args_train()
    train(args)
