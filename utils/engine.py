import datetime
import torch
from tqdm import tqdm
from utils.average_meter import AverageMeter


def reduce_fn(vals):
    return sum(vals)/len(vals)


class Engine:

    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        model_fn=None,
        use_mean_loss=False,
    ):

        """
        model_fn should take batch of data, device and model. Returns loss
        for example:
            def model_fn(data, device, model):
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                _, loss = model(images, targets)
                return loss
        """

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.model_fn = model_fn
        self.use_mean_loss = use_mean_loss

    def train(self, data_loader):
        losses = AverageMeter()
        self.model.train()

        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()

        tk0 = tqdm(data_loader, total=len(data_loader))

        for b_idx, data in enumerate(tk0):
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()

            if self.model_fn is None:
                for key, value in data.items():
                    data[key] = value.to(self.device)

                _, loss = self.model(**data)

                #print(batch_preds)

            else:
                loss = self.model_fn(data, self.device, self.model)

            with torch.set_grad_enabled(True):
                if self.use_mean_loss:
                    loss = loss.mean()

                loss.backward()

                if (b_idx+1) % self.accumulation_steps == 0:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                if b_idx > 0:
                    self.optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)

            tk0.set_postfix(loss=losses.avg)
        
        tk0.close()
        return losses.avg


    def evaluate(self, data_loader):
        losses = AverageMeter()
        self.model.eval()
        final_predictions = []

        with torch.no_grad():
            tk0 = tqdm(data_loader, total = len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(self.device)

                batch_preds, loss = self.model(**data)

                #print(b_idx, batch_pred)

                final_predictions.append(batch_preds.cpu())
                
                if self.use_mean_loss:
                    loss = loss.mean()

                losses.update(loss.item(), data_loader.batch_size)

                tk0.set_postfix(loss=losses.avg)
            tk0.close()

        return losses.avg, final_predictions


    def predict(self, data_loader):
        self.model.eval()
        final_predictions = []

        with torch.no_grad():
            tk0 = tqdm(data_loader, total = len(data_loader))

            for data in tk0:
                for key, value in data.items():
                    data[key] = value.to(self.device)

                predictions, _ = self.model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)

            tk0.close()
        return final_predictions

