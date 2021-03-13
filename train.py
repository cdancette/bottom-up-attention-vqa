import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import wandb

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []

    total_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()

        for i, (v, q, a, b, qid) in tqdm(enumerate(train_loader),
                                    desc="Epoch %d" % (epoch), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()

            pred, loss = model(v, None, q, a, b)

            if (loss != loss).any():
              raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum().item()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score
            wandb.log({"train_loss_batch": total_loss})

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        run_eval = eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            results, predictions = evaluate(model, eval_loader)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            all_results.append(results)

            with open(join(output, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)
            
            with open(join(output, f"predictions-{epoch}.json"), "w") as f:
                json.dump(predictions, f, indent=2)
    
            model.train(True)
            wandb.log(results)

            eval_score = results["score"]
            bound = results["upper_bound"]

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    predictions = []

    all_logits = []
    all_bias = []
    for v, q, a, b, qid in tqdm(dataloader, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred, _ = model(v, None, q, None, None)
        all_logits.append(pred.data.cpu().numpy())

        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        all_bias.append(b)
        labels = torch.argmax(pred, 1) # argmax
        for i in range(len(pred)):
            ansid =  labels[i].item()
            predictions.append({
                "question_id": qid[i].item(),
                "answer": dataloader.dataset.label2ans[ansid],
            })
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score.item(),
        upper_bound=upper_bound.item(),
    )
    return results, predictions
