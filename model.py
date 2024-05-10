#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import wandb

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        out = self.dropout(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, model_id):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.model_id = model_id

    def train(self, save_dir, train_loader, valid_loader, num_epochs, learning_rate, device, lambda_=0.15):
        
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_loss = np.inf

        early_stop = 15

        for epoch in range(num_epochs):


            self.model.train()

            epoch_loss = 0
            correct = 0
            total = 0

            for b, batch in enumerate(train_loader):
                
                batch_input, batch_target = batch
                mask = torch.ones_like(batch_target, dtype=torch.float)
                batch_target = torch.argmax(batch_target,axis=1)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                

                loss_classif = 0
                loss_boundary = 0

                for p in predictions:
                    loss_classif += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_boundary += lambda_*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                loss = loss_classif + loss_boundary

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                print(f"[epoch {epoch +1} / {num_epochs}]({b+1} / {len(train_loader)}): loss = {epoch_loss/(b+1):.2f} ({loss_classif:.2f} / {loss_boundary:.2f}) | Acc = {float(correct)/total:.2f}", end="\r")
            
            self.model.eval()

            loss_val = 0
            correct_val = 0
            total_val = 0

            for batch in valid_loader:

                batch_input, batch_target = batch
                mask = torch.ones_like(batch_target, dtype=torch.float)
                batch_target = torch.argmax(batch_target,axis=1)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                predictions = self.model(batch_input, mask)

                loss_classif_val = 0
                loss_boundary_val = 0

                for p in predictions:
                    loss_classif_val += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_boundary_val += lambda_*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
            
                loss = loss_classif_val + loss_boundary_val

                loss_val += loss.item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_val += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total_val += torch.sum(mask[:, 0, :]).item()
            

            print(f"[epoch {epoch +1} / {num_epochs}]({b+1} / {len(train_loader)}): loss train = {epoch_loss/len(train_loader):.2f} | Acc train = {float(correct)/total:.2f} // ", end=" ")
            print(f"loss validation = {loss_val/len(valid_loader):.2f} | Acc validation = {float(correct_val)/total_val:.2f}", end=" ")
            



            if loss_val < best_loss:
                torch.save(self.model, save_dir+f"/best_model_{self.model_id}.pt")
                best_loss = loss_val
                early_stop = 15
                print("[Best model on validation set]")
            else:
                early_stop -= 1
                print()

            if early_stop == 0:
                print("Early stopping!")
                break
            
            wandb.log({"val_loss": loss_val/len(valid_loader), "train_loss": epoch_loss/len(train_loader),
                        "val_acc": float(correct_val)/total_val, "train_acc":float(correct)/total})

        best_model = torch.load(save_dir+f"/best_model_{self.model_id}.pt")
        self.model.eval()
        correct_val = 0
        total_val = 0

        for batch in valid_loader:

            batch_input, batch_target = batch
            mask = torch.ones_like(batch_target, dtype=torch.float)
            batch_target = torch.argmax(batch_target,axis=1)
            batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

            predictions = best_model(batch_input, mask)

            _, predicted = torch.max(predictions[-1].data, 1)
            correct_val += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
            total_val += torch.sum(mask[:, 0, :]).item()

        
        wandb.log({'best_val_acc':float(correct_val)/total_val})

            # torch.save(self.model, s)
            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
