import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from SDAN.layers import PoolSuper


def train_with_args(GNN_list, labels_list, in_channels, out_channels, args, d, cell_type_str, api=False):
    if api:
        [train_GNN, val_GNN] = GNN_list
        [train_labels, val_labels] = labels_list
    else:
        [train_GNN, val_GNN, test_GNN] = GNN_list
        [train_labels, val_labels, test_labels] = labels_list
    # Model and optimizer
    model = PoolSuper(in_channels=in_channels,
                      hidden_channels1=args.hidden1,
                      hidden_channels2=args.hidden2,
                      n_comp=args.n_comp,
                      out_channels=out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.cuda:
        model.cuda()
        train_GNN = train_GNN.cuda()
        train_labels = train_labels.cuda()

    def train():
        model.train()
        optimizer.zero_grad()
        out_x, out, mc_loss, o_loss = model(train_GNN)
        clf_loss = nn.CrossEntropyLoss()(out_x, train_labels)
        loss = clf_loss + args.mc_weight * mc_loss + args.o_weight * o_loss
        loss.backward()
        optimizer.step()
        return out, clf_loss, args.mc_weight * mc_loss + args.o_weight * o_loss

    @torch.no_grad()
    def test(data_reduced, data_labels):
        model.eval()
        data_score = model.linear_relu_stack(data_reduced)
        clf_loss = nn.CrossEntropyLoss()(data_score, data_labels)
        data_score = torch.nn.Softmax(dim=1)(data_score)
        data_auc = roc_auc_score(data_labels.detach().numpy(), data_score[:, 1].detach().numpy())
        return data_score, clf_loss, data_auc

    best_val_loss = float('inf')
    patience = args.start_patience
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    if not api:
        test_loss_list = []
        test_auc_list = []
    for epoch in range(args.epochs):
        train_s, train_loss, graph_loss = train()
        train_loss = train_loss + graph_loss

        val_data_reduced = val_GNN.x.t() @ train_s
        val_score, val_loss, val_auc = test(val_data_reduced, val_labels)
        val_loss = val_loss + graph_loss

        if not api:
            test_data_reduced = test_GNN.x.t() @ train_s
            test_score, test_loss, test_auc = test(test_data_reduced, test_labels)
            test_loss = test_loss + graph_loss

        train_data_reduced = train_GNN.x.t() @ train_s
        train_score, _, train_auc = test(train_data_reduced, train_labels)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')

        train_loss_list.append(train_loss.detach().numpy())
        val_loss_list.append(val_loss.detach().numpy())
        train_auc_list.append(train_auc)
        val_auc_list.append(val_auc)
        if not api:
            test_loss_list.append(test_loss.detach().numpy())
            test_auc_list.append(test_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_s = train_s
            torch.save(model.state_dict(), f'{d}output/state_{cell_type_str}.pt')
            patience = args.start_patience
        else:
            patience -= 1
            if patience <= 0 and epoch > args.epochs_min:
                print('Early stopping!')
                break

    model.load_state_dict(torch.load(f'{d}output/state_{cell_type_str}.pt'))
    train_s = best_train_s

    if api:
        return model, train_s, [train_loss_list, val_loss_list], [train_auc_list, val_auc_list]
    else:
        return model, train_s, [train_loss_list, val_loss_list, test_loss_list], [train_auc_list, val_auc_list, test_auc_list]


def test_model(model, data_reduced, data_labels):
    model.eval()
    data_score = model.linear_relu_stack(data_reduced)
    clf_loss = nn.CrossEntropyLoss()(data_score, data_labels)
    data_score = torch.nn.Softmax(dim=1)(data_score)
    data_auc = roc_auc_score(data_labels.detach().numpy(), data_score[:, 1].detach().numpy())
    return data_score, clf_loss, data_auc

