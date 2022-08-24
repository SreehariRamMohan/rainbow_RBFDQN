import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# From https://github.com/abagaria/hrl/blob/sree/init-rbf/hrl/agent/dsc/classifier/utils.py

class ObsClassifierMLP(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.l1 = nn.Linear(obs_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, obs):
        return self.l3(
            F.leaky_relu(
                self.l2(
                    F.leaky_relu(
                        self.l1(obs)
                    )
                )
            )
        )

# From https://github.com/abagaria/hrl/blob/sree/init-rbf/hrl/agent/dsc/classifier/mlp_classifier.py

class BinaryMLPClassifier:
    """" Generic binary neural network classifier. """
    def __init__(self,
                obs_dim,
                device,
                threshold=0.5,
                batch_size=128):
        
        self.device = device
        self.is_trained = False
        self.threshold = threshold
        self.batch_size = batch_size

        self.model = ObsClassifierMLP(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Debug variables
        self.losses = []

    @torch.no_grad()
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).to(self.device).float()
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        return probabilities
        
    @torch.no_grad()
    def predict(self, X, threshold=None):
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        threshold = self.threshold if threshold is None else threshold
        return probabilities > threshold

    def determine_pos_weight(self, y):
        n_negatives = len(y[y != 1])
        n_positives = len(y[y == 1])
        if n_positives > 0:
            pos_weight = (1. * n_negatives) / n_positives
            return torch.as_tensor(pos_weight).float()

    def should_train(self, y):
        enough_data = len(y) > self.batch_size
        has_positives = len(y[y == 1]) > 0
        has_negatives = len(y[y != 1]) > 0
        return enough_data and has_positives and has_negatives

    def fit(self, X, y, W=None, n_epochs=5):
        dataset = ClassifierDataset(X, y, W)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.should_train(y):
            losses = []

            for _ in range(n_epochs):
                epoch_loss = self._train(dataloader)                
                losses.append(epoch_loss)
            
            self.is_trained = True

            mean_loss = np.mean(losses)
            print(mean_loss)
            self.losses.append(mean_loss)

    def _train(self, loader):
        """ Single epoch of training. """
        batch_losses = []

        for sample in loader:
            observations, labels, weights = self._extract_sample(sample)

            pos_weight = self.determine_pos_weight(labels)

            if pos_weight is None or pos_weight.nelement == 0:
                continue

            logits = self.model(observations)
            if logits.size[0] != 1:
                squeezed_logits = logits.squeeze()
            else:
                squeezed_logits = torch.reshape(logits, (1,))
            print("!!!!! Observations:", observations)
            print("!!!!! Labels:", labels)
            if logits is not None:
                print("!!!!! Logits shape", logits.shape)
            if squeezed_logits is not None:
                print("!!!!! Squeezed logits shape", squeezed_logits.shape)
            if pos_weight is not None:
                print("!!!!! pos_weight shape", pos_weight.shape)
            if weights is not None:
                print("!!!!! weights shape", weights.shape)
            loss = F.binary_cross_entropy_with_logits(squeezed_logits,
                                                      labels,
                                                      pos_weight=pos_weight,
                                                      weight=weights) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())
        
        return np.mean(batch_losses)
    
    def _extract_sample(self, sample):
        weights = None

        if len(sample) == 3:
            observations, labels, weights = sample
            weights = weights.to(self.device).float().squeeze()
        else:
            observations, labels = sample

        observations = observations.to(self.device).float()
        labels = labels.to(self.device)

        return observations, labels, weights


class ClassifierDataset(Dataset):
    def __init__(self, states, labels, weights=None):
        self.states = states
        self.labels = labels
        self.weights = weights
        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.weights is not None:
            return self.states[i], self.labels[i], self.weights[i]

        return self.states[i], self.labels[i]

def test_matt():
    import random
    num_ex = 10
    X_np = np.random.random((14, 41))
    X = torch.from_numpy(X_np).float()
    y = np.array([random.choice([0., 1.]) for i in range(X.shape[0])])
    W = np.random.randint(0, 10, X_np.shape[0]).astype(float)

    clf = BinaryMLPClassifier(\
        X_np.shape[1], \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'), \
        threshold=0.5, \
        batch_size=5)
    print(X, y, W)
    clf.fit(X, y, W, n_epochs=10)
    probs = clf.predict_proba(X).detach().cpu().numpy()
    probs = probs.reshape((-1))
    from scipy.special import softmax
    probs = softmax(probs)
    print(np.random.choice(a=len(probs), p=probs))
