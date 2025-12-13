from typing import Any, Optional

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch.optim import RMSprop

def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        th.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            th.nn.init.constant_(m.bias, 0)




class FractionProposalNetwork(nn.Module):
    """
    from https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch
    """
    def __init__(self, N=32, embedding_dim=7*7*64):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, N)
        ).apply(lambda x: initialize_weights_xavier(x, gain=0.01))

        self.N = N
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):

        batch_size = state_embeddings.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)

        tau_0 = th.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = th.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = th.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N+1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, tau_hats, entropies


class CosineEmbeddingNetwork(nn.Module):
    """
    from https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch
    """
    def __init__(self, num_cosines=64, embedding_dim=7*7*64):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * th.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = th.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings

class FullyparameterizedQuantileNetwork(BasePolicy):
    """
    Fully parameterized Quantile network for FQF

    :param observation_space: Observation space
    :param action_space: Action space
    :param n_quantiles: Number of quantiles
    :param num_cosines: Number of cosine embedding 
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_quantiles: int = 32,
        num_cosines: int = 64,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        self.num_cosines = num_cosines
        action_dim = int(self.action_space.n)  # number of actions

        mlp_net = create_mlp(self.features_dim, 0, self.net_arch, self.activation_fn)
        self.mlp_net = nn.Sequential(*mlp_net)

        self.fraction_net = FractionProposalNetwork(
            N=self.n_quantiles, 
            embedding_dim=self.net_arch[-1]
        )


        self.embed_model = CosineEmbeddingNetwork(self.num_cosines, self.net_arch[-1])
        
        self.projection = nn.Sequential(
            nn.Linear(self.net_arch[-1], self.net_arch[-1]),
            nn.ReLU(),
            nn.Linear(self.net_arch[-1], action_dim),)
        

    def compute_distribution_quantiles(self, state_features: PyTorchObs,tau_hats, n_quantiles) -> th.Tensor:
        
        batch_size = state_features.shape[0]
        tau_embedding = self.embed_model(tau_hats)

        # element-wise product
        quantiles = state_features.unsqueeze(1) * tau_embedding  


        # Calculate quantile values
        quantiles = self.projection(quantiles)
        quantiles= quantiles.view(batch_size, n_quantiles, int(self.action_space.n))
        return quantiles

    def forward(self, obs: PyTorchObs, return_bounded_quantiles: bool= False) -> th.Tensor:
        """
        Predict the quantiles.

        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        state_features = self.mlp_net(self.extract_features(obs, self.features_extractor))
        taus, tau_hats, entropies = self.fraction_net(state_features.detach())
        
        quantiles=self.compute_distribution_quantiles(state_features, tau_hats, self.n_quantiles)

        if return_bounded_quantiles:
            bounded_quantiles=self.compute_distribution_quantiles(state_features, taus[:, 1:-1], self.n_quantiles-1)
            return quantiles, taus, tau_hats, entropies, bounded_quantiles
        return quantiles, taus, tau_hats, entropies

    def calculate_q(self, obs: PyTorchObs) -> th.Tensor:

        quantiles, taus, _, _ = self.forward(obs)
        batch_size = quantiles.shape[0]
        q_values = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1) 
        assert q_values.shape == (batch_size, int(self.action_space.n))
        return q_values

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self.calculate_q(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_quantiles=self.n_quantiles,
                num_cosines=self.num_cosines,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class FQFPolicy(BasePolicy):
    """
    Policy class with quantile and target networks for FQF.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param num_cosines: Number of cosine embedding 
    :param fraction_lr: lr of the  FractionProposalNetwork
    :param entropy_coef: if > 0 use entropy as a regularization term in fraction loss
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    quantile_net: FullyparameterizedQuantileNetwork
    quantile_net_target: FullyparameterizedQuantileNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 32,
        num_cosines: int = 64,
        fraction_lr: float = 2.5e-9, 
        entropy_coef: float = 0.001,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.n_quantiles = n_quantiles
        self.num_cosines = num_cosines
        self.fraction_lr = fraction_lr
        self.entropy_coef = entropy_coef
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_quantiles": self.n_quantiles,
            "num_cosines": self.num_cosines,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizers.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        excluded_params = set(self.quantile_net.fraction_net.parameters())
        quantile_params = [
            p for p in self.quantile_net.parameters() if p not in excluded_params
        ]
        
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            quantile_params,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        self.fraction_optimizer = RMSprop(
            self.quantile_net.fraction_net.parameters(),
            lr=self.fraction_lr, 
            alpha=0.95, 
            eps=0.00001)

    def make_quantile_net(self) -> FullyparameterizedQuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return FullyparameterizedQuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_quantiles=self.net_args["n_quantiles"],
                num_cosines=self.net_args["num_cosines"],
                net_arch=self.net_args["net_arch"],
                fraction_lr=self.fraction_lr,
                entropy_coef=self.entropy_coef,
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = FQFPolicy


class CnnPolicy(FQFPolicy):
    """
    Policy class for FQF when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param num_cosines: Number of cosine embedding 
    :param fraction_lr: lr of the  FractionProposalNetwork
    :param entropy_coef: if > 0 use entropy as a regularization term in fraction loss
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 32,
        num_cosines: int = 64,
        fraction_lr: float = 2.5e-9, 
        entropy_coef: float = 0.0,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            num_cosines,
            fraction_lr,
            entropy_coef,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(FQFPolicy):
    """
    Policy class for FQF when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param num_cosines: Number of cosine embedding 
    :param fraction_lr: lr of the  FractionProposalNetwork
    :param entropy_coef: if > 0 use entropy as a regularization term in fraction loss 
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 32,
        num_cosines: int = 64,
        fraction_lr: float = 2.5e-9, 
        entropy_coef: float = 0.0,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            num_cosines,
            fraction_lr,
            entropy_coef,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
