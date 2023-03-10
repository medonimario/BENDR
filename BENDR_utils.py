import torch
from dn3_ext import BENDRClassification
from tqdm import tqdm
from dn3.trainable.processes import StandardClassification
from dn3.metrics.base import balanced_accuracy

"""
This is a Python class called BENDR, which inherits from BENDRClassification.
It is a deep learning model that can classify signals into one of several target classes.
The input to the model is a tensor of shape (batch_size, channels, samples),
where batch_size is the number of samples in a batch, channels is the number of
input channels, and samples is the number of time samples in the signal.
The output is a tensor of shape (batch_size, targets), where targets is the number of target classes.

The BENDR class provides methods for loading and saving the model weights,
as well as evaluating the model on input data. It also provides a forward method
for performing a forward pass through the model, and a forward_probs method
for returning class probabilities instead of class predictions.
The BENDR class has several hyperparameters that can be customized, such as
the size of the encoder and contextualizer layers, the number of projection layers,
the dropout rate, and the span and probability of masking.
"""
class BENDR(BENDRClassification):
    def __init__(self, targets: int, samples: int, channels: int, device: str = 'cpu', encoder_h=512, contextualizer_hidden=3076,
                 projection_head=False, new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                 keep_layers=None, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False):
        super().__init__(targets, samples, channels, encoder_h, contextualizer_hidden, projection_head,
                 new_projection_layers, dropout, trial_embeddings, layer_drop, keep_layers,
                 mask_p_t, mask_p_c, mask_t_span, mask_c_span, multi_gpu)
        
        if device is not None:
            self.device = torch.device(device)
            self.to(self.device)
        else:
            self.device = None

        self.input_shape = torch.Size((channels, samples))
        self.output_shape = torch.Size((targets, ))

    def freeze_classifier(self, unfreeze: bool = False) -> None:
        """
        Unfreeze classifier weights
        Parameters
        ----------
        freeze : bool
            Freeze classifier weights
        """
        for param in self.classifier.parameters():
            param.requires_grad = unfreeze
        
    def load_classifier(self, classifier_file: str, freeze: bool = False) -> None:
        """
        Load classifier weights from file
        Parameters
        ----------
        classifier_file : str
            Path to classifier weights file
        freeze : bool
            Freeze classifier weights
        """
        if self.device is not None:
            classifier_state_dict = torch.load(classifier_file, self.device)
        else:
            classifier_state_dict = torch.load(classifier_file)
        self.classifier.load_state_dict(classifier_state_dict, strict=True)

        self.freeze_classifier(not freeze)

    def load_all(self, encoder_file: str, contextualizer_file: str, classifier_file: str,
                strict: bool = True, freeze_encoder: bool = False, freeze_contextualizer: bool = False,
                freeze_classifier: bool = False, verbose: bool = False) -> None:
        """
        Load encoder, contextualizer and classifier weights from files
        Parameters
        ----------
        encoder_file : str
            Path to encoder weights file
        contextualizer_file : str
            Path to contextualizer weights file
        classifier_file : str
            Path to classifier weights file
        strict : bool
            Strict loading of encoder and contextualizer weights
        freeze_encoder : bool
            Freeze encoder weights
        freeze_contextualizer : bool
            Freeze contextualizer weights
        freeze_classifier : bool
            Freeze classifier weights
        """

        if verbose: print("Loading encoder... ", end="")
        if self.device is not None:
            encoder_state_dict = torch.load(encoder_file, self.device)
        else:
            encoder_state_dict = torch.load(encoder_file)
        self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze_encoder)
        if verbose: print("done")

        if verbose: print("Loading contextualizer... ", end="")
        if self.device is not None:
            contextualizer_state_dict = torch.load(contextualizer_file, self.device)
        else:
            contextualizer_state_dict = torch.load(contextualizer_file)
        self.contextualizer.load_state_dict(contextualizer_state_dict, strict=True)
        self.contextualizer.freeze_features(unfreeze=not freeze_contextualizer)
        if verbose: print("done")

        if verbose: print("Loading classifier... ", end="")
        self.load_classifier(classifier_file, freeze=freeze_classifier)
        if verbose: print("done")

    def save_all(self, encoder_file: str, contextualizer_file: str, classifier_file: str):
        """
        Save encoder, contextualizer and classifier weights to files
        Parameters
        ----------
        encoder_file : str
            Path to encoder weights file
        contextualizer_file : str
            Path to contextualizer weights file
        classifier_file : str
            Path to classifier weights file
        """
        torch.save(self.encoder.state_dict(), encoder_file)
        torch.save(self.contextualizer.state_dict(), contextualizer_file)
        torch.save(self.classifier.state_dict(), classifier_file)

    def feedforward(self, x: torch.Tensor, return_features: bool = False, grad: bool = False) -> torch.Tensor:
        """
        Forward pass through the model
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, samples)
        return_features : bool
            Return features from the encoder and contextualizer
        grad : bool
            Enable gradient calculation
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, targets)
        """

        assert isinstance(x, torch.Tensor), "Input has to be of instance torch.Tensor"

        if x.shape == self.input_shape: x = torch.unsqueeze(x, dim=0)
        
        assert x.shape[1:] == self.input_shape, "Input has to be of shape (*, {}, {})".format(*self.input_shape)

        self.return_features = return_features

        if grad:
            output = super().forward(x.to(self.device))
        else:
            with torch.no_grad(): output = super().forward(x.to(self.device))

        if return_features:
            return output[0], output[1]
        else:
            return output
        
    def feedforward_probs(self, x: torch.Tensor, return_features: bool = False, grad: bool = False) -> torch.Tensor:
        """
        Forward pass through the model and return probabilities
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, samples)
        return_features : bool
            Return features from the encoder and contextualizer
        grad : bool
            Enable gradient calculation
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, targets)
        """
        output = self.feedforward(x, return_features, grad)

        if return_features:
            return output[0].softmax(dim=1), output[1]
        else:
            return output.softmax(dim=1)

    def evaluate(self, X: torch.Tensor, batch_size: int = 8, return_probs: bool = False) -> torch.Tensor:
        """
        Evaluate model on input data
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (observation, channels, samples)
        batch_size : int
            Batch size
        return_probs : bool
            Return probabilities
        Returns
        -------
        torch.Tensor
            Predictions tensor of shape (observation, )
        """

        assert isinstance(X, torch.Tensor), "Input has to be of instance torch.Tensor"
        assert X.shape[1:] == self.input_shape, "Input has to be of shape (*, {}, {})".format(*self.input_shape)

        N = len(X)

        probs = torch.empty((N, self.targets))

        prog_bar = tqdm(total = int(torch.ceil(torch.Tensor([len(X) / batch_size])).item()),
                        desc="Evaluating", unit="batches")

        for i, x in zip(range(0, N, batch_size), X.split(batch_size)):
            probs[i:(i + batch_size)] = self.feedforward_probs(x)
            prog_bar.update(1)

        prog_bar.close()

        predictions = probs.argmax(1)
        
        if return_probs:
            return predictions, probs
        else:
            return predictions
        
    def fit(self, training, validation=None, warmup_frac: float = 0.1, retain_best: str = 'bac',
            freeze_contextualizer: bool = False, freeze_encoder: bool = False, freeze_classifier: bool = False,
            learning_rate: float = 1e-05, weight_decay: float = 0.01, pin_memory: bool = False,
            metric = balanced_accuracy, **kwargs):


        self.contextualizer.freeze_features(unfreeze=not freeze_contextualizer)
        self.encoder.freeze_features(unfreeze=not freeze_encoder)
        self.freeze_classifier(unfreeze=not freeze_classifier)

        process = StandardClassification(self, metrics=metric)
        process.set_optimizer(torch.optim.Adam(process.parameters(), learning_rate, weight_decay=weight_decay))

        process.fit(training_dataset=training, validation_dataset=validation, warmup_frac=warmup_frac,
                    retain_best=retain_best, pin_memory=pin_memory, **kwargs)
