# # ODST/NODE
# https://github.com/Qwicen/node
import os
import glob
import hashlib
import gc
import time
import requests
import numpy as np

# from torch.nn.modules.module import Module
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from torch.autograd import Function
from torch.jit import script
from warnings import warn


def to_one_hot(y, depth=None):
    r"""
    Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.
    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.
    Args:
        y: input integer (IntTensor, LongTensor or Variable) of any shape
        depth (int):  the size of the one hot dimension
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(
        1, y_flat, 1
    )
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Args:
            input: any dimension
            dim: dimension along which to apply the sparsemax
        Returns:
            the threshold value
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = lambda input, dim=-1: SparsemaxFunction.apply(input, dim)
sparsemoid = lambda input: (0.5 * input + 0.5).clamp_(0, 1)


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """ A highly optimized equivalent of labda x: Entmax15([x, 0]) """

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = lambda input, dim=-1: Entmax15Function.apply(input, dim)
entmoid15 = Entmoid15.apply


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModuleWithInit(nn.Module):
    """ Base class for pytorch module with data-aware initializer on first batch """

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(
            torch.tensor(0, dtype=torch.uint8), requires_grad=False
        )
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules

    def initialize(self, *args, **kwargs):
        """ initialize module tensors using first batch of data """
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get("content-length")

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename


def iterate_minibatches(
    *tensors,
    batch_size,
    shuffle=True,
    epochs=1,
    allow_incomplete=True,
    callback=lambda x: x,
):
    indices = np.arange(len(tensors[0]))
    upper_bound = (
        int((np.ceil if allow_incomplete else np.floor)(len(indices) / batch_size))
        * batch_size
    )
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large data tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(data), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0:batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(
            *output_shape,
            dtype=first_output.dtype,
            device=first_output.device,
            layout=first_output.layout,
            **kwargs,
        )

    out[0:batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


def get_latest_file(pattern):
    list_of_files = glob.glob(pattern)  # * means all if need specific format then *.csv
    assert len(list_of_files) > 0, "No files found: " + pattern
    return max(list_of_files, key=os.path.getctime)


def md5sum(fname):
    """ Computes mdp checksum of a file """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


class ODST(ModuleWithInit):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights
        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        self.response = nn.Parameter(
            torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True
        )
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(
            torch.zeros([in_features, num_trees, depth]), requires_grad=True
        )
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(
                torch.float32
            )
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(
                *input.shape[:-1], -1
            )
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(
            -self.log_temperatures
        )

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn(
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                "You can do so manually before training. Use with torch.no_grad() for memory efficiency."
            )
        with torch.no_grad():
            feature_selectors = self.choice_function(
                self.feature_selection_logits, dim=0
            )
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(feature_values.flatten(1, 2).t()),
                        percentiles_q.flatten(),
                    )
                ),
                dtype=feature_values.dtype,
                device=feature_values.device,
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(
                check_numpy(abs(feature_values - self.feature_thresholds)),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(
                torch.as_tensor(temperatures) + eps
            )

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__,
            self.feature_selection_logits.shape[0],
            self.num_trees,
            self.depth,
            self.tree_dim,
            self.flatten_output,
        )


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        input_dim,
        layer_dim,
        num_layers,
        tree_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=True,
        Module=ODST,
        **kwargs,
    ):
        layers = []
        for i in range(num_layers):
            oddt = Module(
                input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs
            )
            input_dim = min(
                input_dim + layer_dim * tree_dim, max_features or float("inf")
            )
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = (
                    min(self.max_features, layer_inp.shape[-1]) - initial_features
                )
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(
                *outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim
            )
        return outputs


class NODE(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        layer_dim=16,
        num_layers=4,
        tree_dim=1,
        depth=6,
        input_dropout=0.3,
        flatten_output=True,
        choice_function=entmax15,
        bin_function=entmoid15,
    ):
        super(NODE, self).__init__()
        self.b0 = nn.BatchNorm1d(input_dim)
        self.dense_layer = DenseBlock(
            input_dim,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=tree_dim,
            depth=depth,
            input_dropout=input_dropout,
            flatten_output=flatten_output,
            choice_function=choice_function,
            bin_function=bin_function,
        )
        self.b1 = nn.BatchNorm1d(layer_dim * num_layers * tree_dim)
        self.dropout0 = nn.Dropout(input_dropout)
        self.out_layer = nn.utils.weight_norm(
            nn.Linear(layer_dim * num_layers * tree_dim, out_dim)
        )

    def forward(self, input):
        x = self.b0(input)
        x = self.dense_layer(x)
        x = self.b1(x)
        x = self.dropout0(x)
        x = self.out_layer(x)
        return x


class TAB(nn.Module):
    def __init__(self, input_dim, out_dim, layer_dim):
        super(TAB, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dense1 = nn.utils.weight_norm(nn.Linear(input_dim, layer_dim))

        self.batch_norm2 = nn.BatchNorm1d(layer_dim)
        self.dropout2 = nn.Dropout(0.4)
        self.dense2 = nn.utils.weight_norm(nn.Linear(layer_dim, layer_dim))

        self.batch_norm3 = nn.BatchNorm1d(layer_dim)
        self.dropout3 = nn.Dropout(0.4)
        self.dense3 = nn.utils.weight_norm(nn.Linear(layer_dim, out_dim))

    def recalibrate_layer(self, layer):
        if torch.isnan(layer.weight_v).sum() > 0:
            print("recalibrate layer.weight_v")
            layer.weight_v = torch.nn.Parameter(
                torch.where(
                    torch.isnan(layer.weight_v),
                    torch.zeros_like(layer.weight_v),
                    layer.weight_v,
                )
            )
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)
        if torch.isnan(layer.weight).sum() > 0:
            print("recalibrate layer.weight")
            layer.weight = torch.where(
                torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight
            )
            layer.weight += 1e-7

    def forward(self, x):
        x = self.batch_norm1(x)
        self.recalibrate_layer(self.dense1)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        self.recalibrate_layer(self.dense2)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        self.recalibrate_layer(self.dense3)
        x = self.dense3(x)
        return x


class StackingCNN(nn.Module):
    def __init__(self, out_dim=206, kernel_size=2):
        super(StackingCNN, self).__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size)
        )
        self.batch_norm1 = nn.BatchNorm1d(out_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))

    def recalibrate_layer(self, layer):
        if torch.isnan(layer.weight_v).sum() > 0:
            print("recalibrate layer.weight_v")
            layer.weight_v = torch.nn.Parameter(
                torch.where(
                    torch.isnan(layer.weight_v),
                    torch.zeros_like(layer.weight_v),
                    layer.weight_v,
                )
            )
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)
        if torch.isnan(layer.weight).sum() > 0:
            print("recalibrate layer.weight")
            layer.weight = torch.where(
                torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight
            )
            layer.weight += 1e-7

    def forward(self, x):
        x = self.conv1(x).squeeze(-1)
        x = self.dropout1(self.batch_norm1(x))
        self.recalibrate_layer(self.dense1)
        x = self.dense1(x)
        return x
