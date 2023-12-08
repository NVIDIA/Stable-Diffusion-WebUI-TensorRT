#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
from onnx import shape_inference
import os
from polygraphy.backend.onnx.loader import fold_constants
import tempfile
import torch
import onnx_graphsurgeon as gs

from ldm.modules.diffusionmodules.util import timestep_embedding
from torch import nn
import torch
from diffusers import ControlNetModel
from datastructures import ProfileSettings
from model_manager import CNET_MODEL_PATH

sdxl_hs = [
    *([(320, 1)]) * 3,
    *([(320, 2)]) * 1,
    *([(640, 2)]) * 2,
    *([(640, 4)]) * 1,
    *([(1280, 4)]) * 2,
]

sd1_hs = [
    *([(320, 1)]) * 3,
    *([(320, 2)]) * 1,
    *([(640, 2)]) * 2,
    *([(640, 4)]) * 1,
    *([(1280, 4)]) * 2,
    *([(1280, 8)]) * 3,
]

sd2_hs = sd1_hs


def get_unet_shape_dict(x, encoder_hidden_states, y=None):
    shape_dict = {}
    is_xl = y is not None
    hs = sdxl_hs if is_xl else sd1_hs

    bs, _, h, w = x.shape
    shape_dict["sample"] = tuple(x.shape)
    shape_dict["encoder_hidden_states"] = tuple(encoder_hidden_states.shape)
    shape_dict["timesteps"] = (bs,)
    shape_dict["y"] = (bs, 2816) if is_xl else None

    for i, (hs, s) in enumerate(hs):
        shape_dict[f"down_sample_{i}"] = (bs, hs, h // s, w // s)
    shape_dict["mid_sample"] = (bs, hs, h // s, w // s)
    shape_dict["emb"] = (bs, hs)

    return shape_dict


def get_cnet_shape_dict(x, encoder_hidden_states, y=None):
    shape_dict = {}
    is_xl = y is not None
    hs = sdxl_hs if is_xl else sd1_hs

    bs, _, h, w = x.shape
    shape_dict["sample"] = tuple(x.shape)
    shape_dict["encoder_hidden_states"] = tuple(encoder_hidden_states.shape)
    shape_dict["timesteps"] = (1,)
    shape_dict["controlnet_cond"] = (bs, 3, h * 8, w * 8)
    shape_dict["conditioning_scale"] = (1,)
    shape_dict["y"] = (bs, 2816) if is_xl else None

    for i, (hs, s) in enumerate(hs):
        shape_dict[f"down_sample_{i}"] = (bs, hs, h // s, w // s)
    shape_dict["mid_sample"] = (bs, hs, h // s, w // s)

    return shape_dict


class UNetModelSplit(nn.Module):
    def __init__(self, unet, embedding_dim, text_minlen=77, is_xl=False) -> None:
        super().__init__()
        self.unet = unet
        self.is_xl = is_xl
        self.encoder = UNetModelDown(unet, is_xl=is_xl)
        self.decoder = UNetModelUp(unet, is_xl=is_xl)

        self.text_minlen = text_minlen
        self.embedding_dim = embedding_dim
        self.num_xl_classes = 2816  # Magic number for num_classes

        self.dyn_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B", 1: "77N"},
            "timesteps": {0: "2B"},
            "mid_sample": {0: "2B", 2: "H/8", 3: "W/8"},
            "emb": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
            "y": {0: "2B"},
        }
        for i in range(len(self.unet.input_blocks)):
            self.dyn_axes[f"down_sample_{i}"] = {0: "2B", 2: f"H_{i}", 3: f"W_{i}"}

        self.hs = sdxl_hs if is_xl else sd1_hs
        self.emb_chn = 1280

    def get_encoder_input_names(self):
        names = ["sample", "timesteps", "encoder_hidden_states"]
        if self.is_xl:
            names.append("y")
        return names

    def get_encoder_output_names(self):
        down_samples = [f"down_sample_{i}" for i in range(len(self.unet.input_blocks))]
        return down_samples + ["mid_sample", "emb"]

    def get_encoder_dynamic_axes(self):
        io_names = self.get_encoder_input_names() + self.get_encoder_output_names()
        dyn_axes = {name: self.dyn_axes[name] for name in io_names}
        return dyn_axes

    def get_encoder_sample_input(
        self,
        batch_size,
        latent_height,
        latent_width,
        text_len,
        device="cuda",
        dtype=torch.float32,
    ):
        return (
            torch.randn(
                batch_size,
                self.unet.in_channels,
                latent_height,
                latent_width,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, dtype=dtype, device=device),
            torch.randn(
                batch_size,
                text_len,
                self.embedding_dim,
                dtype=dtype,
                device=device,
            ),
            torch.randn(batch_size, self.num_xl_classes, dtype=dtype, device=device)
            if self.is_xl
            else None,
        )

    def get_encoder_input_profile(self, profile: ProfileSettings):
        min_batch, opt_batch, max_batch = profile.get_a1111_batch_dim()
        (
            min_latent_height,
            latent_height,
            max_latent_height,
            min_latent_width,
            latent_width,
            max_latent_width,
        ) = profile.get_latent_dim()

        return {
            "sample": [
                (min_batch, self.unet.in_channels, min_latent_height, min_latent_width),
                (opt_batch, self.unet.in_channels, latent_height, latent_width),
                (max_batch, self.unet.in_channels, max_latent_height, max_latent_width),
            ],
            "timesteps": [(min_batch,), (opt_batch,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, profile.t_min, self.embedding_dim),
                (opt_batch, profile.t_opt, self.embedding_dim),
                (max_batch, profile.t_max, self.embedding_dim),
            ],
        }

    def get_decoder_input_names(self):
        input_names = ["encoder_hidden_states"]
        down_samples = self.get_encoder_output_names()
        return input_names + down_samples

    def get_decoder_output_names(self):
        return ["latent"]

    def get_decoder_dynamic_axes(self):
        io_names = self.get_decoder_input_names() + self.get_decoder_output_names()
        dyn_axes = {name: self.dyn_axes[name] for name in io_names}
        return dyn_axes

    def get_decoder_sample_input(
        self,
        batch_size,
        latent_height,
        latent_width,
        text_len,
        device="cuda",
        dtype=torch.float32,
    ):
        context = torch.randn(
                batch_size,
                text_len,
                self.embedding_dim,
                dtype=dtype,
                device=device,
        )
        emb = torch.randn(batch_size, self.emb_chn).to(dtype=dtype, device=device)

        hidden_states = []
        for i, (hs, s) in enumerate(self.hs):
            h = latent_height // s
            w = latent_width // s
            hidden_states.append(torch.randn(batch_size, hs, h, w).to(dtype=dtype, device=device))
        mid_states = torch.clone(hidden_states[-1]).to(dtype=dtype, device=device)
        return (context, hidden_states, mid_states, emb)

    def get_decoder_input_profile(self, profile: ProfileSettings):
        min_batch, opt_batch, max_batch = profile.get_a1111_batch_dim()
        (
            min_latent_height,
            latent_height,
            max_latent_height,
            min_latent_width,
            latent_width,
            max_latent_width,
        ) = profile.get_latent_dim()

        shape_dict = {}

        for i, (hs, s) in enumerate(self.hs):
            shape_dict[f"down_sample_{i}"] = [
                (min_batch, hs, min_latent_height // s, min_latent_width // s),
                (opt_batch, hs, latent_height // s, latent_width // s),
                (max_batch, hs, max_latent_height // s, max_latent_width // s),
            ]
        shape_dict["mid_sample"] = [
            (min_batch, self.emb_chn, min_latent_height // s, min_latent_width // s),
            (opt_batch, self.emb_chn, latent_height // s, latent_width // s),
            (max_batch, self.emb_chn, max_latent_height // s, max_latent_width // s),
        ]
        shape_dict["emb"] = [
            (min_batch, self.emb_chn),
            (opt_batch, self.emb_chn),
            (max_batch, self.emb_chn),
        ]

        return shape_dict

    @staticmethod
    def optimize(name, onnx_graph, verbose=False):
        opt = Optimizer(onnx_graph, verbose=verbose)
        opt.info(name + ": original")
        opt.cleanup()
        opt.info(name + ": cleanup")
        opt.fold_constants()
        opt.info(name + ": fold constants")
        opt.infer_shapes()
        opt.info(name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(name + ": finished")
        return onnx_opt_graph


class UNetModelDown(nn.Module):
    def __init__(self, unet, is_xl=False) -> None:
        super().__init__()
        self.unet = unet
        self.is_xl = is_xl

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.unet.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.unet.model_channels, repeat_only=False
        )
        emb = self.unet.time_embed(t_emb)

        if self.unet.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.unet.label_emb(y)

        h = x.type(self.unet.dtype)
        for module in self.unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.unet.middle_block(h, emb, context)

        return hs, h, emb


class UNetModelUp(nn.Module):
    def __init__(self, unet, is_xl=False) -> None:
        super().__init__()
        self.unet = unet
        self.is_xl = is_xl

    def forward(self, context, hs, h, emb):
        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(context.dtype)
        if self.unet.predict_codebook_ids:
            return self.unet.id_predictor(h)
        else:
            return self.unet.out(h)


class CNetModel(nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()

        self.model = ControlNetModel.from_pretrained(
            model_id, torch_dtype=torch.float16, local_dir=CNET_MODEL_PATH
        ).to("cuda")
        self.model.eval()

        self.text_minlen = 77
        self.embedding_dim = self.model.config["cross_attention_dim"]
        self.block_chn = self.model.config["block_out_channels"]
        self.n_down_blocks = len(self.model.controlnet_down_blocks)
        self.out_channels = self.model.config["in_channels"]

    def get_input_names(self):
        return [
            "sample",
            "timesteps",
            "encoder_hidden_states",
            "controlnet_cond",
            "conditioning_scale",
        ]

    def get_output_names(self):
        down_samples = [f"down_sample_{i}" for i in range(self.n_down_blocks)]
        return down_samples + ["mid_sample"]

    def get_dynamic_axes(self):
        dyn_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B", 1: "77N"},
            "controlnet_cond": {0: "2B", 2: "8H", 3: "8W"},
            "mid_sample": {0: "2B", 2: "H/8", 3: "W/8"},
        }
        for i in range(self.n_down_blocks):
            dyn_axes[f"down_sample_{i}"] = {0: "2B", 2: f"H_{i}", 3: f"W_{i}"}
        return dyn_axes

    def get_sample_input(self, batch_size, img_width, img_height, dtype, device):
        assert img_height % 8 == 0
        assert img_width % 8 == 0

        h = img_height // 8
        w = img_width // 8

        return {
            "sample": torch.randn(2 * batch_size, self.out_channels, h, w).to(
                dtype=dtype, device=device
            ),
            "timestep": torch.randn(1).to(
                dtype=dtype, device=device
            ),  # TODO diffusers requires timestep but rest needs to match with ldm
            "encoder_hidden_states": torch.randn(
                2 * batch_size, self.text_minlen, self.embedding_dim
            ).to(dtype=dtype, device=device),
            "controlnet_cond": torch.randn(2 * batch_size, 3, img_height, img_width).to(
                dtype=dtype, device=device
            ),
            "conditioning_scale": torch.randn(1).to(dtype=dtype, device=device),
        }

    def get_input_profile(
        self,
        profile: ProfileSettings,
    ):
        # min_batch, opt_batch, max_batch = get_batch_dim(
        #     min_batch, opt_batch, max_batch, text_optlen, text_maxlen, static_shape
        # ) # TODO proably dosent require the WAR?
        min_batch = profile.bs_min * 2
        opt_batch = profile.bs_opt * 2
        max_batch = profile.bs_max * 2

        (
            min_latent_height,
            latent_height,
            max_latent_height,
            min_latent_width,
            latent_width,
            max_latent_width,
        ) = profile.get_latent_dim()

        return {
            "sample": [
                (min_batch, self.out_channels, min_latent_height, min_latent_width),
                (opt_batch, self.out_channels, latent_height, latent_width),
                (max_batch, self.out_channels, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (min_batch, profile.t_min, self.embedding_dim),
                (opt_batch, profile.t_opt, self.embedding_dim),
                (max_batch, profile.t_max, self.embedding_dim),
            ],
            "controlnet_cond": [
                (min_batch, 3, profile.h_min, profile.w_min),
                (opt_batch, 3, profile.h_opt, profile.w_opt),
                (max_batch, 3, profile.h_max, profile.w_max),
            ],
            "timesteps": [(1,), (1,), (1,)],
            "conditioning_scale": [(1,), (1,), (1,)],
        }

    @staticmethod
    def list_cnet_models():
        controls = [
            "canny",
            "depth",
            "hed",
            "mlsd",
            "normal",
            "openpose",
            "scribble",
            "seg",
        ]
        return controls


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"""{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, 
                {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"""
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(
            gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True
        )
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, "model.onnx")
            onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
            onnx.save_model(
                onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph
