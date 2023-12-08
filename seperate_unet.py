from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding
from torch import nn
import torch
import os

class UNetModelDown(nn.Module):
    def __init__(self, unet) -> None:
        self.unet = unet
        super().__init__()
        
    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
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
        t_emb = timestep_embedding(timesteps, self.unet.model_channels, repeat_only=False)
        emb = self.unet.time_embed(t_emb)

        if self.unet.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.unet.label_emb(y)

        h = x.type(self.unet.dtype)
        for module in self.unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.unet.middle_block(h, emb, context)

        return h, hs, emb

        
class UNetModelUp(nn.Module):
    def __init__(self, unet) -> None:
        self.unet = unet
        super().__init__()
        
    def forward(self, x, h, hs, emb, context):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.unet.predict_codebook_ids:
            return self.unet.id_predictor(h)
        else:
            return self.unet.out(h)


def export_model(model, path: str):
    model_down = UNetModelDown(model)
    model_up = UNetModelUp(model)

    path_vanilla = path + "_vanilla.onnx"
    path_down = path + "_down.onnx"
    path_up = path + "_up.onnx"

    with torch.inference_mode(), torch.autocast("cuda"):
        sample = torch.randn(2, 4, 64, 64).cuda()
        timestep = torch.randn(2).cuda()
        context = torch.randn(2, 77, 768).cuda()

        if not os.path.exists(path_vanilla):
            torch.onnx.export(
                model,
                (sample, timestep, context),
                path_vanilla,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["sample", "timestep", "encoder_hidden_states"],
                output_names=["latent"],
            )

        h, hs, emb = model_down(sample, timestep, context)

        if not os.path.exists(path_down):
            torch.onnx.export(
                model_down,
                (sample, timestep, context),
                path_down,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["sample", "timestep", "encoder_hidden_states"],
                output_names=["down", "mid", "emb"],
            )

        if not os.path.exists(path_up):
            torch.onnx.export(
                model_up,
                (sample, h, hs, emb, context),
                path_up,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["sample", "down", "mid", "emb"],
                output_names=["latent"]
            )
    
model = UNetModel(            
            use_checkpoint=False,
            use_fp16=False,
            image_size=32,
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[4, 2, 1],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_head_channels=8,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=768,
            legacy=False)
model_path = "sd15"

export_model(model, model_path)
