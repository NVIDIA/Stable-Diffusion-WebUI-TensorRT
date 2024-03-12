import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == "cat":
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, item)
        )

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


def aligned_adding(base, x, require_channel_alignment):
    if isinstance(x, float):
        if x == 0.0:
            return base
        return base + x

    if require_channel_alignment:
        zeros = torch.zeros_like(base)
        zeros[:, : x.shape[1], ...] = x
        x = zeros

    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]

    if xh > 1 or xw > 1:
        if base_h != xh or base_w != xw:
            # logger.info('[Warning] ControlNet finds unexpected mis-alignment in tensor shape.')
            x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")

    return base + x


class ControlUNet(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control_0=None,
        control_1=None,
        control_2=None,
        control_3=None,
        control_4=None,
        control_5=None,
        control_6=None,
        control_7=None,
        control_8=None,
        control_9=None,
        control_10=None,
        control_11=None,
        control_12=None,
        **kwargs
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        control = [
            control_0,
            control_1,
            control_2,
            control_3,
            control_4,
            control_5,
            control_6,
            control_7,
            control_8,
            control_9
        ]
        if control_12 is not None:
            control += [control_10, control_11, control_12]
        total_t2i_adapter_embedding = [0.0] * 4
        hs = []
        require_inpaint_hijack = False  # todo: 需要在每一次都确认下是佛为False
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        # encoder blocks
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = module(h, emb, context)
            t2i_injection = [2, 5, 8, 11]
            if i in t2i_injection:
                h = aligned_adding(
                    h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
                )
            hs.append(h)

        self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])

        # middle block
        h = self.middle_block(h, emb, context)
        h = aligned_adding(h, control.pop(), require_inpaint_hijack)

        # decoder blocks
        for i, module in enumerate(self.output_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = th.cat(
                [h, aligned_adding(hs.pop(), control.pop(), require_inpaint_hijack)],
                dim=1,
            )
            h = module(h, emb, context)

        # U-Net Output
        h = h.type(x.dtype)
        h = self.out(h)

        return h
