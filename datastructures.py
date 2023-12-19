from dataclasses import dataclass
from enum import Enum


@dataclass
class UNetEngineArgs:
    idx: int
    hr_idx: int = None
    lora: dict = None
    controlnets: dict = None


class SDVersion(Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    Unknown = -1

    def __str__(self):
        return self.name

    @classmethod
    def from_str(cls, str):
        try:
            return cls[str]
        except KeyError:
            return cls.Unknown

    def match(self, sd_model):
        if sd_model.is_sd1 and self == SDVersion.SD1:
            return True
        elif sd_model.is_sd2 and self == SDVersion.SD2:
            return True
        elif sd_model.is_sdxl and self == SDVersion.SDXL:
            return True
        elif self == SDVersion.Unknown:
            return True
        else:
            return False


class ModelType(Enum):
    UNET = 0
    CONTROLNET = 1
    LORA = 2
    UNDEFINED = -1

    @classmethod
    def from_string(cls, s):
        return getattr(cls, s.upper(), None)

    def __str__(self):
        return self.name.lower()


@dataclass
class ProfileSettings:
    bs_min: int
    bs_opt: int
    bs_max: int
    h_min: int
    h_opt: int
    h_max: int
    w_min: int
    w_opt: int
    w_max: int
    t_min: int
    t_opt: int
    t_max: int
    static_shape: bool = False

    def __str__(self) -> str:
        return "Batch Size: {}-{}-{}\nHeight: {}-{}-{}\nWidth: {}-{}-{}\nToken Count: {}-{}-{}".format(
            self.bs_min,
            self.bs_opt,
            self.bs_max,
            self.h_min,
            self.h_opt,
            self.h_max,
            self.w_min,
            self.w_opt,
            self.w_max,
            self.t_min,
            self.t_opt,
            self.t_max,
        )

    def out(self):
        return (
            self.bs_min,
            self.bs_opt,
            self.bs_max,
            self.h_min,
            self.h_opt,
            self.h_max,
            self.w_min,
            self.w_opt,
            self.w_max,
            self.t_min,
            self.t_opt,
            self.t_max,
        )

    def token_to_dim(self, static_shapes: bool):
        self.t_min = (self.t_min // 75) * 77
        self.t_opt = (self.t_opt // 75) * 77
        self.t_max = (self.t_max // 75) * 77

        if static_shapes:
            self.t_min = self.t_max = self.t_opt
            self.bs_min = self.bs_max = self.bs_opt
            self.h_min = self.h_max = self.h_opt
            self.w_min = self.w_max = self.w_opt
            self.static_shape = True

    def get_latent_dim(self):
        return (
            self.h_min // 8,
            self.h_opt // 8,
            self.h_max // 8,
            self.w_min // 8,
            self.w_opt // 8,
            self.w_max // 8,
        )

    def get_a1111_batch_dim(self):
        static_batch = self.bs_min == self.bs_max == self.bs_opt
        if self.t_max <= 77:
            return (self.bs_min * 2, self.bs_opt * 2, self.bs_max * 2)
        elif self.t_max > 77 and static_batch:
            return (self.bs_opt, self.bs_opt, self.bs_opt)
        elif self.t_max > 77 and not static_batch:
            if self.t_opt > 77:
                return (self.bs_min, self.bs_opt, self.bs_max * 2)
            return (self.bs_min, self.bs_opt * 2, self.bs_max * 2)
        else:
            raise Exception("Uncovered case in get_batch_dim")


class ProfilePrests:
    def __init__(self):
        self.profile_presets = {
            "512x512 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 512, 512, 512, 512, 512, 512, 75, 75, 75
            ),
            "768x768 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 768, 768, 768, 768, 768, 768, 75, 75, 75
            ),
            "1024x1024 | Batch Size 1 (Static)": ProfileSettings(
                1, 1, 1, 1024, 1024, 1024, 1024, 1024, 1024, 75, 75, 75
            ),
            "256x256 - 512x512 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 256, 512, 512, 256, 512, 512, 75, 75, 150
            ),
            "512x512 - 768x768 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150
            ),
            "768x768 - 1024x1024 | Batch Size 1-4": ProfileSettings(
                1, 1, 4, 768, 1024, 1024, 768, 1024, 1024, 75, 75, 150
            ),
        }
        self.default = ProfileSettings(
            1, 1, 4, 512, 512, 768, 512, 512, 768, 75, 75, 150
        )
        self.default_xl = ProfileSettings(
            1, 1, 4, 768, 1024, 1024, 768, 1024, 1024, 75, 75, 150
        )

    def get_settings_from_version(self, version):
        static = False
        if version == "Default":
            return *self.default.out(), static
        if "Static" in version:
            static = True
        return *self.profile_presets[version].out(), static

    def get_choices(self):
        return list(self.profile_presets.keys()) + ["Default"]

    def get_default(self, is_xl: bool):
        if is_xl:
            return self.default_xl
        return self.default
