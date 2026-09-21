import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table

from video_utils import ID2LABEL, LABEL2ID


class BlinkVideoMAE(nn.Module):
    """VideoMAE with optional token-level temporal difference modeling."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        model_name,
        num_labels=2,
        use_tdm=True,
        temporal_layers=2,
        dropout=0.3,
        num_frames=16,
    ):
        super().__init__()
        self.use_tdm = use_tdm
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"
        self._keys_to_ignore_on_save = None
        self._resize_position_embeddings(num_frames)

        hidden_dim = self.config.hidden_size
        self.diff_proj = nn.Linear(hidden_dim, hidden_dim) if use_tdm else None
        if temporal_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=temporal_layers
            )
        else:
            self.temporal_encoder = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def _resize_position_embeddings(self, num_frames):
        """VideoMAE's position embeddings are a fixed (non-learned) sin-cos table sized
        for the pretraining num_frames (16). Regenerate it for a different clip length
        (e.g. the native 10-frame HUST-LEBW span) instead of crashing on a shape mismatch."""
        embeddings = self.backbone.embeddings
        pretrained_num_frames = getattr(self.config, "num_frames", 16)
        if num_frames == pretrained_num_frames:
            return
        tubelet_size = getattr(self.config, "tubelet_size", 2) or 2
        patch_embeddings = embeddings.patch_embeddings
        num_patches_per_frame = (
            (patch_embeddings.image_size[1] // patch_embeddings.patch_size[1])
            * (patch_embeddings.image_size[0] // patch_embeddings.patch_size[0])
        )
        num_patches = (num_frames // tubelet_size) * num_patches_per_frame
        embeddings.position_embeddings = get_sinusoid_encoding_table(
            num_patches, self.config.hidden_size
        )
        embeddings.num_patches = num_patches

    def _temporal_tokens(self, pixel_values, hidden_states):
        batch_size, num_tokens, hidden_dim = hidden_states.shape
        tubelet_size = getattr(self.config, "tubelet_size", 2) or 2
        t_tokens = max(1, pixel_values.shape[1] // tubelet_size)
        if num_tokens % t_tokens == 0:
            s_tokens = num_tokens // t_tokens
            return hidden_states.view(batch_size, t_tokens, s_tokens, hidden_dim).mean(dim=2)
        return hidden_states.mean(dim=1, keepdim=True)

    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values)
        x = self._temporal_tokens(pixel_values, outputs.last_hidden_state)

        if self.use_tdm and self.diff_proj is not None and x.shape[1] > 1:
            diff = x[:, 1:] - x[:, :-1]
            diff = torch.cat([diff, diff[:, -1:].clone()], dim=1)
            x = x + self.diff_proj(diff)

        if self.temporal_encoder is not None:
            x = self.temporal_encoder(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class R3D18BlinkClassifier(nn.Module):
    """Kinetics-pretrained 3D ResNet-18 baseline."""

    accepts_loss_kwargs = False

    def __init__(self, num_labels=2):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_labels)
        self.config = type(
            "Cfg", (), {"num_labels": num_labels, "problem_type": "single_label_classification"}
        )()
        self._keys_to_ignore_on_save = None

    def forward(self, pixel_values, labels=None, **kwargs):
        # VideoMAE layout is (B, T, C, H, W); R3D expects (B, C, T, H, W).
        video = pixel_values.permute(0, 2, 1, 3, 4)
        logits = self.backbone(video)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(kind, model_name, **kwargs):
    if kind == "baseline":
        return VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    if kind == "r3d18":
        return R3D18BlinkClassifier(num_labels=2)
    return BlinkVideoMAE(model_name, **kwargs)
