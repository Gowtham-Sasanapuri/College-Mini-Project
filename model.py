import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from torchvision.transforms import v2
import cv2
import os
import numpy as np

class ViolenceCNNPoseModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()

        # -------- Enhanced Visual Backbone (ResNet34 for better features) --------
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_dim = 512

        # Strategic freezing: Train layer3 and layer4 for better adaptation
        for name, param in self.cnn.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # -------- Enhanced Visual Temporal Modeling --------
        self.visual_lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,  # Deeper for better temporal understanding
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Visual Attention: Focus on important frames
        self.visual_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # -------- Enhanced Pose Branch --------
        self.pose_mlp = nn.Sequential(
            nn.Linear(34, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.pose_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,  # Deeper for better temporal patterns
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Pose Attention: Focus on critical movements
        self.pose_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # -------- Motion Feature Extraction --------
        # Captures frame differences (optical flow approximation)
        self.motion_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # -------- Enhanced Fusion --------
        fusion_dim = (256 * 2) + (128 * 2) + 128  # visual + pose + motion

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, video, pose):
        """
        Args:
            video: [B, 3, T, H, W] - Batch of video sequences
            pose:  [B, T, 34] - Batch of pose sequences (17 keypoints x 2 coords)
        
        Returns:
            logits: [B, num_classes] - Classification logits
        """
        B, C, T, H, W = video.shape

        # -------- Visual Branch with Attention --------
        video_frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        feat = self.cnn(video_frames)  # [B*T, 512, 1, 1]
        feat = feat.view(B, T, 512)

        # LSTM for temporal modeling
        vis_seq, _ = self.visual_lstm(feat)  # [B, T, 512]
        
        # Attention mechanism: weight frames by importance
        attn_weights = torch.softmax(self.visual_attention(vis_seq), dim=1)  # [B, T, 1]
        vis_feat = (vis_seq * attn_weights).sum(dim=1)  # [B, 512] - weighted sum instead of last timestep

        # -------- Pose Branch with Attention --------
        # Reshape for BatchNorm (needs [B*T, features])
        pose_reshaped = pose.reshape(B * T, 34)
        pose_feat = self.pose_mlp(pose_reshaped)
        pose_feat = pose_feat.view(B, T, 128)
        
        # LSTM for temporal pose patterns
        pose_seq, _ = self.pose_lstm(pose_feat)  # [B, T, 256]
        
        # Attention mechanism: weight poses by importance
        pose_attn_weights = torch.softmax(self.pose_attention(pose_seq), dim=1)  # [B, T, 1]
        pose_feat = (pose_seq * pose_attn_weights).sum(dim=1)  # [B, 256] - weighted sum

        # -------- Motion Features (Optical Flow Approximation) --------
        # Compute frame differences to detect rapid movements
        frame_diffs = video[:, :, 1:] - video[:, :, :-1]  # [B, 3, T-1, H, W]
        motion_feat = self.motion_conv(frame_diffs.mean(dim=2))  # [B, 128, 1, 1]
        motion_feat = motion_feat.view(B, -1)  # [B, 128]

        # -------- Multi-Modal Fusion --------
        fused = torch.cat([vis_feat, pose_feat, motion_feat], dim=1)  # [B, fusion_dim]

        return self.classifier(fused)