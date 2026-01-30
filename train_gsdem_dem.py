import os
import sys
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn import metrics
import h5py
from torch.optim import lr_scheduler
import random

from models.network_gsdem import GSDEM, DomainClassifier
from stop_utils import EarlyStopping


# ========== Configuration Options ==========
USE_PRETRAINED = True # Set to False for full training, True to load pretrained model
FINETUNE_MODE = True   # Set to False for pretraining (land only), True for fine-tuning (mixed land/ocean)
USE_DANN = False       # Set to True to enable domain adaptation (DANN), False to disable
FREEZE_LAYERS = True  # Set to True to freeze RSTB layers during fine-tuning, False to train all layers
FREEZE_RSTB_COUNT = 1  # Number of RSTB layers to freeze from the beginning (0-4, e.g. 1=freeze first RSTB only)
# ==========================================

n_epochs = 100
batch_size = 32
patience = 100

# ========== Data Path Configuration ==========
# For pretraining mode (FINETUNE_MODE=False)
train_dir = "your path"
valid_dir = "your path"

# For fine-tuning mode (FINETUNE_MODE=True)
if FINETUNE_MODE:
    LAND_TRAIN_H5 = "your path"
    OCEAN_TRAIN_H5 = "your path"
    OCEAN_VALID_H5 = "your path"
    LAND_OCEAN_RATIO = 0.0  # Adjustable: land ratio (0.0-1.0)
# ============================================


class SingleChannelDataset(Dataset):
    """Dataset for single-channel DEM data from HDF5"""
    def __init__(self, h5_path, dataset_key_hr='480_480', dataset_key_lr='32_32'):
        self.h5_path = h5_path
        self.dataset_key_hr = dataset_key_hr
        self.dataset_key_lr = dataset_key_lr

        with h5py.File(h5_path, 'r') as f:
            self.size = f[dataset_key_hr].shape[0]

        self.h5_file = None

    def _open_h5(self):
        """Lazy open H5 file (in worker process)"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._open_h5()

        data_hr = torch.from_numpy(self.h5_file[self.dataset_key_hr][idx, :, :, :]).float()
        data_lr = torch.from_numpy(self.h5_file[self.dataset_key_lr][idx, :, :, :]).float()
        domain_label = 0  # Default label for pretraining mode (not used)

        return data_hr, data_lr, domain_label


class MixedDataset(Dataset):
    """Dataset mixing land and ocean DEM data"""
    def __init__(self, land_h5_path, ocean_h5_path, land_ratio=0.5):
        """
        Args:
            land_h5_path: Path to land training H5
            ocean_h5_path: Path to ocean training H5
            land_ratio: Proportion of land data (0.0-1.0)
        """
        self.land_h5_path = land_h5_path
        self.ocean_h5_path = ocean_h5_path
        self.land_ratio = land_ratio

        with h5py.File(land_h5_path, 'r') as f:
            self.land_size = f['480_480'].shape[0]

        with h5py.File(ocean_h5_path, 'r') as f:
            self.ocean_size = f['480_480'].shape[0]

        self.size = int(self.land_size * self.land_ratio + self.ocean_size * (1 - self.land_ratio))
        self.land_file = None
        self.ocean_file = None

    def _open_h5(self):
        """Lazy open H5 files (in worker process)"""
        if self.land_file is None:
            self.land_file = h5py.File(self.land_h5_path, 'r')

        if self.ocean_file is None:
            self.ocean_file = h5py.File(self.ocean_h5_path, 'r')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._open_h5()

        if random.random() < self.land_ratio:
            # Land data
            land_idx = idx % self.land_size
            data_hr = torch.from_numpy(self.land_file['480_480'][land_idx, :, :, :]).float()
            data_lr = torch.from_numpy(self.land_file['32_32'][land_idx, :, :, :]).float()
            domain_label = 0  # Land label
        else:
            # Ocean data
            ocean_idx = idx % self.ocean_size
            data_hr = torch.from_numpy(self.ocean_file['480_480'][ocean_idx, :, :, :]).float()
            data_lr = torch.from_numpy(self.ocean_file['32_32'][ocean_idx, :, :, :]).float()
            domain_label = 1  # Ocean label

        return data_hr, data_lr, domain_label




os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize GSDEM model for DEM (single channel, 15x upsampling)
model = GSDEM(
    upscale=15,
    in_chans=1,
    img_size=32,
    window_size=8,
    img_range=1.,
    depths=[12, 12, 12, 12],
    embed_dim=180,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv',
    use_gated_mlp_in_last_layer=True
).to(device)

cudnn.benchmark = True
print(model)
model = nn.DataParallel(model)

# Initialize domain classifier (if using DANN)
if USE_DANN and FINETUNE_MODE:
    domain_classifier = DomainClassifier(feature_dim=180).to(device)
    domain_classifier = nn.DataParallel(domain_classifier)
    print("✓ Domain Classifier initialized")
else:
    domain_classifier = None

# Load pretrained model if specified
if USE_PRETRAINED:
    print("Loading pretrained checkpoint...")
    checkpoint_path = "your path"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("✓ Pretrained model loaded")

        # Freeze layers during fine-tuning if requested
        if FINETUNE_MODE and FREEZE_LAYERS and FREEZE_RSTB_COUNT > 0:
            print(f"Freezing first {FREEZE_RSTB_COUNT} RSTB layer(s) for fine-tuning...")
            # Freeze conv_first
            for param in model.module.conv_first.parameters():
                param.requires_grad = False
            # Freeze first N RSTB layers
            for i in range(FREEZE_RSTB_COUNT):
                for param in model.module.layers[i].parameters():
                    param.requires_grad = False
            # Freeze conv_after_body
            for param in model.module.conv_after_body.parameters():
                param.requires_grad = False
            trainable_rstb_idx = FREEZE_RSTB_COUNT
            print(f"✓ Frozen: conv_first + layers[0-{FREEZE_RSTB_COUNT-1}] + conv_after_body")
            print(f"✓ Trainable: layers[{trainable_rstb_idx}-3] + upsampler + conv_last")
        elif FINETUNE_MODE:
            print("✓ No layers frozen - training all parameters")
    else:
        print("✗ Checkpoint not found, starting from scratch")
        USE_PRETRAINED = False
else:
    print("✓ Training from scratch (no pretrained model)")

# ========== Setup training and validation datasets ==========
if FINETUNE_MODE:
    print(f"Fine-tuning mode: {LAND_OCEAN_RATIO*100:.0f}% land, {(1-LAND_OCEAN_RATIO)*100:.0f}% ocean")
    trainset = MixedDataset(LAND_TRAIN_H5, OCEAN_TRAIN_H5, land_ratio=LAND_OCEAN_RATIO)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=48, pin_memory=True, prefetch_factor=32, persistent_workers=True)

    # Validation on ocean data
    validset = SingleChannelDataset(OCEAN_VALID_H5)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8)
else:
    print("Pretraining mode: Land data only")
    trainset = SingleChannelDataset(train_dir)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=48, pin_memory=True, prefetch_factor=32, persistent_workers=True)

    validset = SingleChannelDataset(valid_dir)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8)
# ============================================================

# Choose loss function based on training mode
if FINETUNE_MODE:
    # Micro-tuning stage: use standard L1 loss
    criterion = nn.L1Loss()
    print("✓ Using L1Loss for fine-tuning")
else:
    # Pretraining stage: use standard L1 loss
    criterion = nn.L1Loss()
    print("✓ Using L1Loss for pretraining")

domain_criterion = nn.BCELoss()  # DANN domain classification loss
optimizer = optim.AdamW(model.parameters(), lr=0.00001, betas=(0.9, 0.999))

# If using DANN, create optimizer for domain classifier
if USE_DANN and FINETUNE_MODE:
    domain_optimizer = optim.AdamW(domain_classifier.parameters(), lr=0.00001, betas=(0.9, 0.999))
else:
    domain_optimizer = None

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    verbose=False,
    threshold=0.000001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08
)


def train_model(model, batch_size, patience, n_epochs):
    train_losses = []
    R2_all = []
    avg_r2_epoch = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        if domain_classifier is not None:
            domain_classifier.train()

        for batch, data in enumerate(trainloader):
            if FINETUNE_MODE and USE_DANN:
                # Fine-tuning mode: data includes domain_label
                data_hr, data_lr, domain_label = data
                domain_label = domain_label.float().to(device).view(-1, 1)
            else:
                # Pretraining mode: data doesn't include domain_label
                data_hr, data_lr, _ = data

            data_hr = data_hr.to(device)
            data_lr = data_lr.to(device)

            # Data from H5 is already normalized
            HR = Variable(data_hr)

            optimizer.zero_grad()

            # Forward pass
            pred = model(Variable(data_lr))
            loss_sr = criterion(HR, pred)

            # ========== DANN Loss (only in fine-tuning mode with USE_DANN=True) ==========
            if FINETUNE_MODE and USE_DANN:
                # Extract features for domain classification
                with torch.no_grad():
                    features = model.module.extract_features(Variable(data_lr))
                features = features.detach()

                # Domain classifier prediction
                domain_pred = domain_classifier(features)

                # Domain classification loss
                loss_domain = domain_criterion(domain_pred, domain_label)

                # Adversarial loss (flip labels to confuse domain classifier)
                adversarial_label = 1.0 - domain_label
                loss_adversarial = domain_criterion(domain_pred, adversarial_label)

                # Total loss
                loss = loss_sr + 0.1 * loss_domain + 0.01 * loss_adversarial

                # Optimize both networks
                loss.backward()
                optimizer.step()
                domain_optimizer.zero_grad()

                # Domain classifier backprop (standard domain classification, not adversarial)
                domain_pred_for_domain = domain_classifier(features.detach())
                loss_domain_only = domain_criterion(domain_pred_for_domain, domain_label)
                loss_domain_only.backward()
                domain_optimizer.step()
            else:
                # Standard training (SR loss only)
                loss = loss_sr
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            sys.stdout.write('\r[%d/%d][%d/%d] Loss: %.4f' % (epoch, n_epochs, batch, len(trainloader), loss.item()))

        scheduler.step(np.average(train_losses))

        ######################
        # validate the model #
        ######################
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for i, data in enumerate(validloader):
                # SingleChannelDataset now always returns 3 values
                data_hr, data_lr, _ = data

                data_hr = data_hr.to(device)
                data_lr = data_lr.to(device)

                HR = Variable(data_hr)
                output = model(Variable(data_lr))
                loss = criterion(output, HR)
                valid_losses.append(loss.item())

        valid_loss = np.average(valid_losses)
        model.train()

        # print training/validation statistics
        train_loss = np.average(train_losses)
        r2_epoch = np.average(R2_all) if len(R2_all) > 0 else 0

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_r2_epoch.append(r2_epoch)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        print('train_loss:', train_loss)
        print('valid_loss:', valid_loss)

        # clear lists to track next epoch
        train_losses = []
        R2_all = []
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return avg_train_losses, avg_valid_losses, avg_r2_epoch


if __name__ == '__main__':
    print("\n========== GSDEM DEM Super-Resolution Training ==========")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: 0.00001")
    print(f"Number of epochs: {n_epochs}")
    print(f"Patience: {patience}")
    print("========================================================\n")

    avg_train_losses, avg_valid_losses, avg_r2_epoch = train_model(model, batch_size, patience, n_epochs)

    # Save loss curves to H5 file (filename includes freeze config)
    if FREEZE_LAYERS:
        h5_filename = f'avg_train_Ocean_losses_freeze{FREEZE_RSTB_COUNT}.h5'
    else:
        h5_filename = 'avg_train_Ocean_losses_nofreeze.h5'

    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('avg_train_losses', data=avg_train_losses)
        f.create_dataset('avg_valid_losses', data=avg_valid_losses)
        f.create_dataset('avg_r2_epoch', data=avg_r2_epoch)
    print(f"✓ Loss curves saved to {h5_filename}")

