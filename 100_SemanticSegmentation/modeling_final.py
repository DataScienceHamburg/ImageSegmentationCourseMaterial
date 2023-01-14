#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sem_seg_dataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import seaborn as sns
import matplotlib.pyplot as plt

#%% Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Hyperparameters
EPOCHS = 5
BS = 4

#%% Instantiate Dataset and Dataloader
train_ds = SegmentationDataset(path_name='train')
train_dataloader = DataLoader(train_ds, batch_size=BS, shuffle=True)
val_ds = SegmentationDataset(path_name='val')
val_dataloader = DataLoader(val_ds, batch_size=BS, shuffle=True)

#%% HYPERPARAMETERS

model = smp.FPN(
    encoder_name='se_resnext50_32x4d', 
    encoder_weights='imagenet', 
    classes=6, 
    activation='sigmoid',
)
model.to(DEVICE)

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

#%% TRAIN MODEL
criterion = nn.CrossEntropyLoss()
# criterion = smp.losses.DiceLoss(mode='multiclass')
train_losses, val_losses = [], []

#%%
for e in range(EPOCHS):
    model.train()
    running_train_loss, running_val_loss = 0, 0
    for i, data in enumerate(train_dataloader):
        #training phase
        image_i, mask_i = data
        image = image_i.to(DEVICE)
        mask = mask_i.to(DEVICE)
        
        # reset gradients
        optimizer.zero_grad() 
        #forward
        output = model(image.float())
        
        # calc losses
        train_loss = criterion(output.float(), mask.long())

        # back propagation
        train_loss.backward()
        optimizer.step() #update weight          
        
        running_train_loss += train_loss.item()
    train_losses.append(running_train_loss) 
    
    # validation
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            image_i, mask_i = data
            image = image_i.to(DEVICE)
            mask = mask_i.to(DEVICE)
            #forward
            output = model(image.float())
            # calc losses
            val_loss = criterion(output.float(), mask.long())
            running_val_loss += val_loss.item()
    val_losses.append(running_val_loss) 
    
        
    print(f"Epoch: {e}: Train Loss: {np.median(running_train_loss)}, Val Loss: {np.median(running_val_loss)}")

#%% TRAIN LOSS
sns.lineplot(x = range(len(train_losses)), y= train_losses).set(title='Train Loss')
plt.show()
sns.lineplot(x = range(len(train_losses)), y= val_losses).set(title='Validation Loss')
plt.show()

# %% save model
torch.save(model.state_dict(), f'models/FPN_epochs_{EPOCHS}_crossentropy_state_dict.pth')