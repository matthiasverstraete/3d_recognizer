# RandLA-Net Point Cloud Semantic Segmentation

[RandLA-Net](https://arxiv.org/abs/1911.11236) is an efficient semantic segmentation network for large-scale point clouds. As input, it accepts 3D point clouds, possibly augmented with features such as colors, and returns a class index for each point in the point cloud.

## Analysis

An extensive analysis of the netwerk has been performed. The results are summarized on [Confluence](https://robovision.atlassian.net/wiki/spaces/2040VLAIOH/pages/2742026263/Point+Cloud+Semantic+Segmentation).

## Common usage

Training, evaluating and predicting of RandLA-Net is performed on a `Model` instance:

```python
from deep_learning.randlanet import (
    Model,
    RandLANetSettings,
    AugmentationSettings,
    TrainingSettings,
)

# define model with default settings
model_settings = RandLANetSettings()
model = Model(model_settings)

# train model with default settings
augmentation_settings = AugmentationSettings()
training_settings = TrainingSettings()
log_dir = "training_dir"
class_names = ["main_stem", "leaf", "side_branch"]

# dataset_train and dataset_validation are Sequences return points (N, 3), features (N, F), labels (N,) as numpy arrays
model.train(
    dataset_train,
    dataset_validation,
    log_dir,
    training_settings,
    augmentation_settings,
    class_names,
)
model.save("/path/to/my_trained_model")

# load model
model = Model.load("/path/to/my_model")

# evaluate model
metrics = model.evaluate(dataset_test)

# predict on model
predictions = model.toggle_predict(xyz)
```
