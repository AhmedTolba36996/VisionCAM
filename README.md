# VisionCAM
![example](https://github.com/AhmedTolba36996/VisionCAM/assets/55206978/e697d656-6c46-4e72-b963-7ec93ad13b8f)

# Code Example
```
# Import libraries
from visioncam.xgradcam import XgradCAM
import keras
from keras.applications import Xception
import matplotlib.pyplot as plt

# Build Your Model
model = Xception(weights="imagenet")

# Preapare Images
image = XgradCAM.get_img_array(img_path = '/content/sample.jpg',target_size = (299,299,3) )

# Model Predcition
preds = model.predict(image)
class_idx = np.argmax(preds[0])

# Define index for your classes
class_index = {'tiger_cat': 282, 'beagle': 162 ,'african_elephant' : 386 , 'bird' : 10 }
```
# Set your CAM extractor
```
# For All Methods
grad_cam = XgradCAM(model ,class_index['tiger_cat'] )

# For AblationCAM
last_conv_l_name = "block14_sepconv2_act"
classifier_l_names = ["avg_pool","predictions"]
grad_cam = AblationCAM(model,last_conv_layer_name = last_conv_l_name, classifier_layer_names = classifier_l_names )
```


If you want to visualize your heatmap, you only need to cast the CAM to a numpy ndarray:
```
preprocess_input = keras.applications.xception.preprocess_input
img_array = preprocess_input(image)
grad_heatmap = grad_cam.compute_cam_features(img_array)
plt.imshow(grad_heatmap)
```

![heatmap](https://github.com/AhmedTolba36996/VisionCAM/assets/55206978/4bf5de7e-d906-462c-baaa-a9bc0e6bebe8)

For display heatmap on image
```
XgradCAM.save_and_display_gradcam('/content/sample.jpg', grad_heatmap)
```
![XgradCAM](https://github.com/AhmedTolba36996/VisionCAM/assets/55206978/c811f8ff-a3d3-428d-887f-3eacf4e955d6)
